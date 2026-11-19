"""
tests/unit/test_app_middleware.py
─────────────────────────────────
The app's middleware order (CORS outermost) and its shutdown steps. The
app is built without running its lifespan, so no service is contacted.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from slowapi.middleware import SlowAPIMiddleware

from src import main
from src.core.audit import AuditMiddleware
from src.db import session

pytestmark = pytest.mark.unit

ORIGIN = main.settings.allowed_origins[0]


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.create_app())  # no lifespan: no service probes


def test_cors_is_the_outermost_middleware() -> None:
    # Starlette puts the middleware added last first in user_middleware,
    # and that one runs first.
    order = [m.cls for m in main.create_app().user_middleware]
    assert order == [CORSMiddleware, AuditMiddleware, SlowAPIMiddleware]


def test_preflight_is_answered_with_cors_headers(client: TestClient) -> None:
    response = client.options(
        "/api/v1/auth/login",
        headers={"Origin": ORIGIN, "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ORIGIN
    assert response.headers["access-control-allow-credentials"] == "true"


def test_audited_response_has_cors_and_request_id_headers(client: TestClient) -> None:
    # An unmapped POST is still audited ("api.post"); the 404 passes back
    # through the audit layer and then the CORS layer.
    response = client.post("/api/v1/no-such-route", headers={"Origin": ORIGIN})
    assert response.status_code == 404
    assert response.headers["access-control-allow-origin"] == ORIGIN
    assert response.headers["x-request-id"]


def test_other_origins_get_no_cors_headers(client: TestClient) -> None:
    response = client.post(
        "/api/v1/no-such-route", headers={"Origin": "https://evil.example"}
    )
    assert "access-control-allow-origin" not in response.headers


async def test_shutdown_cancels_the_warmup_and_closes_the_pool(monkeypatch) -> None:
    engine = SimpleNamespace(dispose=AsyncMock())
    monkeypatch.setattr(session, "engine", engine)
    warmup = asyncio.create_task(asyncio.sleep(3600))
    app = SimpleNamespace(state=SimpleNamespace(warmup_task=warmup))

    await main._shutdown(app)

    assert warmup.cancelled()
    engine.dispose.assert_awaited_once()


async def test_shutdown_without_a_running_warmup_still_closes_the_pool(
    monkeypatch,
) -> None:
    engine = SimpleNamespace(dispose=AsyncMock())
    monkeypatch.setattr(session, "engine", engine)

    await main._shutdown(SimpleNamespace(state=SimpleNamespace()))

    engine.dispose.assert_awaited_once()
