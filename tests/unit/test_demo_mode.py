"""Unit tests for demo mode: the public info endpoint and the new-user hook."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.v1.demo import router as demo_router
from src.configs.config import get_settings
from src.schemas.auth import RegisterRequest
from src.services.auth_service import AuthService

pytestmark = pytest.mark.unit


def _demo_client() -> TestClient:
    app = FastAPI()
    app.include_router(demo_router, prefix="/api/v1")
    return TestClient(app)


def test_demo_info_reports_limits_and_llm_host(monkeypatch: pytest.MonkeyPatch) -> None:
    s = get_settings()
    for name, value in {
        "demo_mode": True,
        "quota_max_documents": 5,
        "quota_max_upload_mb": 25,
        "quota_questions_per_day": 30,
        "demo_wipe_hour_utc": 3,
        "llm_provider": "openai",
        "openai_base_url": "https://api.groq.com/openai/v1",
    }.items():
        monkeypatch.setattr(s, name, value)

    body = _demo_client().get("/api/v1/demo").json()

    assert body == {
        "enabled": True,
        "max_documents": 5,
        "max_upload_mb": 25,
        "questions_per_day": 30,
        "wipe_hour_utc": 3,
        "llm_host": "api.groq.com",
    }


def test_demo_info_when_demo_mode_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "demo_mode", False)
    monkeypatch.setattr(get_settings(), "llm_provider", "ollama")

    body = _demo_client().get("/api/v1/demo").json()

    assert body["enabled"] is False
    assert body["wipe_hour_utc"] is None
    assert body["llm_host"] is None


async def test_register_calls_the_new_user_hook() -> None:
    db = MagicMock()
    db.flush = AsyncMock()
    db.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )
    created: list[uuid.UUID] = []
    service = AuthService(db, on_user_created=created.append)

    async def flush() -> None:
        db.add.call_args.args[0].id = uuid.uuid4()

    db.flush.side_effect = flush
    user = await service.register(
        RegisterRequest(
            email="new@example.com", password="NewPass123!", full_name="New"
        )
    )

    assert created == [user.id]
