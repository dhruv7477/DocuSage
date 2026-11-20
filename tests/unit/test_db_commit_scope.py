"""
tests/unit/test_db_commit_scope.py
──────────────────────────────────
`get_db` commits after the endpoint. With scope="function" FastAPI runs that
commit before it sends the response; with the default "request" scope only
afterwards, so a client's next request can miss the row (a new chat that
"does not exist" when the client streams into it). Every route uses the
function scope except the streaming chat endpoint, whose session must stay
open while it streams.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi import Depends, FastAPI
from fastapi.routing import APIRoute

from src.db import session as db_session
from src.db.session import get_db

pytestmark = pytest.mark.unit


class _RecordingSession:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def commit(self) -> None:
        self._events.append("commit")

    async def rollback(self) -> None:
        self._events.append("rollback")

    async def close(self) -> None:
        pass


async def _order_of_commit_and_response(monkeypatch, scope: str | None) -> list[str]:
    events: list[str] = []

    @asynccontextmanager
    async def fake_session_factory() -> AsyncIterator[_RecordingSession]:
        yield _RecordingSession(events)

    monkeypatch.setattr(db_session, "AsyncSessionLocal", fake_session_factory)
    depends = Depends(get_db, scope=scope) if scope else Depends(get_db)
    app = FastAPI()

    @app.post("/things")
    async def create_thing(db=depends) -> dict:
        return {"ok": True}

    async def recording_app(asgi_scope, receive, send):
        async def recording_send(message):
            if message["type"] == "http.response.start":
                events.append("response")
            await send(message)

        await app(asgi_scope, receive, recording_send)

    transport = httpx.ASGITransport(app=recording_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.post("/things")).status_code == 200
    return events


async def test_function_scope_commits_before_the_response(monkeypatch) -> None:
    assert await _order_of_commit_and_response(monkeypatch, "function") == [
        "commit",
        "response",
    ]


async def test_request_scope_commits_only_after_the_response(monkeypatch) -> None:
    # The FastAPI behaviour the function scope avoids.
    assert await _order_of_commit_and_response(monkeypatch, None) == [
        "response",
        "commit",
    ]


def _get_db_scopes(dependant) -> list[str | None]:
    """The scope of every get_db dependency below `dependant`."""
    found = []
    for sub in dependant.dependencies:
        if sub.call is get_db:
            found.append(sub.scope)
        found.extend(_get_db_scopes(sub))
    return found


def _api_routes(routes) -> list[APIRoute]:
    """Every APIRoute, including those of included routers."""
    found = []
    for route in routes:
        if isinstance(route, APIRoute):
            found.append(route)
        nested = getattr(route, "original_router", None)  # an included router
        if nested is not None:
            found.extend(_api_routes(nested.routes))
    return found


@pytest.fixture(scope="module")
def routes() -> list[APIRoute]:
    from src.main import create_app

    found = _api_routes(create_app().routes)
    assert len(found) > 30, "the route walk found too few routes"
    return found


def test_routes_commit_before_responding(routes: list[APIRoute]) -> None:
    from src.api.v1.chat import stream_query

    late = {
        route.name
        for route in routes
        if route.endpoint is not stream_query
        and any(scope != "function" for scope in _get_db_scopes(route.dependant))
    }
    assert not late, f"routes that commit after responding: {sorted(late)}"


def test_the_stream_keeps_its_session_open_while_it_streams(
    routes: list[APIRoute],
) -> None:
    from src.api.v1.chat import stream_query

    stream = next(r for r in routes if r.endpoint is stream_query)
    direct = [d.scope for d in stream.dependant.dependencies if d.call is get_db]
    assert direct == [None]  # the default "request" scope
