"""
tests/integration/test_abuse_controls.py
────────────────────────────────────────
Rate limits, per-user quotas and the access rules a public demo relies
on, through the real API against a real Postgres. Storage is in memory;
the LLM, embedder, vector store and Celery are mocked.
"""

import socket
import uuid
from collections.abc import Iterator
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.configs.config import get_settings
from src.core.rate_limit import limiter
from src.db.session import get_db
from src.services.document_service import DocumentService
from src.services.quota_service import DEMO_DAILY_LIMIT_MESSAGE, QuotaService
from tests.fakes import InMemoryStorage

PASSWORD = "AbusePass123!"


def _postgres_reachable() -> bool:
    s = get_settings()
    try:
        with socket.create_connection((s.postgres_host, s.postgres_port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _postgres_reachable(), reason="needs a live Postgres"),
]


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    from src.api.v1.chat import _get_orchestrator
    from src.api.v1.documents import _get_service
    from src.main import create_app

    app = create_app()
    storage = InMemoryStorage()

    def _documents(db: AsyncSession = Depends(get_db)) -> DocumentService:
        return DocumentService(db, storage, QuotaService(db))

    app.dependency_overrides[_get_service] = _documents
    app.dependency_overrides[_get_orchestrator] = lambda: MagicMock()
    with (
        patch("src.workers.tasks.ingest_document_task"),
        patch("src.vector_store.factory.get_vector_store", return_value=MagicMock()),
        TestClient(app, raise_server_exceptions=False) as c,
    ):
        yield c


@pytest.fixture
def limits_on() -> Iterator[None]:
    """Turn the limiter on with an empty counter store for one test."""
    limiter.reset()
    limiter.enabled = True
    yield
    limiter.enabled = False
    limiter.reset()


@pytest.fixture
def settings_override(monkeypatch: pytest.MonkeyPatch):
    """Set fields on the shared settings object for one test."""

    def _set(**values: object) -> None:
        for name, value in values.items():
            monkeypatch.setattr(get_settings(), name, value)

    return _set


# ── Helpers ───────────────────────────────────────────────────────


def _signup(c: TestClient) -> dict[str, str]:
    email = f"abuse-{uuid.uuid4().hex[:10]}@example.com"
    r = c.post(
        "/api/v1/auth/register",
        json={"email": email, "password": PASSWORD, "full_name": "Abuse Test"},
    )
    assert r.status_code == 201, r.text
    token = c.post(
        "/api/v1/auth/login", json={"email": email, "password": PASSWORD}
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}", "email": email}


def _headers(user: dict[str, str]) -> dict[str, str]:
    return {"Authorization": user["Authorization"]}


def _workspace(c: TestClient, user: dict[str, str]) -> str:
    r = c.post("/api/v1/workspaces", json={"name": "abuse"}, headers=_headers(user))
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _upload(c: TestClient, user: dict[str, str], ws: str, name: str, size: int = 64):
    return c.post(
        "/api/v1/documents/upload",
        data={"workspace_id": ws},
        files={"file": (name, BytesIO(b"x" * size), "text/plain")},
        headers=_headers(user),
    )


def _sql(statement: str, **params: object) -> object:
    """Run one statement on its own connection; return the first column, if any."""
    engine = create_engine(get_settings().database_url_sync)
    try:
        with engine.begin() as conn:
            result = conn.execute(text(statement), params)
            return result.scalar() if result.returns_rows else None
    finally:
        engine.dispose()


def _add_questions(user: dict[str, str], count: int) -> None:
    for _ in range(count):
        _sql(
            "INSERT INTO question_usage (user_id) "
            "SELECT id FROM users WHERE email = :email",
            email=user["email"],
        )


def _questions_today(user: dict[str, str] | None = None) -> int:
    where = "asked_at >= date_trunc('day', now() AT TIME ZONE 'utc') AT TIME ZONE 'utc'"
    if user is None:
        return int(_sql(f"SELECT count(*) FROM question_usage WHERE {where}"))
    return int(
        _sql(
            "SELECT count(*) FROM question_usage q JOIN users u ON u.id = q.user_id "
            f"WHERE u.email = :email AND {where}",
            email=user["email"],
        )
    )


def _new_chat(c: TestClient, user: dict[str, str]) -> str:
    ws = _workspace(c, user)
    r = c.post("/api/v1/chat", json={"workspace_id": ws}, headers=_headers(user))
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _ask(c: TestClient, user: dict[str, str], chat_id: str):
    return c.get(
        f"/api/v1/chat/{chat_id}/stream",
        params={"query": "a question"},
        headers=_headers(user),
    )


# ── Rate limits ───────────────────────────────────────────────────


def test_login_is_limited_per_ip(
    client: TestClient, limits_on, settings_override
) -> None:
    settings_override(rate_limit_auth="3/minute")
    body = {"email": "nobody@example.com", "password": "Wrong-pass-1"}

    codes = [client.post("/api/v1/auth/login", json=body).status_code for _ in range(4)]

    assert codes == [401, 401, 401, 429]


def test_register_is_limited_per_ip(
    client: TestClient, limits_on, settings_override
) -> None:
    settings_override(rate_limit_register="2/hour")

    def register() -> int:
        email = f"bot-{uuid.uuid4().hex[:10]}@example.com"
        return client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": PASSWORD, "full_name": "bot"},
        ).status_code

    assert [register() for _ in range(3)] == [201, 201, 429]


def test_upload_limit_is_per_user(client: TestClient, settings_override) -> None:
    alice, bob = _signup(client), _signup(client)  # before the limiter is on
    ws_a, ws_b = _workspace(client, alice), _workspace(client, bob)
    settings_override(rate_limit_upload="2/hour")
    limiter.reset()
    limiter.enabled = True
    try:
        alice_codes = [
            _upload(client, alice, ws_a, f"a{i}.txt").status_code for i in range(3)
        ]
        bob_code = _upload(client, bob, ws_b, "b0.txt").status_code
    finally:
        limiter.enabled = False
        limiter.reset()

    assert alice_codes == [202, 202, 429]
    assert bob_code == 202


# ── Quotas ────────────────────────────────────────────────────────


def test_document_quota_allows_replacing_but_not_adding(
    client: TestClient, settings_override
) -> None:
    settings_override(quota_max_documents=1)
    user = _signup(client)
    ws = _workspace(client, user)

    assert _upload(client, user, ws, "first.txt").status_code == 202
    assert _upload(client, user, ws, "first.txt").status_code == 202  # re-upload
    second = _upload(client, user, ws, "second.txt")

    assert second.status_code == 429
    assert second.json()["error_code"] == "QUOTA_EXCEEDED"


def test_storage_quota_counts_total_size(client: TestClient, settings_override) -> None:
    settings_override(quota_max_upload_mb=1)
    user = _signup(client)
    ws = _workspace(client, user)

    assert _upload(client, user, ws, "half.txt", size=600 * 1024).status_code == 202
    too_much = _upload(client, user, ws, "more.txt", size=600 * 1024)

    assert too_much.status_code == 429
    assert "Storage limit" in too_much.json()["message"]


def test_daily_question_quota_blocks_the_stream_before_it_starts(
    client: TestClient, settings_override
) -> None:
    settings_override(quota_questions_per_day=2)
    user = _signup(client)
    ws = _workspace(client, user)
    chat_id = client.post(
        "/api/v1/chat", json={"workspace_id": ws}, headers=_headers(user)
    ).json()["id"]
    _add_questions(user, 2)

    stream = client.get(
        f"/api/v1/chat/{chat_id}/stream",
        params={"query": "one more?"},
        headers=_headers(user),
    )
    query = client.post(
        f"/api/v1/chat/{chat_id}/query",
        json={"query": "one more?"},
        headers=_headers(user),
    )

    for resp in (stream, query):
        assert resp.status_code == 429
        assert resp.headers["content-type"].startswith("application/json")
        assert "Daily question limit" in resp.json()["message"]


def test_deleting_chats_does_not_give_questions_back(
    client: TestClient, settings_override
) -> None:
    # Messages are deleted with their chat, so the quota counts
    # question_usage rows instead (DECISIONS D7).
    settings_override(quota_questions_per_day=1)
    user = _signup(client)
    first = _new_chat(client, user)
    assert _ask(client, user, first).status_code == 200

    assert (
        client.delete(f"/api/v1/chat/{first}", headers=_headers(user)).status_code
        == 204
    )
    refused = _ask(client, user, _new_chat(client, user))

    assert refused.status_code == 429
    assert refused.json()["error_code"] == "QUOTA_EXCEEDED"


def test_site_wide_budget_refuses_with_the_demo_message(
    client: TestClient, settings_override
) -> None:
    _sql("DELETE FROM question_usage")  # start the day's site-wide count at 0
    settings_override(demo_global_questions_per_day=2)
    first, second, third = (_signup(client) for _ in range(3))
    assert _ask(client, first, _new_chat(client, first)).status_code == 200
    assert _ask(client, second, _new_chat(client, second)).status_code == 200

    chat_id = _new_chat(client, third)
    stream = _ask(client, third, chat_id)
    query = client.post(
        f"/api/v1/chat/{chat_id}/query",
        json={"query": "a question"},
        headers=_headers(third),
    )

    for resp in (stream, query):
        assert resp.status_code == 429
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json()["error_code"] == "DEMO_DAILY_LIMIT"
        assert resp.json()["message"] == DEMO_DAILY_LIMIT_MESSAGE
    assert _questions_today() == 2


def test_admins_skip_the_site_wide_budget_but_still_count(
    client: TestClient, settings_override
) -> None:
    _sql("DELETE FROM question_usage")
    settings_override(demo_global_questions_per_day=1)
    member, admin = _signup(client), _signup(client)
    _sql("UPDATE users SET role = 'admin' WHERE email = :e", e=admin["email"])
    assert _ask(client, member, _new_chat(client, member)).status_code == 200

    assert _ask(client, admin, _new_chat(client, admin)).status_code == 200
    assert _questions_today() == 2


def test_no_usage_is_recorded_while_question_limits_are_off(
    client: TestClient, settings_override
) -> None:
    settings_override(quota_questions_per_day=0, demo_global_questions_per_day=0)
    user = _signup(client)

    assert _ask(client, user, _new_chat(client, user)).status_code == 200
    assert _questions_today(user) == 0


# ── Access rules (A9) ─────────────────────────────────────────────


def test_chat_in_a_foreign_workspace_is_refused(client: TestClient) -> None:
    owner, outsider = _signup(client), _signup(client)
    ws = _workspace(client, owner)

    r = client.post(
        "/api/v1/chat", json={"workspace_id": ws}, headers=_headers(outsider)
    )

    assert r.status_code == 403


def test_removed_member_cannot_ask_in_an_old_chat(client: TestClient) -> None:
    owner, member = _signup(client), _signup(client)
    ws = _workspace(client, owner)
    added = client.post(
        f"/api/v1/workspaces/{ws}/members",
        json={"email": member["email"]},
        headers=_headers(owner),
    )
    assert added.status_code == 201, added.text
    chat_id = client.post(
        "/api/v1/chat", json={"workspace_id": ws}, headers=_headers(member)
    ).json()["id"]
    member_id = added.json()["user_id"]
    removed = client.delete(
        f"/api/v1/workspaces/{ws}/members/{member_id}", headers=_headers(owner)
    )
    assert removed.status_code == 204, removed.text

    query = client.post(
        f"/api/v1/chat/{chat_id}/query",
        json={"query": "still here?"},
        headers=_headers(member),
    )
    stream = client.get(
        f"/api/v1/chat/{chat_id}/stream",
        params={"query": "still here?"},
        headers=_headers(member),
    )
    history = client.get(f"/api/v1/chat/{chat_id}", headers=_headers(member))

    assert query.status_code == 403
    assert stream.status_code == 403
    assert history.status_code == 200  # reading old chats is unchanged


def test_plain_member_cannot_manage_members(client: TestClient) -> None:
    owner, member, third = _signup(client), _signup(client), _signup(client)
    ws = _workspace(client, owner)
    client.post(
        f"/api/v1/workspaces/{ws}/members",
        json={"email": member["email"]},
        headers=_headers(owner),
    )

    r = client.post(
        f"/api/v1/workspaces/{ws}/members",
        json={"email": third["email"]},
        headers=_headers(member),
    )

    assert r.status_code == 403


def test_eval_is_admin_only(client: TestClient) -> None:
    user = _signup(client)
    ws = _workspace(client, user)

    r = client.post(
        "/api/v1/eval/suite",
        json={
            "workspace_id": ws,
            "test_cases": [{"question": "q", "answer": "a", "contexts": ["c"]}],
        },
        headers=_headers(user),
    )

    assert r.status_code == 403


def test_deep_health_is_hidden_from_remote_clients(
    client: TestClient, settings_override
) -> None:
    assert client.get("/health/deep").status_code == 404

    settings_override(health_deep_public=True)
    assert client.get("/health/deep").status_code in (200, 503)


def test_question_budget_is_shared_across_chats(
    client: TestClient, settings_override
) -> None:
    """One per-minute budget per user, not one per chat ID in the URL."""
    user = _signup(client)
    ws = _workspace(client, user)
    chats = [
        client.post(
            "/api/v1/chat", json={"workspace_id": ws}, headers=_headers(user)
        ).json()["id"]
        for _ in range(3)
    ]
    settings_override(rate_limit_chat_per_minute=2)
    limiter.reset()
    limiter.enabled = True
    try:
        codes = [
            client.get(
                f"/api/v1/chat/{chat_id}/stream",
                params={"query": "q"},
                headers=_headers(user),
            ).status_code
            for chat_id in chats
        ]
    finally:
        limiter.enabled = False
        limiter.reset()

    assert codes == [200, 200, 429]
