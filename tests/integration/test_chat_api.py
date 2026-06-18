"""
tests/integration/test_chat_api.py
────────────────────────────────────
Integration tests for /api/v1/chat/* endpoints.

Tests the full request → route → service → RAG → response cycle
with all external dependencies (DB, LLM, Qdrant) mocked.
"""

import json
import uuid
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import create_app


@contextmanager
def _auth_as(test_app, mock_user):
    """
    Override the get_current_user dependency for the duration of a test.

    NOTE: `unittest.mock.patch("src.dependencies.get_current_user")` does
    NOT work here — FastAPI captured the original function object in
    `Depends(...)` at import time, so swapping the module attribute has
    no effect. `dependency_overrides` is the supported mechanism.
    """
    from src.dependencies import get_current_user
    test_app.dependency_overrides[get_current_user] = lambda: mock_user
    try:
        yield
    finally:
        test_app.dependency_overrides.pop(get_current_user, None)


@contextmanager
def _rag_as(test_app, mock_rag):
    """Override the _get_rag_service dependency (same patch caveat as above)."""
    from src.api.v1.chat import _get_rag_service
    test_app.dependency_overrides[_get_rag_service] = lambda: mock_rag
    try:
        yield
    finally:
        test_app.dependency_overrides.pop(_get_rag_service, None)


def _make_mock_user(uid: uuid.UUID):
    from src.models.user import User, UserRole
    user = User()
    user.id = uid
    user.email = "alice@example.com"
    user.full_name = "Alice"
    user.role = UserRole.MEMBER
    user.is_active = True
    user.is_email_verified = True
    return user


def _make_auth_headers(user_id: uuid.UUID) -> dict:
    from src.configs.config import get_settings
    from src.core.security import create_access_token

    settings = get_settings()
    token = create_access_token(
        subject=str(user_id),
        secret_key=settings.secret_key,
        algorithm=settings.algorithm,
        expires_minutes=settings.access_token_expire_minutes,
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def test_app():
    return create_app()


@pytest.fixture
def user_id():
    return uuid.uuid4()


@pytest.fixture
def workspace_id():
    return uuid.uuid4()


@pytest.fixture
def chat_id():
    return uuid.uuid4()


@pytest.fixture
def headers(user_id):
    return _make_auth_headers(user_id)


class TestCreateChat:

    def test_create_chat_requires_auth(self, test_app):
        with TestClient(test_app, raise_server_exceptions=False) as c:
            resp = c.post("/api/v1/chat", json={"workspace_id": str(uuid.uuid4())})
        assert resp.status_code in (401, 403)

    def test_create_chat_missing_workspace_id_returns_422(self, test_app, user_id, headers):
        mock_user = _make_mock_user(user_id)
        with TestClient(test_app, raise_server_exceptions=False) as c:
            with _auth_as(test_app, mock_user):
                resp = c.post(
                    "/api/v1/chat",
                    json={"title": "No workspace"},
                    headers=headers,
                )
        assert resp.status_code == 422

    def test_create_chat_success(self, test_app, user_id, workspace_id, headers):
        mock_user = _make_mock_user(user_id)

        mock_chat = MagicMock()
        mock_chat.id = uuid.uuid4()
        mock_chat.workspace_id = workspace_id
        mock_chat.user_id = user_id
        mock_chat.title = "New Chat"
        mock_chat.message_count = 0
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        mock_chat.created_at = now
        mock_chat.updated_at = now

        with TestClient(test_app, raise_server_exceptions=False) as c:
            with patch("src.api.v1.chat.ChatService") as MockChatService:
                mock_svc = AsyncMock()
                mock_svc.create_chat.return_value = mock_chat
                MockChatService.return_value = mock_svc

                with _auth_as(test_app, mock_user):
                    resp = c.post(
                        "/api/v1/chat",
                        json={"workspace_id": str(workspace_id), "title": "My Chat"},
                        headers=headers,
                    )
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert data["title"] == "New Chat"


class TestListChats:

    def test_list_chats_requires_auth(self, test_app, workspace_id):
        with TestClient(test_app, raise_server_exceptions=False) as c:
            resp = c.get(f"/api/v1/chat?workspace_id={workspace_id}")
        assert resp.status_code in (401, 403)

    def test_list_chats_returns_array(self, test_app, user_id, workspace_id, headers):
        mock_user = _make_mock_user(user_id)

        with TestClient(test_app, raise_server_exceptions=False) as c:
            with patch("src.api.v1.chat.ChatService") as MockChatService:
                mock_svc = AsyncMock()
                mock_svc.list_chats.return_value = []
                MockChatService.return_value = mock_svc
                with _auth_as(test_app, mock_user):
                    resp = c.get(
                        f"/api/v1/chat?workspace_id={workspace_id}",
                        headers=headers,
                    )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestDeleteChat:

    def test_delete_chat_requires_auth(self, test_app, chat_id):
        with TestClient(test_app, raise_server_exceptions=False) as c:
            resp = c.delete(f"/api/v1/chat/{chat_id}")
        assert resp.status_code in (401, 403)

    def test_delete_chat_not_found(self, test_app, user_id, chat_id, headers):
        from src.core.exceptions import ChatNotFoundError
        mock_user = _make_mock_user(user_id)

        with TestClient(test_app, raise_server_exceptions=False) as c:
            with patch("src.api.v1.chat.ChatService") as MockChatService:
                mock_svc = AsyncMock()
                mock_svc.delete_chat.side_effect = ChatNotFoundError("Not found.")
                MockChatService.return_value = mock_svc
                with _auth_as(test_app, mock_user):
                    resp = c.delete(
                        f"/api/v1/chat/{chat_id}",
                        headers=headers,
                    )
        assert resp.status_code == 404
        assert resp.json()["error_code"] == "CHAT_NOT_FOUND"


class TestQueryEndpoint:

    def test_query_requires_auth(self, test_app, chat_id):
        with TestClient(test_app, raise_server_exceptions=False) as c:
            resp = c.post(
                f"/api/v1/chat/{chat_id}/query",
                json={"query": "What is the refund policy?"},
            )
        assert resp.status_code in (401, 403)

    def test_query_empty_string_returns_422(self, test_app, user_id, chat_id, headers):
        mock_user = _make_mock_user(user_id)
        with TestClient(test_app, raise_server_exceptions=False) as c:
            # _get_rag_service is also a Depends — override it so a 422
            # doesn't construct real providers as a side effect.
            with _auth_as(test_app, mock_user), _rag_as(test_app, MagicMock()):
                resp = c.post(
                    f"/api/v1/chat/{chat_id}/query",
                    json={"query": ""},
                    headers=headers,
                )
        assert resp.status_code == 422

    def test_query_too_long_returns_422(self, test_app, user_id, chat_id, headers):
        mock_user = _make_mock_user(user_id)
        with TestClient(test_app, raise_server_exceptions=False) as c:
            with _auth_as(test_app, mock_user), _rag_as(test_app, MagicMock()):
                resp = c.post(
                    f"/api/v1/chat/{chat_id}/query",
                    json={"query": "q" * 5000},
                    headers=headers,
                )
        assert resp.status_code == 422

    def test_query_chat_not_found_returns_404(self, test_app, user_id, chat_id, headers):
        from src.core.exceptions import ChatNotFoundError
        mock_user = _make_mock_user(user_id)

        with TestClient(test_app, raise_server_exceptions=False) as c:
            with patch("src.api.v1.chat.ChatService") as MockCS:
                mock_svc = AsyncMock()
                mock_svc.get_chat.side_effect = ChatNotFoundError("Chat not found.")
                MockCS.return_value = mock_svc
                with _auth_as(test_app, mock_user), _rag_as(test_app, MagicMock()):
                    resp = c.post(
                        f"/api/v1/chat/{chat_id}/query",
                        json={"query": "What is X?"},
                        headers=headers,
                    )
        assert resp.status_code == 404

    def test_query_success_returns_answer_and_sources(
        self, test_app, user_id, workspace_id, chat_id, headers
    ):
        from src.services.rag_service import CitedSource
        mock_user = _make_mock_user(user_id)

        mock_chat = MagicMock()
        mock_chat.id = chat_id
        mock_chat.workspace_id = workspace_id
        mock_chat.user_id = user_id
        mock_chat.message_count = 0

        mock_assistant_msg = MagicMock()
        mock_assistant_msg.id = uuid.uuid4()

        from src.services.rag_service import RAGResponse
        mock_rag_response = RAGResponse(
            answer="The refund policy is 30 days.",
            sources=[
                CitedSource(
                    document_id=str(uuid.uuid4()),
                    chunk_id=str(uuid.uuid4()),
                    source_filename="policy.pdf",
                    page_number=1,
                    chunk_index=0,
                    relevance_score=0.92,
                    text="The refund policy allows returns within 30 days.",
                )
            ],
            context_chunks_used=1,
            confidence=0.92,
            abstained=False,
        )

        with TestClient(test_app, raise_server_exceptions=False) as c:
            with patch("src.api.v1.chat.ChatService") as MockCS:
                mock_svc = AsyncMock()
                mock_svc.get_chat.return_value = mock_chat
                mock_svc.load_history.return_value = []
                mock_svc.save_user_message.return_value = MagicMock()
                mock_svc.save_assistant_message.return_value = mock_assistant_msg
                mock_svc.update_chat_title.return_value = mock_chat
                MockCS.generate_title = MagicMock(return_value="Test Chat")
                MockCS.return_value = mock_svc

                mock_rag = AsyncMock()
                mock_rag.generate.return_value = mock_rag_response
                with _auth_as(test_app, mock_user), _rag_as(test_app, mock_rag):
                    resp = c.post(
                        f"/api/v1/chat/{chat_id}/query",
                        json={"query": "What is the refund policy?"},
                        headers=headers,
                    )

        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert isinstance(data["sources"], list)


class TestSSEStreamEndpoint:

    def test_stream_requires_auth(self, test_app, chat_id):
        with TestClient(test_app, raise_server_exceptions=False) as c:
            resp = c.get(f"/api/v1/chat/{chat_id}/stream?query=hello")
        assert resp.status_code in (401, 403)

    def test_stream_empty_query_returns_422(self, test_app, user_id, chat_id, headers):
        from src.dependencies import get_current_user

        mock_user = _make_mock_user(user_id)
        test_app.dependency_overrides[get_current_user] = lambda: mock_user
        try:
            with TestClient(test_app, raise_server_exceptions=False) as c:
                resp = c.get(
                    f"/api/v1/chat/{chat_id}/stream?query=",
                    headers=headers,
                )
        finally:
            test_app.dependency_overrides.clear()
        assert resp.status_code == 422

    @staticmethod
    def _mock_orchestrator(tokens: list[str]):
        """Orchestrator whose prepare() emits agentic events then streams tokens."""
        from src.services.chat_orchestrator import StreamPlan

        orch = MagicMock()

        async def fake_prepare(*args, emit=None, **kwargs):
            if emit:
                emit({"type": "thinking", "stage": "analyzing"})
                emit({"type": "route", "intent": "rag_query"})

            async def _tokens():
                for token in tokens:
                    yield token

            return StreamPlan(
                intent="rag_query",
                token_iter=_tokens(),
                sources=[],
                confidence=1.0,
                abstained=False,
                allow_citations=True,
            )

        orch.prepare = fake_prepare
        return orch

    def _run_stream(self, test_app, user_id, workspace_id, chat_id, headers, tokens, query="hello"):
        """Shared harness: override the FastAPI deps, mock services, GET the stream."""
        from src.api.v1.chat import _get_orchestrator
        from src.db.session import get_db
        from src.dependencies import get_current_user

        mock_user = _make_mock_user(user_id)
        mock_chat = MagicMock()
        mock_chat.id = chat_id
        mock_chat.workspace_id = workspace_id
        mock_chat.user_id = user_id
        mock_chat.message_count = 0
        mock_assistant_msg = MagicMock()
        mock_assistant_msg.id = uuid.uuid4()

        orch = self._mock_orchestrator(tokens)
        mock_db = AsyncMock()

        async def _fake_db():
            yield mock_db

        test_app.dependency_overrides[_get_orchestrator] = lambda: orch
        test_app.dependency_overrides[get_current_user] = lambda: mock_user
        test_app.dependency_overrides[get_db] = _fake_db

        try:
            with TestClient(test_app, raise_server_exceptions=False) as c:
                with patch("src.api.v1.chat.ChatService") as MockCS:
                    mock_svc = AsyncMock()
                    mock_svc.get_chat.return_value = mock_chat
                    mock_svc.load_history.return_value = []
                    mock_svc.save_user_message.return_value = MagicMock()
                    mock_svc.save_assistant_message.return_value = mock_assistant_msg
                    mock_svc.update_chat_title.return_value = mock_chat
                    # generate_title is a @staticmethod on the real class.
                    MockCS.generate_title = MagicMock(return_value="Test Chat")
                    MockCS.return_value = mock_svc

                    with patch("src.services.document_service.DocumentService") as MockDS:
                        mock_ds = AsyncMock()
                        mock_ds.list_documents.return_value = []
                        MockDS.return_value = mock_ds

                        return c.get(
                            f"/api/v1/chat/{chat_id}/stream?query={query}",
                            headers=headers,
                        )
        finally:
            test_app.dependency_overrides.clear()

    def test_stream_returns_sse_content_type(self, test_app, user_id, workspace_id, chat_id, headers):
        """SSE endpoint must return text/event-stream content type."""
        resp = self._run_stream(
            test_app, user_id, workspace_id, chat_id, headers,
            tokens=["Hello ", "world."],
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_stream_events_are_valid_json(self, test_app, user_id, workspace_id, chat_id, headers):
        """Each SSE data line must parse as valid JSON with a 'type' field."""
        resp = self._run_stream(
            test_app, user_id, workspace_id, chat_id, headers,
            tokens=["Hello world."], query="test",
        )

        # Parse SSE events
        events = []
        for line in resp.text.split("\n"):
            if line.startswith("data: "):
                payload = line[6:]
                try:
                    event = json.loads(payload)
                    events.append(event)
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON in SSE event: {payload}")

        assert len(events) > 0
        event_types = [e.get("type") for e in events]
        # Agentic event contract: route + thinking + tokens + sources + done.
        assert "route" in event_types
        assert "thinking" in event_types
        assert "token" in event_types
        assert "sources" in event_types
        assert "done" in event_types
        # Validating stage emitted after the token loop.
        stages = [e.get("stage") for e in events if e.get("type") == "thinking"]
        assert "validating" in stages

    def test_stream_commits_user_message_before_generation(
        self, test_app, user_id, workspace_id, chat_id, headers
    ):
        """
        When generation fails (e.g. Ollama times out), the user message must
        already be committed so it survives in the thread. Regression: it used
        to share the assistant's transaction, committed only at the end, so a
        failure rolled it back and the message vanished.
        """
        from src.api.v1.chat import _get_orchestrator
        from src.core.exceptions import LLMProviderError
        from src.db.session import get_db
        from src.dependencies import get_current_user

        mock_user = _make_mock_user(user_id)
        mock_chat = MagicMock()
        mock_chat.id = chat_id
        mock_chat.workspace_id = workspace_id
        mock_chat.user_id = user_id
        mock_chat.message_count = 0

        # Orchestrator whose prepare() fails the way a dead/slow Ollama does.
        orch = MagicMock()

        async def failing_prepare(*args, emit=None, **kwargs):
            raise LLMProviderError(
                "The model timed out — it may still be loading. "
                "Give it a moment and try again."
            )

        orch.prepare = failing_prepare

        mock_db = AsyncMock()

        async def _fake_db():
            yield mock_db

        test_app.dependency_overrides[_get_orchestrator] = lambda: orch
        test_app.dependency_overrides[get_current_user] = lambda: mock_user
        test_app.dependency_overrides[get_db] = _fake_db

        try:
            with TestClient(test_app, raise_server_exceptions=False) as c:
                with patch("src.api.v1.chat.ChatService") as MockCS:
                    mock_svc = AsyncMock()
                    mock_svc.get_chat.return_value = mock_chat
                    mock_svc.load_history.return_value = []
                    mock_svc.save_user_message.return_value = MagicMock()
                    MockCS.return_value = mock_svc

                    with patch("src.services.document_service.DocumentService") as MockDS:
                        mock_ds = AsyncMock()
                        mock_ds.list_documents.return_value = []
                        MockDS.return_value = mock_ds

                        resp = c.get(
                            f"/api/v1/chat/{chat_id}/stream?query=summarize",
                            headers=headers,
                        )
        finally:
            test_app.dependency_overrides.clear()

        assert resp.status_code == 200
        # User message saved, then committed — before generation failed.
        mock_svc.save_user_message.assert_awaited_once()
        mock_db.commit.assert_awaited()
        # No assistant message was saved (generation never produced one).
        mock_svc.save_assistant_message.assert_not_called()

        # The provider's user-safe message is forwarded in the SSE error event.
        events = [
            json.loads(line[6:])
            for line in resp.text.split("\n")
            if line.startswith("data: ")
        ]
        error_events = [e for e in events if e.get("type") == "error"]
        assert error_events, "expected an SSE error event"
        assert "timed out" in error_events[0]["message"].lower()
