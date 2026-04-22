"""
tests/integration/test_chat_api.py
────────────────────────────────────
Integration tests for /api/v1/chat/* endpoints.

Tests the full request → route → service → RAG → response cycle
with all external dependencies (DB, LLM, Qdrant) mocked.
"""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import create_app


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
    from src.core.security import create_access_token
    token = create_access_token(subject=str(user_id), token_type="access")
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
            with patch("src.dependencies.get_current_user", return_value=mock_user):
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

                with patch("src.dependencies.get_current_user", return_value=mock_user):
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
                with patch("src.dependencies.get_current_user", return_value=mock_user):
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
                with patch("src.dependencies.get_current_user", return_value=mock_user):
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
            with patch("src.dependencies.get_current_user", return_value=mock_user):
                resp = c.post(
                    f"/api/v1/chat/{chat_id}/query",
                    json={"query": ""},
                    headers=headers,
                )
        assert resp.status_code == 422

    def test_query_too_long_returns_422(self, test_app, user_id, chat_id, headers):
        mock_user = _make_mock_user(user_id)
        with TestClient(test_app, raise_server_exceptions=False) as c:
            with patch("src.dependencies.get_current_user", return_value=mock_user):
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
                with patch("src.dependencies.get_current_user", return_value=mock_user):
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

        mock_rag_response = MagicMock()
        mock_rag_response.answer = "The refund policy is 30 days."
        mock_rag_response.sources = [
            CitedSource(
                document_id=str(uuid.uuid4()),
                source_filename="policy.pdf",
                page_number=1,
                chunk_index=0,
                relevance_score=0.92,
            )
        ]

        with TestClient(test_app, raise_server_exceptions=False) as c:
            with patch("src.api.v1.chat.ChatService") as MockCS:
                mock_svc = AsyncMock()
                mock_svc.get_chat.return_value = mock_chat
                mock_svc.load_history.return_value = []
                mock_svc.save_user_message.return_value = MagicMock()
                mock_svc.save_assistant_message.return_value = mock_assistant_msg
                mock_svc.update_chat_title.return_value = mock_chat
                MockCS.return_value = mock_svc

                with patch("src.api.v1.chat._get_rag_service") as MockRAG:
                    mock_rag = AsyncMock()
                    mock_rag.generate.return_value = mock_rag_response
                    MockRAG.return_value = mock_rag

                    with patch("src.dependencies.get_current_user", return_value=mock_user):
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
        mock_user = _make_mock_user(user_id)
        with TestClient(test_app, raise_server_exceptions=False) as c:
            with patch("src.dependencies.get_current_user", return_value=mock_user):
                resp = c.get(
                    f"/api/v1/chat/{chat_id}/stream?query=",
                    headers=headers,
                )
        assert resp.status_code == 422

    def test_stream_returns_sse_content_type(self, test_app, user_id, workspace_id, chat_id, headers):
        """SSE endpoint must return text/event-stream content type."""
        mock_user = _make_mock_user(user_id)

        mock_chat = MagicMock()
        mock_chat.id = chat_id
        mock_chat.workspace_id = workspace_id
        mock_chat.user_id = user_id
        mock_chat.message_count = 0

        mock_assistant_msg = MagicMock()
        mock_assistant_msg.id = uuid.uuid4()

        async def fake_stream(*args, **kwargs):
            for token in ["Hello ", "world."]:
                yield token

        with TestClient(test_app, raise_server_exceptions=False) as c:
            with patch("src.api.v1.chat.ChatService") as MockCS:
                mock_svc = AsyncMock()
                mock_svc.get_chat.return_value = mock_chat
                mock_svc.load_history.return_value = []
                mock_svc.save_user_message.return_value = MagicMock()
                mock_svc.save_assistant_message.return_value = mock_assistant_msg
                mock_svc.update_chat_title.return_value = mock_chat
                MockCS.return_value = mock_svc

                with patch("src.api.v1.chat._get_rag_service") as MockRAG:
                    mock_rag = MagicMock()
                    mock_rag.get_sources_for_query = AsyncMock(return_value=[])
                    mock_rag.stream = fake_stream
                    MockRAG.return_value = mock_rag

                    with patch("src.dependencies.get_current_user", return_value=mock_user):
                        resp = c.get(
                            f"/api/v1/chat/{chat_id}/stream?query=hello",
                            headers=headers,
                        )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_stream_events_are_valid_json(self, test_app, user_id, workspace_id, chat_id, headers):
        """Each SSE data line must parse as valid JSON with a 'type' field."""
        mock_user = _make_mock_user(user_id)
        mock_chat = MagicMock()
        mock_chat.id = chat_id
        mock_chat.workspace_id = workspace_id
        mock_chat.user_id = user_id
        mock_chat.message_count = 0
        mock_assistant_msg = MagicMock()
        mock_assistant_msg.id = uuid.uuid4()

        async def fake_stream(*args, **kwargs):
            yield "Hello world."

        with TestClient(test_app, raise_server_exceptions=False) as c:
            with patch("src.api.v1.chat.ChatService") as MockCS:
                mock_svc = AsyncMock()
                mock_svc.get_chat.return_value = mock_chat
                mock_svc.load_history.return_value = []
                mock_svc.save_user_message.return_value = MagicMock()
                mock_svc.save_assistant_message.return_value = mock_assistant_msg
                mock_svc.update_chat_title.return_value = mock_chat
                MockCS.return_value = mock_svc

                with patch("src.api.v1.chat._get_rag_service") as MockRAG:
                    mock_rag = MagicMock()
                    mock_rag.get_sources_for_query = AsyncMock(return_value=[])
                    mock_rag.stream = fake_stream
                    MockRAG.return_value = mock_rag

                    with patch("src.dependencies.get_current_user", return_value=mock_user):
                        resp = c.get(
                            f"/api/v1/chat/{chat_id}/stream?query=test",
                            headers=headers,
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
        # Should have at least one token event and a done event
        assert "token" in event_types
        assert "done" in event_types
