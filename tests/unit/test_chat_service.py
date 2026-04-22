"""
tests/unit/test_chat_service.py
─────────────────────────────────
Unit tests for ChatService — all DB operations mocked.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.exceptions import ChatAccessDeniedError, ChatNotFoundError
from src.models.chat import Chat, Message, MessageRole
from src.models.user import User, UserRole
from src.providers.llm.base import LLMMessage
from src.services.chat_service import ChatService


def _make_user(user_id: uuid.UUID | None = None) -> User:
    user = User()
    user.id = user_id or uuid.uuid4()
    user.email = "test@example.com"
    user.role = UserRole.MEMBER
    user.is_active = True
    return user


def _make_chat(user_id: uuid.UUID) -> Chat:
    chat = Chat()
    chat.id = uuid.uuid4()
    chat.workspace_id = uuid.uuid4()
    chat.user_id = user_id
    chat.title = "Test Chat"
    chat.message_count = 0
    return chat


def _make_message(role: MessageRole, content: str) -> Message:
    msg = Message()
    msg.id = uuid.uuid4()
    msg.chat_id = uuid.uuid4()
    msg.role = role
    msg.content = content
    msg.token_count = len(content) // 4
    msg.sources_json = None
    from datetime import datetime, timezone
    msg.created_at = datetime.now(timezone.utc)
    return msg


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.flush = AsyncMock()
    db.execute = AsyncMock()
    return db


@pytest.fixture
def service(mock_db):
    return ChatService(mock_db)


class TestGetChat:

    @pytest.mark.asyncio
    async def test_returns_chat_for_owner(self, service, mock_db):
        user = _make_user()
        chat = _make_chat(user.id)
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = chat
        mock_db.execute.return_value = result_mock

        returned = await service.get_chat(chat.id, user)
        assert returned.id == chat.id

    @pytest.mark.asyncio
    async def test_raises_not_found_for_missing_chat(self, service, mock_db):
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = result_mock

        with pytest.raises(ChatNotFoundError):
            await service.get_chat(uuid.uuid4(), _make_user())

    @pytest.mark.asyncio
    async def test_raises_access_denied_for_wrong_user(self, service, mock_db):
        owner = _make_user()
        other = _make_user()
        chat = _make_chat(owner.id)
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = chat
        mock_db.execute.return_value = result_mock

        with pytest.raises(ChatAccessDeniedError):
            await service.get_chat(chat.id, other)


class TestLoadHistory:

    @pytest.mark.asyncio
    async def test_empty_history_returns_empty(self, service, mock_db):
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = result_mock

        history = await service.load_history(uuid.uuid4())
        assert history == []

    @pytest.mark.asyncio
    async def test_history_returned_as_llm_messages(self, service, mock_db):
        messages = [
            _make_message(MessageRole.ASSISTANT, "X is Y."),  # newest first (matches DB DESC order)
            _make_message(MessageRole.USER, "What is X?"),
        ]
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = messages
        mock_db.execute.return_value = result_mock

        history = await service.load_history(uuid.uuid4())
        assert len(history) == 2
        assert all(isinstance(m, LLMMessage) for m in history)
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_history_trimmed_by_token_budget(self, service, mock_db):
        # Create messages that exceed token budget
        big_messages = [
            _make_message(MessageRole.USER, "Q " * 500),     # ~250 tokens
            _make_message(MessageRole.ASSISTANT, "A " * 500), # ~250 tokens
            _make_message(MessageRole.USER, "Q2 " * 500),
            _make_message(MessageRole.ASSISTANT, "A2 " * 500),
        ]
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = big_messages
        mock_db.execute.return_value = result_mock

        history = await service.load_history(uuid.uuid4())
        # Token budget is 800 — should trim to fit
        total_tokens = sum(len(m.content) // 4 for m in history)
        assert total_tokens <= 800

    @pytest.mark.asyncio
    async def test_system_messages_excluded(self, service, mock_db):
        from src.models.chat import MessageRole
        messages = [
            _make_message(MessageRole.SYSTEM, "You are an assistant."),
            _make_message(MessageRole.USER, "Hello"),
            _make_message(MessageRole.ASSISTANT, "Hi there!"),
        ]
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = messages
        mock_db.execute.return_value = result_mock

        history = await service.load_history(uuid.uuid4())
        roles = [m.role for m in history]
        assert "system" not in roles


class TestGenerateTitle:

    def test_short_query_unchanged(self):
        title = ChatService.generate_title("What is the refund policy?")
        assert title == "What is the refund policy?"

    def test_long_query_truncated(self):
        long_query = "A" * 100
        title = ChatService.generate_title(long_query)
        assert len(title) <= 63
        assert title.endswith("...")

    def test_whitespace_stripped(self):
        title = ChatService.generate_title("  Hello world  ")
        assert title == "Hello world"
