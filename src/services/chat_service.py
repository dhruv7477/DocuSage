"""
src/services/chat_service.py
──────────────────────────────
Chat session and memory management service.

Responsibilities:
- Create and manage chat sessions
- Persist messages to PostgreSQL
- Load conversation history with token-budget enforcement
- Guard against context window overflow
- Provide history in LLMMessage format for the RAG service

Memory strategy:
    Load the last N messages where total estimated tokens < MAX_HISTORY_TOKENS.
    This is a sliding window — oldest messages fall off as the conversation grows.
    Future enhancement: summarise dropped messages and prepend a summary.

Design:
    ChatService owns all DB operations for chats and messages.
    RAGService receives pre-built LLMMessage history from here.
    Clean separation: memory management ≠ retrieval ≠ generation.
"""

import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import ChatAccessDeniedError, ChatNotFoundError
from src.core.logger import get_logger
from src.models.chat import Chat, Message, MessageRole
from src.models.user import User
from src.providers.llm.base import LLMMessage
from src.services.rag_service import CitedSource

log = get_logger(__name__)

# Token budget for history loaded into each RAG call
MAX_HISTORY_TOKENS = 800
CHARS_PER_TOKEN = 4
# Maximum messages fetched from DB before token-trimming
MAX_MESSAGES_FETCH = 20


class ChatService:

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ── Session Management ────────────────────────────────────────

    async def create_chat(
        self,
        workspace_id: uuid.UUID,
        user: User,
        title: str | None = None,
    ) -> Chat:
        """Create a new chat session."""
        chat = Chat(
            workspace_id=workspace_id,
            user_id=user.id,
            title=title or "New Chat",
            message_count=0,
        )
        self._db.add(chat)
        await self._db.flush()
        log.info("chat_created", chat_id=str(chat.id), user_id=str(user.id))
        return chat

    async def get_chat(self, chat_id: uuid.UUID, user: User) -> Chat:
        """
        Fetch a chat session, verifying ownership.

        Raises:
            ChatNotFoundError, ChatAccessDeniedError
        """
        result = await self._db.execute(
            select(Chat).where(Chat.id == chat_id)
        )
        chat = result.scalar_one_or_none()
        if not chat:
            raise ChatNotFoundError(f"Chat '{chat_id}' not found.")
        if chat.user_id != user.id:
            raise ChatAccessDeniedError("You do not have access to this chat.")
        return chat

    async def list_chats(
        self, workspace_id: uuid.UUID, user: User, limit: int = 50, offset: int = 0
    ) -> list[Chat]:
        """List chats for this user in this workspace, newest first, paginated."""
        result = await self._db.execute(
            select(Chat)
            .where(
                Chat.workspace_id == workspace_id,
                Chat.user_id == user.id,
            )
            .order_by(Chat.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def delete_chat(self, chat_id: uuid.UUID, user: User) -> None:
        """Delete a chat session and all its messages (cascade)."""
        chat = await self.get_chat(chat_id, user)
        await self._db.delete(chat)
        log.info("chat_deleted", chat_id=str(chat_id))

    async def update_chat_title(
        self, chat_id: uuid.UUID, title: str, user: User
    ) -> Chat:
        """Rename a chat session."""
        chat = await self.get_chat(chat_id, user)
        chat.title = title[:500]
        return chat

    # ── Message Persistence ───────────────────────────────────────

    async def save_user_message(
        self, chat_id: uuid.UUID, content: str
    ) -> Message:
        """Persist a user turn to the messages table."""
        return await self._save_message(chat_id, MessageRole.USER, content)

    async def save_assistant_message(
        self,
        chat_id: uuid.UUID,
        content: str,
        sources: list[CitedSource] | None = None,
    ) -> Message:
        """
        Persist an assistant turn to the messages table.
        Sources are serialised as JSON for display in the UI.
        """
        # Persist the enriched citation shape so the UI can render
        # heading paths and (eventually) link to the exact passage on a
        # "view in document" click. The column is JSONB, so additive
        # field changes don't need a migration.
        sources_list = [
            {
                "document_id": s.document_id,
                "chunk_id": getattr(s, "chunk_id", None),
                "source": s.source_filename,
                "page": s.page_number,
                "score": s.relevance_score,
                "heading_path": list(getattr(s, "heading_path", []) or []),
                "char_start": getattr(s, "char_start", 0),
                "char_end": getattr(s, "char_end", 0),
                "text": getattr(s, "text", "") or "",
            }
            for s in sources
        ] if sources else None
        return await self._save_message(
            chat_id, MessageRole.ASSISTANT, content, sources_json=sources_list
        )

    async def _save_message(
        self,
        chat_id: uuid.UUID,
        role: MessageRole,
        content: str,
        sources_json: list | None = None,
    ) -> Message:
        token_count = max(1, len(content) // CHARS_PER_TOKEN)
        message = Message(
            chat_id=chat_id,
            role=role,
            content=content,
            token_count=token_count,
            sources_json=sources_json,
        )
        self._db.add(message)
        # Increment message counter on the parent chat
        await self._db.execute(
            update(Chat)
            .where(Chat.id == chat_id)
            .values(message_count=Chat.message_count + 1)
        )
        await self._db.flush()
        return message

    # ── Memory Loading ────────────────────────────────────────────

    async def load_history(self, chat_id: uuid.UUID) -> list[LLMMessage]:
        """
        Load recent messages formatted as LLMMessage for the RAG service.

        Strategy:
            1. Fetch last MAX_MESSAGES_FETCH messages from DB (newest first)
            2. Reverse to chronological order
            3. Walk forward keeping messages within token budget
            4. Return only user/assistant turns (skip system messages)

        Returns:
            List of LLMMessage in chronological order (oldest first).
        """
        result = await self._db.execute(
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.created_at.desc())
            .limit(MAX_MESSAGES_FETCH)
        )
        messages = list(result.scalars().all())  # newest-first from DB

        # Token-budget trim: iterate newest→oldest, keep as many as fit
        selected: list[LLMMessage] = []
        token_budget = MAX_HISTORY_TOKENS

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                continue
            tokens = max(1, len(msg.content) // CHARS_PER_TOKEN)
            if tokens > token_budget:
                break
            selected.insert(0, LLMMessage(role=str(msg.role), content=msg.content))
            token_budget -= tokens

        log.debug(
            "history_loaded",
            chat_id=str(chat_id),
            messages_fetched=len(messages),
            messages_used=len(selected),
        )
        return selected

    async def get_messages(
        self, chat_id: uuid.UUID, user: User, limit: int = 50
    ) -> list[Message]:
        """Fetch messages for display in the UI."""
        await self.get_chat(chat_id, user)  # Verify ownership
        result = await self._db.execute(
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.created_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())

    # ── Auto-Title ────────────────────────────────────────────────

    @staticmethod
    def generate_title(first_query: str) -> str:
        """
        Generate a short title from the first user query.
        Used to auto-name a new chat session.
        """
        title = first_query.strip()
        if len(title) > 60:
            title = title[:57] + "..."
        return title
