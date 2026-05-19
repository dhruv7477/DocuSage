"""
src/models/chat.py
──────────────────
Chat session and Message models for conversation history.

Chat:    A named session within a workspace (one user, one workspace)
Message: Individual turn in a chat (user query or assistant response)
"""

import uuid
from enum import Enum

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class MessageRole(str, Enum):
    """
    Persisted as a plain VARCHAR(50). We override __str__ because
    Python 3.11 changed `str(StrEnum)` from returning `self.value`
    ("user") to the qualified name ("MessageRole.USER"), which then
    leaks into the database column via SQLAlchemy's stringification
    of the ORM value. The override keeps the on-disk values stable
    regardless of Python version, so existing rows and downstream
    consumers (LLMMessage.role, sources_json) keep working.
    """
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

    def __str__(self) -> str:
        return self.value


class Chat(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Represents a conversation session.

    One user can have many chats in the same workspace.
    Chat memory is scoped per chat — no cross-chat context leakage.
    """

    __tablename__ = "chats"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str | None] = mapped_column(
        String(500), nullable=True, default="New Chat"
    )
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # ── Relationships ────────────────────────────────────────────
    workspace: Mapped["Workspace"] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "Workspace", back_populates="chats"
    )
    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="chat",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
    user: Mapped["User"] = relationship("User")  # type: ignore[name-defined]  # noqa: F821

    def __repr__(self) -> str:
        return f"<Chat id={self.id} user={self.user_id} workspace={self.workspace_id}>"


class Message(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    A single message turn in a chat.

    token_count: tracked for LLM context window management.
    sources_json: JSON list of document IDs used to answer (for citations).
    """

    __tablename__ = "messages"

    chat_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[MessageRole] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Stores list of cited sources as JSONB — no manual json.dumps/loads needed
    sources_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    # ── Relationships ────────────────────────────────────────────
    chat: Mapped["Chat"] = relationship("Chat", back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message id={self.id} role={self.role} chat={self.chat_id}>"