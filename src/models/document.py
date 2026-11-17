"""
src/models/document.py
──────────────────────
Document model — tracks every uploaded file and its processing state.

Lifecycle:  PENDING → PROCESSING → COMPLETED / FAILED

Key design: user_id is stored on every document.
This is the enforcement point for user-level isolation in vector searches.
"""

import uuid
from enum import Enum

from sqlalchemy import BigInteger, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class DocumentStatus(str, Enum):
    """
    Granular ingestion lifecycle.

    The frontend renders any non-terminal status as "indexing" with the
    `progress` field as the bar value, so adding intermediate states is a
    UI no-op while giving operators much better visibility on stuck jobs.

    PENDING / PROCESSING are kept for backward compatibility with already-
    persisted rows. New code should use the granular states.

    `__str__` is overridden because Python 3.11 changed StrEnum
    stringification to return the qualified name; without this override
    SQLAlchemy persists "DocumentStatus.PENDING" instead of "pending"
    into the VARCHAR column.
    """

    PENDING = "pending"  # Uploaded, awaiting pipeline pickup
    PROCESSING = "processing"  # Generic in-flight (legacy, still valid)
    PARSING = "parsing"  # Storage download + parser running
    CHUNKING = "chunking"  # Splitting into RAG chunks
    EMBEDDING = "embedding"  # Computing embeddings (often the slow stage)
    INDEXING = "indexing"  # Upserting vectors into Qdrant
    COMPLETED = "completed"  # Vectors stored, ready to query
    FAILED = "failed"  # Pipeline error — see error_message

    def __str__(self) -> str:
        return self.value


# Statuses still considered "in flight" by the frontend / readiness checks
DOCUMENT_IN_FLIGHT_STATUSES: frozenset[DocumentStatus] = frozenset(
    {
        DocumentStatus.PENDING,
        DocumentStatus.PROCESSING,
        DocumentStatus.PARSING,
        DocumentStatus.CHUNKING,
        DocumentStatus.EMBEDDING,
        DocumentStatus.INDEXING,
    }
)


class DocumentType(str, Enum):
    def __str__(self) -> str:  # noqa: D401 — see DocumentStatus comment
        return self.value

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Represents an uploaded document.

    Isolation guarantee:
        Every query to Qdrant MUST include `user_id` in the filter.
        This is enforced at the service layer (rag_service.py).
        The user_id here is the ground truth.
    """

    __tablename__ = "documents"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # ★ USER-LEVEL ISOLATION KEY
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_type: Mapped[DocumentType] = mapped_column(String(50), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[DocumentStatus] = mapped_column(
        String(50), nullable=False, default=DocumentStatus.PENDING, index=True
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 0–100. Frontend reads this for the progress bar; the pipeline updates
    # it after each stage so a stuck ingestion is visible.
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # ── Relationships ────────────────────────────────────────────
    workspace: Mapped["Workspace"] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "Workspace", back_populates="documents"
    )
    uploader: Mapped["User"] = relationship("User")  # type: ignore[name-defined]  # noqa: F821

    def __repr__(self) -> str:
        return (
            f"<Document id={self.id} filename={self.filename} "
            f"status={self.status} user={self.user_id}>"
        )
