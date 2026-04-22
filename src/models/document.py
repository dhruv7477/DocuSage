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
    PENDING = "pending"         # Uploaded, awaiting pipeline
    PROCESSING = "processing"   # Celery worker is ingesting
    COMPLETED = "completed"     # Vectors stored in Qdrant
    FAILED = "failed"           # Pipeline error — see error_message


class DocumentType(str, Enum):
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
