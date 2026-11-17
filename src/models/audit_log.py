"""
src/models/audit_log.py
────────────────────────
AuditLog ORM model.

Every significant API action (auth, document ops, chat, admin actions)
is written here for security review, compliance, and debugging.

Design decisions:
- Append-only: rows are never updated or deleted by the application.
- user_id is nullable: captures failed auth attempts (no user yet).
- ip_address + user_agent: for security forensics.
- request_id: correlation ID set by middleware, links log ↔ request.
- status_code: lets you query for all 4xx/5xx events easily.
- extra_json: JSONB escape hatch for action-specific metadata
  (e.g. document_id for upload, workspace_id for member add).
"""

from __future__ import annotations

from sqlalchemy import DateTime, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class AuditLog(Base):
    """
    Immutable record of every significant application event.

    Written by AuditMiddleware on every request that modifies state,
    and explicitly for security-sensitive reads (e.g. document download).
    """

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Who ──────────────────────────────────────────────────────
    user_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), nullable=True, index=True
    )
    user_email: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # ── What ─────────────────────────────────────────────────────
    action: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )
    # e.g. "auth.register", "document.upload", "chat.query", "admin.disable_user"

    resource_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # e.g. "user", "document", "workspace", "chat"

    resource_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # UUID of the resource being acted upon

    # ── HTTP context ─────────────────────────────────────────────
    method: Mapped[str] = mapped_column(String(10), nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # ── Client context ───────────────────────────────────────────
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    # 45 chars handles IPv6

    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # ── Extra metadata ───────────────────────────────────────────
    extra: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # Arbitrary action-specific data: {"workspace_id": "...", "chunk_count": 12}

    # ── Timing ───────────────────────────────────────────────────
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    # ── Compound indexes for common query patterns ────────────────
    __table_args__ = (
        Index("ix_audit_logs_user_action", "user_id", "action"),
        Index("ix_audit_logs_resource", "resource_type", "resource_id"),
        Index("ix_audit_logs_created_status", "created_at", "status_code"),
    )

    def __repr__(self) -> str:
        return (
            f"<AuditLog id={self.id} action={self.action!r} "
            f"user={self.user_email!r} status={self.status_code}>"
        )
