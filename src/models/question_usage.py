"""
src/models/question_usage.py
────────────────────────────
One row per question accepted while a daily question limit is on.

Kept apart from `messages` on purpose: deleting a chat deletes its
messages, and must not hand a user's questions back. Rows are never
updated; the user link is cleared (not cascaded) when a user is deleted,
so the site-wide count for the day stays correct.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, UUIDPrimaryKeyMixin


class QuestionUsage(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "question_usage"
    __table_args__ = (
        Index("ix_question_usage_user_id_asked_at", "user_id", "asked_at"),
    )

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    asked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
