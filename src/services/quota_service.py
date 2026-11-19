"""
src/services/quota_service.py
─────────────────────────────
Per-user quotas for public deployments: stored documents, total upload
size and questions per UTC day. Sample documents do not count.

Every limit is a setting where 0 means unlimited, so self-hosted installs
have no quotas unless they opt in. Admins and super admins are exempt.
"""

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.configs.config import Settings, get_settings
from src.core.exceptions import QuotaExceededError
from src.demo.samples import library_storage_paths
from src.models.chat import Chat, Message, MessageRole
from src.models.document import Document
from src.models.user import User, UserRole

_BYTES_PER_MB = 1024 * 1024
_EXEMPT_ROLES = frozenset({UserRole.ADMIN, UserRole.SUPER_ADMIN})


class QuotaService:
    """Checks a user's usage against the configured quotas."""

    def __init__(self, db: AsyncSession, settings: Settings | None = None) -> None:
        self._db = db
        self._settings = settings or get_settings()

    async def check_upload(
        self, user: User, new_bytes: int, replaces_bytes: int | None = None
    ) -> None:
        """Raise QuotaExceededError if this upload would exceed a quota.

        `replaces_bytes` is the size of the document a re-upload replaces,
        or None when the upload creates a new document.
        """
        if self._is_exempt(user):
            return
        max_docs = self._settings.quota_max_documents
        max_mb = self._settings.quota_max_upload_mb
        if not max_docs and not max_mb:
            return

        count, total_bytes = await self._document_usage(user)
        if max_docs and replaces_bytes is None and count >= max_docs:
            raise QuotaExceededError(
                f"Document limit reached ({max_docs} documents). "
                "Delete a document to upload another."
            )
        if (
            max_mb
            and total_bytes - (replaces_bytes or 0) + new_bytes > max_mb * _BYTES_PER_MB
        ):
            raise QuotaExceededError(
                f"Storage limit reached ({max_mb} MB in total). "
                "Delete a document or upload a smaller file."
            )

    async def check_question(self, user: User) -> None:
        """Raise QuotaExceededError if the user has used today's questions."""
        limit = self._settings.quota_questions_per_day
        if not limit or self._is_exempt(user):
            return
        midnight = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        asked = await self._db.scalar(
            select(func.count(Message.id))
            .join(Chat, Message.chat_id == Chat.id)
            .where(
                Chat.user_id == user.id,
                Message.role == MessageRole.USER,
                Message.created_at >= midnight,
            )
        )
        if (asked or 0) >= limit:
            raise QuotaExceededError(
                f"Daily question limit reached ({limit} per day). "
                "It resets at 00:00 UTC."
            )

    @staticmethod
    def _is_exempt(user: User) -> bool:
        return user.role in _EXEMPT_ROLES

    async def _document_usage(self, user: User) -> tuple[int, int]:
        row = (
            await self._db.execute(
                select(
                    func.count(Document.id),
                    func.coalesce(func.sum(Document.file_size_bytes), 0),
                ).where(
                    Document.user_id == user.id,
                    # Sample documents (demo mode) do not count.
                    Document.storage_path.not_in(library_storage_paths()),
                )
            )
        ).one()
        return int(row[0]), int(row[1])
