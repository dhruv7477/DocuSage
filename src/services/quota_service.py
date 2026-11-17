"""
src/services/quota_service.py
─────────────────────────────
Quotas for public deployments: per-user stored documents, total upload
size and questions per UTC day, and a site-wide question budget per UTC
day. Sample documents do not count.

Every limit is a setting where 0 means unlimited, so self-hosted installs
have no quotas unless they opt in. Admins and super admins are exempt.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.configs.config import Settings, get_settings
from src.core.exceptions import DemoDailyLimitError, QuotaExceededError
from src.demo.samples import library_storage_paths
from src.models.document import Document
from src.models.question_usage import QuestionUsage
from src.models.user import User, UserRole

_BYTES_PER_MB = 1024 * 1024
_EXEMPT_ROLES = frozenset({UserRole.ADMIN, UserRole.SUPER_ADMIN})

DEMO_DAILY_LIMIT_MESSAGE = (
    "The demo has reached today's limit. Try again tomorrow, "
    "or self-host it from the README."
)


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

    async def reserve_question(self, user: User) -> None:
        """Check today's question limits, then count this question.

        Raises QuotaExceededError when the user has used their questions
        for the day, and DemoDailyLimitError when the whole site has. The
        count is committed at once, so a question counts once accepted,
        whatever happens to the answer. Nothing is recorded while both
        limits are off.
        """
        per_user = self._settings.quota_questions_per_day
        site_wide = self._settings.demo_global_questions_per_day
        if not per_user and not site_wide:
            return
        if not self._is_exempt(user):
            since = _utc_midnight()
            if per_user and await self._questions_since(since, user.id) >= per_user:
                raise QuotaExceededError(
                    f"Daily question limit reached ({per_user} per day). "
                    "It resets at 00:00 UTC."
                )
            if site_wide and await self._questions_since(since) >= site_wide:
                raise DemoDailyLimitError(DEMO_DAILY_LIMIT_MESSAGE)
        self._db.add(QuestionUsage(user_id=user.id))
        await self._db.commit()

    @staticmethod
    def _is_exempt(user: User) -> bool:
        return user.role in _EXEMPT_ROLES

    async def _questions_since(
        self, since: datetime, user_id: uuid.UUID | None = None
    ) -> int:
        """Questions counted since `since`, for one user or for everyone."""
        query = select(func.count(QuestionUsage.id)).where(
            QuestionUsage.asked_at >= since
        )
        if user_id is not None:
            query = query.where(QuestionUsage.user_id == user_id)
        return int(await self._db.scalar(query) or 0)

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


def _utc_midnight() -> datetime:
    return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
