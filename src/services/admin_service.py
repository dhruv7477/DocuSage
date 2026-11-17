"""
src/services/admin_service.py
──────────────────────────────
Admin service: user management, platform statistics, audit log queries.

Only accessible to users with role=SUPER_ADMIN (enforced at API layer
via require_super_admin dependency).

Methods:
    list_users()        Paginated list of all users with filters.
    get_user()          Get a single user by ID.
    update_user()       Enable/disable users, change roles.
    get_stats()         Platform-wide aggregate statistics.
    list_audit_logs()   Paginated, filterable audit log query.
    get_audit_log()     Single audit log entry by ID.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import (
    InsufficientPermissionsError,
    UserNotFoundError,
)
from src.core.logger import get_logger
from src.models.audit_log import AuditLog
from src.models.chat import Chat, Message
from src.models.document import Document
from src.models.user import User, UserRole
from src.models.workspace import Workspace

log = get_logger(__name__)


# ── Response dataclasses ──────────────────────────────────────────


@dataclass
class UserAdminView:
    id: str
    email: str
    full_name: str
    role: str
    is_active: bool
    is_email_verified: bool
    created_at: datetime
    workspace_count: int
    document_count: int


@dataclass
class PlatformStats:
    total_users: int
    active_users: int
    total_workspaces: int
    total_documents: int
    total_chats: int
    total_messages: int
    documents_by_status: dict[str, int]


@dataclass
class AuditLogView:
    id: int
    user_id: str | None
    user_email: str | None
    action: str
    resource_type: str | None
    resource_id: str | None
    method: str
    path: str
    status_code: int
    ip_address: str | None
    duration_ms: int | None
    extra: dict | None
    created_at: datetime


class AdminService:
    """
    Privileged operations available only to SUPER_ADMIN users.
    All methods take an explicit `db` session (no global state).
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ── User Management ───────────────────────────────────────────

    async def list_users(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
        role: str | None = None,
        is_active: bool | None = None,
        search: str | None = None,
    ) -> tuple[list[UserAdminView], int]:
        """
        Paginated user list with optional filters.

        Returns:
            (list of UserAdminView, total_count)
        """
        stmt = select(User)

        if role is not None:
            stmt = stmt.where(User.role == role)
        if is_active is not None:
            stmt = stmt.where(User.is_active == is_active)
        if search:
            pattern = f"%{search.lower()}%"
            stmt = stmt.where(
                func.lower(User.email).like(pattern)
                | func.lower(User.full_name).like(pattern)
            )

        # Count before pagination
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self._db.execute(count_stmt)).scalar_one()

        # Paginate
        stmt = stmt.order_by(User.created_at.desc())
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        users = (await self._db.execute(stmt)).scalars().all()

        views = []
        for user in users:
            ws_count = await self._count_user_workspaces(user.id)
            doc_count = await self._count_user_documents(user.id)
            views.append(
                UserAdminView(
                    id=str(user.id),
                    email=user.email,
                    full_name=user.full_name,
                    role=user.role.value
                    if hasattr(user.role, "value")
                    else str(user.role),
                    is_active=user.is_active,
                    is_email_verified=user.is_email_verified,
                    created_at=user.created_at,
                    workspace_count=ws_count,
                    document_count=doc_count,
                )
            )

        return views, total

    async def get_user(self, user_id: uuid.UUID) -> UserAdminView:
        """Get a single user by ID for admin review."""
        user = await self._fetch_user(user_id)
        ws_count = await self._count_user_workspaces(user.id)
        doc_count = await self._count_user_documents(user.id)
        return UserAdminView(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            role=user.role.value if hasattr(user.role, "value") else str(user.role),
            is_active=user.is_active,
            is_email_verified=user.is_email_verified,
            created_at=user.created_at,
            workspace_count=ws_count,
            document_count=doc_count,
        )

    async def update_user(
        self,
        user_id: uuid.UUID,
        *,
        is_active: bool | None = None,
        role: str | None = None,
        acting_admin_id: uuid.UUID,
    ) -> UserAdminView:
        """
        Update user status or role.

        Guards:
        - Admins cannot deactivate themselves.
        - Cannot promote to SUPER_ADMIN via this endpoint
          (requires direct DB change — prevents privilege escalation).
        """
        if user_id == acting_admin_id and is_active is False:
            raise InsufficientPermissionsError(
                "Admins cannot deactivate their own account."
            )

        if role == UserRole.SUPER_ADMIN.value:
            raise InsufficientPermissionsError("Cannot promote to SUPER_ADMIN via API.")

        user = await self._fetch_user(user_id)

        updates: dict = {}
        if is_active is not None:
            updates["is_active"] = is_active
        if role is not None:
            updates["role"] = role

        if updates:
            stmt = update(User).where(User.id == user_id).values(**updates)
            await self._db.execute(stmt)
            await self._db.commit()
            await self._db.refresh(user)

        log.info(
            "admin_user_updated",
            target_user=str(user_id),
            acting_admin=str(acting_admin_id),
            changes=updates,
        )

        return await self.get_user(user_id)

    # ── Platform Statistics ───────────────────────────────────────

    async def get_stats(self) -> PlatformStats:
        """Aggregate platform statistics for admin dashboard."""
        total_users = await self._scalar(select(func.count(User.id)))
        active_users = await self._scalar(
            select(func.count(User.id)).where(User.is_active == True)  # noqa: E712
        )
        total_workspaces = await self._scalar(select(func.count(Workspace.id)))
        total_documents = await self._scalar(select(func.count(Document.id)))
        total_chats = await self._scalar(select(func.count(Chat.id)))
        total_messages = await self._scalar(select(func.count(Message.id)))

        # Documents grouped by status
        status_rows = (
            await self._db.execute(
                select(Document.status, func.count(Document.id)).group_by(
                    Document.status
                )
            )
        ).all()
        documents_by_status = {
            (row[0].value if hasattr(row[0], "value") else str(row[0])): row[1]
            for row in status_rows
        }

        return PlatformStats(
            total_users=total_users,
            active_users=active_users,
            total_workspaces=total_workspaces,
            total_documents=total_documents,
            total_chats=total_chats,
            total_messages=total_messages,
            documents_by_status=documents_by_status,
        )

    # ── Audit Logs ────────────────────────────────────────────────

    async def list_audit_logs(
        self,
        *,
        page: int = 1,
        page_size: int = 100,
        user_id: str | None = None,
        action: str | None = None,
        status_code: int | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> tuple[list[AuditLogView], int]:
        """
        Paginated, filtered audit log query.

        Returns:
            (list of AuditLogView, total_count)
        """
        stmt = select(AuditLog)

        if user_id:
            stmt = stmt.where(AuditLog.user_id == user_id)
        if action:
            stmt = stmt.where(AuditLog.action == action)
        if status_code:
            stmt = stmt.where(AuditLog.status_code == status_code)
        if since:
            stmt = stmt.where(AuditLog.created_at >= since)
        if until:
            stmt = stmt.where(AuditLog.created_at <= until)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self._db.execute(count_stmt)).scalar_one()

        stmt = stmt.order_by(AuditLog.created_at.desc())
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        rows = (await self._db.execute(stmt)).scalars().all()

        return [_audit_to_view(r) for r in rows], total

    async def get_audit_log(self, log_id: int) -> AuditLogView:
        """Fetch a single audit log entry by integer ID."""
        result = await self._db.execute(select(AuditLog).where(AuditLog.id == log_id))
        entry = result.scalar_one_or_none()
        if not entry:
            from src.core.exceptions import PrivateAIException
            from http import HTTPStatus

            raise PrivateAIException(
                message=f"Audit log {log_id} not found.",
                status_code=HTTPStatus.NOT_FOUND.value,
                error_code="AUDIT_LOG_NOT_FOUND",
            )
        return _audit_to_view(entry)

    # ── Private helpers ───────────────────────────────────────────

    async def _fetch_user(self, user_id: uuid.UUID) -> User:
        result = await self._db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise UserNotFoundError(f"User {user_id} not found.")
        return user

    async def _count_user_workspaces(self, user_id) -> int:
        return await self._scalar(
            select(func.count(Workspace.id)).where(Workspace.owner_id == user_id)
        )

    async def _count_user_documents(self, user_id) -> int:
        return await self._scalar(
            select(func.count(Document.id)).where(Document.user_id == user_id)
        )

    async def _scalar(self, stmt) -> int:
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none() or 0


def _audit_to_view(entry: AuditLog) -> AuditLogView:
    return AuditLogView(
        id=entry.id,
        user_id=entry.user_id,
        user_email=entry.user_email,
        action=entry.action,
        resource_type=entry.resource_type,
        resource_id=entry.resource_id,
        method=entry.method,
        path=entry.path,
        status_code=entry.status_code,
        ip_address=entry.ip_address,
        duration_ms=entry.duration_ms,
        extra=entry.extra,
        created_at=entry.created_at,
    )
