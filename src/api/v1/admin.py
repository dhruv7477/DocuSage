"""
src/api/v1/admin.py
────────────────────
Admin API endpoints. All require role=SUPER_ADMIN.

Endpoints:
    GET    /admin/users                List all users (paginated, filterable)
    GET    /admin/users/{id}           Get single user
    PATCH  /admin/users/{id}           Enable/disable user, change role
    GET    /admin/stats                Platform-wide statistics
    GET    /admin/audit-logs           Paginated audit log (filterable)
    GET    /admin/audit-logs/{id}      Single audit log entry

Authentication: JWT required + role must be SUPER_ADMIN.
Rate limiting: 60 req/min (same as default, admin actions are rare).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, Field

from src.core.rate_limit import default_limit, limiter
from src.dependencies import get_current_user, require_super_admin
from src.db.session import get_db
from src.models.user import User
from src.services.admin_service import AdminService, AuditLogView, PlatformStats, UserAdminView
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/admin", tags=["Admin"])


# ── Response Schemas ──────────────────────────────────────────────

class UserAdminResponse(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    is_active: bool
    is_email_verified: bool
    created_at: datetime
    workspace_count: int
    document_count: int

    model_config = {"from_attributes": True}


class PaginatedUsersResponse(BaseModel):
    users: list[UserAdminResponse]
    total: int
    page: int
    page_size: int


class UpdateUserRequest(BaseModel):
    is_active: bool | None = None
    role: str | None = Field(
        default=None,
        description="member | admin — cannot set super_admin via API",
    )


class PlatformStatsResponse(BaseModel):
    total_users: int
    active_users: int
    total_workspaces: int
    total_documents: int
    total_chats: int
    total_messages: int
    documents_by_status: dict[str, int]


class AuditLogResponse(BaseModel):
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


class PaginatedAuditLogsResponse(BaseModel):
    logs: list[AuditLogResponse]
    total: int
    page: int
    page_size: int


# ── Dependency ────────────────────────────────────────────────────

def _admin_service(db: AsyncSession = Depends(get_db)) -> AdminService:
    return AdminService(db)


# ── User Endpoints ────────────────────────────────────────────────

@router.get(
    "/users",
    response_model=PaginatedUsersResponse,
    summary="List all users",
)
@limiter.limit(default_limit())
async def list_users(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    role: str | None = Query(None, description="Filter by role"),
    is_active: bool | None = Query(None, description="Filter by active status"),
    search: str | None = Query(None, description="Search by email or name"),
    _admin: User = Depends(require_super_admin),
    service: AdminService = Depends(_admin_service),
) -> PaginatedUsersResponse:
    """
    List all users across the platform. Supports pagination and filtering.
    Only accessible to SUPER_ADMIN.
    """
    users, total = await service.list_users(
        page=page,
        page_size=page_size,
        role=role,
        is_active=is_active,
        search=search,
    )
    return PaginatedUsersResponse(
        users=[_user_view_to_response(u) for u in users],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/users/{user_id}",
    response_model=UserAdminResponse,
    summary="Get single user",
)
@limiter.limit(default_limit())
async def get_user(
    request: Request,
    user_id: uuid.UUID,
    _admin: User = Depends(require_super_admin),
    service: AdminService = Depends(_admin_service),
) -> UserAdminResponse:
    """Get a single user by ID with usage statistics."""
    user = await service.get_user(user_id)
    return _user_view_to_response(user)


@router.patch(
    "/users/{user_id}",
    response_model=UserAdminResponse,
    summary="Update user status or role",
)
@limiter.limit(default_limit())
async def update_user(
    request: Request,
    user_id: uuid.UUID,
    payload: UpdateUserRequest,
    admin: User = Depends(require_super_admin),
    service: AdminService = Depends(_admin_service),
) -> UserAdminResponse:
    """
    Enable/disable a user or change their role.

    Guards:
    - Cannot deactivate your own account.
    - Cannot promote to SUPER_ADMIN via API.
    """
    user = await service.update_user(
        user_id,
        is_active=payload.is_active,
        role=payload.role,
        acting_admin_id=admin.id,
    )
    return _user_view_to_response(user)


# ── Stats Endpoint ────────────────────────────────────────────────

@router.get(
    "/stats",
    response_model=PlatformStatsResponse,
    summary="Platform statistics",
)
@limiter.limit(default_limit())
async def get_stats(
    request: Request,
    _admin: User = Depends(require_super_admin),
    service: AdminService = Depends(_admin_service),
) -> PlatformStatsResponse:
    """
    Aggregate platform statistics: user counts, document counts,
    storage usage, and document status breakdown.
    """
    stats = await service.get_stats()
    return PlatformStatsResponse(
        total_users=stats.total_users,
        active_users=stats.active_users,
        total_workspaces=stats.total_workspaces,
        total_documents=stats.total_documents,
        total_chats=stats.total_chats,
        total_messages=stats.total_messages,
        documents_by_status=stats.documents_by_status,
    )


# ── Audit Log Endpoints ───────────────────────────────────────────

@router.get(
    "/audit-logs",
    response_model=PaginatedAuditLogsResponse,
    summary="List audit logs",
)
@limiter.limit(default_limit())
async def list_audit_logs(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
    user_id: str | None = Query(None, description="Filter by user UUID"),
    action: str | None = Query(None, description="Filter by action e.g. 'document.upload'"),
    status_code: int | None = Query(None, description="Filter by HTTP status code"),
    since: datetime | None = Query(None, description="ISO datetime lower bound"),
    until: datetime | None = Query(None, description="ISO datetime upper bound"),
    _admin: User = Depends(require_super_admin),
    service: AdminService = Depends(_admin_service),
) -> PaginatedAuditLogsResponse:
    """
    Paginated audit log with filtering by user, action, status code,
    and time range. Most recent entries first.
    """
    logs, total = await service.list_audit_logs(
        page=page,
        page_size=page_size,
        user_id=user_id,
        action=action,
        status_code=status_code,
        since=since,
        until=until,
    )
    return PaginatedAuditLogsResponse(
        logs=[_audit_view_to_response(entry) for entry in logs],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/audit-logs/{log_id}",
    response_model=AuditLogResponse,
    summary="Get single audit log entry",
)
@limiter.limit(default_limit())
async def get_audit_log(
    request: Request,
    log_id: int,
    _admin: User = Depends(require_super_admin),
    service: AdminService = Depends(_admin_service),
) -> AuditLogResponse:
    """Fetch a single audit log entry by integer ID."""
    entry = await service.get_audit_log(log_id)
    return _audit_view_to_response(entry)


# ── Helpers ───────────────────────────────────────────────────────

def _user_view_to_response(u: UserAdminView) -> UserAdminResponse:
    return UserAdminResponse(
        id=u.id,
        email=u.email,
        full_name=u.full_name,
        role=u.role,
        is_active=u.is_active,
        is_email_verified=u.is_email_verified,
        created_at=u.created_at,
        workspace_count=u.workspace_count,
        document_count=u.document_count,
    )


def _audit_view_to_response(entry: AuditLogView) -> AuditLogResponse:
    return AuditLogResponse(
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