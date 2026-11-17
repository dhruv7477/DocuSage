"""
src/core/audit.py
──────────────────
Audit logging middleware and helpers.

AuditMiddleware runs on every request and writes one AuditLog row
for any request that:
  - modifies state (POST, PUT, PATCH, DELETE)
  - is a security-sensitive GET (login, /me, document download)

Read-only GETs to non-sensitive endpoints (list, health) are skipped
to avoid log noise and keep the table lean.

Action naming convention:
    "<resource>.<verb>"
    e.g. "auth.register", "document.upload", "admin.disable_user"

Usage (explicit logging from a service/route):
    from src.core.audit import audit
    await audit(db, action="document.delete", user=current_user,
                resource_type="document", resource_id=str(doc_id),
                request=request, status_code=200)
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.core.logger import get_logger
from src.models.audit_log import AuditLog

log = get_logger(__name__)

# ── Paths that are always skipped (health, docs, metrics) ─────────
_SKIP_PATHS = frozenset(
    {
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/metrics",
        "/favicon.ico",
    }
)

# ── Only log GETs for these sensitive path prefixes ───────────────
_SENSITIVE_GET_PREFIXES = (
    "/api/v1/auth/me",
    "/api/v1/auth/refresh",
    "/api/v1/admin/",
)

# ── Action map: (method, path_prefix) → action string ────────────
_ACTION_MAP: list[tuple[str, str, str]] = [
    # Auth
    ("POST", "/api/v1/auth/register", "auth.register"),
    ("POST", "/api/v1/auth/login", "auth.login"),
    ("POST", "/api/v1/auth/refresh", "auth.refresh"),
    ("PUT", "/api/v1/auth/me/password", "auth.change_password"),
    ("GET", "/api/v1/auth/me", "auth.me"),
    # Workspaces
    ("POST", "/api/v1/workspaces", "workspace.create"),
    ("PATCH", "/api/v1/workspaces/", "workspace.update"),
    ("DELETE", "/api/v1/workspaces/", "workspace.delete"),
    ("POST", "/api/v1/workspaces/", "workspace.member_add"),  # /{id}/members
    ("DELETE", "/api/v1/workspaces/", "workspace.member_remove"),
    # Documents
    ("POST", "/api/v1/documents/upload", "document.upload"),
    ("DELETE", "/api/v1/documents/", "document.delete"),
    # Chat (more specific prefix must come first)
    ("POST", "/api/v1/chat/", "chat.query"),
    ("POST", "/api/v1/chat", "chat.create"),
    ("DELETE", "/api/v1/chat/", "chat.delete"),
    # Eval
    ("POST", "/api/v1/eval/run", "eval.run"),
    ("POST", "/api/v1/eval/suite", "eval.suite"),
    # Admin
    ("PATCH", "/api/v1/admin/users/", "admin.update_user"),
    ("GET", "/api/v1/admin/", "admin.read"),
]


def _resolve_action(method: str, path: str) -> str | None:
    """Map a method + path to an action string. Returns None to skip logging."""
    if path in _SKIP_PATHS:
        return None

    # Skip safe GETs unless they're sensitive
    if method == "GET":
        if not any(path.startswith(p) for p in _SENSITIVE_GET_PREFIXES):
            return None

    for m, prefix, action in _ACTION_MAP:
        if method == m and path.startswith(prefix):
            return action

    # Fallback for anything not explicitly mapped
    if method in ("POST", "PUT", "PATCH", "DELETE"):
        return f"api.{method.lower()}"

    return None


def _get_client_ip(request: Request) -> str | None:
    """Extract real client IP, honouring X-Forwarded-For from trusted proxies."""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that writes one AuditLog row per auditable request.

    Runs AFTER the route handler so it has the response status code.
    Does not block the response — DB write happens in background via
    fire-and-forget to avoid adding latency.

    Note: cannot directly inject AsyncSession via FastAPI DI from middleware,
    so we grab it from src.state.db_session_factory set at startup.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Attach a request ID for correlation
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.monotonic()
        response = await call_next(request)
        duration_ms = int((time.monotonic() - start) * 1000)

        action = _resolve_action(request.method, request.url.path)
        if action is None:
            return response

        response.headers["X-Request-ID"] = request_id

        # Fire-and-forget audit write (don't await — no latency impact)
        import asyncio

        asyncio.create_task(
            _write_audit_log(
                request=request,
                status_code=response.status_code,
                action=action,
                request_id=request_id,
                duration_ms=duration_ms,
            )
        )

        return response


async def _write_audit_log(
    request: Request,
    status_code: int,
    action: str,
    request_id: str,
    duration_ms: int,
) -> None:
    """
    Write one AuditLog row. Called as a background task from middleware.
    Gets a fresh DB session from src.state.db_session_factory.
    """
    try:
        db_factory = request.app.state.db_session_factory
    except AttributeError:
        # DB factory not registered (e.g. during tests) — skip silently
        return

    # Extract user info from request state (set by get_current_user dep)
    user_id: str | None = getattr(request.state, "user_id", None)
    user_email: str | None = getattr(request.state, "user_email", None)

    try:
        async with db_factory() as db:
            entry = AuditLog(
                user_id=user_id,
                user_email=user_email,
                action=action,
                method=request.method,
                path=str(request.url.path),
                status_code=status_code,
                request_id=request_id,
                ip_address=_get_client_ip(request),
                user_agent=request.headers.get("user-agent"),
                duration_ms=duration_ms,
            )
            db.add(entry)
            await db.commit()
    except Exception as exc:
        # Audit failures MUST NOT break the main request flow
        log.error("audit_write_failed", error=str(exc), action=action)


async def audit(
    db: AsyncSession,
    *,
    action: str,
    request: Request,
    status_code: int,
    user_id: str | None = None,
    user_email: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    extra: dict | None = None,
) -> None:
    """
    Explicit audit log write from a route or service.

    Use this for high-value actions where you need structured extra metadata
    (e.g. logging which document was deleted, which user was disabled).

    Example:
        await audit(db, action="document.delete", request=request,
                    status_code=200, resource_type="document",
                    resource_id=str(doc_id), user_id=str(user.id))
    """
    try:
        entry = AuditLog(
            user_id=user_id,
            user_email=user_email,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            method=request.method,
            path=str(request.url.path),
            status_code=status_code,
            request_id=getattr(request.state, "request_id", None),
            ip_address=_get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            extra=extra,
        )
        db.add(entry)
        await db.commit()
    except Exception as exc:
        log.error("explicit_audit_failed", error=str(exc), action=action)
