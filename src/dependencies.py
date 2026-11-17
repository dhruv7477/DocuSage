"""
src/dependencies.py
────────────────────
FastAPI dependency injection — shared guards used across all routes.

Usage in routes:
    async def my_route(current_user: User = Depends(get_current_user)): ...
    async def admin_route(user: User = Depends(require_admin)): ...

These are the ONLY place OAuth2 bearer extraction happens.
Services receive User objects, never raw tokens.
"""

import uuid

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.configs.config import get_settings
from src.core.exceptions import (
    InsufficientPermissionsError,
    TokenInvalidError,
    UserInactiveError,
    UserNotFoundError,
)
from src.core.security import decode_token, extract_user_id
from src.core.token_blacklist import is_token_blacklisted
from src.db.session import get_db
from src.models.user import User, UserRole

# Extracts "Authorization: Bearer <token>" header
_bearer = HTTPBearer(auto_error=True)

_settings = get_settings()


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Core auth dependency.
    Decodes JWT, checks blacklist, fetches User from DB, checks is_active.
    Inject into any route that requires authentication.
    """
    payload = decode_token(
        credentials.credentials, _settings.secret_key, _settings.algorithm
    )

    jti = payload.get("jti")
    if jti and await is_token_blacklisted(jti):
        raise TokenInvalidError("Token has been revoked. Please log in again.")

    user_id = extract_user_id(payload)
    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()

    if not user:
        raise UserNotFoundError("User not found.")
    if not user.is_active:
        raise UserInactiveError("Account is deactivated.")

    request.state.user_id = str(user.id)
    request.state.user_email = user.email
    return user


async def require_active_user(
    user: User = Depends(get_current_user),
) -> User:
    """Alias for get_current_user — use when you want explicit naming."""
    return user


async def require_admin(
    user: User = Depends(get_current_user),
) -> User:
    """
    Requires global SUPER_ADMIN or ADMIN role.
    Use for platform-level admin endpoints.
    """
    if user.role not in (UserRole.SUPER_ADMIN, UserRole.ADMIN):
        raise InsufficientPermissionsError(
            "Admin role required.",
            detail="Your account does not have admin privileges.",
        )
    return user


async def require_super_admin(
    user: User = Depends(get_current_user),
) -> User:
    """Requires global SUPER_ADMIN role. Use for billing/platform endpoints."""
    if user.role != UserRole.SUPER_ADMIN:
        raise InsufficientPermissionsError("Super admin role required.")
    return user


def get_auth_service(db: AsyncSession = Depends(get_db)):
    """Dependency factory for AuthService."""
    from src.services.auth_service import AuthService

    return AuthService(db)
