"""
src/api/v1/auth.py
──────────────────
Authentication endpoints.

Design: Routes are intentionally thin — no business logic here.
Every route does: validate input → call service → return schema.
"""

import uuid
from collections.abc import Callable

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.configs.config import get_settings
from src.core.rate_limit import client_ip_key, limiter, setting_limit
from src.db.session import get_db
from src.dependencies import get_current_user
from src.models.user import User
from src.schemas.auth import (
    ChangePasswordRequest,
    GoogleLoginRequest,
    LoginRequest,
    RefreshTokenRequest,
    RegisterRequest,
    RegisterResponse,
    TokenResponse,
    UserResponse,
)
from src.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])


def _new_user_hook() -> Callable[[uuid.UUID], None] | None:
    """In demo mode, every new account is queued for the sample documents."""
    if not get_settings().demo_mode:
        return None
    from src.workers.demo_tasks import queue_sample_documents

    return queue_sample_documents


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
@limiter.limit(setting_limit("rate_limit_register"), key_func=client_ip_key)
async def register(
    request: Request,
    payload: RegisterRequest,
    db: AsyncSession = Depends(get_db, scope="function"),
) -> RegisterResponse:
    service = AuthService(db, on_user_created=_new_user_hook())
    user = await service.register(payload)
    return RegisterResponse(user=UserResponse.model_validate(user))


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and receive access + refresh tokens",
)
@limiter.limit(setting_limit("rate_limit_auth"), key_func=client_ip_key)
async def login(
    request: Request,
    payload: LoginRequest,
    db: AsyncSession = Depends(get_db, scope="function"),
) -> TokenResponse:
    service = AuthService(db)
    return await service.login(payload)


@router.post(
    "/google",
    response_model=TokenResponse,
    summary="Sign in / sign up with a Google ID token",
)
@limiter.limit(setting_limit("rate_limit_auth"), key_func=client_ip_key)
async def google_login(
    request: Request,
    payload: GoogleLoginRequest,
    db: AsyncSession = Depends(get_db, scope="function"),
) -> TokenResponse:
    service = AuthService(db, on_user_created=_new_user_hook())
    return await service.google_login(payload)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Exchange refresh token for new token pair",
)
@limiter.limit(setting_limit("rate_limit_auth"), key_func=client_ip_key)
async def refresh_tokens(
    request: Request,
    payload: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db, scope="function"),
) -> TokenResponse:
    service = AuthService(db)
    return await service.refresh_tokens(payload.refresh_token)


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current authenticated user profile",
)
async def get_me(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    return UserResponse.model_validate(current_user)


@router.put(
    "/me/password",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Change current user password",
)
async def change_password(
    payload: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db, scope="function"),
) -> None:
    service = AuthService(db)
    await service.change_password(
        current_user, payload.current_password, payload.new_password
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke the current access token",
)
async def logout(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db, scope="function"),
) -> None:
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    service = AuthService(db)
    await service.logout(token)
