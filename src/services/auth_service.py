"""
src/services/auth_service.py
─────────────────────────────
Authentication business logic.

Responsibilities:
- Register a new user
- Authenticate (login) and issue tokens
- Refresh access token via refresh token
- Change password
- Get current user from token

Design:
- No FastAPI imports — purely domain logic
- Receives AsyncSession; does NOT create its own
- All DB queries use select() — no lazy loading (async-safe)
"""

import uuid

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.configs.config import get_settings
from src.core.exceptions import (
    AuthenticationError,
    TokenInvalidError,
    UserAlreadyExistsError,
    UserInactiveError,
    UserNotFoundError,
)
from src.core.logger import get_logger
from src.core.security import (
    REFRESH_TOKEN_TYPE,
    create_access_token,
    create_refresh_token,
    decode_token,
    extract_user_id,
    hash_password,
    verify_password,
)
from src.core.token_blacklist import blacklist_token
from src.models.user import User, UserRole
from src.schemas.auth import (
    GoogleLoginRequest,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
)

log = get_logger(__name__)
settings = get_settings()


class AuthService:
    """
    Stateless service — instantiated per request with a DB session.
    All methods are async to match the async DB driver.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ── Registration ──────────────────────────────────────────────

    async def register(self, payload: RegisterRequest) -> User:
        """
        Create a new user account.

        Raises:
            UserAlreadyExistsError: if email already registered
        """
        existing = await self._get_user_by_email(payload.email)
        if existing:
            raise UserAlreadyExistsError(
                f"Email '{payload.email}' is already registered.",
                detail="Please use a different email or log in.",
            )

        user = User(
            email=payload.email,
            password_hash=hash_password(payload.password),
            full_name=payload.full_name,
            role=UserRole.MEMBER,
            is_active=True,
            is_email_verified=False,
        )
        self._db.add(user)
        await self._db.flush()  # get ID without full commit

        log.info("user_registered", user_id=str(user.id), email=user.email)
        return user

    # ── Login ─────────────────────────────────────────────────────

    async def login(self, payload: LoginRequest) -> TokenResponse:
        """
        Authenticate user and return access + refresh tokens.

        Raises:
            AuthenticationError: invalid credentials
            UserInactiveError: account disabled
        """
        user = await self._get_user_by_email(payload.email)

        # Use same error for "not found" and "wrong password" — prevents
        # user enumeration attacks via timing or error message differences
        if not user or not user.password_hash or not verify_password(payload.password, user.password_hash):
            raise AuthenticationError(
                "Invalid email or password.",
                detail="Please check your credentials and try again.",
            )

        if not user.is_active:
            raise UserInactiveError(
                "Your account has been deactivated.",
                detail="Please contact support.",
            )

        tokens = self._issue_tokens(user)
        log.info("user_login", user_id=str(user.id), email=user.email)
        return tokens

    # ── Google OAuth ──────────────────────────────────────────────

    async def google_login(self, payload: GoogleLoginRequest) -> TokenResponse:
        """
        Verify a Google ID token and return our JWT pair.
        Creates the user on first sign-in (social sign-up).

        Raises:
            AuthenticationError: if the token is invalid or client_id not configured
            UserInactiveError: if the matched account is disabled
        """
        if not settings.google_client_id:
            raise AuthenticationError(
                "Google sign-in is not configured.",
                detail="GOOGLE_CLIENT_ID is not set on the server.",
            )

        try:
            id_info = google_id_token.verify_oauth2_token(
                payload.credential,
                google_requests.Request(),
                settings.google_client_id,
            )
        except Exception:
            raise AuthenticationError(
                "Invalid Google credential.",
                detail="The provided Google ID token could not be verified.",
            )

        email: str = id_info["email"].lower().strip()
        full_name: str = id_info.get("name", email.split("@")[0])
        avatar_url: str | None = id_info.get("picture")

        user = await self._get_user_by_email(email)

        if user is None:
            user = User(
                email=email,
                password_hash=None,
                full_name=full_name,
                role=UserRole.MEMBER,
                is_active=True,
                is_email_verified=True,
                avatar_url=avatar_url,
            )
            self._db.add(user)
            await self._db.flush()
            log.info("google_user_registered", user_id=str(user.id), email=email)
        else:
            if not user.is_active:
                raise UserInactiveError(
                    "Your account has been deactivated.",
                    detail="Please contact support.",
                )
            if avatar_url and not user.avatar_url:
                user.avatar_url = avatar_url
            log.info("google_user_login", user_id=str(user.id), email=email)

        return self._issue_tokens(user)

    # ── Token Refresh ─────────────────────────────────────────────

    async def refresh_tokens(self, refresh_token: str) -> TokenResponse:
        """
        Exchange a valid refresh token for a new token pair (rotation).
        The old refresh token is blacklisted immediately after use.

        Raises:
            TokenExpiredError / TokenInvalidError from decode_token
            UserNotFoundError / UserInactiveError if account changed
        """
        from datetime import UTC, datetime
        payload = decode_token(
            refresh_token,
            settings.secret_key,
            settings.algorithm,
            expected_type=REFRESH_TOKEN_TYPE,
        )

        # Blacklist the consumed refresh token (rotation — one-time use)
        jti = payload.get("jti")
        if jti:
            from src.core.token_blacklist import is_token_blacklisted
            if await is_token_blacklisted(jti):
                raise TokenInvalidError("Refresh token has already been used.")
            exp = payload.get("exp")
            ttl = max(0, int(exp - datetime.now(UTC).timestamp())) if exp else 0
            await blacklist_token(jti, ttl)

        user_id = extract_user_id(payload)
        user = await self._get_user_by_id(uuid.UUID(user_id))

        if not user:
            raise UserNotFoundError("User associated with token no longer exists.")
        if not user.is_active:
            raise UserInactiveError("Account is deactivated.")

        tokens = self._issue_tokens(user)
        log.info("tokens_refreshed", user_id=str(user.id))
        return tokens

    # ── Change Password ───────────────────────────────────────────

    async def change_password(
        self, user: User, current_password: str, new_password: str
    ) -> None:
        """
        Verify current password then update to new hash.

        Raises:
            AuthenticationError: if current password is wrong
        """
        # Re-fetch user in this session so SQLAlchemy tracks the change
        tracked_user = await self._get_user_by_id(user.id)
        if not tracked_user:
            raise AuthenticationError("Current password is incorrect.")

        if not tracked_user.password_hash or not verify_password(current_password, tracked_user.password_hash):
            raise AuthenticationError("Current password is incorrect.")

        tracked_user.password_hash = hash_password(new_password)
        await self._db.flush()
        log.info("password_changed", user_id=str(user.id))

    # ── Logout ───────────────────────────────────────────────────

    async def logout(self, token: str) -> None:
        """Blacklist the access token so it cannot be used after logout."""
        from datetime import UTC, datetime
        try:
            payload = decode_token(token, settings.secret_key, settings.algorithm)
        except Exception:
            return  # already expired or invalid — nothing to revoke
        jti = payload.get("jti")
        if not jti:
            return
        exp = payload.get("exp")
        ttl = max(0, int(exp - datetime.now(UTC).timestamp())) if exp else 0
        await blacklist_token(jti, ttl)
        log.info("token_revoked", jti=jti)

    # ── Get Current User ──────────────────────────────────────────

    async def get_current_user(self, token: str) -> User:
        """
        Decode access token and return the corresponding User.
        Used by FastAPI dependencies (dependencies.py).

        Raises:
            TokenExpiredError / TokenInvalidError
            UserNotFoundError / UserInactiveError
        """
        payload = decode_token(
            token, settings.secret_key, settings.algorithm
        )
        user_id = extract_user_id(payload)
        user = await self._get_user_by_id(uuid.UUID(user_id))

        if not user:
            raise UserNotFoundError("User not found.")
        if not user.is_active:
            raise UserInactiveError("Account is deactivated.")

        return user

    # ── Private Helpers ───────────────────────────────────────────

    def _issue_tokens(self, user: User) -> TokenResponse:
        """Create access + refresh token pair for a user."""
        access_token = create_access_token(
            subject=user.id,
            secret_key=settings.secret_key,
            algorithm=settings.algorithm,
            expires_minutes=settings.access_token_expire_minutes,
            extra_claims={"role": str(user.role), "email": user.email},
        )
        refresh_token = create_refresh_token(
            subject=user.id,
            secret_key=settings.secret_key,
            algorithm=settings.algorithm,
            expires_days=settings.refresh_token_expire_days,
        )
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.access_token_expire_minutes * 60,
        )

    async def _get_user_by_email(self, email: str) -> User | None:
        result = await self._db.execute(
            select(User).where(User.email == email.lower())
        )
        return result.scalar_one_or_none()

    async def _get_user_by_id(self, user_id: uuid.UUID) -> User | None:
        result = await self._db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
