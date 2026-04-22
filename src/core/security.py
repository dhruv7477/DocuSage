"""
src/core/security.py
────────────────────
All cryptographic operations live here and ONLY here:
- Password hashing / verification (bcrypt)
- JWT access token creation
- JWT refresh token creation
- Token decoding & validation

Design: Pure functions, no FastAPI dependencies.
This makes it trivially unit-testable and reusable.
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from jose import JWTError, ExpiredSignatureError, jwt
from passlib.context import CryptContext

from src.core.exceptions import TokenExpiredError, TokenInvalidError
from src.core.logger import get_logger

log = get_logger(__name__)

# bcrypt context — auto-rehashes if work factor changes
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Token type constants — prevents confusion between access & refresh tokens
ACCESS_TOKEN_TYPE = "access"
REFRESH_TOKEN_TYPE = "refresh"


# ── Password ─────────────────────────────────────────────────────

def hash_password(plain_password: str) -> str:
    """Return bcrypt hash of the given password."""
    return _pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Constant-time comparison — safe against timing attacks."""
    return _pwd_context.verify(plain_password, hashed_password)


# ── Token Creation ───────────────────────────────────────────────

def create_access_token(
    subject: str | UUID,
    secret_key: str,
    algorithm: str,
    expires_minutes: int,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """
    Create a short-lived JWT access token.

    Args:
        subject:       User ID (UUID or str) — stored in 'sub' claim
        secret_key:    App secret key from settings
        algorithm:     Signing algorithm (HS256)
        expires_minutes: Token lifetime in minutes
        extra_claims:  Optional dict (e.g. {"role": "admin"}) merged into payload

    Returns:
        Signed JWT string
    """
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "jti": str(uuid4()),
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
        "type": ACCESS_TOKEN_TYPE,
    }
    if extra_claims:
        payload.update(extra_claims)

    return jwt.encode(payload, secret_key, algorithm=algorithm)


def create_refresh_token(
    subject: str | UUID,
    secret_key: str,
    algorithm: str,
    expires_days: int,
) -> str:
    """
    Create a long-lived JWT refresh token.
    Refresh tokens only carry 'sub' and 'type' — minimal surface area.
    """
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "jti": str(uuid4()),
        "iat": now,
        "exp": now + timedelta(days=expires_days),
        "type": REFRESH_TOKEN_TYPE,
    }
    return jwt.encode(payload, secret_key, algorithm=algorithm)


# ── Token Decoding ────────────────────────────────────────────────

def decode_token(
    token: str,
    secret_key: str,
    algorithm: str,
    expected_type: str = ACCESS_TOKEN_TYPE,
) -> dict[str, Any]:
    """
    Decode and validate a JWT token.

    Raises:
        TokenExpiredError: if token is past expiry
        TokenInvalidError: if signature is invalid, malformed, or wrong type
    """
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
    except ExpiredSignatureError:
        raise TokenExpiredError("Token has expired. Please log in again.")
    except JWTError as exc:
        log.warning("jwt_decode_failed", error=str(exc))
        raise TokenInvalidError("Token is invalid or tampered.") from exc

    token_type = payload.get("type")
    if token_type != expected_type:
        raise TokenInvalidError(
            f"Expected token type '{expected_type}', got '{token_type}'."
        )

    return payload


def extract_user_id(payload: dict[str, Any]) -> str:
    """Extract 'sub' claim from decoded token payload."""
    subject = payload.get("sub")
    if not subject:
        raise TokenInvalidError("Token missing 'sub' claim.")
    return subject