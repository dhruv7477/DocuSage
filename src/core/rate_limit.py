"""
src/core/rate_limit.py
───────────────────────
Centralised rate limiting using slowapi (Starlette-native).

Design:
- One shared Limiter instance imported everywhere.
- Limits are applied per authenticated user (by JWT sub claim).
  Falls back to IP address for unauthenticated endpoints (login, register).
- Two tiers:
    DEFAULT_LIMIT  — general API calls (60/min from config)
    CHAT_LIMIT     — chat/query endpoints (20/min from config) — more
                     expensive because they hit the LLM.

Usage in routes:
    from src.core.rate_limit import limiter, chat_limit, default_limit

    @router.post("/query")
    @limiter.limit(chat_limit())
    async def query(request: Request, ...):
        ...

Note: `request: Request` MUST be a parameter of the route function for
slowapi to extract the client key. It doesn't have to be used in the body.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from src.core.logger import get_logger

log = get_logger(__name__)


def _get_rate_limit_key(request) -> str:  # type: ignore[no-untyped-def]
    """
    Rate limit key strategy:
    - Authenticated requests: keyed by user ID (from JWT sub claim).
      This means all devices for the same user share a quota.
    - Unauthenticated requests: keyed by IP address.
    """
    # FastAPI auth dep stores user_id on request.state after verifying JWT
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        return f"user:{user_id}"
    return get_remote_address(request)


# ── Shared limiter instance ───────────────────────────────────────
# Imported by main.py and individual routers.
limiter = Limiter(key_func=_get_rate_limit_key, default_limits=[])


def default_limit() -> str:
    """Return the default rate limit string (reads from settings lazily)."""
    from src.configs.config import get_settings

    s = get_settings()
    return f"{s.rate_limit_per_minute}/minute"


def chat_limit() -> str:
    """Return the chat-specific rate limit string (reads from settings lazily)."""
    from src.configs.config import get_settings

    s = get_settings()
    return f"{s.rate_limit_chat_per_minute}/minute"
