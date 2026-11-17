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

from collections.abc import Callable

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
def _build_limiter() -> Limiter:
    from src.configs.config import get_settings

    s = get_settings()
    return Limiter(
        key_func=_get_rate_limit_key,
        default_limits=[],
        storage_uri=s.rate_limit_storage_uri,
        enabled=s.rate_limit_enabled,
        # Count per route, not per URL: with "url", every chat ID and
        # document ID in the path would get its own budget.
        key_style="endpoint",
        # If the limit store (Redis) is unreachable, serve the request
        # instead of failing it. slowapi logs every such error.
        swallow_errors=True,
    )


limiter = _build_limiter()


def client_ip_key(request) -> str:  # type: ignore[no-untyped-def]
    """Key for routes used before sign-in (register, login, refresh).

    Behind a proxy this is the real client address only when uvicorn trusts
    the proxy (FORWARDED_ALLOW_IPS), as the demo compose sets up.
    """
    return f"ip:{get_remote_address(request)}"


def setting_limit(name: str) -> Callable[[], str]:
    """Limit read from a settings field each time, for example "rate_limit_auth"."""

    def _limit() -> str:
        from src.configs.config import get_settings

        return str(getattr(get_settings(), name))

    return _limit


# Scope shared by every route that asks the LLM a question (stream and
# query), so one budget covers all of a user's chats.
QUESTION_SCOPE = "questions"


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
