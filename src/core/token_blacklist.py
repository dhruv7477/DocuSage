"""
src/core/token_blacklist.py
────────────────────────────
Redis-backed JWT blacklist for token revocation on logout.

Each blacklisted token is stored under key "blacklist:{jti}" with a TTL
matching the token's remaining lifetime so Redis auto-expires entries.
"""

from __future__ import annotations

import redis.asyncio as aioredis

from src.core.logger import get_logger

log = get_logger(__name__)

_redis_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        from src.configs.config import get_settings

        settings = get_settings()
        _redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis_client


async def blacklist_token(jti: str, ttl_seconds: int) -> None:
    """Add a JTI to the blacklist with the token's remaining TTL."""
    if ttl_seconds <= 0:
        return
    try:
        await get_redis().setex(f"blacklist:{jti}", ttl_seconds, "1")
    except Exception as exc:
        log.error("blacklist_set_failed", jti=jti, error=str(exc))


async def is_token_blacklisted(jti: str) -> bool:
    """Return True if the JTI has been revoked."""
    try:
        return bool(await get_redis().exists(f"blacklist:{jti}"))
    except Exception as exc:
        log.error("blacklist_check_failed", jti=jti, error=str(exc))
        return False
