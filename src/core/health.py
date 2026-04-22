"""
src/core/health.py
───────────────────
Startup and runtime health checks.

Called during app lifespan startup to verify all required services
are reachable before accepting traffic. Also powers the /health/ready
endpoint used by load balancers and Kubernetes readiness probes.

Checks:
    PostgreSQL   — async DB connection + simple SELECT 1
    Qdrant       — REST /healthz endpoint
    Redis        — PING command
    MinIO        — bucket existence check
    LLM          — provider.health() (skipped if SKIP_LLM_HEALTH=true)
    Embeddings   — provider.health()

All checks run concurrently (asyncio.gather).
Any failure is logged but does NOT crash the server — the /health/ready
endpoint returns 503 so the load balancer stops sending traffic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from src.core.logger import get_logger

log = get_logger(__name__)


@dataclass
class ServiceHealth:
    name: str
    healthy: bool
    message: str = ""
    latency_ms: float = 0.0


@dataclass
class HealthReport:
    healthy: bool
    services: list[ServiceHealth] = field(default_factory=list)

    @property
    def summary(self) -> dict:
        return {
            "healthy": self.healthy,
            "services": {
                s.name: {
                    "healthy": s.healthy,
                    "message": s.message,
                    "latency_ms": round(s.latency_ms, 1),
                }
                for s in self.services
            },
        }


async def check_postgres() -> ServiceHealth:
    """Verify PostgreSQL is reachable and responsive."""
    import time
    from sqlalchemy import text
    from src.db.session import AsyncSessionLocal

    start = time.monotonic()
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(text("SELECT 1"))
        return ServiceHealth(
            name="postgres",
            healthy=True,
            latency_ms=(time.monotonic() - start) * 1000,
        )
    except Exception as exc:
        return ServiceHealth(
            name="postgres",
            healthy=False,
            message=str(exc),
        )


async def check_qdrant() -> ServiceHealth:
    """Verify Qdrant REST API is reachable."""
    import time
    import httpx
    from src.configs.config import get_settings

    settings = get_settings()
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.qdrant_url}/healthz")
            healthy = resp.status_code == 200
        return ServiceHealth(
            name="qdrant",
            healthy=healthy,
            message="" if healthy else f"status {resp.status_code}",
            latency_ms=(time.monotonic() - start) * 1000,
        )
    except Exception as exc:
        return ServiceHealth(name="qdrant", healthy=False, message=str(exc))


async def check_redis() -> ServiceHealth:
    """Verify Redis is reachable via PING."""
    import time
    from src.configs.config import get_settings
    import redis.asyncio as aioredis

    settings = get_settings()
    start = time.monotonic()
    try:
        r = aioredis.from_url(settings.redis_url, socket_timeout=5)
        await r.ping()
        await r.aclose()
        return ServiceHealth(
            name="redis",
            healthy=True,
            latency_ms=(time.monotonic() - start) * 1000,
        )
    except Exception as exc:
        return ServiceHealth(name="redis", healthy=False, message=str(exc))


async def check_minio() -> ServiceHealth:
    """Verify MinIO bucket exists (implicitly checks credentials too)."""
    import time
    import asyncio
    from src.configs.config import get_settings
    from src.services.storage_service import StorageService

    settings = get_settings()
    start = time.monotonic()
    try:
        storage = StorageService()
        # Run sync check in executor
        loop = asyncio.get_event_loop()
        client = await loop.run_in_executor(None, storage._get_client)
        exists = await loop.run_in_executor(
            None, client.bucket_exists, settings.minio_bucket
        )
        return ServiceHealth(
            name="minio",
            healthy=exists,
            message="" if exists else f"bucket '{settings.minio_bucket}' not found",
            latency_ms=(time.monotonic() - start) * 1000,
        )
    except Exception as exc:
        return ServiceHealth(name="minio", healthy=False, message=str(exc))


async def check_llm() -> ServiceHealth:
    """Verify the configured LLM provider is reachable."""
    import time
    import os
    from src.providers.llm.factory import get_llm_provider

    if os.getenv("SKIP_LLM_HEALTH", "").lower() in ("true", "1"):
        return ServiceHealth(name="llm", healthy=True, message="skipped")

    start = time.monotonic()
    try:
        provider = get_llm_provider()
        healthy = await provider.health()
        return ServiceHealth(
            name="llm",
            healthy=healthy,
            message=provider.provider_name if healthy else "provider unreachable",
            latency_ms=(time.monotonic() - start) * 1000,
        )
    except Exception as exc:
        return ServiceHealth(name="llm", healthy=False, message=str(exc))


async def check_embeddings() -> ServiceHealth:
    """Verify the embedding provider can load and respond."""
    import time
    import os
    from src.providers.embedding.factory import get_embedding_provider

    if os.getenv("SKIP_EMBEDDING_HEALTH", "").lower() in ("true", "1"):
        return ServiceHealth(name="embeddings", healthy=True, message="skipped")

    start = time.monotonic()
    try:
        provider = get_embedding_provider()
        healthy = await provider.health()
        return ServiceHealth(
            name="embeddings",
            healthy=healthy,
            message=provider.provider_name if healthy else "provider unreachable",
            latency_ms=(time.monotonic() - start) * 1000,
        )
    except Exception as exc:
        return ServiceHealth(name="embeddings", healthy=False, message=str(exc))


async def run_all_checks() -> HealthReport:
    """
    Run all health checks concurrently.
    Returns a HealthReport — overall healthy only if ALL services pass.
    """
    results = await asyncio.gather(
        check_postgres(),
        check_qdrant(),
        check_redis(),
        check_minio(),
        check_llm(),
        check_embeddings(),
        return_exceptions=True,
    )

    services: list[ServiceHealth] = []
    for result in results:
        if isinstance(result, Exception):
            services.append(ServiceHealth(
                name="unknown", healthy=False, message=str(result)
            ))
        else:
            services.append(result)
            if not result.healthy:
                log.warning(
                    "health_check_failed",
                    service=result.name,
                    message=result.message,
                )
            else:
                log.info(
                    "health_check_ok",
                    service=result.name,
                    latency_ms=round(result.latency_ms, 1),
                )

    overall = all(s.healthy for s in services)
    return HealthReport(healthy=overall, services=services)