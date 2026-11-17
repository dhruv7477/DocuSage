"""
src/core/readiness.py
──────────────────────
Centralized readiness/health probes for every external dependency.

Used in two places:

* `src.main.lifespan` — at startup, runs every probe and (when
  `STRICT_STARTUP=true`) refuses to boot if any required dependency is
  unreachable. The previous behavior was to let the API come up and
  silently 500 on the first request that needed Qdrant / Redis / MinIO,
  which is exactly the failure mode this audit caught.

* `GET /api/v1/health/deep` — runs the same probes on demand for ops /
  CI smoke tests. Cheap to call (every probe runs in parallel with a
  configurable timeout).

Probe contract
--------------
Each probe returns a `ProbeResult` with:
  * `name`       – stable identifier (used by the JSON response)
  * `ok`         – pass/fail boolean
  * `required`   – when False the probe is informational and never
                   blocks startup even under STRICT_STARTUP
  * `detail`     – short human-readable message; always populated on
                   failure with a remediation hint when possible
  * `latency_ms` – probe wall time, useful for spotting slow deps
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import text

from src.configs.config import get_settings
from src.core.logger import get_logger

log = get_logger(__name__)


def redact_url(url: str) -> str:
    """Drop the user and password from a URL, e.g. redis://:secret@redis:6379/0.

    Probe details go to logs and to /health/deep, so they must never carry
    credentials.
    """
    parts = urlsplit(url)
    if parts.username is None and parts.password is None:
        return url
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


@dataclass
class ProbeResult:
    name: str
    ok: bool
    required: bool
    detail: str
    latency_ms: int


@dataclass
class ReadinessReport:
    ok: bool
    probes: list[ProbeResult]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "probes": [
                {
                    "name": p.name,
                    "ok": p.ok,
                    "required": p.required,
                    "detail": p.detail,
                    "latency_ms": p.latency_ms,
                }
                for p in self.probes
            ],
        }


class StartupReadinessError(RuntimeError):
    """Raised when STRICT_STARTUP is on and a required probe failed."""


# ── Individual probes ─────────────────────────────────────────────


async def _probe_postgres(timeout: float) -> ProbeResult:
    started = time.monotonic()
    log.info("probe_starting", probe="postgres", timeout=timeout)
    try:
        from src.db.session import AsyncSessionLocal

        async def _ping() -> None:
            async with AsyncSessionLocal() as db:
                await db.execute(text("SELECT 1"))

        await asyncio.wait_for(_ping(), timeout=timeout)
        return ProbeResult(
            name="postgres",
            ok=True,
            required=True,
            detail="ok",
            latency_ms=int((time.monotonic() - started) * 1000),
        )
    except Exception as exc:
        settings = get_settings()
        return ProbeResult(
            name="postgres",
            ok=False,
            required=True,
            detail=(
                f"Cannot reach PostgreSQL at {settings.postgres_host}:"
                f"{settings.postgres_port}. Run database migrations and "
                "verify the container is up. (" + type(exc).__name__ + ")"
            ),
            latency_ms=int((time.monotonic() - started) * 1000),
        )


async def _probe_redis(timeout: float) -> ProbeResult:
    started = time.monotonic()
    log.info("probe_starting", probe="redis", timeout=timeout)
    try:
        from src.core.token_blacklist import get_redis

        await asyncio.wait_for(get_redis().ping(), timeout=timeout)
        return ProbeResult(
            name="redis",
            ok=True,
            required=True,
            detail="ok",
            latency_ms=int((time.monotonic() - started) * 1000),
        )
    except Exception as exc:
        settings = get_settings()
        return ProbeResult(
            name="redis",
            ok=False,
            required=True,
            detail=(
                f"Cannot reach Redis at {redact_url(settings.redis_url)}. "
                "Start Redis (e.g. `docker compose up redis`). "
                f"({type(exc).__name__})"
            ),
            latency_ms=int((time.monotonic() - started) * 1000),
        )


async def _probe_qdrant(timeout: float) -> ProbeResult:
    started = time.monotonic()
    log.info("probe_starting", probe="qdrant", timeout=timeout)
    try:
        from src.vector_store.factory import get_vector_store

        ok = await asyncio.wait_for(get_vector_store().health(), timeout=timeout)
        return ProbeResult(
            name="qdrant",
            ok=bool(ok),
            required=True,
            detail="ok" if ok else "Qdrant did not respond healthy.",
            latency_ms=int((time.monotonic() - started) * 1000),
        )
    except Exception as exc:
        settings = get_settings()
        return ProbeResult(
            name="qdrant",
            ok=False,
            required=True,
            detail=(
                f"Cannot reach Qdrant at {redact_url(settings.qdrant_url)}. "
                "Start Qdrant (e.g. `docker compose up qdrant`). "
                f"({type(exc).__name__})"
            ),
            latency_ms=int((time.monotonic() - started) * 1000),
        )


async def _probe_storage(timeout: float) -> ProbeResult:
    """
    Verify the S3 server is reachable AND the configured bucket exists.

    Note: `StorageService._get_client()` is synchronous and lazy. We run
    it in the default thread pool so a hung connection respects the
    probe timeout.
    """
    started = time.monotonic()
    log.info("probe_starting", probe="storage", timeout=timeout)
    settings = get_settings()
    try:
        from src.services.storage_service import StorageService

        def _ping_storage() -> None:
            svc = StorageService()
            client, bucket = svc._get_client()  # noqa: SLF001 — same module family
            # `bucket_exists` is the cheapest call that proves connectivity
            # AND auth; it's already invoked inside _get_client during
            # initial setup, but call again to keep the probe self-contained
            # against future refactors.
            if settings.storage_backend == "s3":
                client.head_bucket(Bucket=bucket)
            else:
                client.bucket_exists(bucket)

        await asyncio.wait_for(asyncio.to_thread(_ping_storage), timeout=timeout)
        return ProbeResult(
            name="storage",
            ok=True,
            required=True,
            detail=f"ok ({settings.storage_backend} bucket={settings.s3_bucket})",
            latency_ms=int((time.monotonic() - started) * 1000),
        )
    except Exception as exc:
        endpoint = (
            f"s3 ({settings.aws_region})"
            if settings.storage_backend == "s3"
            else settings.s3_endpoint
        )
        return ProbeResult(
            name="storage",
            ok=False,
            required=True,
            detail=(
                f"Cannot reach storage at {endpoint}. "
                "Start the S3 server (`make dev-up` starts SeaweedFS) and ensure "
                f"the bucket exists. ({type(exc).__name__})"
            ),
            latency_ms=int((time.monotonic() - started) * 1000),
        )


async def _probe_llm(timeout: float) -> ProbeResult:
    """
    For Ollama this verifies the API is up AND the configured model is
    pulled — a missing model is the single most common cause of
    "ingestion / chat hangs" in local dev.
    """
    started = time.monotonic()
    log.info("probe_starting", probe="llm", timeout=timeout)
    settings = get_settings()
    try:
        from src.providers.llm.factory import get_llm_provider

        ok = await asyncio.wait_for(get_llm_provider().health(), timeout=timeout)
        if ok:
            return ProbeResult(
                name="llm",
                ok=True,
                required=True,
                detail=f"ok ({settings.llm_provider} / {settings.llm_model})",
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        # Provider reached but model not available
        if settings.llm_provider == "ollama":
            hint = (
                f"Pull it with: `ollama pull {settings.llm_model}` "
                f"(or set LLM_MODEL to a model you already have)."
            )
        else:
            hint = "Verify the configured model name and API key."
        return ProbeResult(
            name="llm",
            ok=False,
            required=True,
            detail=(
                f"{settings.llm_provider} is reachable but model "
                f"'{settings.llm_model}' is not available. {hint}"
            ),
            latency_ms=int((time.monotonic() - started) * 1000),
        )
    except Exception as exc:
        return ProbeResult(
            name="llm",
            ok=False,
            required=True,
            detail=(
                f"Cannot reach LLM provider '{settings.llm_provider}'. "
                f"({type(exc).__name__})"
            ),
            latency_ms=int((time.monotonic() - started) * 1000),
        )


async def _probe_embedder(timeout: float) -> ProbeResult:
    """
    For local BGE this loads the model into RAM (~3–5s on first call).
    The probe is therefore slow on cold start but fast thereafter
    because the model is cached on the singleton.
    """
    started = time.monotonic()
    log.info("probe_starting", probe="embedder", timeout=timeout)
    settings = get_settings()
    try:
        from src.providers.embedding.factory import get_embedding_provider

        ok = await asyncio.wait_for(get_embedding_provider().health(), timeout=timeout)
        return ProbeResult(
            name="embedder",
            ok=bool(ok),
            required=True,
            detail=(
                f"ok ({settings.embedding_provider} / {settings.embedding_model}, "
                f"dim={settings.embedding_dimension})"
                if ok
                else "embedder did not return a vector of the expected dimension."
            ),
            latency_ms=int((time.monotonic() - started) * 1000),
        )
    except Exception as exc:
        return ProbeResult(
            name="embedder",
            ok=False,
            required=True,
            detail=(
                f"Cannot initialize embedding provider "
                f"'{settings.embedding_provider}'. ({type(exc).__name__})"
            ),
            latency_ms=int((time.monotonic() - started) * 1000),
        )


# ── Public API ────────────────────────────────────────────────────


async def run_readiness_checks(*, timeout: float | None = None) -> ReadinessReport:
    """
    Run every probe in parallel. Never raises — failures are reported
    in the returned `ReadinessReport`.
    """
    settings = get_settings()
    t = timeout if timeout is not None else settings.startup_probe_timeout_seconds

    # Embedder probe is intentionally given a longer budget because the
    # first call may load a multi-hundred-MB sentence-transformers model.
    embedder_timeout = max(t, 30.0)

    log.info(
        "readiness_gather_start", probe_timeout=t, embedder_timeout=embedder_timeout
    )
    probes: list[ProbeResult] = await asyncio.gather(
        _probe_postgres(t),
        _probe_redis(t),
        _probe_qdrant(t),
        _probe_storage(t),
        _probe_llm(t),
        _probe_embedder(embedder_timeout),
    )
    log.info("readiness_gather_done", results={p.name: p.ok for p in probes})

    overall_ok = all(p.ok for p in probes if p.required)
    return ReadinessReport(ok=overall_ok, probes=probes)


async def enforce_startup_readiness() -> ReadinessReport:
    """
    Run every probe; if STRICT_STARTUP is on and any required probe
    failed, raise StartupReadinessError so the app refuses to boot.
    """
    settings = get_settings()
    report = await run_readiness_checks()

    for p in report.probes:
        level = "info" if p.ok else "error"
        getattr(log, level)(
            "readiness_probe",
            probe=p.name,
            ok=p.ok,
            required=p.required,
            latency_ms=p.latency_ms,
            detail=p.detail,
        )

    if not report.ok and settings.strict_startup:
        failing = [p for p in report.probes if not p.ok and p.required]
        message = "Startup readiness check failed. " + " | ".join(
            f"[{p.name}] {p.detail}" for p in failing
        )
        raise StartupReadinessError(message)

    return report
