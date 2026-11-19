"""
src/main.py
───────────
FastAPI application factory.

Responsibilities:
- Configure logging at startup
- Register global exception handlers (maps domain exceptions → HTTP responses)
- Mount CORS middleware
- Mount rate limiting middleware
- Include all API routers
- Health check endpoint
"""

import asyncio
import time as _time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import sentry_sdk
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.api.v1 import api_v1_router
from src.configs.config import get_settings
from src.core.audit import AuditMiddleware
from src.core.exceptions import PrivateAIException
from src.core.logger import configure_logging, get_logger
from src.core.rate_limit import limiter
from src.db.session import AsyncSessionLocal

settings = get_settings()
log = get_logger(__name__)

# Client addresses treated as "this host" (the API container itself).
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


# ── Lifespan ──────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Runs once at startup, once at shutdown.
    Use for: DB pool warm-up, loading ML models, connecting to services.
    """
    # Configure structured logging
    configure_logging(
        log_level=settings.log_level,
        is_production=settings.app_env == "production",
    )

    # Initialize Sentry error tracking (no-op if DSN is empty)
    if settings.sentry_dsn:
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.app_env,
            traces_sample_rate=0.1,
        )
        log.info("sentry_initialized", env=settings.app_env)

    # Register DB session factory on app.state so AuditMiddleware can use it
    app.state.db_session_factory = AsyncSessionLocal

    # Pre-warm the embedder BEFORE the readiness probes run. The first
    # call to `model.encode()` in a thread triggers a multi-second
    # SentenceTransformer load (and on Windows can take 60–120 s on a
    # cold disk). If we let that happen during the readiness probe it
    # blows the per-probe timeout. Doing it here means the embedder
    # probe later just runs a tiny inference against an already-loaded
    # model — completes in milliseconds.
    _t0 = _time.monotonic()
    log.info("STARTUP_STEP_1_embedder_warmup_start")
    try:
        from src.providers.embedding.factory import get_embedding_provider

        embedder = get_embedding_provider()
        log.info("embedding_warmup_start", provider=embedder.provider_name)
        warmup_ok = await embedder.health()
        log.info(
            "embedding_warmup_done",
            provider=embedder.provider_name,
            ok=warmup_ok,
            dim=embedder.dimension,
            elapsed_ms=int((_time.monotonic() - _t0) * 1000),
        )
    except Exception as exc:
        log.warning(
            "embedding_warmup_failed",
            error=str(exc),
            elapsed_ms=int((_time.monotonic() - _t0) * 1000),
        )

    log.info(
        "STARTUP_STEP_2_vector_store_prewarm_start",
        elapsed_ms=int((_time.monotonic() - _t0) * 1000),
    )
    # Pre-warm provider singletons BEFORE the concurrent readiness probes.
    #
    # Root cause: `get_vector_store()` uses a lazy `from qdrant_client import
    # AsyncQdrantClient` inside its @lru_cache body. On Windows, that first
    # import loads gRPC/protobuf DLLs synchronously and takes 5–7 s on the
    # main event-loop thread — blocking every other probe in the
    # asyncio.gather from running until the import finishes. Redis, Qdrant,
    # MinIO and Ollama probes all pile up and hit their 15 s timeout even
    # though every service is healthy.
    #
    # Fix: call each singleton factory ONCE here (sequentially, before the
    # gather) so by the time the probes run the cache is warm and each
    # probe's factory call is a nanosecond dict lookup. The Qdrant store is
    # run in a thread pool (asyncio.to_thread) so its blocking import never
    # touches the event loop.
    try:
        from src.vector_store.factory import get_vector_store

        await asyncio.to_thread(get_vector_store)  # heavy gRPC import in thread
        log.info(
            "vector_store_prewarm_done",
            elapsed_ms=int((_time.monotonic() - _t0) * 1000),
        )
    except Exception as exc:
        log.warning(
            "vector_store_prewarm_failed",
            error=str(exc),
            elapsed_ms=int((_time.monotonic() - _t0) * 1000),
        )

    log.info(
        "STARTUP_STEP_3_reranker_prewarm_start",
        elapsed_ms=int((_time.monotonic() - _t0) * 1000),
    )
    # Pre-warm the cross-encoder reranker BEFORE the first query arrives.
    # On a cold cache the model download (~568 MB for bge-reranker-v2-m3)
    # takes minutes; doing it here makes the wait visible in the startup
    # log instead of appearing as a frozen hang on the first chat query.
    try:
        _rs = get_settings()
        if _rs.reranker == "cross_encoder":
            from src.retrieval.reranker import CrossEncoderReranker

            await asyncio.to_thread(
                CrossEncoderReranker._load_model, _rs.reranker_model
            )
            log.info(
                "reranker_prewarm_done",
                model=_rs.reranker_model,
                elapsed_ms=int((_time.monotonic() - _t0) * 1000),
            )
        else:
            log.info("reranker_prewarm_skipped", strategy=_rs.reranker)
    except Exception as exc:
        log.warning(
            "reranker_prewarm_failed",
            error=str(exc),
            elapsed_ms=int((_time.monotonic() - _t0) * 1000),
        )

    log.info(
        "STARTUP_STEP_4_llm_redis_init_start",
        elapsed_ms=int((_time.monotonic() - _t0) * 1000),
    )
    _router_llm = None
    try:
        from src.providers.llm.factory import get_llm_provider, get_router_llm_provider

        _llm = get_llm_provider()  # fast — just stores config + httpx client
        _router_llm = get_router_llm_provider()  # small intent-classifier model
        from src.core.token_blacklist import get_redis

        get_redis()  # fast — just stores aioredis client
        log.info(
            "provider_singletons_initialized",
            elapsed_ms=int((_time.monotonic() - _t0) * 1000),
        )
    except Exception as exc:
        log.warning(
            "provider_init_failed",
            error=str(exc),
            elapsed_ms=int((_time.monotonic() - _t0) * 1000),
        )
        _llm = None

    # Pre-load the LLMs into RAM so the first query doesn't wait 30-90 s for
    # Ollama to pull a model off disk. warmup() sends a zero-generation
    # request that loads the weights and keeps them there for 30 min.
    #
    # Run it in the BACKGROUND — never `await` it inline. A 7B load on CPU
    # can take minutes (longer under memory pressure), and awaiting it here
    # blocks the whole lifespan, so the server never binds its port and the
    # entire API is unreachable until warmup returns. Scheduling it lets
    # startup finish and bind immediately; the first query just pays the
    # load cost if it arrives before warmup completes.
    log.info(
        "STARTUP_STEP_4b_llm_warmup_scheduled",
        elapsed_ms=int((_time.monotonic() - _t0) * 1000),
    )

    async def _warm_models() -> None:
        try:
            if _llm is not None and hasattr(_llm, "warmup"):
                await _llm.warmup()
            # Warm the small router model only if the LLM intent router is
            # actually enabled. With router_use_llm=False (retrieval-score
            # routing) it's never called, so warming it would waste ~1 GB.
            if (
                settings.router_use_llm
                and _router_llm is not None
                and _router_llm is not _llm
                and hasattr(_router_llm, "warmup")
            ):
                await _router_llm.warmup()
            log.info("llm_warmup_done")
        except Exception as exc:
            log.warning("llm_warmup_failed", error=str(exc))

    # Keep a reference on app.state so the task isn't garbage-collected.
    app.state.warmup_task = asyncio.create_task(_warm_models())

    log.info(
        "STARTUP_STEP_5_readiness_probes_start",
        elapsed_ms=int((_time.monotonic() - _t0) * 1000),
    )

    # Run readiness probes against every external dependency. With
    # STRICT_STARTUP=true (default) the app refuses to boot if anything
    # required is unreachable, surfacing problems immediately instead of
    # the previous behavior where requests would silently fail later.
    from src.core.readiness import enforce_startup_readiness, StartupReadinessError

    try:
        report = await enforce_startup_readiness()
        app.state.last_readiness_report = report
        log.info(
            "readiness_summary",
            ok=report.ok,
            probes={p.name: p.ok for p in report.probes},
        )
    except StartupReadinessError as exc:
        log.error("startup_aborted_readiness", error=str(exc))
        raise
    except Exception as exc:
        # Defensive: never let an unexpected probe error mask the rest
        # of the app's startup story. Log and continue (STRICT_STARTUP
        # already gates the actual fail-fast behavior above).
        log.error("readiness_unexpected_error", error=str(exc))

    log.info(
        "app_startup",
        app_name=settings.app_name,
        env=settings.app_env,
        version="1.0.0",
    )

    yield  # App runs here

    log.info("app_shutdown")
    # Cancel the background warmup if it's still running at shutdown.
    warmup_task = getattr(app.state, "warmup_task", None)
    if warmup_task is not None and not warmup_task.done():
        warmup_task.cancel()
        try:
            await warmup_task
        except (asyncio.CancelledError, Exception):
            pass

    # Close pooled database connections on the loop that opened them. A
    # pooled asyncpg connection cannot be reused from another event loop,
    # which is what a second app instance in the same process would do.
    from src.db.session import engine

    await engine.dispose()


# ── App Factory ───────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="Multi-Tenant AI-Powered Private Knowledge Assistant",
        version="1.0.0",
        docs_url="/docs" if settings.app_env != "production" else None,
        redoc_url="/redoc" if settings.app_env != "production" else None,
        lifespan=lifespan,
    )

    _register_middleware(app)
    _register_exception_handlers(app)
    _register_routers(app)

    return app


def _register_middleware(app: FastAPI) -> None:
    # Rate limiting — must be registered before CORSMiddleware
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
    app.add_middleware(SlowAPIMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(AuditMiddleware)


def _register_exception_handlers(app: FastAPI) -> None:
    """
    Convert all PrivateAIException subclasses to consistent JSON error responses.
    Format: { "error_code": "...", "message": "...", "detail": "..." }

    Why: Service layer raises domain exceptions; routes never catch them —
    the handler here is the single catch point.
    """

    @app.exception_handler(PrivateAIException)
    async def domain_exception_handler(
        request: Request, exc: PrivateAIException
    ) -> JSONResponse:
        log.warning(
            "domain_exception",
            error_code=exc.error_code,
            message=exc.message,
            path=str(request.url),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        log.error("unhandled_exception", error=str(exc), path=str(request.url))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred.",
                "detail": str(exc) if settings.app_debug else None,
            },
        )


def _register_routers(app: FastAPI) -> None:
    # Liveness check — proves the API process is up, no dep checks.
    @app.get("/health", tags=["Health"], include_in_schema=False)
    async def health_check() -> dict:
        return {"status": "ok", "app": settings.app_name, "env": settings.app_env}

    # Deep readiness — runs every dependency probe on demand. Returns
    # 200 when everything required is healthy, 503 otherwise. Use this
    # from CI smoke tests, k8s readinessProbe, and ops dashboards.
    # Only callers on this host may see it (provider, model and bucket
    # names), unless HEALTH_DEEP_PUBLIC=true. Through the demo's proxy the
    # path is not exposed at all.
    @app.get("/health/deep", tags=["Health"], include_in_schema=False)
    async def health_deep(request: Request) -> JSONResponse:
        from src.core.readiness import run_readiness_checks

        client_host = request.client.host if request.client else ""
        if not settings.health_deep_public and client_host not in _LOOPBACK_HOSTS:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND, content={"detail": "Not Found"}
            )

        report = await run_readiness_checks()
        return JSONResponse(
            status_code=(
                status.HTTP_200_OK if report.ok else status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            content=report.to_dict(),
        )

    app.include_router(api_v1_router, prefix=settings.api_v1_prefix)


# ── Entry Point ───────────────────────────────────────────────────
app = create_app()
