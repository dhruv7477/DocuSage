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

    # Pre-warm embedding model so it's loaded before the first request arrives
    try:
        from src.providers.embedding.factory import get_embedding_provider
        embedder = get_embedding_provider()
        log.info("embedding_warmup_start", provider=embedder.provider_name)
        await embedder.health()
        log.info("embedding_warmup_done", provider=embedder.provider_name)
    except Exception as exc:
        log.warning("embedding_warmup_failed", error=str(exc))

    log.info(
        "app_startup",
        app_name=settings.app_name,
        env=settings.app_env,
        version="1.0.0",
    )

    yield  # App runs here

    log.info("app_shutdown")


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
    # Health check — no auth, no versioning
    @app.get("/health", tags=["Health"], include_in_schema=False)
    async def health_check() -> dict:
        return {"status": "ok", "app": settings.app_name, "env": settings.app_env}

    app.include_router(api_v1_router, prefix=settings.api_v1_prefix)


# ── Entry Point ───────────────────────────────────────────────────
app = create_app()