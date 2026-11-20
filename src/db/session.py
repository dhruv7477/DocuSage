"""
src/db/session.py
─────────────────
Async SQLAlchemy engine + session factory.

Design:
- AsyncEngine with connection pooling tuned for production
- AsyncSessionLocal: factory for request-scoped sessions
- get_db(): FastAPI dependency — yields session, auto-commits or rolls back
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.configs.config import get_settings
from src.core.logger import get_logger

log = get_logger(__name__)
settings = get_settings()

# ── Engine ────────────────────────────────────────────────────────
# pool_size: number of persistent connections
# max_overflow: extra connections allowed under burst load
# pool_pre_ping: validates connection is alive before handing to request
engine = create_async_engine(
    settings.database_url,
    echo=settings.app_debug,  # Log SQL only in debug mode
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,  # Recycle connections every 1hr
)

# ── Session Factory ───────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Keep objects usable after commit (important for async)
    autocommit=False,
    autoflush=False,
)


# ── FastAPI Dependency ────────────────────────────────────────────


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yields an AsyncSession scoped to a single HTTP request.

    Usage in route:
        async def my_route(db: AsyncSession = Depends(get_db, scope="function")): ...

    Commits on clean exit, rolls back on any exception.
    This prevents partial writes from leaking into the DB.

    Use scope="function". FastAPI runs the code after `yield` (the commit)
    when the endpoint returns with that scope, but only after the response
    has been sent with the default "request" scope. A client's next request
    (create a chat, then stream into it) can then arrive before the commit
    and not find the row. Only the streaming chat endpoint keeps the request
    scope, because its session must stay open while the answer streams.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
