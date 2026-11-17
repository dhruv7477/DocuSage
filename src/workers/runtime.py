"""
src/workers/runtime.py
──────────────────────
One asyncio event loop and one database engine per Celery worker process.

Celery tasks are synchronous, and the pipeline is async. Calling
asyncio.run() per task creates and closes a fresh loop every time, but
pooled async clients keep sockets bound to the loop that opened them:
the asyncpg pool behind SQLAlchemy and the httpx pool inside the cached
AsyncQdrantClient. The next task then fails with "Event loop is closed"
or "attached to a different loop". Running every task on one long-lived
loop per process keeps those pooled connections valid.

The loop and engine are keyed by PID, so a forked prefork child never
reuses objects it inherited from the parent.
"""

import asyncio
import os
from collections.abc import Coroutine
from typing import Any, TypeVar

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.configs.config import get_settings

T = TypeVar("T")

_pid: int | None = None
_loop: asyncio.AbstractEventLoop | None = None
_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def _ensure_current_process() -> None:
    """Drop state inherited from another process (after fork)."""
    global _pid, _loop, _engine, _sessionmaker
    if _pid == os.getpid():
        return
    _pid = os.getpid()
    _loop = None
    _engine = None
    _sessionmaker = None


def get_loop() -> asyncio.AbstractEventLoop:
    """Return this process's worker loop, creating it on first use."""
    global _loop
    _ensure_current_process()
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
    return _loop


def run(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion on this process's worker loop."""
    return get_loop().run_until_complete(coro)


def session_factory() -> async_sessionmaker[AsyncSession]:
    """Return a session factory bound to this process's worker engine.

    Kept separate from the API's engine in src.db.session so a worker
    never shares pooled connections with code running on another loop.
    """
    global _engine, _sessionmaker
    _ensure_current_process()
    if _sessionmaker is None:
        settings = get_settings()
        # One task runs at a time per process; a small pool is enough.
        _engine = create_async_engine(
            settings.database_url,
            pool_size=2,
            max_overflow=2,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        _sessionmaker = async_sessionmaker(
            bind=_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _sessionmaker
