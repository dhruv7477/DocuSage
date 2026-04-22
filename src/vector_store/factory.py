"""
src/vector_store/factory.py
────────────────────────────
Vector store factory — maps VECTOR_STORE env var to implementation.

Usage:
    from src.vector_store.factory import get_vector_store
    vs = get_vector_store()
    await vs.search(workspace_id, query_vector, search_filter)
"""

from functools import lru_cache

from src.configs.config import get_settings
from src.core.logger import get_logger
from src.vector_store.base import BaseVectorStore

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_vector_store() -> BaseVectorStore:
    settings = get_settings()
    store = settings.vector_store

    log.info("vector_store_init", store=store, url=settings.qdrant_url)

    if store == "qdrant":
        from src.vector_store.qdrant_store import QdrantVectorStore
        return QdrantVectorStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

    raise ValueError(
        f"Unsupported VECTOR_STORE='{store}'. Valid options: qdrant"
    )


def reset_vector_store() -> None:
    """Clear cached store. Used in tests."""
    get_vector_store.cache_clear()