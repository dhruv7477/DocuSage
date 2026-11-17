"""
src/providers/embedding/factory.py
────────────────────────────────────
Embedding provider factory — maps EMBEDDING_PROVIDER env var to implementation.

Usage:
    from src.providers.embedding.factory import get_embedding_provider
    embedder = get_embedding_provider()
    vector = await embedder.embed_query("What is our refund policy?")

Supported:
    bge     → BGEProvider (default, BAAI/bge-m3, runs locally)
    openai  → OpenAIEmbeddingProvider (text-embedding-3-large)
"""

from functools import lru_cache

from src.configs.config import get_settings
from src.core.logger import get_logger
from src.providers.embedding.base import BaseEmbeddingProvider

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_provider() -> BaseEmbeddingProvider:
    """
    Returns the configured embedding provider as a cached singleton.

    The model is loaded once on first call and reused for the process lifetime.
    BGE-M3 is ~580MB on disk; loading takes ~3-5 seconds on first call.
    """
    settings = get_settings()
    provider = settings.embedding_provider

    log.info(
        "embedding_provider_init",
        provider=provider,
        model=settings.embedding_model,
        device=settings.embedding_device,
    )

    if provider == "bge":
        from src.providers.embedding.bge_provider import BGEProvider

        return BGEProvider(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
            # Pass through the user's configured dim so the provider's
            # `dimension` property is correct *before* the model has
            # loaded (e.g. when Qdrant collection creation runs first).
            # The provider replaces this with the model's real reported
            # dim once loaded.
            expected_dimension=settings.embedding_dimension,
        )

    if provider == "openai":
        from src.providers.embedding.openai_embed import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_embed_model,
        )

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER='{provider}'. " "Valid options: bge | openai"
    )


def reset_embedding_provider() -> None:
    """Clear cached provider. Used in tests to swap providers."""
    get_embedding_provider.cache_clear()
