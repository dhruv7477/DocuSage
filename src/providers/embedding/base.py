"""
src/providers/embedding/base.py
────────────────────────────────
Abstract base class for all embedding providers.

Design mirrors the LLM provider pattern:
- Single interface, multiple interchangeable implementations
- embed_query()     → single text → vector (for RAG retrieval)
- embed_documents() → batch texts → vectors (for ingestion pipeline)
- Batching is handled internally — callers don't need to worry about it

All vectors are normalised to unit length (L2) before returning.
This ensures cosine similarity == dot product, which Qdrant optimises for.
"""

from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """
    Contract every embedding provider must satisfy.

    The application ONLY imports this class — never a concrete provider.
    Use get_embedding_provider() from factory.py to get the active instance.
    """

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.
        Used at RAG retrieval time — must be fast.

        Returns:
            L2-normalised dense vector of length == embedding_dimension
        """
        ...

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of document chunks.
        Used during ingestion — can be slower, batched internally.

        Returns:
            List of L2-normalised vectors, same length as input texts.
        """
        ...

    @abstractmethod
    async def health(self) -> bool:
        """Returns True if the embedding backend is reachable."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output vector dimension. Must match Qdrant collection config."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier for logs."""
        ...
