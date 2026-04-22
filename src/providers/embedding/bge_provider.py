"""
src/providers/embedding/bge_provider.py
─────────────────────────────────────────
BGE-M3 embedding provider using sentence-transformers.

Why BGE-M3:
- #1 on MTEB (Massive Text Embedding Benchmark) leaderboard for open-source
- Supports DENSE + SPARSE + MULTI-VECTOR retrieval natively
- 1024-dimensional output — high information density
- Multilingual (100+ languages) — future-proof
- Runs well on CPU for development; GPU gives ~10x speedup

Model is loaded ONCE at startup via lru_cache in factory.py.
Inference is run in a thread pool executor to avoid blocking the event loop
since sentence-transformers is synchronous.

Install: pip install sentence-transformers torch
"""

import asyncio
from functools import lru_cache

import numpy as np

from src.core.exceptions import EmbeddingError
from src.core.logger import get_logger
from src.providers.embedding.base import BaseEmbeddingProvider

log = get_logger(__name__)

_SUPPORTED_MODELS = {
    "BAAI/bge-m3":         1024,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5":  768,
    "intfloat/e5-large-v2":   1024,
}


class BGEProvider(BaseEmbeddingProvider):
    """
    Local embedding using BAAI/bge-m3 (or compatible BGE/E5 models).

    Thread-safety: SentenceTransformer is not async-native.
    We use asyncio.get_event_loop().run_in_executor() to offload
    inference to a thread pool, keeping the event loop unblocked.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._dim = _SUPPORTED_MODELS.get(model_name, 1024)
        self._model = None  # Lazy load — don't block import time

    def _load_model(self):
        """Lazy-loads the model on first use. Thread-safe via GIL."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                log.info(
                    "embedding_model_loading",
                    model=self._model_name,
                    device=self._device,
                )
                self._model = SentenceTransformer(
                    self._model_name,
                    device=self._device,
                )
                log.info("embedding_model_ready", model=self._model_name)
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers not installed.",
                    detail="Run: pip install sentence-transformers torch",
                ) from exc
        return self._model

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def provider_name(self) -> str:
        return f"bge:{self._model_name}"

    # ── Embed Query ───────────────────────────────────────────────

    async def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query.
        BGE models perform better with a query prefix: "Represent this sentence: "
        """
        # BGE-M3 instruction prefix for retrieval queries
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        vectors = await self._run_in_executor([prefixed])
        return vectors[0]

    # ── Embed Documents ───────────────────────────────────────────

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Batch-embed document chunks.
        Processes in batches of self._batch_size to manage memory.
        """
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            batch_vectors = await self._run_in_executor(batch)
            all_vectors.extend(batch_vectors)

        log.debug(
            "documents_embedded",
            count=len(texts),
            model=self._model_name,
        )
        return all_vectors

    # ── Health ────────────────────────────────────────────────────

    async def health(self) -> bool:
        try:
            model = self._load_model()
            # Embed a tiny test string
            vec = model.encode(["health check"], normalize_embeddings=True)
            return len(vec[0]) == self._dim
        except Exception:
            return False

    # ── Private ───────────────────────────────────────────────────

    async def _run_in_executor(self, texts: list[str]) -> list[list[float]]:
        """
        Run synchronous SentenceTransformer inference in a thread pool.
        This prevents blocking the FastAPI event loop during embedding.
        """
        loop = asyncio.get_event_loop()

        def _encode():
            model = self._load_model()
            vectors = model.encode(
                texts,
                normalize_embeddings=True,   # L2 normalise — cosine = dot product
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
            return vectors.tolist()

        try:
            return await loop.run_in_executor(None, _encode)
        except Exception as exc:
            raise EmbeddingError(
                "BGE embedding failed.", detail=str(exc)
            ) from exc
