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


# Eagerly import sentence-transformers (and transitively torch) at module
# import time. The first time `import sentence_transformers` runs in a
# *thread* on Windows it takes 30–120 s — there are ~100 native DLLs to
# load, and the per-thread loader is dramatically slower than on the main
# import path. Doing it here pays that cost once during uvicorn startup
# (where it shows up as "loading the API") rather than during a readiness
# probe (where it busts the timeout). Subsequent calls — including the
# Celery worker's `worker_process_init` warmup — are near-instant.
try:  # pragma: no cover — exercised at every real boot
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except ImportError:  # pragma: no cover
    _SentenceTransformer = None  # type: ignore[assignment]

from src.core.exceptions import EmbeddingError
from src.core.logger import get_logger
from src.providers.embedding.base import BaseEmbeddingProvider

log = get_logger(__name__)

# Best-known output dimensions for BGE / E5 models. The list isn't
# authoritative — once a model is loaded we ask it for its real
# `get_sentence_embedding_dimension()` and use that as ground truth.
# This map is only consulted before the first load (e.g. for the
# `dimension` property on a freshly constructed provider).
_KNOWN_MODEL_DIMENSIONS = {
    "BAAI/bge-m3":              1024,
    "BAAI/bge-large-en-v1.5":   1024,
    "BAAI/bge-base-en-v1.5":     768,
    "BAAI/bge-small-en-v1.5":    384,
    "intfloat/e5-large-v2":     1024,
    "intfloat/e5-base-v2":       768,
    "intfloat/e5-small-v2":      384,
}


class BGEProvider(BaseEmbeddingProvider):
    """
    Local embedding using BAAI BGE / intfloat E5 models via
    sentence-transformers.

    Dimension handling
    ------------------
    The provider holds an *expected* dimension (`expected_dimension` or
    a value looked up in `_KNOWN_MODEL_DIMENSIONS`). After the model
    loads we ask it for its real output size; if the two disagree we
    log a warning and trust the model — the model can't be wrong about
    itself, and a stale `EMBEDDING_DIMENSION` setting would otherwise
    silently break Qdrant upserts via `EmbeddingDimensionMismatch`.

    Thread-safety: SentenceTransformer is not async-native; inference
    is offloaded to a thread pool via `loop.run_in_executor()` so the
    FastAPI event loop stays free.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        batch_size: int = 32,
        expected_dimension: int | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        # Pick the best estimate available pre-load: explicit user
        # config wins, otherwise the known-model map, otherwise a
        # neutral default that won't cause spurious dim-mismatch errors.
        self._dim = (
            expected_dimension
            or _KNOWN_MODEL_DIMENSIONS.get(model_name)
            or 384
        )
        self._model = None  # Lazy load — don't block import time

    def _load_model(self):
        """Lazy-loads the model on first use. Thread-safe via GIL."""
        if self._model is None:
            try:
                if _SentenceTransformer is None:
                    raise ImportError("sentence-transformers not importable")
                log.info(
                    "embedding_model_loading",
                    model=self._model_name,
                    device=self._device,
                )
                self._model = _SentenceTransformer(
                    self._model_name,
                    device=self._device,
                )
                # Source of truth: the model itself. Override our
                # pre-load estimate, and warn loudly if the user's
                # configured EMBEDDING_DIMENSION disagrees so they can
                # update .env (or reindex Qdrant) before the first
                # ingestion fails with `EmbeddingDimensionMismatch`.
                actual_dim = self._model.get_sentence_embedding_dimension()
                if actual_dim and actual_dim != self._dim:
                    log.warning(
                        "embedding_dimension_overridden",
                        model=self._model_name,
                        expected=self._dim,
                        actual=actual_dim,
                        hint=(
                            f"Set EMBEDDING_DIMENSION={actual_dim} in .env "
                            "to silence this warning."
                        ),
                    )
                    self._dim = actual_dim
                log.info(
                    "embedding_model_ready",
                    model=self._model_name,
                    dim=self._dim,
                )
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
        """
        Round-trip a single embedding to prove the model is loaded and
        produces vectors of the expected dimension.

        IMPORTANT: `_load_model()` and `model.encode()` are both blocking
        — on a cold start the model has to download (~133 MB for BGE-small,
        ~2 GB for BGE-M3). Running them directly in this `async def`
        would stall the event loop and starve every other readiness
        probe; we saw a 5+ minute stall block Postgres / Redis / Qdrant
        / Ollama probes on first boot. Always route through the
        executor.
        """
        loop = asyncio.get_event_loop()

        def _check() -> bool:
            try:
                model = self._load_model()
                vec = model.encode(["health check"], normalize_embeddings=True)
                return len(vec[0]) == self._dim
            except Exception:
                return False

        try:
            return await loop.run_in_executor(None, _check)
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
