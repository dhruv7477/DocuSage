"""
src/retrieval/reranker.py
──────────────────────────
Cross-encoder reranking + confidence-based abstention.

A cross-encoder reads (query, candidate) together and models
fine-grained relevance — negation, numerics, conditionals — that
bi-encoder retrieval can't see. Default model is `bge-reranker-v2-m3`
(multilingual, ~568 MB), which is materially stronger than the legacy
ms-marco-MiniLM on the kinds of grounded-document questions DocuSage
is asked.

Pipeline position:
    hybrid retrieve (top 50) → rerank (cross-encoder) → top_k → LLM

Output convention
-----------------
Reranker scores are passed through a sigmoid so they live in [0, 1]
regardless of the underlying model's logit range. This means
downstream code can use a single `abstention_threshold` setting (0.3
by default) without knowing which cross-encoder is loaded.

`compute_confidence()` returns the mean of the top-3 (post-sigmoid)
scores. The RAG service uses it to decide whether to send the LLM a
full grounded-answer prompt or the abstention branch.
"""

from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from src.core.logger import get_logger
from src.providers.llm.base import BaseLLMProvider, LLMConfig, LLMMessage
from src.vector_store.base import SearchResult

log = get_logger(__name__)

# Thread pool for running the cross-encoder synchronously without blocking
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="reranker")


class BaseReranker(ABC):
    """Abstract reranker contract."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Rerank results by true relevance to query.

        Args:
            query:    The user's question (not the rewritten form).
            results:  Candidate results from hybrid retrieval.
            top_k:    Number of results to return after reranking.

        Returns:
            Reranked list (best first), length <= top_k.
        """
        ...


class NoOpReranker(BaseReranker):
    """
    Pass-through reranker — returns top_k results in original order.
    Used when reranking is disabled via RERANKER=none in env.
    """

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        return results[:top_k]


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker (sentence-transformers `CrossEncoder`).

    Default model: BAAI/bge-reranker-v2-m3 (multilingual, ~568 MB).
    Configurable via `RERANKER_MODEL`. Any ms-marco-* alternative works
    too; we sigmoid the raw logits so the score scale is consistent.

    The model is loaded lazily on first use and cached on the class so
    we pay the load cost (a few seconds on cold start, dominated by
    weight downloads on the very first run) exactly once per process.
    """

    _model = None
    _loaded_model_name: str | None = None

    def __init__(self, model_name: str | None = None) -> None:
        # Read default lazily so tests / CI can monkey-patch settings.
        from src.configs.config import get_settings
        self._model_name = model_name or get_settings().reranker_model

    @classmethod
    def _load_model(cls, model_name: str):
        if cls._model is not None and cls._loaded_model_name == model_name:
            return cls._model
        # First call: model must be downloaded from HuggingFace (~568 MB for
        # bge-reranker-v2-m3). Runs in the reranker thread pool so the event
        # loop stays free, but may take 1-5 minutes on a cold cache. Log
        # upfront so the user can see it in the API console.
        log.info("cross_encoder_loading_model", model=model_name,
                 note="first-run download may take several minutes")
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            cls._model = CrossEncoder(model_name)
            cls._loaded_model_name = model_name
            log.info("cross_encoder_loaded", model=model_name)
        except Exception as exc:
            log.error("cross_encoder_load_failed", model=model_name, error=str(exc))
            raise
        return cls._model

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        if not results:
            return []

        loop = asyncio.get_event_loop()
        try:
            reranked = await loop.run_in_executor(
                _EXECUTOR,
                self._rerank_sync,
                query,
                results,
                top_k,
            )
            log.debug(
                "cross_encoder_rerank",
                query_preview=query[:50],
                input=len(results),
                output=len(reranked),
            )
            return reranked
        except Exception as exc:
            log.warning("cross_encoder_rerank_failed", error=str(exc))
            # Fallback: return original order
            return results[:top_k]

    def _rerank_sync(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Run in thread pool — blocks, not async-safe."""
        model = self._load_model(self._model_name)

        pairs = [(query, r.payload.text) for r in results]
        scores = model.predict(pairs)

        # Pass raw logits through a sigmoid → [0, 1]. This makes the
        # confidence threshold (`abstention_threshold`) meaningful
        # across cross-encoders with different output ranges. For
        # bge-reranker-v2-m3 logits sit roughly in [-10, +10].
        scored: list[tuple[SearchResult, float]] = []
        for result, raw in zip(results, scores):
            scored.append((result, _sigmoid(float(raw))))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked: list[SearchResult] = []
        for result, sigmoid_score in scored[:top_k]:
            reranked.append(SearchResult(
                chunk_id=result.chunk_id,
                score=round(sigmoid_score, 4),
                payload=result.payload,
            ))
        return reranked


class LLMReranker(BaseReranker):
    """
    LLM-based relevance scoring as a fallback when no cross-encoder is available.

    Asks the LLM to score each passage on a 1-10 scale.
    Much slower than CrossEncoderReranker but requires no extra model weights.
    Only practical for top_k <= 5 due to latency.
    """

    _SCORE_PROMPT = """\
Rate how relevant this passage is to the question on a scale of 1-10.
Respond with ONLY a single integer — nothing else.

Question: {query}

Passage: {passage}

Relevance (1-10):"""

    def __init__(self, llm: BaseLLMProvider) -> None:
        self._llm = llm
        self._config = LLMConfig(temperature=0.0, max_tokens=5)

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        if not results:
            return []

        # Score in parallel (gather all LLM calls concurrently)
        scores = await asyncio.gather(
            *[self._score_one(query, r.payload.text) for r in results],
            return_exceptions=True,
        )

        scored = []
        for result, score in zip(results, scores):
            if isinstance(score, Exception):
                score = 5.0  # Default middle score on error
            scored.append((result, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        max_score = scored[0][1] if scored else 10.0

        reranked = []
        for result, score in scored[:top_k]:
            reranked.append(SearchResult(
                chunk_id=result.chunk_id,
                score=round(score / max_score, 4),
                payload=result.payload,
            ))

        log.debug(
            "llm_rerank_done",
            query_preview=query[:50],
            input=len(results),
            output=len(reranked),
        )
        return reranked

    async def _score_one(self, query: str, passage: str) -> float:
        prompt = self._SCORE_PROMPT.format(
            query=query, passage=passage[:600]
        )
        response = await self._llm.generate(
            [LLMMessage(role="user", content=prompt)],
            self._config,
        )
        raw = response.content.strip()
        # Parse first integer found
        import re
        match = re.search(r"\d+", raw)
        if match:
            return min(10.0, max(1.0, float(match.group())))
        return 5.0


def get_reranker(
    strategy: str = "cross_encoder",
    llm: BaseLLMProvider | None = None,
    model_name: str | None = None,
) -> BaseReranker:
    """
    Factory: returns the appropriate reranker based on strategy string.

    Args:
        strategy:    "cross_encoder" | "llm" | "none"
        llm:         Required when strategy="llm"
        model_name:  Override cross-encoder model name

    Returns:
        A BaseReranker instance.
    """
    if strategy == "cross_encoder":
        return CrossEncoderReranker(model_name=model_name)
    elif strategy == "llm":
        if llm is None:
            raise ValueError("LLM required for LLMReranker")
        return LLMReranker(llm=llm)
    else:
        return NoOpReranker()


# ── Confidence + abstention helpers ───────────────────────────────


def _sigmoid(x: float) -> float:
    # Numerically stable for large negative x.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def compute_confidence(reranked: list[SearchResult]) -> float:
    """
    Aggregate reranker confidence as the mean of the top-3 scores.

    Mean-of-top-3 is the cheapest signal that handles two failure
    modes well: (a) one freak high-scoring chunk that doesn't actually
    answer the question, and (b) lots of mid-ranked chunks that
    collectively look promising but aren't. The retrieval pipeline
    feeds this into `abstention_threshold` to decide whether to send
    the LLM the grounded-answer prompt or the no-answer prompt.

    Returns 0.0 for an empty list so empty retrieval is treated as
    maximum-uncertainty.
    """
    if not reranked:
        return 0.0
    top = [r.score for r in reranked[:3]]
    return sum(top) / len(top)
