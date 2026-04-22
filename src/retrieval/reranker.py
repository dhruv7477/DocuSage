"""
src/retrieval/reranker.py
──────────────────────────
Cross-encoder reranking of retrieved document chunks.

The retrieval pipeline fetches candidates quickly (approximate nearest
neighbour + BM25).  But bi-encoder similarity is approximate — it
encodes query and document independently and compares embeddings.

A cross-encoder reads query AND document TOGETHER, which allows
it to model fine-grained relevance that bi-encoders miss (e.g.
negation, conditional statements, exact-match within context).

Pipeline position:
    retrieve (top 20-50) → rerank (cross-encoder) → take top K → LLM

Two backends:
    CrossEncoderReranker   Uses sentence-transformers CrossEncoder.
                           Model: ms-marco-MiniLM-L-6-v2 (fast, good quality)
                           Runs in a thread pool to avoid blocking the event loop.

    LLMReranker            Uses the LLM to score relevance as a fallback
                           when no cross-encoder model is available.
                           Slower but requires no extra model weights.

    NoOpReranker           Pass-through, preserves original order.
                           Used when reranking is disabled via config.

All three implement BaseReranker so the pipeline is reranker-agnostic.
"""

from __future__ import annotations

import asyncio
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
    Cross-encoder reranker using sentence-transformers.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - 22M params, fast inference
    - Trained on MS MARCO passage ranking
    - ~2ms per (query, chunk) pair on CPU

    The model is loaded lazily on first use and cached as a class
    variable to avoid reloading across requests.
    """

    _model = None
    _model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str | None = None) -> None:
        if model_name:
            self._model_name = model_name

    @classmethod
    def _load_model(cls):
        if cls._model is None:
            try:
                from sentence_transformers.cross_encoder import CrossEncoder
                cls._model = CrossEncoder(cls._model_name)
                log.info("cross_encoder_loaded", model=cls._model_name)
            except Exception as exc:
                log.error("cross_encoder_load_failed", error=str(exc))
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
        model = self._load_model()

        # Build (query, passage) pairs
        pairs = [(query, r.payload.text) for r in results]
        scores = model.predict(pairs)

        # Attach scores and sort descending
        scored = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Rebuild SearchResult with cross-encoder score replacing original
        reranked = []
        max_score = float(scored[0][1]) if scored else 1.0
        for result, ce_score in scored[:top_k]:
            # Normalise to [0,1] using sigmoid-like clamping
            normalised = max(0.0, min(1.0, (float(ce_score) + 10) / 20))
            reranked.append(SearchResult(
                chunk_id=result.chunk_id,
                score=round(normalised, 4),
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
