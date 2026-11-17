"""
src/retrieval/hybrid.py
────────────────────────
True hybrid retrieval = dense (vector) + sparse (Qdrant FT index)
fused with Reciprocal Rank Fusion.

Why this replaces the legacy "hybrid" path
------------------------------------------
The previous implementation built an in-memory BM25 index over the top
20 candidates returned by vector search and called that "hybrid". It
couldn't rescue any keyword-exact match the dense embedder happened to
miss because such matches never made it into the candidate pool in the
first place. Real hybrid retrieval needs two *independent* index reads
that are fused after the fact — which is exactly what this module does
now that Phase 2.3 added a proper Qdrant text payload index.

Algorithm
---------
1. Embed every query in `query_enrichment.queries` (1 query for the
   common path, +1 for HyDE when enabled). Embedding calls run in
   parallel via `asyncio.gather`.
2. In parallel:
   * `vector_store.search(...)` for each embedding → dense candidates
   * `vector_store.text_search(query_enrichment.original, ...)` →
     sparse candidates against the FT index. We always use the
     **original** query (not the HyDE / rewritten form) for the
     lexical branch — the user's literal wording is what BM25-style
     retrieval is good at.
3. Merge the dense candidate sets (taking the best score per chunk_id)
   and fuse with the sparse list via Reciprocal Rank Fusion (k=60).
4. Return the top-N candidates carrying merged rank info.

Reranking and abstention happen in `reranker.py` (Phase 3.3).
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

from src.core.logger import get_logger
from src.providers.embedding.base import BaseEmbeddingProvider
from src.retrieval.query_understanding import QueryEnrichment
from src.vector_store.base import BaseVectorStore, SearchFilter, SearchResult

log = get_logger(__name__)


@dataclass
class HybridRetrievalResult:
    """Fused candidate list plus per-stage diagnostics."""

    results: list[SearchResult]
    dense_count: int = 0
    sparse_count: int = 0
    fused_count: int = 0
    queries_embedded: int = 0


class HybridRetriever:
    """
    Stateless retriever — every method is per-query. The dense and
    sparse branches both run inside the workspace's mandatory user_id
    isolation filter; there is no path that bypasses it.
    """

    def __init__(
        self,
        embedder: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        candidate_k: int = 50,
        sparse_k: int | None = None,
        rrf_k: int = 60,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._candidate_k = candidate_k
        # Sparse retrieval is cheap on the FT index — pull the same
        # number as dense by default so RRF has equal contribution.
        self._sparse_k = sparse_k if sparse_k is not None else candidate_k
        self._rrf_k = rrf_k

    async def retrieve(
        self,
        *,
        workspace_id: uuid.UUID,
        user_id: uuid.UUID,
        enrichment: QueryEnrichment,
        document_ids: list[str] | None = None,
    ) -> HybridRetrievalResult:
        ws_id = str(workspace_id)
        search_filter = SearchFilter(
            user_id=str(user_id),
            workspace_id=ws_id,
            document_ids=document_ids or [],
        )

        # ── Stage 1: embed each retrieval query in parallel ──────
        vectors = await asyncio.gather(
            *(self._embedder.embed_query(q) for q in enrichment.queries)
        )

        # ── Stage 2: dense + sparse in parallel ──────────────────
        dense_tasks = [
            self._vector_store.search(
                workspace_id=ws_id,
                query_vector=vec,
                search_filter=search_filter,
                top_k=self._candidate_k,
                score_threshold=0.0,  # Recall-first; reranker filters
            )
            for vec in vectors
        ]
        sparse_task = self._safe_text_search(
            ws_id=ws_id,
            query=enrichment.original,
            search_filter=search_filter,
        )

        dense_results_list_and_sparse = await asyncio.gather(
            *dense_tasks, sparse_task, return_exceptions=True
        )
        *dense_lists, sparse_results = dense_results_list_and_sparse

        # Surface (but don't fail on) per-branch errors.
        dense_results: list[list[SearchResult]] = []
        for dl in dense_lists:
            if isinstance(dl, Exception):
                log.warning("dense_branch_failed", error=str(dl))
                continue
            dense_results.append(dl)
        if isinstance(sparse_results, Exception):
            log.warning("sparse_branch_failed", error=str(sparse_results))
            sparse_results = []

        # ── Stage 3: merge dense (best score per chunk) ──────────
        dense_merged = _merge_dense(dense_results)

        # ── Stage 4: RRF fuse dense + sparse ─────────────────────
        if sparse_results:
            fused = _reciprocal_rank_fusion(
                ranked_lists=[dense_merged, sparse_results],
                k=self._rrf_k,
            )
        else:
            fused = dense_merged

        result = HybridRetrievalResult(
            results=fused[: self._candidate_k],
            dense_count=len(dense_merged),
            sparse_count=len(sparse_results),
            fused_count=len(fused),
            queries_embedded=len(enrichment.queries),
        )

        log.info(
            "hybrid_retrieval",
            user_id=str(user_id),
            workspace_id=ws_id,
            dense=result.dense_count,
            sparse=result.sparse_count,
            fused=result.fused_count,
            queries=result.queries_embedded,
        )
        return result

    async def _safe_text_search(
        self,
        *,
        ws_id: str,
        query: str,
        search_filter: SearchFilter,
    ) -> list[SearchResult]:
        """Wrap text_search so a backend without FT support returns []."""
        try:
            return await self._vector_store.text_search(
                workspace_id=ws_id,
                query=query,
                search_filter=search_filter,
                top_k=self._sparse_k,
            )
        except NotImplementedError:
            # Acceptable fallback for non-Qdrant backends that don't
            # have a text payload index. Log once per run.
            log.debug("text_search_not_supported")
            return []


# ── Pure helpers ──────────────────────────────────────────────────


def _merge_dense(branches: list[list[SearchResult]]) -> list[SearchResult]:
    """Merge multiple dense branches keeping the best score per chunk_id."""
    best: dict[str, SearchResult] = {}
    for branch in branches:
        for r in branch:
            existing = best.get(r.chunk_id)
            if existing is None or r.score > existing.score:
                best[r.chunk_id] = r
    return sorted(best.values(), key=lambda r: r.score, reverse=True)


def _reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
) -> list[SearchResult]:
    """
    Standard RRF: each chunk's fused score is sum(1 / (k + rank_i))
    across each list it appears in. Robust to score-scale differences
    between dense (cosine 0–1) and sparse (synthetic rank-based)
    branches — the only thing that matters is rank position.
    """
    scores: dict[str, float] = {}
    canonical: dict[str, SearchResult] = {}
    for branch in ranked_lists:
        for rank, result in enumerate(branch):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (
                k + rank + 1
            )
            # Prefer the dense-branch SearchResult when both branches see
            # the same chunk (it carries the meaningful cosine score).
            existing = canonical.get(result.chunk_id)
            if existing is None or result.score > existing.score:
                canonical[result.chunk_id] = result

    fused = sorted(canonical.values(), key=lambda r: scores[r.chunk_id], reverse=True)
    # Replace each result's score with the fused score for downstream
    # transparency. Reranker uses its own model score anyway, so this
    # is purely diagnostic.
    return [
        SearchResult(chunk_id=r.chunk_id, score=scores[r.chunk_id], payload=r.payload)
        for r in fused
    ]
