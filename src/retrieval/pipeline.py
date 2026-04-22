"""
src/retrieval/pipeline.py
──────────────────────────
Advanced retrieval pipeline: the upgraded replacement for the simple
vector-only search in RAGService._retrieve().

Full pipeline per query:
    1. Query rewriting  (HYD or multi-query or pass-through)
    2. Parallel vector search for each rewritten query
    3. BM25 search over the same candidate pool
    4. Reciprocal Rank Fusion to merge vector + BM25 results
    5. Near-duplicate deduplication
    6. Cross-encoder reranking of top candidates
    7. Return top_k results with scores and provenance

Integration:
    RetrievalPipeline replaces RAGService._retrieve() via injection.
    RAGService accepts an optional `retrieval_pipeline` argument.
    If not provided, it falls back to the original simple vector search.
    This maintains backward compatibility.

Config (.env):
    REWRITER_STRATEGY=hyd        # none | hyd | multi
    RERANKER=cross_encoder       # none | cross_encoder | llm
    HYBRID_ALPHA=0.5             # Weight of BM25 (0=pure vector, 1=pure BM25)
    RETRIEVAL_CANDIDATE_K=20     # Candidates to fetch before reranking
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field

from src.core.logger import get_logger
from src.models.user import User
from src.providers.embedding.base import BaseEmbeddingProvider
from src.retrieval.bm25_index import BM25Index, build_chunks_from_qdrant_results
from src.retrieval.hybrid_fusion import deduplicate_results, reciprocal_rank_fusion
from src.retrieval.query_rewriter import QueryRewriter, RewriteStrategy
from src.retrieval.reranker import BaseReranker, NoOpReranker
from src.vector_store.base import BaseVectorStore, SearchFilter, SearchResult

log = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Results from the advanced retrieval pipeline with full provenance."""
    results: list[SearchResult]
    rewritten_queries: list[str] = field(default_factory=list)
    vector_result_count: int = 0
    bm25_result_count: int = 0
    after_fusion_count: int = 0
    after_dedup_count: int = 0
    after_rerank_count: int = 0


class RetrievalPipeline:
    """
    Advanced multi-stage retrieval pipeline.

    Replace simple vector search with:
        query rewrite → hybrid search (vector + BM25) → RRF fusion
        → deduplication → cross-encoder reranking

    All stages are optional and configurable via constructor args.
    The pipeline degrades gracefully: if reranking fails, it returns
    fused results; if BM25 fails, it returns vector-only results.
    """

    def __init__(
        self,
        embedder: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        rewriter: QueryRewriter | None = None,
        reranker: BaseReranker | None = None,
        rewrite_strategy: RewriteStrategy = RewriteStrategy.HYD,
        candidate_k: int = 20,       # Candidates before reranking
        final_top_k: int = 5,        # Results returned to RAG service
        score_threshold: float = 0.3, # Lower threshold for candidates
        use_bm25: bool = True,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._rewriter = rewriter
        self._reranker = reranker or NoOpReranker()
        self._rewrite_strategy = rewrite_strategy
        self._candidate_k = candidate_k
        self._final_top_k = final_top_k
        self._score_threshold = score_threshold
        self._use_bm25 = use_bm25
        self._bm25 = BM25Index() if use_bm25 else None

    async def retrieve(
        self,
        query: str,
        workspace_id: uuid.UUID,
        user: User,
        document_ids: list[str] | None = None,
        history_summary: str | None = None,
    ) -> RetrievalResult:
        """
        Run the full advanced retrieval pipeline.

        Args:
            query:            Raw user query.
            workspace_id:     Tenant workspace ID.
            user:             Authenticated user (enforces isolation).
            document_ids:     Optional: scope to specific documents.
            history_summary:  Optional: for anaphora resolution in rewriter.

        Returns:
            RetrievalResult with final ranked results and pipeline metrics.
        """
        result = RetrievalResult(results=[])

        # ── Stage 1: Query Rewriting ──────────────────────────────
        if self._rewriter:
            try:
                rewritten = await self._rewriter.rewrite(
                    query=query,
                    strategy=self._rewrite_strategy,
                    history_summary=history_summary,
                )
            except Exception as exc:
                log.warning("rewrite_stage_failed", error=str(exc))
                rewritten = [query]
        else:
            rewritten = [query]

        result.rewritten_queries = rewritten

        # ── Stage 2: Parallel Vector Search ──────────────────────
        search_filter = SearchFilter(
            user_id=str(user.id),
            workspace_id=str(workspace_id),
            document_ids=document_ids or [],
        )

        vector_results = await self._multi_vector_search(
            queries=rewritten,
            workspace_id=str(workspace_id),
            search_filter=search_filter,
        )
        result.vector_result_count = len(vector_results)

        # ── Stage 3: BM25 Search ──────────────────────────────────
        bm25_results: list[SearchResult] = []
        if self._use_bm25 and self._bm25 and vector_results:
            try:
                # BM25 runs over the vector candidates (not full corpus)
                # This is "late fusion" — cheap and effective
                chunks = build_chunks_from_qdrant_results(vector_results)
                bm25_results = await self._bm25.search(
                    query=query,  # Use original query for BM25 (keyword match)
                    chunks=chunks,
                    top_k=self._candidate_k,
                )
                result.bm25_result_count = len(bm25_results)
            except Exception as exc:
                log.warning("bm25_stage_failed", error=str(exc))

        # ── Stage 4: Hybrid Fusion (RRF) ─────────────────────────
        if bm25_results:
            fused = reciprocal_rank_fusion(
                vector_results=vector_results,
                bm25_results=bm25_results,
                top_k=self._candidate_k,
            )
        else:
            fused = vector_results[: self._candidate_k]

        result.after_fusion_count = len(fused)

        # ── Stage 5: Deduplication ────────────────────────────────
        deduped = deduplicate_results(fused, similarity_threshold=0.85)
        result.after_dedup_count = len(deduped)

        # ── Stage 6: Cross-Encoder Reranking ─────────────────────
        try:
            reranked = await self._reranker.rerank(
                query=query,  # Always use original query for reranking
                results=deduped,
                top_k=self._final_top_k,
            )
            result.after_rerank_count = len(reranked)
            result.results = reranked
        except Exception as exc:
            log.warning("rerank_stage_failed", error=str(exc))
            result.results = deduped[: self._final_top_k]
            result.after_rerank_count = len(result.results)

        log.info(
            "retrieval_pipeline_done",
            workspace_id=str(workspace_id),
            user_id=str(user.id),
            queries=len(rewritten),
            vector=result.vector_result_count,
            bm25=result.bm25_result_count,
            fused=result.after_fusion_count,
            deduped=result.after_dedup_count,
            final=result.after_rerank_count,
        )

        return result

    # ── Private ───────────────────────────────────────────────────

    async def _multi_vector_search(
        self,
        queries: list[str],
        workspace_id: str,
        search_filter: SearchFilter,
    ) -> list[SearchResult]:
        """
        Run vector search for each rewritten query in parallel.
        Merge and deduplicate by chunk_id (keep highest score).
        """
        # Embed all queries concurrently
        vectors = await asyncio.gather(
            *[self._embedder.embed_query(q) for q in queries]
        )

        # Search all query vectors concurrently
        search_tasks = [
            self._vector_store.search(
                workspace_id=workspace_id,
                query_vector=vec,
                search_filter=search_filter,
                top_k=self._candidate_k,
                score_threshold=self._score_threshold,
            )
            for vec in vectors
        ]
        all_result_lists = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Merge: keep best score per chunk_id
        best: dict[str, SearchResult] = {}
        for result_list in all_result_lists:
            if isinstance(result_list, Exception):
                continue
            for r in result_list:
                if r.chunk_id not in best or r.score > best[r.chunk_id].score:
                    best[r.chunk_id] = r

        merged = sorted(best.values(), key=lambda r: r.score, reverse=True)
        return merged