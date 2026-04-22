"""
src/retrieval/bm25_index.py
────────────────────────────
BM25 sparse keyword retrieval over document chunks stored in PostgreSQL.

Why BM25 alongside vector search?
    Vector search excels at semantic similarity but can miss exact-match
    keywords (product codes, names, acronyms, numbers).
    BM25 is the inverse — weak at paraphrase but perfect at exact keyword
    matching.  Hybrid search combines both to get best of both worlds.

Architecture:
    BM25 is computed in-memory over chunks fetched from PostgreSQL.
    We do NOT maintain a persistent BM25 index because:
    - Chunks change as documents are added/deleted
    - The BM25 computation is fast (< 50ms for 10K chunks)
    - This avoids a separate search infrastructure

    For production at scale (>100K chunks per user), consider:
    - Elasticsearch/OpenSearch for persistent BM25
    - Qdrant sparse vectors (SPLADE model)

Implementation:
    Uses rank_bm25 library (pure Python, no C deps).
    Tokenisation: lowercase + split on non-alphanumeric.
    Chunks are fetched once per query and cached per request.

Usage:
    index = BM25Index()
    results = await index.search(
        query="refund policy 30 days",
        workspace_id=ws_id,
        user_id=user_id,
        db=db_session,
        top_k=10,
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logger import get_logger
from src.vector_store.base import SearchResult, VectorPayload

log = get_logger(__name__)


@dataclass
class BM25Chunk:
    """A minimal chunk record for BM25 indexing."""
    chunk_id: str
    document_id: str
    text: str
    source: str
    page_number: int
    chunk_index: int
    user_id: str
    workspace_id: str


class BM25Index:
    """
    In-memory BM25 index over chunks fetched from the vector store payload.

    Because Qdrant stores full chunk text in payload, we can fetch all
    chunks for a user+workspace, build the BM25 index, and query it —
    all in a single request context.
    """

    def __init__(self) -> None:
        # rank_bm25 import is lazy to avoid hard dependency at module load
        self._bm25_available = self._check_bm25()

    def _check_bm25(self) -> bool:
        try:
            import rank_bm25  # noqa: F401
            return True
        except ImportError:
            log.warning(
                "rank_bm25_not_installed",
                hint="pip install rank-bm25 to enable BM25 hybrid search",
            )
            return False

    async def search(
        self,
        query: str,
        chunks: list[BM25Chunk],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Run BM25 search over the provided chunk list.

        Args:
            query:   Raw query string (not embedded).
            chunks:  Pre-fetched chunks for this user+workspace.
            top_k:   Maximum results to return.

        Returns:
            SearchResult list sorted by BM25 score descending.
            Scores are normalised to [0.0, 1.0].
        """
        if not self._bm25_available or not chunks:
            return []

        from rank_bm25 import BM25Okapi

        # Tokenise all chunks
        tokenised = [_tokenise(c.text) for c in chunks]
        query_tokens = _tokenise(query)

        if not query_tokens:
            return []

        bm25 = BM25Okapi(tokenised)
        raw_scores = bm25.get_scores(query_tokens)

        # Normalise scores to [0, 1]
        max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
        norm_scores = [s / max_score for s in raw_scores]

        # Pair with chunks, filter zeros, sort, take top_k
        scored = [
            (chunks[i], norm_scores[i])
            for i in range(len(chunks))
            if norm_scores[i] > 0.0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for chunk, score in scored[:top_k]:
            results.append(SearchResult(
                chunk_id=chunk.chunk_id,
                score=round(score, 4),
                payload=VectorPayload(
                    user_id=chunk.user_id,
                    workspace_id=chunk.workspace_id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    source=chunk.source,
                    page_number=chunk.page_number,
                ),
            ))

        log.debug(
            "bm25_search_done",
            query_preview=query[:50],
            total_chunks=len(chunks),
            results=len(results),
        )
        return results


def build_chunks_from_qdrant_results(
    results: list[SearchResult],
) -> list[BM25Chunk]:
    """
    Convert SearchResult list (from Qdrant) to BM25Chunk list.
    Used when we want to apply BM25 re-scoring over vector results.
    """
    return [
        BM25Chunk(
            chunk_id=r.chunk_id,
            document_id=r.payload.document_id,
            text=r.payload.text,
            source=r.payload.source,
            page_number=r.payload.page_number,
            chunk_index=r.payload.chunk_index,
            user_id=r.payload.user_id,
            workspace_id=r.payload.workspace_id,
        )
        for r in results
    ]


def _tokenise(text: str) -> list[str]:
    """
    Lowercase and split on non-alphanumeric characters.
    Removes tokens shorter than 2 chars to reduce noise.
    """
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 2]
