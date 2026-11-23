"""
tests/unit/test_hybrid_retrieval.py
────────────────────────────────────
The hybrid retriever reports the best dense cosine, which the answer gate
needs, even though fusion replaces each result's score with a rank score.
"""

import uuid
from unittest.mock import AsyncMock

import pytest

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.query_understanding import QueryEnrichment
from src.vector_store.base import SearchResult, VectorPayload

pytestmark = pytest.mark.unit


def _result(score: float, text: str = "chunk") -> SearchResult:
    return SearchResult(
        chunk_id=str(uuid.uuid4()),
        score=score,
        payload=VectorPayload(
            user_id="u",
            workspace_id="ws",
            document_id="doc",
            chunk_index=0,
            text=text,
            source="a.pdf",
            page_number=1,
        ),
    )


def _enrichment(query: str = "question") -> QueryEnrichment:
    return QueryEnrichment(
        original=query,
        standalone=query,
        queries=[query],
        rewritten=False,
        used_hyde=False,
    )


async def _retrieve(dense: list[SearchResult], sparse: list[SearchResult]):
    embedder = AsyncMock()
    embedder.embed_query.return_value = [0.1] * 8
    store = AsyncMock()
    store.search.return_value = dense
    store.text_search.return_value = sparse
    retriever = HybridRetriever(embedder=embedder, vector_store=store, candidate_k=5)
    return await retriever.retrieve(
        workspace_id=uuid.uuid4(), user_id=uuid.uuid4(), enrichment=_enrichment()
    )


class TestDenseTopScore:
    @pytest.mark.asyncio
    async def test_reports_the_best_cosine_after_fusion(self):
        out = await _retrieve(
            dense=[_result(0.62), _result(0.41)], sparse=[_result(1.0)]
        )
        assert out.dense_top_score == 0.62
        # Fusion replaced every score with a small rank score.
        assert all(r.score < 0.1 for r in out.results)

    @pytest.mark.asyncio
    async def test_reports_the_best_cosine_without_fusion(self):
        out = await _retrieve(dense=[_result(0.55)], sparse=[])
        assert out.dense_top_score == 0.55

    @pytest.mark.asyncio
    async def test_no_dense_hits_reports_zero(self):
        out = await _retrieve(dense=[], sparse=[])
        assert out.dense_top_score == 0.0
