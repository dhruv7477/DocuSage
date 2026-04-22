"""
tests/unit/test_hybrid_fusion.py
──────────────────────────────────
Unit tests for Reciprocal Rank Fusion and deduplication.
Pure logic — no external deps.
"""

import pytest

from src.retrieval.bm25_index import BM25Chunk, BM25Index, _tokenise
from src.retrieval.hybrid_fusion import (
    deduplicate_results,
    reciprocal_rank_fusion,
)
from src.vector_store.base import SearchResult, VectorPayload


def _make_result(chunk_id: str, score: float, text: str = "some text") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        payload=VectorPayload(
            user_id="user-1",
            workspace_id="ws-1",
            document_id="doc-1",
            chunk_index=0,
            text=text,
            source="doc.pdf",
            page_number=1,
        ),
    )


class TestRRF:
    def test_empty_inputs_return_empty(self):
        assert reciprocal_rank_fusion([], [], top_k=5) == []

    def test_vector_only_returns_top_k(self):
        results = [_make_result(f"c{i}", 1.0 - i * 0.1) for i in range(10)]
        fused = reciprocal_rank_fusion(results, [], top_k=3)
        assert len(fused) == 3

    def test_bm25_only_returns_top_k(self):
        results = [_make_result(f"c{i}", 1.0 - i * 0.1) for i in range(10)]
        fused = reciprocal_rank_fusion([], results, top_k=3)
        assert len(fused) == 3

    def test_result_appearing_in_both_ranks_higher(self):
        """A chunk that appears in both vector and BM25 should rank highest."""
        shared_id = "shared-chunk"
        vector = [
            _make_result(shared_id, 0.9),
            _make_result("v-only-1", 0.8),
            _make_result("v-only-2", 0.7),
        ]
        bm25 = [
            _make_result(shared_id, 0.95),
            _make_result("b-only-1", 0.85),
        ]
        fused = reciprocal_rank_fusion(vector, bm25, top_k=5)
        assert fused[0].chunk_id == shared_id

    def test_scores_are_normalised_to_0_1(self):
        results = [_make_result(f"c{i}", 0.9 - i * 0.1) for i in range(5)]
        fused = reciprocal_rank_fusion(results, [], top_k=5)
        for r in fused:
            assert 0.0 <= r.score <= 1.0

    def test_fused_sorted_descending(self):
        vector = [_make_result(f"c{i}", 1.0 - i * 0.1) for i in range(5)]
        bm25 = [_make_result(f"b{i}", 1.0 - i * 0.1) for i in range(5)]
        fused = reciprocal_rank_fusion(vector, bm25, top_k=10)
        scores = [r.score for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_deduplication_by_chunk_id(self):
        """Same chunk_id from both lists should only appear once."""
        same = [_make_result("c1", 0.9)]
        fused = reciprocal_rank_fusion(same, same, top_k=10)
        ids = [r.chunk_id for r in fused]
        assert len(ids) == len(set(ids))


class TestDeduplication:
    def test_exact_duplicate_text_removed(self):
        text = "The policy is exactly the same text."
        results = [
            _make_result("c1", 0.9, text),
            _make_result("c2", 0.8, text),  # Same text — should be removed
        ]
        deduped = deduplicate_results(results, similarity_threshold=0.85)
        assert len(deduped) == 1
        assert deduped[0].chunk_id == "c1"  # Keep best (first)

    def test_different_texts_not_removed(self):
        results = [
            _make_result("c1", 0.9, "The refund policy allows 30 day returns."),
            _make_result("c2", 0.8, "Customer support is available 24/7 by phone."),
        ]
        deduped = deduplicate_results(results, similarity_threshold=0.85)
        assert len(deduped) == 2

    def test_empty_list_returns_empty(self):
        assert deduplicate_results([]) == []

    def test_single_result_unchanged(self):
        results = [_make_result("c1", 0.9, "Some text")]
        assert deduplicate_results(results) == results


class TestBM25Tokenise:
    def test_lowercase(self):
        tokens = _tokenise("Hello World")
        assert tokens == ["hello", "world"]

    def test_strips_punctuation(self):
        tokens = _tokenise("Hello, world!")
        assert tokens == ["hello", "world"]

    def test_filters_short_tokens(self):
        tokens = _tokenise("a be cat")
        # "a" is filtered (< 2 chars), "be" stays (2 chars)
        assert "a" not in tokens
        assert "be" in tokens
        assert "cat" in tokens

    def test_empty_returns_empty(self):
        assert _tokenise("") == []

    def test_numbers_included(self):
        tokens = _tokenise("Article 42 section 7")
        assert "42" in tokens
        assert "section" in tokens


class TestBM25Index:
    def _make_chunk(self, chunk_id: str, text: str) -> BM25Chunk:
        return BM25Chunk(
            chunk_id=chunk_id,
            document_id="doc-1",
            text=text,
            source="doc.pdf",
            page_number=1,
            chunk_index=0,
            user_id="user-1",
            workspace_id="ws-1",
        )

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_chunks(self):
        index = BM25Index()
        if not index._bm25_available:
            pytest.skip("rank_bm25 not installed")
        results = await index.search("query", [], top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_returns_relevant_chunk_first(self):
        index = BM25Index()
        if not index._bm25_available:
            pytest.skip("rank_bm25 not installed")

        chunks = [
            self._make_chunk("c1", "The refund policy allows returns within 30 days."),
            self._make_chunk("c2", "Our office hours are Monday to Friday."),
            self._make_chunk("c3", "The warranty covers manufacturing defects."),
        ]
        results = await index.search("refund policy returns", chunks, top_k=3)
        assert results[0].chunk_id == "c1"

    @pytest.mark.asyncio
    async def test_scores_normalised_0_to_1(self):
        index = BM25Index()
        if not index._bm25_available:
            pytest.skip("rank_bm25 not installed")

        chunks = [
            self._make_chunk("c1", "refund policy returns"),
            self._make_chunk("c2", "unrelated topic about weather"),
        ]
        results = await index.search("refund", chunks, top_k=2)
        for r in results:
            assert 0.0 <= r.score <= 1.0
