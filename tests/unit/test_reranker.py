"""
tests/unit/test_reranker.py
─────────────────────────────
Unit tests for reranker implementations.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.providers.llm.base import LLMResponse
from src.retrieval.reranker import (
    LLMReranker,
    NoOpReranker,
    get_reranker,
)
from src.vector_store.base import SearchResult, VectorPayload


def _make_result(
    chunk_id: str, score: float, text: str = "passage text"
) -> SearchResult:
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


class TestNoOpReranker:
    @pytest.mark.asyncio
    async def test_returns_top_k_in_original_order(self):
        reranker = NoOpReranker()
        results = [_make_result(f"c{i}", 1.0 - i * 0.1) for i in range(10)]
        reranked = await reranker.rerank("query", results, top_k=3)
        assert len(reranked) == 3
        assert reranked[0].chunk_id == "c0"
        assert reranked[1].chunk_id == "c1"

    @pytest.mark.asyncio
    async def test_empty_returns_empty(self):
        reranker = NoOpReranker()
        assert await reranker.rerank("q", [], top_k=5) == []

    @pytest.mark.asyncio
    async def test_top_k_larger_than_results_returns_all(self):
        reranker = NoOpReranker()
        results = [_make_result(f"c{i}", 0.9) for i in range(3)]
        reranked = await reranker.rerank("q", results, top_k=10)
        assert len(reranked) == 3


class TestLLMReranker:
    def _make_llm(self, score: int = 8):
        llm = AsyncMock()
        llm.generate = AsyncMock(
            return_value=LLMResponse(
                content=str(score),
                model="mock",
                prompt_tokens=30,
                completion_tokens=2,
            )
        )
        return llm

    @pytest.mark.asyncio
    async def test_reranks_by_llm_score(self):
        """Chunk with higher LLM score should rank first."""
        llm = AsyncMock()
        # First call returns 3 (low relevance), second returns 9 (high relevance)
        llm.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="3", model="mock", prompt_tokens=10, completion_tokens=2
                ),
                LLMResponse(
                    content="9", model="mock", prompt_tokens=10, completion_tokens=2
                ),
            ]
        )
        reranker = LLMReranker(llm=llm)
        results = [
            _make_result("c1", 0.9, "Unrelated passage about weather."),
            _make_result("c2", 0.8, "Refund policy: 30 days for full return."),
        ]
        reranked = await reranker.rerank("What is the refund policy?", results, top_k=2)
        assert reranked[0].chunk_id == "c2"  # Higher score should be first

    @pytest.mark.asyncio
    async def test_scores_normalised_to_0_1(self):
        reranker = LLMReranker(llm=self._make_llm(8))
        results = [_make_result("c1", 0.9)]
        reranked = await reranker.rerank("query", results, top_k=1)
        assert 0.0 <= reranked[0].score <= 1.0

    @pytest.mark.asyncio
    async def test_handles_llm_error_gracefully(self):
        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=Exception("LLM error"))
        reranker = LLMReranker(llm=llm)
        results = [_make_result("c1", 0.9), _make_result("c2", 0.8)]
        # Should not raise — LLM errors assign default score 5
        reranked = await reranker.rerank("query", results, top_k=2)
        assert len(reranked) == 2

    @pytest.mark.asyncio
    async def test_empty_returns_empty(self):
        reranker = LLMReranker(llm=self._make_llm())
        assert await reranker.rerank("q", [], top_k=5) == []

    @pytest.mark.asyncio
    async def test_non_numeric_response_defaults_to_5(self):
        """If LLM returns non-numeric, default to score 5."""
        llm = AsyncMock()
        llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="very relevant",
                model="mock",
                prompt_tokens=10,
                completion_tokens=3,
            )
        )
        reranker = LLMReranker(llm=llm)
        results = [_make_result("c1", 0.9)]
        reranked = await reranker.rerank("q", results, top_k=1)
        # Score of 5/5 = 1.0 (only one result, gets max score normalisation)
        assert 0.0 <= reranked[0].score <= 1.0


class TestGetReranker:
    def test_cross_encoder_strategy(self):
        from src.retrieval.reranker import CrossEncoderReranker

        r = get_reranker("cross_encoder")
        assert isinstance(r, CrossEncoderReranker)

    def test_none_strategy_returns_noop(self):
        r = get_reranker("none")
        assert isinstance(r, NoOpReranker)

    def test_llm_strategy_requires_llm(self):
        with pytest.raises(ValueError):
            get_reranker("llm", llm=None)

    def test_llm_strategy_with_llm(self):
        mock_llm = MagicMock()
        r = get_reranker("llm", llm=mock_llm)
        assert isinstance(r, LLMReranker)


class TestCrossEncoderBatchSize:
    def _scored(self, monkeypatch, reranker) -> MagicMock:
        from src.retrieval.reranker import CrossEncoderReranker

        model = MagicMock()
        model.predict.return_value = [2.0, -1.0, 0.5]
        monkeypatch.setattr(CrossEncoderReranker, "_load_model", lambda *_: model)
        results = [_make_result(f"c{i}", 0.5) for i in range(3)]
        ranked = reranker._rerank_sync("q", results, top_k=2)
        assert [r.chunk_id for r in ranked] == ["c0", "c2"]
        return model

    def test_uses_the_configured_batch_size(self, monkeypatch):
        from src.configs.config import get_settings
        from src.retrieval.reranker import CrossEncoderReranker

        monkeypatch.setattr(get_settings(), "reranker_batch_size", 8)
        model = self._scored(monkeypatch, CrossEncoderReranker(model_name="m"))
        assert model.predict.call_args.kwargs["batch_size"] == 8

    def test_an_explicit_batch_size_wins(self, monkeypatch):
        from src.retrieval.reranker import CrossEncoderReranker

        model = self._scored(
            monkeypatch, CrossEncoderReranker(model_name="m", batch_size=4)
        )
        assert model.predict.call_args.kwargs["batch_size"] == 4
