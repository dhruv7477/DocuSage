"""
tests/unit/test_eval_service.py
────────────────────────────────
Unit tests for EvalService metrics — all LLM and embedding calls mocked.
"""

import pytest
from unittest.mock import AsyncMock

from src.providers.llm.base import LLMResponse
from src.services.eval_service import (
    EvalService,
    EvalSuiteResult,
    _cosine_sim,
    _mean,
    _parse_json,
)


def _mock_llm(content: str = "{}"):
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=LLMResponse(
        content=content,
        model="mock",
        prompt_tokens=50,
        completion_tokens=30,
    ))
    return llm


def _mock_embedder(vector: list[float] | None = None):
    v = vector or [0.1] * 1024
    embedder = AsyncMock()
    embedder.embed_query = AsyncMock(return_value=v)
    embedder.embed_documents = AsyncMock(return_value=[v, v])
    return embedder


class TestFaithfulness:
    @pytest.mark.asyncio
    async def test_all_supported_returns_1(self):
        llm = _mock_llm('{"claims": [{"claim": "Refunds in 30 days", "verdict": "supported"}]}')
        service = EvalService(llm=llm)
        score = await service._faithfulness("Q", "Refunds in 30 days", "Context about refunds")
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_all_contradicted_returns_0(self):
        llm = _mock_llm('{"claims": [{"claim": "Refunds in 90 days", "verdict": "contradicted"}]}')
        service = EvalService(llm=llm)
        score = await service._faithfulness("Q", "A", "Context")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_partial_support(self):
        payload = '{"claims": [{"verdict": "supported"}, {"verdict": "contradicted"}, {"verdict": "supported"}]}'
        llm = _mock_llm(payload)
        service = EvalService(llm=llm)
        score = await service._faithfulness("Q", "A", "C")
        assert abs(score - 2/3) < 0.01

    @pytest.mark.asyncio
    async def test_empty_claims_returns_1(self):
        llm = _mock_llm('{"claims": []}')
        service = EvalService(llm=llm)
        score = await service._faithfulness("Q", "A", "C")
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_parse_failure_returns_neutral(self):
        llm = _mock_llm("invalid json {{{{")
        service = EvalService(llm=llm)
        score = await service._faithfulness("Q", "A", "C")
        assert score == 0.5  # Neutral on failure


class TestContextPrecision:
    @pytest.mark.asyncio
    async def test_all_useful_returns_1(self):
        llm = _mock_llm('{"useful": true}')
        service = EvalService(llm=llm)
        score = await service._context_precision("Q", "A", ["chunk1", "chunk2"])
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_none_useful_returns_0(self):
        llm = _mock_llm('{"useful": false}')
        service = EvalService(llm=llm)
        score = await service._context_precision("Q", "A", ["chunk1", "chunk2"])
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_empty_contexts_returns_0(self):
        service = EvalService(llm=_mock_llm())
        score = await service._context_precision("Q", "A", [])
        assert score == 0.0


class TestAnswerSimilarity:
    @pytest.mark.asyncio
    async def test_identical_answers_return_high_score(self):
        # Two identical vectors → cosine sim = 1.0
        embedder = AsyncMock()
        identical_vec = [1.0, 0.0, 0.0]
        embedder.embed_documents = AsyncMock(return_value=[identical_vec, identical_vec])
        service = EvalService(llm=_mock_llm(), embedder=embedder)
        score = await service._answer_similarity("Same answer.", "Same answer.")
        assert abs(score - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_orthogonal_answers_return_low_score(self):
        embedder = AsyncMock()
        embedder.embed_documents = AsyncMock(return_value=[
            [1.0, 0.0], [0.0, 1.0]
        ])
        service = EvalService(llm=_mock_llm(), embedder=embedder)
        score = await service._answer_similarity("Answer A", "Answer B")
        assert score < 0.1  # Orthogonal → near-zero similarity


class TestEvaluateComposite:
    @pytest.mark.asyncio
    async def test_composite_averages_available_metrics(self):
        from src.services.eval_service import EvalResult
        r = EvalResult(question="Q", answer="A")
        r.faithfulness = 0.8
        r.answer_relevance = 0.6
        r.context_precision = 1.0
        assert abs(r.composite_score - (0.8 + 0.6 + 1.0) / 3) < 0.001

    @pytest.mark.asyncio
    async def test_composite_ignores_none_metrics(self):
        from src.services.eval_service import EvalResult
        r = EvalResult(question="Q", answer="A")
        r.faithfulness = 0.9
        # Others are None — should not divide by 4
        assert r.composite_score == 0.9


class TestParseJson:
    def test_valid_json(self):
        assert _parse_json('{"key": "value"}') == {"key": "value"}

    def test_json_with_markdown_fence(self):
        result = _parse_json("```json\n{\"key\": 1}\n```")
        assert result == {"key": 1}

    def test_json_embedded_in_text(self):
        result = _parse_json('Some preamble {"claims": []} end')
        assert result == {"claims": []}

    def test_invalid_json_returns_none(self):
        assert _parse_json("not json at all") is None

    def test_empty_returns_none(self):
        assert _parse_json("") is None


class TestCosinesim:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_sim(v, v) - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        assert _cosine_sim([1.0, 0.0], [0.0, 1.0]) < 0.001

    def test_empty_vectors_return_0(self):
        assert _cosine_sim([], []) == 0.0

    def test_clamped_to_1(self):
        # Floating point might push above 1.0 — must be clamped
        v = [0.9999999] * 10
        result = _cosine_sim(v, v)
        assert result <= 1.0


class TestMean:
    def test_normal_case(self):
        assert _mean([0.8, 0.6, 1.0]) == pytest.approx(0.8)

    def test_empty_returns_0(self):
        assert _mean([]) == 0.0

    def test_single_value(self):
        assert _mean([0.75]) == 0.75


class TestEvalSuiteResult:
    def test_mean_faithfulness(self):
        from src.services.eval_service import EvalResult
        r1 = EvalResult(question="Q", answer="A")
        r1.faithfulness = 0.8
        r2 = EvalResult(question="Q2", answer="A2")
        r2.faithfulness = 0.6
        suite = EvalSuiteResult(results=[r1, r2])
        assert abs(suite.mean_faithfulness - 0.7) < 0.001

    def test_summary_keys(self):
        suite = EvalSuiteResult(results=[])
        summary = suite.summary()
        assert "n_questions" in summary
        assert "mean_faithfulness" in summary
        assert "mean_composite_score" in summary
