"""
tests/unit/test_query_rewriter.py
────────────────────────────────────
Unit tests for QueryRewriter — all LLM calls mocked.
"""

import pytest
from unittest.mock import AsyncMock

from src.providers.llm.base import LLMResponse
from src.retrieval.query_rewriter import (
    QueryRewriter,
    RewriteStrategy,
    _clean_query_line,
)


def _mock_llm(content: str):
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=LLMResponse(
        content=content,
        model="mock",
        prompt_tokens=10,
        completion_tokens=20,
    ))
    return llm


class TestQueryRewriterNone:
    @pytest.mark.asyncio
    async def test_none_strategy_returns_original(self):
        rewriter = QueryRewriter(llm=_mock_llm("irrelevant"))
        result = await rewriter.rewrite("What is the policy?", RewriteStrategy.NONE)
        assert result == ["What is the policy?"]

    @pytest.mark.asyncio
    async def test_none_strategy_does_not_call_llm(self):
        llm = _mock_llm("irrelevant")
        rewriter = QueryRewriter(llm=llm)
        await rewriter.rewrite("Any query", RewriteStrategy.NONE)
        llm.generate.assert_not_called()


class TestQueryRewriterHYD:
    @pytest.mark.asyncio
    async def test_hyd_returns_hypothesis_list(self):
        hypothesis = "The refund policy allows customers to return items within 30 days of purchase for a full refund."
        rewriter = QueryRewriter(llm=_mock_llm(hypothesis))
        result = await rewriter.rewrite("What is the refund policy?", RewriteStrategy.HYD)
        assert len(result) == 1
        assert result[0] == hypothesis

    @pytest.mark.asyncio
    async def test_hyd_falls_back_on_short_response(self):
        """Short LLM responses (< 20 chars) should fall back to original query."""
        rewriter = QueryRewriter(llm=_mock_llm("Refunds: 30 days"))
        # "Refunds: 30 days" is 16 chars < 20 → fallback
        result = await rewriter.rewrite("What is the refund policy?", RewriteStrategy.HYD)
        assert result == ["What is the refund policy?"]

    @pytest.mark.asyncio
    async def test_hyd_falls_back_on_empty_response(self):
        rewriter = QueryRewriter(llm=_mock_llm(""))
        result = await rewriter.rewrite("Query", RewriteStrategy.HYD)
        assert result == ["Query"]

    @pytest.mark.asyncio
    async def test_hyd_falls_back_on_llm_error(self):
        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=Exception("LLM timeout"))
        rewriter = QueryRewriter(llm=llm)
        result = await rewriter.rewrite("Query", RewriteStrategy.HYD)
        assert result == ["Query"]  # Must never raise


class TestQueryRewriterMulti:
    @pytest.mark.asyncio
    async def test_multi_returns_original_plus_variants(self):
        variants_text = "What are the return guidelines?\nHow long do I have to return items?\nCan I get a refund after purchase?"
        rewriter = QueryRewriter(llm=_mock_llm(variants_text))
        result = await rewriter.rewrite("What is the refund policy?", RewriteStrategy.MULTI_QUERY)
        # Should include original + up to 3 variants
        assert result[0] == "What is the refund policy?"
        assert len(result) > 1

    @pytest.mark.asyncio
    async def test_multi_deduplicates_variants(self):
        # All variants are the same as original (case-insensitive)
        variants_text = "What is the refund policy?\nWhat is the refund policy?\nwhat is the refund policy?"
        rewriter = QueryRewriter(llm=_mock_llm(variants_text))
        result = await rewriter.rewrite("What is the refund policy?", RewriteStrategy.MULTI_QUERY)
        # Should only have original (all duplicates removed)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_multi_strips_numbering(self):
        variants_text = "1. How to request a refund?\n2. What items can be returned?\n3. Is there a time limit?"
        rewriter = QueryRewriter(llm=_mock_llm(variants_text))
        result = await rewriter.rewrite("Refund process", RewriteStrategy.MULTI_QUERY)
        for q in result[1:]:  # Skip original
            assert not q.startswith("1.")
            assert not q.startswith("2.")
            assert not q[0].isdigit() or q[1] not in ".)"


class TestCleanQueryLine:
    def test_strips_number_dot(self):
        assert _clean_query_line("1. What is X?") == "What is X?"

    def test_strips_bullet(self):
        assert _clean_query_line("- What is X?") == "What is X?"

    def test_strips_star(self):
        assert _clean_query_line("* What is X?") == "What is X?"

    def test_strips_number_paren(self):
        assert _clean_query_line("3) What is X?") == "What is X?"

    def test_no_prefix_unchanged(self):
        assert _clean_query_line("What is X?") == "What is X?"

    def test_empty_returns_empty(self):
        assert _clean_query_line("") == ""
