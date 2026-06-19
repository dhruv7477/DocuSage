"""
tests/unit/test_question_analyzer.py
─────────────────────────────────────
Router tests. Tier 1 = instant regex heuristics (greetings, summaries);
tier 2 = opt-in LLM classification (use_llm=True) with timeout. The
critical property is fail-open: a broken or slow router must never
block a real question.
"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from src.providers.llm.base import LLMResponse
from src.services.question_analyzer import (
    FALLBACK_OFF_TOPIC_REPLY,
    GREETING_REPLY,
    QuestionAnalyzer,
)

pytestmark = pytest.mark.unit

DOCS = [("doc-aaa", "i-9instr.pdf"), ("doc-bbb", "Quarterly Report.docx")]


def _llm_returning(payload: dict | str) -> AsyncMock:
    content = json.dumps(payload) if isinstance(payload, dict) else payload
    llm = AsyncMock()
    llm.generate.return_value = LLMResponse(content=content, model="test")
    return llm


class TestHeuristicTier:
    """Tier 1 — must never touch the LLM."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "greeting",
        ["Hi", "hi there!", "Hello", "hey", "Thanks!", "good morning",
         # common typos the LLM tier shouldn't have to handle
         "hlo", "helo", "heya", "hiya", "yo", "sup"],
    )
    async def test_greetings_route_off_topic_without_llm(self, greeting):
        llm = AsyncMock()
        decision = await QuestionAnalyzer(llm).analyze(greeting, documents=DOCS)
        assert decision.intent == "off_topic"
        assert decision.reply == GREETING_REPLY
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_greeting_prefix_with_real_question_is_not_off_topic(self):
        llm = AsyncMock()
        decision = await QuestionAnalyzer(llm).analyze(
            "hi, what does section 1 require?", documents=DOCS
        )
        assert decision.intent == "rag_query"
        llm.generate.assert_not_called()  # use_llm defaults to False

    @pytest.mark.asyncio
    async def test_summary_request_routes_without_llm(self):
        llm = AsyncMock()
        decision = await QuestionAnalyzer(llm).analyze("summarize everything", documents=DOCS)
        assert decision.intent == "doc_summary"
        assert decision.target_document_ids == []  # [] = all documents
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_summary_with_filename_mention_narrows_targets(self):
        llm = AsyncMock()
        decision = await QuestionAnalyzer(llm).analyze(
            "give me an executive summary of the quarterly report", documents=DOCS
        )
        assert decision.intent == "doc_summary"
        assert decision.target_document_ids == ["doc-bbb"]
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_plain_question_defaults_to_rag_without_llm(self):
        llm = AsyncMock()
        decision = await QuestionAnalyzer(llm).analyze(
            "Who signs Section 1?", documents=DOCS
        )
        assert decision.intent == "rag_query"
        llm.generate.assert_not_called()


class TestLLMTier:
    """Tier 2 — only with use_llm=True, for non-heuristic queries."""

    @pytest.mark.asyncio
    async def test_off_topic_uses_fixed_refusal_never_answers(self):
        # The router classifies only. Even if the model tries to ANSWER the
        # off-topic question in a 'reply' field, we discard it and use the
        # fixed refusal — so off-topic questions are never answered.
        llm = _llm_returning({
            "intent": "off_topic",
            "reply": "A boy is typically a male child.",  # model trying to answer
            "documents": [],
        })
        decision = await QuestionAnalyzer(llm, use_llm=True).analyze(
            "what is a boy?", documents=DOCS
        )
        assert decision.intent == "off_topic"
        assert decision.reply == FALLBACK_OFF_TOPIC_REPLY
        assert "male child" not in decision.reply

    @pytest.mark.asyncio
    async def test_generic_knowledge_question_reaches_llm_and_redirects(self):
        # The user's repro: "What is the capital of India?" is neither a
        # greeting nor a summary, so heuristics must NOT short-circuit it —
        # it reaches the LLM classifier, which routes it off_topic with a
        # redirect instead of a blind rag_query that abstains.
        llm = _llm_returning({
            "intent": "off_topic",
            "reply": "I can only help with the documents in this workspace.",
            "documents": [],
        })
        decision = await QuestionAnalyzer(llm, use_llm=True).analyze(
            "What is the capital of India?", documents=DOCS
        )
        llm.generate.assert_awaited_once()  # heuristics did not catch it
        assert decision.intent == "off_topic"
        assert "documents" in decision.reply

    @pytest.mark.asyncio
    async def test_off_topic_empty_reply_falls_back_to_canned(self):
        llm = _llm_returning({"intent": "off_topic", "reply": "  ", "documents": []})
        decision = await QuestionAnalyzer(llm, use_llm=True).analyze(
            "tell me a joke about cats", documents=DOCS
        )
        assert decision.reply == FALLBACK_OFF_TOPIC_REPLY

    @pytest.mark.asyncio
    async def test_doc_summary_resolves_filename_fragment(self):
        llm = _llm_returning({
            "intent": "doc_summary",
            "reply": "",
            "documents": ["quarterly report"],
        })
        decision = await QuestionAnalyzer(llm, use_llm=True).analyze(
            "condense the quarterly numbers for me", documents=DOCS
        )
        assert decision.intent == "doc_summary"
        assert decision.target_document_ids == ["doc-bbb"]

    @pytest.mark.asyncio
    async def test_uses_json_mode_and_zero_temperature(self):
        llm = _llm_returning({"intent": "rag_query", "reply": "", "documents": []})
        await QuestionAnalyzer(llm, max_tokens=99, use_llm=True).analyze(
            "an ambiguous request", documents=DOCS
        )
        _, config = llm.generate.call_args.args
        assert config.response_format == "json"
        assert config.temperature == 0.0
        assert config.max_tokens == 99

    @pytest.mark.asyncio
    async def test_documents_and_history_in_prompt(self):
        llm = _llm_returning({"intent": "rag_query", "reply": "", "documents": []})
        await QuestionAnalyzer(llm, use_llm=True).analyze(
            "and that one?",
            history=[("user", "condense i-9instr"), ("assistant", "Here is a condensed view...")],
            documents=DOCS,
        )
        messages, _ = llm.generate.call_args.args
        user_msg = messages[-1].content
        assert "i-9instr.pdf" in user_msg
        assert "Quarterly Report.docx" in user_msg
        assert "condense i-9instr" in user_msg
        assert "and that one?" in user_msg


class TestFailOpen:

    @pytest.mark.asyncio
    async def test_invalid_json_routes_to_rag(self):
        llm = _llm_returning("this is not json {{{")
        decision = await QuestionAnalyzer(llm, use_llm=True).analyze(
            "a real question", documents=DOCS
        )
        assert decision.intent == "rag_query"

    @pytest.mark.asyncio
    async def test_unknown_intent_routes_to_rag(self):
        llm = _llm_returning({"intent": "make_coffee", "reply": "", "documents": []})
        decision = await QuestionAnalyzer(llm, use_llm=True).analyze(
            "a real question", documents=DOCS
        )
        assert decision.intent == "rag_query"

    @pytest.mark.asyncio
    async def test_llm_exception_routes_to_rag(self):
        llm = AsyncMock()
        llm.generate.side_effect = RuntimeError("ollama down")
        decision = await QuestionAnalyzer(llm, use_llm=True).analyze(
            "a real question", documents=DOCS
        )
        assert decision.intent == "rag_query"

    @pytest.mark.asyncio
    async def test_slow_llm_times_out_and_fails_open(self):
        llm = AsyncMock()

        async def _slow(*args, **kwargs):
            await asyncio.sleep(1.0)  # far beyond the 0.05s budget below
            return LLMResponse(content="{}", model="test")

        llm.generate.side_effect = _slow
        decision = await QuestionAnalyzer(
            llm, use_llm=True, llm_timeout_seconds=0.05
        ).analyze("a real question", documents=DOCS)
        assert decision.intent == "rag_query"
