"""
tests/unit/test_chat_orchestrator.py
─────────────────────────────────────
Orchestrator routing tests — analyzer / RAG / summary all mocked.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.user import User, UserRole
from src.retrieval.query_understanding import QueryEnrichment
from src.services.chat_orchestrator import ChatOrchestrator
from src.services.doc_summary_service import SummaryOutcome
from src.services.prompt_builder import BuiltPrompt
from src.services.question_analyzer import RouteDecision
from src.services.rag_service import _RetrievalOutcome
from src.vector_store.base import SearchResult, VectorPayload

pytestmark = pytest.mark.unit

DOCS = [("doc-1", "report.pdf"), ("doc-2", "notes.md")]


def _make_user() -> User:
    user = User()
    user.id = uuid.uuid4()
    user.email = "test@example.com"
    user.role = UserRole.MEMBER
    user.is_active = True
    return user


def _result(text: str = "chunk text") -> SearchResult:
    return SearchResult(
        chunk_id=str(uuid.uuid4()),
        score=0.9,
        payload=VectorPayload(
            user_id="u",
            workspace_id="w",
            document_id="doc-1",
            chunk_index=0,
            text=text,
            source="report.pdf",
            page_number=1,
        ),
    )


async def _tokens():
    yield "Hello "
    yield "world."


def _mock_rag(abstained: bool = False) -> MagicMock:
    rag = MagicMock()
    included = [_result()]
    outcome = _RetrievalOutcome(
        results=included,
        confidence=0.9,
        enrichment=QueryEnrichment(
            original="q",
            standalone="q",
            queries=["q"],
            rewritten=False,
            used_hyde=False,
        ),
        prompt=BuiltPrompt(
            messages=[],
            context_chunks_used=1,
            history_messages_used=0,
            estimated_tokens=10,
            included_results=included,
        ),
    )
    rag.prepare_stream = AsyncMock(return_value=(outcome, _tokens()))
    rag._is_abstaining = MagicMock(return_value=abstained)
    return rag


def _mock_summary() -> MagicMock:
    svc = MagicMock()
    svc.prepare_summary_stream = AsyncMock(
        return_value=(SummaryOutcome(documents=DOCS), _tokens())
    )
    return svc


def _analyzer(decision: RouteDecision) -> AsyncMock:
    analyzer = AsyncMock()
    analyzer.analyze.return_value = decision
    return analyzer


async def _prepare(orch, **kwargs):
    events: list[dict] = []
    plan = await orch.prepare(
        query=kwargs.get("query", "question"),
        workspace_id=uuid.uuid4(),
        user=_make_user(),
        history_messages=None,
        document_ids=kwargs.get("document_ids"),
        workspace_documents=DOCS,
        emit=events.append,
    )
    return plan, events


class TestRouting:
    @pytest.mark.asyncio
    async def test_router_disabled_goes_straight_to_rag(self):
        analyzer = AsyncMock()
        rag = _mock_rag()
        orch = ChatOrchestrator(analyzer, rag, _mock_summary(), enable_router=False)
        plan, events = await _prepare(orch)

        analyzer.analyze.assert_not_called()
        assert plan.intent == "rag_query"
        assert plan.allow_citations
        # No "analyzing" stage when the router is off.
        stages = [e["stage"] for e in events if e.get("type") == "thinking"]
        assert "analyzing" not in stages

    @pytest.mark.asyncio
    async def test_off_topic_short_circuits_rag(self):
        rag = _mock_rag()
        orch = ChatOrchestrator(
            _analyzer(
                RouteDecision(intent="off_topic", reply="Hi! Ask about your docs.")
            ),
            rag,
            _mock_summary(),
            enable_router=True,
        )
        plan, _ = await _prepare(orch, query="hello!")

        rag.prepare_stream.assert_not_called()
        assert plan.intent == "off_topic"
        assert not plan.allow_citations
        text = "".join([t async for t in plan.token_iter])
        assert text == "Hi! Ask about your docs."

    @pytest.mark.asyncio
    async def test_doc_summary_calls_summary_service_with_targets(self):
        summary = _mock_summary()
        orch = ChatOrchestrator(
            _analyzer(
                RouteDecision(intent="doc_summary", target_document_ids=["doc-2"])
            ),
            _mock_rag(),
            summary,
            enable_router=True,
        )
        plan, _ = await _prepare(orch)

        assert plan.intent == "doc_summary"
        call_kwargs = summary.prepare_summary_stream.call_args.kwargs
        assert call_kwargs["documents"] == [("doc-2", "notes.md")]

    @pytest.mark.asyncio
    async def test_doc_summary_defaults_to_all_documents(self):
        summary = _mock_summary()
        orch = ChatOrchestrator(
            _analyzer(RouteDecision(intent="doc_summary")),
            _mock_rag(),
            summary,
            enable_router=True,
        )
        await _prepare(orch)
        call_kwargs = summary.prepare_summary_stream.call_args.kwargs
        assert call_kwargs["documents"] == DOCS

    @pytest.mark.asyncio
    async def test_rag_route_sources_are_context_ordered(self):
        rag = _mock_rag()
        orch = ChatOrchestrator(
            _analyzer(RouteDecision(intent="rag_query")),
            rag,
            _mock_summary(),
            enable_router=True,
        )
        plan, _ = await _prepare(orch)

        assert plan.intent == "rag_query"
        assert len(plan.sources) == 1
        assert plan.sources[0].source_filename == "report.pdf"
        assert plan.confidence == 0.9

    @pytest.mark.asyncio
    async def test_rag_abstention_declines_as_off_topic(self):
        # The retrieval score is the "is this about the documents?" classifier:
        # when RAG finds nothing relevant (abstains), the orchestrator declines
        # with the fixed reply instead of generating a "can't find" answer.
        from src.services.question_analyzer import FALLBACK_OFF_TOPIC_REPLY

        rag = _mock_rag(abstained=True)
        orch = ChatOrchestrator(
            _analyzer(RouteDecision(intent="rag_query")),
            rag,
            _mock_summary(),
            enable_router=True,
        )
        plan, events = await _prepare(orch, query="what is the capital of france?")

        assert plan.intent == "off_topic"
        assert not plan.allow_citations
        text = "".join([t async for t in plan.token_iter])
        assert text == FALLBACK_OFF_TOPIC_REPLY
        # The route event is corrected to off_topic for the UI.
        intents = [e["intent"] for e in events if e.get("type") == "route"]
        assert "off_topic" in intents


class TestEmittedEvents:
    @pytest.mark.asyncio
    async def test_stage_sequence_for_rag_route(self):
        rag = _mock_rag()

        # Make prepare_stream invoke its on_stage callback like the real one.
        async def _prepare_stream(*args, on_stage=None, **kwargs):
            if on_stage:
                on_stage("retrieving")
                on_stage("reranking")
            return rag._outcome, _tokens()

        rag._outcome = rag.prepare_stream.return_value[0]
        rag.prepare_stream = AsyncMock(side_effect=_prepare_stream)

        orch = ChatOrchestrator(
            _analyzer(RouteDecision(intent="rag_query")),
            rag,
            _mock_summary(),
            enable_router=True,
        )
        _, events = await _prepare(orch)

        stages = [e["stage"] for e in events if e.get("type") == "thinking"]
        assert stages == ["analyzing", "routing", "retrieving", "reranking"]
        intents = [e["intent"] for e in events if e.get("type") == "route"]
        assert intents == ["rag_query"]
