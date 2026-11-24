"""
src/services/chat_orchestrator.py
──────────────────────────────────
Chat orchestrator — composes the agentic pipeline behind the SSE endpoint.

    Question Analyzer (router, 1 small JSON call)
        ├─ off_topic   → router's own reply, streamed in slices
        ├─ doc_summary → DocSummaryService (map-reduce + exec summary, or a
        │                one-call overview when full summaries are off)
        └─ rag_query   → RAGService (retrieve → rerank → grounded stream)

The orchestrator's job is the "prepare" half: classify, route, and hand
the SSE generator a `StreamPlan` carrying the token stream plus
everything needed after it ends (sources, confidence, abstention,
whether [n] citations are legitimate). Validation of the finished
answer happens in the endpoint via `answer_validator.validate_answer`
— programmatic, not an LLM pass.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

from src.core.logger import get_logger
from src.models.user import User
from src.services.doc_summary_service import DocSummaryService
from src.services.question_analyzer import (
    QuestionAnalyzer,
    RouteDecision,
    decline_reply,
)
from src.services.rag_service import CitedSource, RAGService, _results_to_cited

log = get_logger(__name__)

# Slice width for pseudo-streaming the router's off-topic reply. The
# reply already exists in full; slicing preserves the streaming feel
# in the UI without a second LLM call.
_OFF_TOPIC_SLICE = 40


@dataclass
class StreamPlan:
    """Everything the SSE generator needs to run and finish a turn."""

    intent: str  # rag_query | doc_summary | off_topic
    token_iter: AsyncIterator[str]
    sources: list[CitedSource] = field(default_factory=list)  # context-ordered
    confidence: float = 1.0
    abstained: bool = False
    allow_citations: bool = False


class ChatOrchestrator:
    """Stateless per-request composition of analyzer + workers."""

    def __init__(
        self,
        analyzer: QuestionAnalyzer,
        rag_service: RAGService,
        summary_service: DocSummaryService,
        *,
        enable_router: bool = True,
        enable_summaries: bool = True,
    ) -> None:
        self._analyzer = analyzer
        self._rag = rag_service
        self._summary = summary_service
        self._enable_router = enable_router
        self._enable_summaries = enable_summaries

    async def prepare(
        self,
        *,
        query: str,
        workspace_id: uuid.UUID,
        user: User,
        history_messages: list | None,
        document_ids: list[str] | None,
        workspace_documents: list[tuple[str, str]],
        emit: Callable[[dict], None],
    ) -> StreamPlan:
        """
        Classify the query, run the chosen route up to the point where
        tokens start flowing, and return the plan.

        `emit` enqueues SSE event dicts (thinking/route/warning) onto the
        endpoint's queue — safe to call from this coroutine because the
        producer task runs on the same event loop.
        """
        # ── Route ─────────────────────────────────────────────────
        if self._enable_router:
            emit({"type": "thinking", "stage": "analyzing"})
            history = _history_tuples(history_messages)
            decision = await self._analyzer.analyze(
                query, history=history, documents=workspace_documents
            )
        else:
            decision = RouteDecision(intent="rag_query")

        emit({"type": "route", "intent": decision.intent})
        emit({"type": "thinking", "stage": "routing"})

        # ── Off-topic: the router already wrote the reply ─────────
        if decision.intent == "off_topic":
            reply = decision.reply or ""

            async def _slices() -> AsyncIterator[str]:
                for i in range(0, len(reply), _OFF_TOPIC_SLICE):
                    yield reply[i : i + _OFF_TOPIC_SLICE]

            return StreamPlan(
                intent="off_topic",
                token_iter=_slices(),
                allow_citations=False,
            )

        # ── Document summary ──────────────────────────────────────
        if decision.intent == "doc_summary":
            emit({"type": "thinking", "stage": "summarizing"})
            targets = _resolve_targets(decision, document_ids, workspace_documents)
            if not self._enable_summaries:
                # A full summary costs several LLM calls. Searching for the
                # request's words instead matches no passage and was always
                # declined, so give a one-call overview of the documents'
                # headings and opening chunks; it opens with a note saying so.
                outcome, token_iter = await self._summary.prepare_overview_stream(
                    workspace_id=workspace_id,
                    user=user,
                    documents=targets,
                    query=query,
                )
                return StreamPlan(
                    intent="doc_summary",
                    token_iter=token_iter,
                    sources=outcome.sources,
                    allow_citations=False,
                )
            outcome, token_iter = await self._summary.prepare_summary_stream(
                workspace_id=workspace_id,
                user=user,
                documents=targets,
                query=query,
            )
            if outcome.truncated:
                emit(
                    {
                        "type": "warning",
                        "code": "summary_truncated",
                        "message": (
                            "The document set is very large, so the summary "
                            "covers only the first part of it."
                        ),
                        "confidence": 1.0,
                    }
                )
            return StreamPlan(
                intent="doc_summary",
                token_iter=token_iter,
                sources=outcome.sources,
                allow_citations=False,
            )

        # ── RAG (default) ─────────────────────────────────────────
        outcome, token_iter = await self._rag.prepare_stream(
            query=query,
            workspace_id=workspace_id,
            user=user,
            history_messages=history_messages,
            document_ids=document_ids,
            on_stage=lambda s: emit({"type": "thinking", "stage": s}),
        )

        # The answer gate (RAGService._gate): a question that is clearly
        # unrelated to the workspace documents, low on both the reranker and
        # the dense cosine, is declined with a fixed reply and no LLM call.
        # Everything else is answered from the passages, and the grounded
        # prompt lets the model say the documents don't cover it. Tune via
        # ABSTENTION_RERANKER_THRESHOLD and ABSTENTION_COSINE_THRESHOLD.
        if self._rag._is_abstaining(outcome):
            reply = decline_reply(summaries_enabled=self._enable_summaries)

            async def _decline() -> AsyncIterator[str]:
                for i in range(0, len(reply), _OFF_TOPIC_SLICE):
                    yield reply[i : i + _OFF_TOPIC_SLICE]

            emit({"type": "route", "intent": "off_topic"})
            return StreamPlan(
                intent="off_topic",
                token_iter=_decline(),
                allow_citations=False,
            )

        included = (
            outcome.prompt.included_results if outcome.prompt else outcome.results
        )
        return StreamPlan(
            intent="rag_query",
            token_iter=token_iter,
            sources=_results_to_cited(included),
            confidence=outcome.confidence,
            abstained=False,
            allow_citations=True,
        )


# ── Pure helpers ──────────────────────────────────────────────────


def _history_tuples(history_messages: list | None) -> list[tuple[str, str]]:
    """Last 2 turns as (role, content) for the router prompt."""
    out: list[tuple[str, str]] = []
    if history_messages:
        for msg in history_messages[-2:]:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", None)
            if role and content:
                out.append((str(role), str(content)))
    return out


def _resolve_targets(
    decision: RouteDecision,
    request_document_ids: list[str] | None,
    workspace_documents: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """
    Pick the documents to summarize. Priority:
    1. Explicit request `document_ids` intersected with router targets
       (request param wins when both exist).
    2. Router-resolved targets.
    3. All completed workspace documents.
    """
    by_id = dict(workspace_documents)

    if request_document_ids:
        ids = [d for d in request_document_ids if d in by_id]
        if decision.target_document_ids:
            narrowed = [d for d in ids if d in decision.target_document_ids]
            ids = narrowed or ids
        if ids:
            return [(d, by_id[d]) for d in ids]

    if decision.target_document_ids:
        ids = [d for d in decision.target_document_ids if d in by_id]
        if ids:
            return [(d, by_id[d]) for d in ids]

    return list(workspace_documents)
