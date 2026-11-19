"""
src/services/question_analyzer.py
──────────────────────────────────
Question Analyzer — the intent router at the front of the chat pipeline.

Implements the "routing" workflow from Anthropic's Building Effective
Agents: classify the input, then hand it to a specialized path. One
small JSON-mode LLM call (same warm model, max_tokens≈150) decides:

    off_topic    — greetings / chit-chat / unrelated questions. The
                   router writes the 1-2 sentence reply itself in the
                   same call, so off-topic turns never touch retrieval.
    doc_summary  — "summarize report.pdf", "give me an overview" →
                   DocSummaryService.
    rag_query    — everything else → the standard RAG pipeline.

Fail-open by design: any parse error, unknown intent, or LLM failure
routes to rag_query. A broken router must never block a real question.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Literal

from src.core.logger import get_logger
from src.providers.llm.base import BaseLLMProvider, LLMConfig, LLMMessage

log = get_logger(__name__)

# The fixed reply shown when a question can't be answered from the documents
# — either it's off-topic, or retrieval found nothing relevant. It is never
# composed by an LLM (a small model answers the off-topic question itself
# even when told not to), so it's hard-coded to guarantee a clean decline.
FALLBACK_OFF_TOPIC_REPLY = (
    "I couldn't find anything about that in your documents. I can only answer "
    "questions about the documents in this workspace — try rephrasing, or ask "
    "for a summary."
)

GREETING_REPLY = (
    "Hello! Ask me anything about the documents in this workspace, or ask "
    "for a summary of one of them."
)

# Whole-message match for greetings / pleasantries / chitchat openers.
# Anchored so "hi, what does section 1 require?" does NOT match.
_GREETING_RE = re.compile(
    r"^\s*("
    # greetings + common typos (hlo, helo, heya, hiya, yo, sup, wassup)
    r"(hi+|hel+o+|helo+|hl+o|hey+|heya|hiya|yo+|sup|wass?up|hola|namaste|greetings?)(\s+there)?|"
    r"good\s+(morning|afternoon|evening|night)|"
    r"how\s+are\s+you|how'?s\s+it\s+going|what'?s\s+up|"
    r"thanks?|thank\s+you|thx|ty|"
    r"ok(ay)?|cool|nice|great|bye|goodbye|see\s+ya)"
    r"\s*[!.?😀-🙏]*\s*$",
    re.IGNORECASE,
)

# Summary-style requests — routed without an LLM call.
_SUMMARY_RE = re.compile(
    r"\b(summari[sz]e|summary|overview|tl;?dr|executive\s+summary|key\s+points|"
    r"what\s+is\s+(this|the)\s+(doc(ument)?|file|pdf)\s+about)\b",
    re.IGNORECASE,
)

_VALID_INTENTS = ("off_topic", "doc_summary", "rag_query")

_SYSTEM_PROMPT = """You are the request router for a private document Q&A assistant. Your ONLY job is to classify the user's message. You never answer it.

Classify into exactly one intent:
- "rag_query": a question that could plausibly be answered from the content of the user's uploaded documents.
- "doc_summary": asks for a summary, overview, executive summary, or "what is this document about" for one or more whole documents.
- "off_topic": greetings, small talk, or ANY question that is not about the user's documents — general knowledge, trivia, definitions, weather, news, math, coding help, opinions (e.g. "what is a boy", "what is the capital of India", "who won the world cup", "how are you").

When unsure between rag_query and off_topic, choose off_topic only for clearly general-knowledge questions; otherwise use rag_query.

Respond with ONLY this JSON object — no other text, no answer, no explanation:
{"intent": "rag_query" | "doc_summary" | "off_topic", "documents": ["<filename fragments the user referenced; empty array means all documents>"]}"""


@dataclass
class RouteDecision:
    """Outcome of the router call (always usable — fail-open)."""

    intent: Literal["off_topic", "doc_summary", "rag_query"]
    reply: str | None = None  # populated only for off_topic
    target_document_ids: list[str] = field(default_factory=list)  # [] = all
    raw: dict = field(default_factory=dict)


class QuestionAnalyzer:
    """
    Intent router: instant heuristics first, optional LLM for the rest.

    The heuristic tier (regex) handles the two cases with unambiguous
    surface forms — greetings/chitchat and summary requests — at zero
    cost. Everything else defaults to rag_query, unless `use_llm=True`
    (sensible only on GPU hosts: a 7B JSON classification measures
    ~90s on a CPU laptop, which is worse than just retrieving).
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        *,
        max_tokens: int = 150,
        use_llm: bool = False,
        llm_timeout_seconds: float = 10.0,
    ) -> None:
        self._llm = llm
        self._max_tokens = max_tokens
        self._use_llm = use_llm
        self._llm_timeout = llm_timeout_seconds

    async def analyze(
        self,
        query: str,
        history: list[tuple[str, str]] | None = None,
        documents: list[tuple[str, str]] | None = None,
    ) -> RouteDecision:
        """
        Args:
            query:     The user's message.
            history:   Recent (role, content) turns — last 2 are used so
                       follow-ups like "summarize that one" route correctly.
            documents: [(document_id, original_filename)] of COMPLETED docs
                       in the workspace; shown to the router and used to
                       resolve filename fragments back to IDs.
        """
        documents = documents or []

        # ── Tier 1: instant heuristics ────────────────────────────
        heuristic = self._heuristic_route(query, documents)
        if heuristic is not None:
            log.info(
                "router_heuristic", intent=heuristic.intent, query_preview=query[:80]
            )
            return heuristic

        # ── Tier 2: optional LLM classification ───────────────────
        if not self._use_llm:
            return RouteDecision(intent="rag_query")

        user_content = self._build_user_message(query, history or [], documents)

        try:
            response = await asyncio.wait_for(
                self._llm.generate(
                    [
                        LLMMessage(role="system", content=_SYSTEM_PROMPT),
                        LLMMessage(role="user", content=user_content),
                    ],
                    LLMConfig(
                        temperature=0.0,
                        max_tokens=self._max_tokens,
                        response_format="json",
                    ),
                ),
                timeout=self._llm_timeout,
            )
            raw = json.loads(response.content)
        except Exception as exc:
            # Fail-open: a broken/slow router must never block a real question.
            log.warning("router_failed_open", error=str(exc), query_preview=query[:80])
            return RouteDecision(intent="rag_query")

        intent = raw.get("intent") if isinstance(raw, dict) else None
        if intent not in _VALID_INTENTS:
            log.warning("router_unknown_intent", raw_intent=intent)
            return RouteDecision(
                intent="rag_query", raw=raw if isinstance(raw, dict) else {}
            )

        reply: str | None = None
        if intent == "off_topic":
            # Router classifies only — it does NOT write the reply. Always use
            # the fixed refusal so off-topic questions are reliably declined,
            # never answered (a small model answers them even when told not to).
            reply = FALLBACK_OFF_TOPIC_REPLY

        target_ids = _resolve_documents(raw.get("documents") or [], documents)

        log.info(
            "router_decision",
            intent=intent,
            targets=len(target_ids),
            query_preview=query[:80],
        )
        return RouteDecision(
            intent=intent,  # type: ignore[arg-type]
            reply=reply,
            target_document_ids=target_ids,
            raw=raw,
        )

    # ── Private ───────────────────────────────────────────────────

    @staticmethod
    def _heuristic_route(
        query: str,
        documents: list[tuple[str, str]],
    ) -> RouteDecision | None:
        """Zero-cost routing for unambiguous surface forms; None = no match."""
        stripped = query.strip()

        # Greetings / pleasantries — answer instantly, skip retrieval.
        if len(stripped.split()) <= 6 and _GREETING_RE.match(stripped):
            return RouteDecision(intent="off_topic", reply=GREETING_REPLY)

        # Summary requests — route to the doc-summary agent. Filename
        # mentions in the query narrow the target set ([] = all docs).
        if _SUMMARY_RE.search(stripped):
            lowered = stripped.lower()
            targets = [
                doc_id
                for doc_id, filename in documents
                # match on the stem so "summarize i-9instr" hits "i-9instr.pdf"
                if filename.lower() in lowered
                or filename.rsplit(".", 1)[0].lower() in lowered
            ]
            return RouteDecision(intent="doc_summary", target_document_ids=targets)

        return None

    @staticmethod
    def _build_user_message(
        query: str,
        history: list[tuple[str, str]],
        documents: list[tuple[str, str]],
    ) -> str:
        parts: list[str] = []
        if documents:
            doc_lines = "\n".join(
                f"  {i + 1}. {filename}"
                for i, (_, filename) in enumerate(documents[:15])
            )
            parts.append(f"Workspace documents:\n{doc_lines}")
        else:
            parts.append("Workspace documents: (none indexed yet)")

        recent = history[-2:]
        if recent:
            hist_lines = "\n".join(
                f"  {role}: {content[:300]}" for role, content in recent
            )
            parts.append(f"Recent conversation:\n{hist_lines}")

        parts.append(f"User message: {query}")
        return "\n\n".join(parts)


def _resolve_documents(
    fragments: list,
    documents: list[tuple[str, str]],
) -> list[str]:
    """
    Map the router's filename fragments back to document IDs via
    case-insensitive substring match. Zero matches → [] (= all docs),
    never an error.
    """
    resolved: list[str] = []
    for fragment in fragments:
        if not isinstance(fragment, str) or not fragment.strip():
            continue
        needle = fragment.strip().lower()
        for doc_id, filename in documents:
            if needle in filename.lower() and doc_id not in resolved:
                resolved.append(doc_id)
    return resolved
