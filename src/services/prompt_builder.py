"""
src/services/prompt_builder.py
────────────────────────────────
Constructs the full LLM prompt for RAG-based Q&A.

What changed in Phase 4
-----------------------
* **Real tokenizer-based budgeting.** All counts use tiktoken
  (cl100k_base) via `_count_tokens`; the legacy 4-chars-per-token
  heuristic over-counted by 30–50% on dense PDFs and under-counted on
  Markdown — both of which produced prompts that didn't fit the LLM's
  context window in practice.
* **Citation-enforcing system prompt.** The model is told (and shown
  examples) that every claim must be followed by a bracketed source
  number, and that "I cannot find this in your documents." is the
  required output for missing information. The frontend renders [n]
  markers as clickable chips.
* **Heading-aware context block.** Each chunk is prefixed with its
  `source > heading_path > p.N` so the model sees structural context.
  The grounding prompt directly references this.
* **Abstention branch.** When `abstain=True` the builder swaps to a
  short prompt that asks for the canonical "I cannot find this in
  your documents." answer plus, if any partial context exists, a
  one-line list of related topics found.
* **History compression hook.** When the recent history overflows
  `max_history_tokens` we drop the oldest turns instead of failing —
  full LLM-driven summarisation is wired through `Chat.history_summary`
  by the chat service so it doesn't re-summarise on every turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.providers.llm.base import LLMMessage
from src.vector_store.base import SearchResult


# ── Token budgets ─────────────────────────────────────────────────
# Sized for a 7B local LLM with 8K context. Any spare room goes to
# the model's response (LLM_MAX_TOKENS).
DEFAULT_CONTEXT_TOKENS = 4000
DEFAULT_HISTORY_TOKENS = 1000


# Required exact phrase for abstention. The frontend matches on it to
# render the "no answer found" UI state. Keep it stable; if you change
# the wording, update both the system prompt and the frontend.
NO_ANSWER_PHRASE = "I cannot find this in your documents."


SYSTEM_PROMPT_GROUNDED = f"""You are a private knowledge assistant. Answer the user's question using ONLY the information in the CONTEXT block below.

Citation format
  - Every factual claim MUST end with one or more bracketed source numbers, e.g. [1], [2][4].
  - The numbers refer to the bracketed sources in CONTEXT. Use them exactly as numbered.
  - If two sources support the same claim, list both: "Form I-9 must be completed within three business days of hire [1][3]."

Grounding rules
  - Do not use outside knowledge.
  - Do not invent facts, figures, dates, or quotes that are not present in CONTEXT.
  - If CONTEXT does not contain enough information to answer, reply EXACTLY: "{NO_ANSWER_PHRASE}" and stop.
  - Be concise. Prefer direct answers over restating the question.

Style
  - Write like a sharp, helpful colleague: professional, informative, human-friendly.
  - Lead with the answer. Use short paragraphs; use bullet points for enumerations.
  - No filler openers ("Based on the provided context...", "According to the documents...") — go straight to the substance.
"""


SYSTEM_PROMPT_ABSTAIN = f"""You are a private knowledge assistant. The retrieval system did NOT find content that answers the user's question.

Output rules
  - Reply EXACTLY: "{NO_ANSWER_PHRASE}"
  - If, and only if, the (weak) CONTEXT below contains topics that are clearly related to the user's question, you MAY follow the required reply with one short sentence beginning "Related topics in your documents include:" and list those topics. Do not answer the question itself.
  - Do not use outside knowledge.
  - Do not apologize, do not pad. The user wants a clean signal that the answer is not in their corpus.
"""


@dataclass
class BuiltPrompt:
    """The assembled prompt ready to send to the LLM."""

    messages: list[LLMMessage]
    context_chunks_used: int
    history_messages_used: int
    estimated_tokens: int
    abstained: bool = False
    # The results actually included in CONTEXT, in numbering order:
    # included_results[n-1] is the ground truth for citation [n].
    # Anything deduplicated or dropped for budget is NOT in this list.
    included_results: list[SearchResult] = field(default_factory=list)


class PromptBuilder:
    """
    Assembles the full RAG prompt from retrieved chunks and chat history.

    Usage:
        builder = PromptBuilder()
        prompt = builder.build(
            query="What is the refund policy?",
            search_results=results,
            history=recent_messages,
            abstain=False,
        )
        response = await llm.generate(prompt.messages)
    """

    def __init__(
        self,
        max_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        max_history_tokens: int = DEFAULT_HISTORY_TOKENS,
    ) -> None:
        self._max_context_tokens = max_context_tokens
        self._max_history_tokens = max_history_tokens

    def build(
        self,
        query: str,
        search_results: list[SearchResult],
        history: list[LLMMessage] | None = None,
        abstain: bool = False,
    ) -> BuiltPrompt:
        """
        Build the complete prompt for a RAG query.

        Args:
            query:           The user's question (preferably the standalone form).
            search_results:  Reranked chunks (best first).
            history:         Prior chat messages (oldest first).
            abstain:         If True, build the no-answer prompt instead.
        """
        context_text, included = self._build_context(search_results)
        history_messages, history_used = self._trim_history(history or [])

        system_text = SYSTEM_PROMPT_ABSTAIN if abstain else SYSTEM_PROMPT_GROUNDED
        system_content = f"{system_text}\n\n---\nCONTEXT:\n{context_text}\n---"

        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=system_content),
            *history_messages,
            LLMMessage(role="user", content=query),
        ]

        estimated_tokens = _count_tokens(
            system_content + query + "".join(m.content for m in history_messages)
        )

        return BuiltPrompt(
            messages=messages,
            context_chunks_used=len(included),
            history_messages_used=history_used,
            estimated_tokens=estimated_tokens,
            abstained=abstain,
            included_results=included,
        )

    def fits(self, results: list[SearchResult]) -> bool:
        """True when every distinct chunk in `results` fits the context budget."""
        _, included = self._build_context(results)
        distinct = {r.payload.text.strip() for r in results} - {""}
        return len(included) == len(distinct)

    # ── Private ───────────────────────────────────────────────────

    def _build_context(
        self, results: list[SearchResult]
    ) -> tuple[str, list[SearchResult]]:
        """
        Numbered context block, ordered by reranker score (best first).

        Each chunk is rendered as:

            [1] Source: foo.pdf — p.3 — Section 2 > 2.1 Acceptable Documents
            <chunk text>

        Numbering is contiguous over *included* chunks only — duplicates
        and budget-dropped chunks never consume a number. This keeps
        `[n]` ↔ `included_results[n-1]` a perfect bijection, which the
        answer validator and the sources SSE event rely on.
        """
        if not results:
            return "(no documents matched the query)", []

        seen_texts: set[str] = set()
        parts: list[str] = []
        included: list[SearchResult] = []
        token_budget = self._max_context_tokens

        for result in results:
            text = result.payload.text.strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)

            heading = (
                " > ".join(result.payload.heading_path)
                if result.payload.heading_path
                else None
            )
            header_bits = [
                f"Source: {result.payload.source}",
                f"p.{result.payload.page_number}"
                if result.payload.page_number
                else None,
                heading,
            ]
            header = " — ".join(b for b in header_bits if b)
            chunk_text = f"[{len(included) + 1}] {header}\n{text}"

            chunk_tokens = _count_tokens(chunk_text)
            if chunk_tokens > token_budget:
                break

            parts.append(chunk_text)
            included.append(result)
            token_budget -= chunk_tokens

        if not parts:
            return "(no documents matched the query)", []
        return "\n\n".join(parts), included

    def _trim_history(self, history: list[LLMMessage]) -> tuple[list[LLMMessage], int]:
        """
        Newest-first walk that keeps as many turns as fit in the
        history token budget. Older turns are dropped — the chat
        service is responsible for compressing them into a rolling
        summary stored on `Chat.history_summary` so they're not lost.
        """
        if not history:
            return [], 0

        token_budget = self._max_history_tokens
        selected: list[LLMMessage] = []
        for msg in reversed(history):
            tokens = _count_tokens(msg.content)
            if tokens > token_budget:
                break
            selected.insert(0, msg)
            token_budget -= tokens
        return selected, len(selected)


# ── Token counting ────────────────────────────────────────────────


try:
    import tiktoken as _tiktoken

    _enc = _tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        """Public tokenizer-based count (cl100k_base). Shared with DocSummaryService."""
        return max(1, len(_enc.encode(text)))
except Exception:  # pragma: no cover — tiktoken is in requirements.txt

    def count_tokens(text: str) -> int:  # type: ignore[misc]
        return max(1, len(text) // 4)


# Backward-compat alias — existing tests and internal callers use the
# underscore name.
_count_tokens = count_tokens
