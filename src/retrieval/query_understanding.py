"""
src/retrieval/query_understanding.py
─────────────────────────────────────
Lightweight query understanding for the Phase 3 retrieval pipeline.

What this module does (and what it deliberately does NOT do)
-----------------------------------------------------------
Two responsibilities, both cheap:

1. **Standalone-question rewriting.** When the user asks a follow-up
   like "When?" or "And what about Section 2?", embedding the literal
   question retrieves nothing useful. We rewrite it into a self-
   contained query using the most recent few turns of chat history.
   This is *only* invoked when both:
     - the chat already has prior turns, AND
     - the query contains anaphoric markers (pronouns, "this/that",
       "the previous", "above", etc.)
   The regex gate avoids burning a free LLM call on every first turn.

2. **Optional HyDE.** Generates a hypothetical answer and embeds *that*
   instead of the question. Recall booster, but on a 7B local model it
   adds 2–3 s of latency per query for a modest gain — so it's gated
   behind `enable_hyde` and off by default. The cross-encoder reranker
   in Phase 3.3 picks up most of what HyDE used to fix.

Out of scope (intentionally removed from the legacy `QueryRewriter`)
-----------------------------------------------------------------
* Multi-query expansion: nice in theory, but parallel embedding +
  fusion already gives us most of its benefit, and on a local model
  it doubles latency without doubling recall.
* Combining rewriting modes: each strategy was a fork in the legacy
  code; here `enrich_query` is a single, predictable function.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.logger import get_logger
from src.providers.llm.base import BaseLLMProvider, LLMConfig, LLMMessage

log = get_logger(__name__)


# Anaphora regex — if a query matches none of these we skip the
# standalone-rewrite LLM call entirely. Keep this conservative; missing
# a rewrite costs us recall, but rewriting unnecessarily costs latency
# and an LLM round-trip.
_ANAPHORA_PATTERN = re.compile(
    r"\b("
    r"it|its|itself|"
    r"they|them|their|theirs|themselves|"
    r"this|that|these|those|"
    r"he|him|his|himself|she|her|hers|herself|"
    r"the (?:previous|above|former|latter|first|second|third|prior|aforementioned)|"
    r"that one|the same|same as|earlier|before"
    r")\b",
    flags=re.IGNORECASE,
)


_STANDALONE_PROMPT = """\
You will be given a short conversation between a user and an assistant, followed by the user's latest follow-up question. Rewrite the follow-up so it is self-contained: replace every pronoun and indirect reference ("this", "that", "the previous one", "it", etc.) with the explicit subject from the conversation.

Output rules:
  * Output ONE rewritten question.
  * No preamble, no explanation, no quotes.
  * If the question is already self-contained, return it unchanged.

Conversation so far:
{history}

Follow-up question:
{query}

Rewritten question:"""


_HYDE_PROMPT = """\
Write a SHORT passage (2-4 sentences) that would directly answer the following question. Write ONLY the passage — no preamble, no labels, no quotes.

Question: {query}"""


@dataclass
class QueryEnrichment:
    """
    The output of `QueryUnderstanding.enrich_query`. Exactly one item
    in `queries` is the embedding-search input; the original query is
    *always* preserved in `original` so the BM25 / lexical branch can
    use the user's actual wording.
    """

    original: str
    standalone: str  # Same as `original` if no rewrite happened
    queries: list[str]  # 1+ queries to embed (HyDE adds a synthetic doc)
    rewritten: bool
    used_hyde: bool


class QueryUnderstanding:
    """
    Stateless. Holds an LLM provider so it can run rewrites lazily.
    """

    # Both prompts are tightly bounded — rewriting must never cost more
    # than the actual answer call.
    _REWRITE_CONFIG = LLMConfig(temperature=0.0, max_tokens=200)
    _HYDE_CONFIG = LLMConfig(temperature=0.2, max_tokens=300)

    def __init__(self, llm: BaseLLMProvider, *, enable_hyde: bool = False) -> None:
        self._llm = llm
        self._enable_hyde = enable_hyde

    async def enrich_query(
        self,
        query: str,
        history: list[tuple[str, str]] | None = None,
    ) -> QueryEnrichment:
        """
        Args:
            query:    The raw user question.
            history:  Optional `[(role, content), ...]` of recent turns.
                      Pass the most recent ~4 turns; older context is
                      compressed elsewhere via the prompt builder.

        Returns:
            `QueryEnrichment` with at least the original query in
            `queries`. Failure of either rewrite is non-fatal — we log
            and fall back to the original query.
        """
        # Stage 1: standalone-question rewrite
        standalone = query
        rewritten = False
        if history and self._needs_rewrite(query):
            standalone = await self._rewrite_standalone(query, history)
            rewritten = standalone != query
            if rewritten:
                log.info(
                    "query_rewritten_standalone",
                    original=query[:80],
                    standalone=standalone[:80],
                )

        # Stage 2: optional HyDE (synthetic answer for embedding)
        queries: list[str] = [standalone]
        used_hyde = False
        if self._enable_hyde:
            hyde = await self._hyde(standalone)
            if hyde:
                queries.append(hyde)
                used_hyde = True

        return QueryEnrichment(
            original=query,
            standalone=standalone,
            queries=queries,
            rewritten=rewritten,
            used_hyde=used_hyde,
        )

    # ── Internals ─────────────────────────────────────────────────

    @staticmethod
    def _needs_rewrite(query: str) -> bool:
        """Quick anaphora gate — see module docstring."""
        if len(query) < 4:
            # "When?", "Why?" — definitely needs context
            return True
        return bool(_ANAPHORA_PATTERN.search(query))

    async def _rewrite_standalone(
        self,
        query: str,
        history: list[tuple[str, str]],
    ) -> str:
        # Take the last 4 turns max — anything further back is unlikely
        # to bear on the immediate follow-up and just bloats the prompt.
        recent = history[-4:]
        rendered = "\n".join(f"{role.upper()}: {content}" for role, content in recent)
        prompt = _STANDALONE_PROMPT.format(history=rendered, query=query)
        try:
            response = await self._llm.generate(
                [LLMMessage(role="user", content=prompt)],
                self._REWRITE_CONFIG,
            )
        except Exception as exc:
            log.warning("standalone_rewrite_failed", error=str(exc))
            return query

        # Defensive cleanup: the model occasionally returns a quote-
        # wrapped or numbered version. Strip both.
        text = response.content.strip()
        text = text.lstrip("\"'`-*").rstrip("\"'`").strip()
        text = re.sub(r"^\d+[.\)]\s*", "", text).strip()
        if not text or len(text) < 3:
            return query
        return text

    async def _hyde(self, query: str) -> str | None:
        prompt = _HYDE_PROMPT.format(query=query)
        try:
            response = await self._llm.generate(
                [LLMMessage(role="user", content=prompt)],
                self._HYDE_CONFIG,
            )
        except Exception as exc:
            log.warning("hyde_failed", error=str(exc))
            return None
        text = response.content.strip()
        if not text or len(text) < 20:
            return None
        return text
