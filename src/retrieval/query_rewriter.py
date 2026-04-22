"""
src/retrieval/query_rewriter.py
────────────────────────────────
Query rewriting: transforms a user's raw question into one or more
optimised retrieval queries before hitting the vector store.

Why this matters:
    Users ask conversational questions ("what about the refund thing?")
    that embed poorly because they contain pronouns, anaphora, and vague
    references.  A rewritten query ("What is the refund and return policy?")
    produces a better embedding and retrieves more relevant chunks.

Two strategies, selectable via RewriteStrategy:
    HYD (Hypothetical Document Embedding):
        Ask the LLM to write a SHORT hypothetical passage that would
        ANSWER the question, then embed THAT instead of the question.
        Works because the hypothetical answer lives in the same vector
        space as real document chunks.

    MULTI_QUERY:
        Generate N alternative phrasings of the same question.
        Run separate searches for each, then union + deduplicate results.
        Increases recall at the cost of latency.

    NONE:
        Bypass rewriting entirely.  Used in eval pipeline and for
        short, precise queries that don't need expansion.

Design:
    - QueryRewriter is injected into RetrievalPipeline.
    - It calls the LLM directly (not the full RAG pipeline — no context).
    - If the LLM call fails, it falls back to the original query
      so the pipeline never hard-fails due to rewriting.
    - Token cost is kept low: prompts are minimal and max_tokens is capped.
"""

from __future__ import annotations

import re
from enum import Enum

from src.core.logger import get_logger
from src.providers.llm.base import BaseLLMProvider, LLMConfig, LLMMessage

log = get_logger(__name__)


class RewriteStrategy(str, Enum):
    NONE = "none"
    HYD = "hyd"           # Hypothetical Document Embedding
    MULTI_QUERY = "multi"  # Multiple query variants


# ── Prompts ───────────────────────────────────────────────────────

_HYD_PROMPT = """\
Write a SHORT passage (2-4 sentences) that would directly answer the \
following question. Write ONLY the passage — no preamble, no labels.

Question: {query}"""

_MULTI_QUERY_PROMPT = """\
Generate {n} different search queries that all ask for the same \
information as the original question below. Each query should use \
different vocabulary and phrasing. Write ONLY the queries, one per line, \
with no numbering, labels, or preamble.

Original question: {query}"""

_STANDALONE_PROMPT = """\
Rewrite the following question as a standalone search query that makes \
full sense without any prior conversation context. Replace all pronouns \
and vague references with explicit terms. Output ONLY the rewritten \
query — nothing else.

Conversation context:
{history_summary}

Question: {query}"""


class QueryRewriter:
    """
    Rewrites a user query into one or more retrieval-optimised queries.

    Usage:
        rewriter = QueryRewriter(llm=llm_provider)
        queries = await rewriter.rewrite(
            query="what was that policy again?",
            strategy=RewriteStrategy.HYD,
            history_summary="User asked about refund policies.",
        )
        # queries: ["A company's refund policy states that customers may..."]
    """

    # Caps for safety — we don't want rewriting to cost more than the answer
    _REWRITE_CONFIG = LLMConfig(temperature=0.2, max_tokens=300)
    _MULTI_QUERY_N = 3

    def __init__(self, llm: BaseLLMProvider) -> None:
        self._llm = llm

    async def rewrite(
        self,
        query: str,
        strategy: RewriteStrategy = RewriteStrategy.HYD,
        history_summary: str | None = None,
    ) -> list[str]:
        """
        Rewrite query using the chosen strategy.

        Returns:
            List of query strings to use for vector search.
            Always contains at least the original query as fallback.
        """
        if strategy == RewriteStrategy.NONE:
            return [query]

        # Step 1: resolve anaphora if there's conversation context
        resolved = query
        if history_summary:
            resolved = await self._resolve_anaphora(query, history_summary)

        # Step 2: apply chosen strategy
        try:
            if strategy == RewriteStrategy.HYD:
                return await self._hyd(resolved)
            elif strategy == RewriteStrategy.MULTI_QUERY:
                return await self._multi_query(resolved)
        except Exception as exc:
            log.warning("query_rewrite_failed", strategy=strategy, error=str(exc))

        # Fallback: use resolved (or original) query
        return [resolved]

    # ── Strategies ────────────────────────────────────────────────

    async def _hyd(self, query: str) -> list[str]:
        """
        Hypothetical Document Embedding.
        Returns [hypothetical_passage] — embed this instead of the query.
        """
        prompt = _HYD_PROMPT.format(query=query)
        response = await self._llm.generate(
            [LLMMessage(role="user", content=prompt)],
            self._REWRITE_CONFIG,
        )
        hypothesis = response.content.strip()
        if not hypothesis or len(hypothesis) < 20:
            log.debug("hyd_returned_short_response", hypothesis=hypothesis)
            return [query]

        log.debug("hyd_rewrite", original=query[:60], hypothesis=hypothesis[:80])
        return [hypothesis]

    async def _multi_query(self, query: str) -> list[str]:
        """
        Generate N alternative query phrasings.
        Returns [original, variant_1, variant_2, ...].
        Deduplicates by lowercased text.
        """
        prompt = _MULTI_QUERY_PROMPT.format(
            query=query, n=self._MULTI_QUERY_N
        )
        response = await self._llm.generate(
            [LLMMessage(role="user", content=prompt)],
            self._REWRITE_CONFIG,
        )
        raw_lines = response.content.strip().splitlines()
        variants = [
            _clean_query_line(line)
            for line in raw_lines
            if _clean_query_line(line)
        ]

        # Always include original; deduplicate
        all_queries: list[str] = [query]
        seen = {query.lower()}
        for v in variants[: self._MULTI_QUERY_N]:
            if v.lower() not in seen:
                all_queries.append(v)
                seen.add(v.lower())

        log.debug("multi_query_rewrite", original=query[:60], variants=len(all_queries))
        return all_queries

    async def _resolve_anaphora(self, query: str, history_summary: str) -> str:
        """
        Resolve pronouns/references using conversation history.
        Returns a standalone version of the query.
        """
        prompt = _STANDALONE_PROMPT.format(
            query=query, history_summary=history_summary[:400]
        )
        try:
            response = await self._llm.generate(
                [LLMMessage(role="user", content=prompt)],
                LLMConfig(temperature=0.0, max_tokens=200),
            )
            resolved = response.content.strip()
            if resolved:
                return resolved
        except Exception:
            pass
        return query


def _clean_query_line(line: str) -> str:
    """Strip numbering, bullets, and whitespace from a query line."""
    line = line.strip()
    # Remove leading numbers/bullets: "1. ", "- ", "* ", "1) "
    line = re.sub(r"^[\d\-\*\•]+[\.\)]\s*", "", line)
    line = re.sub(r"^[\-\*\•]\s*", "", line)
    return line.strip()
