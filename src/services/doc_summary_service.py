"""
src/services/doc_summary_service.py
────────────────────────────────────
Document summarization agent (map-reduce + streamed executive summary).

Route: the Question Analyzer classifies "summarize report.pdf" as
`doc_summary` and the orchestrator calls `prepare_summary_stream()`.

Algorithm
---------
1. Fetch ALL chunks for each target document straight from Qdrant by
   document_id (no vector search — summaries need the whole document,
   not the most-similar slices).
2. If the combined text fits `summary_context_token_budget`, skip the map
   phase entirely (single streamed LLM pass) — the common case for most
   real documents, and far faster on a CPU model than N map calls.
3. Otherwise MAP: pack consecutive chunks into `summary_map_token_budget`
   segments (never splitting a chunk, never mixing documents) and
   bullet-point each segment with a streamed LLM call, so only the time
   to the first token has to fit the timeout. At most
   `summary_map_concurrency` calls run at once (default 1, because a
   local CPU Ollama serializes requests anyway).
4. REDUCE: join the bullets; if still over the context budget, one more
   map round (2 levels max, then hard-truncate — logged).
5. FINAL (the "Executive Summary agent"): one streamed LLM call that
   turns the condensed material into a professional, human-friendly
   executive summary. The user's original phrasing steers the focus
   ("summarize the methodology" ≠ "summarize everything").
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from src.configs.config import get_settings
from src.core.exceptions import LLMProviderError
from src.core.logger import get_logger
from src.models.user import User
from src.providers.llm.base import BaseLLMProvider, LLMConfig, LLMMessage
from src.services.prompt_builder import count_tokens
from src.services.rag_service import CitedSource
from src.vector_store.base import BaseVectorStore, SearchResult

log = get_logger(__name__)
settings = get_settings()


_MAP_SYSTEM_PROMPT = (
    "You summarize one section of a document into 3-6 dense bullet points. "
    "Keep concrete facts, figures, names, and dates. No preamble, no closing "
    "remarks — bullets only."
)

_FINAL_SYSTEM_PROMPT = """You are writing an executive summary of the user's document(s) for a busy professional.

Structure:
- One short overview paragraph (2-3 sentences) capturing what the document is and its main thrust.
- "Key Points" — the essential facts as bullets, keeping concrete figures, names, and dates.
- "Notable Details" — only if there are specifics worth flagging.

Style: professional, informative, human-friendly. No filler, no apologies.
Do NOT use bracketed citation markers like [1].
If the user asked about a specific aspect, focus the summary on that aspect."""


@dataclass
class SummaryOutcome:
    """Metadata about a summary run, returned alongside the token stream."""

    documents: list[tuple[str, str]]  # (doc_id, filename) summarized
    total_chunks: int = 0
    map_calls: int = 0  # 0 = single-pass
    truncated: bool = False  # hit summary_max_total_chunks
    sources: list[CitedSource] = field(default_factory=list)  # one per document


class DocSummaryService:
    """Map-reduce summarizer over a document's Qdrant chunks."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        llm: BaseLLMProvider,
        *,
        context_budget: int | None = None,
        map_token_budget: int | None = None,
        section_max_tokens: int | None = None,
        final_max_tokens: int | None = None,
        max_total_chunks: int | None = None,
        timeout_seconds: float | None = None,
        map_concurrency: int | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._llm = llm
        # Single-pass threshold: material at or under this is summarized in
        # one streamed call (no map phase). Larger docs map-reduce to here.
        self._context_budget = context_budget or settings.summary_context_token_budget
        self._map_budget = map_token_budget or settings.summary_map_token_budget
        self._section_max_tokens = (
            section_max_tokens or settings.summary_section_max_tokens
        )
        self._final_max_tokens = final_max_tokens or settings.summary_final_max_tokens
        self._max_total_chunks = max_total_chunks or settings.summary_max_total_chunks
        self._timeout = timeout_seconds or settings.summary_timeout_seconds
        self._map_concurrency = max(
            1, map_concurrency or settings.summary_map_concurrency
        )

    async def prepare_summary_stream(
        self,
        *,
        workspace_id: uuid.UUID,
        user: User,
        documents: list[tuple[str, str]],
        query: str,
    ) -> tuple[SummaryOutcome, AsyncIterator[str]]:
        """
        Returns (outcome, token_stream). The map phase runs HERE (before
        the stream starts) — callers should run this inside a background
        task so SSE heartbeats keep flowing during the map calls.
        """
        ws_id = str(workspace_id)
        outcome = SummaryOutcome(documents=list(documents))

        # ── Fetch all chunks per document ─────────────────────────
        doc_texts: list[str] = []  # one rendered blob per document
        chunk_budget = self._max_total_chunks
        for doc_id, filename in documents:
            chunks = await self._vector_store.list_document_chunks(
                ws_id, doc_id, str(user.id)
            )
            if not chunks:
                continue
            if len(chunks) > chunk_budget:
                chunks = chunks[:chunk_budget]
                outcome.truncated = True
            chunk_budget -= len(chunks)
            outcome.total_chunks += len(chunks)

            rendered = "\n\n".join(_render_chunk(c) for c in chunks)
            doc_texts.append(f"## {filename}\n\n{rendered}")

            first = chunks[0]
            outcome.sources.append(
                CitedSource(
                    document_id=doc_id,
                    chunk_id=first.chunk_id,
                    source_filename=filename,
                    page_number=first.payload.page_number,
                    chunk_index=0,
                    relevance_score=1.0,
                    heading_path=[],
                    text="",
                )
            )
            if chunk_budget <= 0:
                outcome.truncated = True
                break

        material = "\n\n".join(doc_texts)
        if not material.strip():
            # No chunks at all — stream a graceful note instead of erroring.
            async def _empty() -> AsyncIterator[str]:
                yield (
                    "I couldn't find any indexed content for the requested "
                    "document(s). Make sure they finished processing, then try again."
                )

            return outcome, _empty()

        # ── Map-reduce only if the doc is too big for a single pass ──
        # Material within context_budget streams directly (no map calls) —
        # the common case. Bigger docs are compressed down to context_budget
        # by mapping into segment-budget chunks, at most 2 levels.
        level = 0
        while count_tokens(material) > self._context_budget and level < 2:
            level += 1
            segments = _pack_segments(
                doc_texts if level == 1 else [material], self._map_budget
            )
            log.info(
                "summary_map_level",
                level=level,
                segments=len(segments),
                tokens=count_tokens(material),
                concurrency=self._map_concurrency,
            )
            bullets = await self._map_segments(segments)
            outcome.map_calls += len(segments)
            material = "\n\n".join(bullets)

        if count_tokens(material) > self._context_budget:
            # 2 reduce levels weren't enough — hard truncate, keep the head.
            log.warning("summary_truncated_after_reduce", tokens=count_tokens(material))
            material = _truncate_to_tokens(material, self._context_budget)
            outcome.truncated = True

        # ── Final pass: streamed executive summary ────────────────
        final_messages = [
            LLMMessage(role="system", content=_FINAL_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=f"User request: {query}\n\nMATERIAL:\n{material}",
            ),
        ]
        llm_config = LLMConfig(
            temperature=settings.llm_temperature,
            max_tokens=self._final_max_tokens,
        )

        log.info(
            "summary_stream_start",
            documents=len(outcome.documents),
            chunks=outcome.total_chunks,
            map_calls=outcome.map_calls,
            truncated=outcome.truncated,
        )

        async def _token_iter() -> AsyncIterator[str]:
            try:
                async for token in self._llm.stream(
                    final_messages, llm_config, timeout=self._timeout
                ):
                    yield token
            except LLMProviderError:
                raise
            except Exception as exc:
                raise LLMProviderError(
                    "Summary generation failed.", detail=str(exc)
                ) from exc

        return outcome, _token_iter()

    async def _map_segments(self, segments: list[str]) -> list[str]:
        """
        Summarize each segment into bullets — CONCURRENTLY.

        The map phase is embarrassingly parallel (each part is summarized
        independently), so we fire the calls together via asyncio.gather
        instead of awaiting them one-by-one. A semaphore caps in-flight
        requests at `map_concurrency` to stay within Ollama's parallel
        slots / memory. Results keep input order (gather preserves it).
        """
        sem = asyncio.Semaphore(self._map_concurrency)
        map_config = LLMConfig(temperature=0.2, max_tokens=self._section_max_tokens)

        async def _one(segment: str) -> str:
            # Stream, don't generate(): a non-streaming call must deliver the
            # whole response (prefill + every decode token) inside ONE read
            # timeout, so on a slow CPU the decode tail pushes a fine-prefill
            # call past the limit. Streaming resets the read clock per token,
            # so only time-to-first-token (the prefill) must fit the timeout
            # and the decode can take as long as it needs. Accumulate the
            # streamed bullets back into the string the reduce step expects.
            async with sem:
                parts: list[str] = []
                async for token in self._llm.stream(
                    [
                        LLMMessage(role="system", content=_MAP_SYSTEM_PROMPT),
                        LLMMessage(role="user", content=segment),
                    ],
                    map_config,
                    timeout=self._timeout,
                ):
                    parts.append(token)
            return "".join(parts).strip()

        return list(await asyncio.gather(*(_one(s) for s in segments)))


# ── Pure helpers ──────────────────────────────────────────────────


def _render_chunk(chunk: SearchResult) -> str:
    heading = (
        " > ".join(chunk.payload.heading_path) if chunk.payload.heading_path else ""
    )
    return f"{heading}\n{chunk.payload.text}" if heading else chunk.payload.text


def _pack_segments(texts: list[str], budget: int) -> list[str]:
    """
    Greedily pack paragraphs into segments of ≤ budget tokens. Inputs
    are per-document blobs, so documents never mix within a segment;
    paragraphs (chunk renderings) are never split.
    """
    segments: list[str] = []
    for text in texts:
        paragraphs = text.split("\n\n")
        current: list[str] = []
        current_tokens = 0
        for para in paragraphs:
            para_tokens = count_tokens(para)
            if current and current_tokens + para_tokens > budget:
                segments.append("\n\n".join(current))
                current = []
                current_tokens = 0
            current.append(para)
            current_tokens += para_tokens
        if current:
            segments.append("\n\n".join(current))
    return segments


def _truncate_to_tokens(text: str, budget: int) -> str:
    """Drop trailing paragraphs until the text fits the budget."""
    paragraphs = text.split("\n\n")
    kept: list[str] = []
    total = 0
    for para in paragraphs:
        tokens = count_tokens(para)
        if total + tokens > budget:
            break
        kept.append(para)
        total += tokens
    return "\n\n".join(kept) if kept else text[: budget * 3]
