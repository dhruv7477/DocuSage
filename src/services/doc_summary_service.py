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

When full summaries are turned off (DOC_SUMMARY_ENABLED=false, to stay
within a hosted LLM's limits), `prepare_overview_stream()` answers a
summary request with ONE call instead: each target document's outline
(its headings) and opening chunks, packed into the normal answer context
budget, so it costs about the same as one question.
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
from src.providers.llm.base import (
    BaseLLMProvider,
    LLMConfig,
    LLMMessage,
    require_text,
)
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

_OVERVIEW_SYSTEM_PROMPT = """You are giving a short overview of the user's document(s). For each document you are shown its headings and its opening sections, not the full text.

- Say in one or two sentences what each document is and what it covers, using its headings.
- Then give a few key points from the opening sections, keeping concrete names, figures, and dates.
- Do not guess at content you were not shown, and do not say a document lacks something only because these excerpts do not show it.
- No preamble. Do NOT use bracketed citation markers like [1].
If the user asked about a specific aspect, focus on that aspect."""

# Shown above every overview, so the reader knows what it covers.
OVERVIEW_NOTE = (
    "Short overview: full summaries are turned off in this demo, so this "
    "covers each document's headings and opening sections.\n\n"
)

_NO_CONTENT_NOTE = (
    "I couldn't find any indexed content for the requested "
    "document(s). Make sure they finished processing, then try again."
)

# The smallest share of the overview budget worth giving one document; with
# more documents than fit, the overview covers the first ones.
_MIN_OVERVIEW_SHARE = 300
# At most this fraction (1/n) of a document's share goes to its outline.
_OUTLINE_SHARE = 3


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
        overview_token_budget: int | None = None,
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
        # The overview fills the same context budget as a normal answer.
        self._overview_budget = overview_token_budget or settings.rag_context_tokens

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

            outcome.sources.append(_document_source(doc_id, filename, chunks[0]))
            if chunk_budget <= 0:
                outcome.truncated = True
                break

        material = "\n\n".join(doc_texts)
        if not material.strip():
            # No chunks at all — stream a graceful note instead of erroring.
            return outcome, _single(_NO_CONTENT_NOTE)

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

        return outcome, self._stream(final_messages, llm_config, purpose="summary")

    async def prepare_overview_stream(
        self,
        *,
        workspace_id: uuid.UUID,
        user: User,
        documents: list[tuple[str, str]],
        query: str,
    ) -> tuple[SummaryOutcome, AsyncIterator[str]]:
        """
        A short overview in ONE call, for when full summaries are off.

        Each target document gets an equal share of the answer context
        budget, filled with its outline (headings, in order) and then its
        opening chunks. The words of the request are never searched for:
        "Summary of the document" matches no passage.
        """
        ws_id = str(workspace_id)
        max_docs = max(1, self._overview_budget // _MIN_OVERVIEW_SHARE)
        targets = list(documents)[:max_docs]
        outcome = SummaryOutcome(
            documents=targets, truncated=len(documents) > len(targets)
        )
        share = self._overview_budget // max(1, len(targets))

        sections: list[str] = []
        for doc_id, filename in targets:
            chunks = await self._vector_store.list_document_chunks(
                ws_id, doc_id, str(user.id)
            )
            if not chunks:
                continue
            section, used = _overview_section(filename, chunks, share)
            sections.append(section)
            outcome.total_chunks += used
            outcome.truncated = outcome.truncated or used < len(chunks)
            outcome.sources.append(_document_source(doc_id, filename, chunks[0]))

        if not sections:
            return outcome, _single(_NO_CONTENT_NOTE)

        material = "\n\n".join(sections)
        messages = [
            LLMMessage(role="system", content=_OVERVIEW_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=f"User request: {query}\n\nMATERIAL:\n{material}",
            ),
        ]
        # The answer limit, not the summary's: a reasoning model counts its
        # reasoning toward max_tokens, and only generated tokens are billed.
        llm_config = LLMConfig(
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        log.info(
            "overview_stream_start",
            documents=len(sections),
            chunks=outcome.total_chunks,
            tokens=count_tokens(material),
            truncated=outcome.truncated,
        )

        async def _with_note() -> AsyncIterator[str]:
            # The note goes out with the first token, so a call that returns
            # nothing leaves no note above an answer that never came.
            note = OVERVIEW_NOTE
            async for token in self._stream(messages, llm_config, purpose="overview"):
                yield note + token
                note = ""

        return outcome, _with_note()

    async def _stream(
        self, messages: list[LLMMessage], llm_config: LLMConfig, *, purpose: str
    ) -> AsyncIterator[str]:
        """
        The final streamed call. Provider errors are made uniform, and an
        empty answer becomes a retry message (see `require_text`).
        """
        try:
            async for token in require_text(
                self._llm.stream(messages, llm_config, timeout=self._timeout),
                purpose=purpose,
            ):
                yield token
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                "Summary generation failed.", detail=str(exc)
            ) from exc

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


async def _single(text: str) -> AsyncIterator[str]:
    """A stream of one fixed piece of text."""
    yield text


def _document_source(doc_id: str, filename: str, first: SearchResult) -> CitedSource:
    """One source per summarized document, pointing at its first chunk."""
    return CitedSource(
        document_id=doc_id,
        chunk_id=first.chunk_id,
        source_filename=filename,
        page_number=first.payload.page_number,
        chunk_index=0,
        relevance_score=1.0,
        heading_path=[],
        text="",
    )


def _outline(chunks: list[SearchResult], budget: int) -> str:
    """The document's distinct heading paths, in order, within `budget` tokens."""
    lines: list[str] = []
    seen: set[str] = set()
    total = count_tokens("Headings:")
    for chunk in chunks:
        path = " > ".join(chunk.payload.heading_path)
        if not path or path in seen:
            continue
        seen.add(path)
        line = f"- {path}"
        cost = count_tokens(line)
        if total + cost > budget:
            break
        lines.append(line)
        total += cost
    return "Headings:\n" + "\n".join(lines) if lines else ""


def _overview_section(
    filename: str, chunks: list[SearchResult], budget: int
) -> tuple[str, int]:
    """
    One document's part of an overview: its outline, then its opening
    chunks, within `budget` tokens. Returns the text and the number of
    chunks it includes.
    """
    parts = [f"## {filename}"]
    outline = _outline(chunks, budget // _OUTLINE_SHARE)
    if outline:
        parts.append(outline)
    # Each "\n\n" separator costs a token or two; reserve them up front.
    remaining = budget - count_tokens("\n\n".join(parts)) - 2 * len(chunks)
    used = 0
    for chunk in chunks:
        text = chunk.payload.text
        cost = count_tokens(text)
        if cost > remaining:
            if used == 0 and remaining > 0:
                # Always show the start of the document, cut to fit.
                parts.append(_truncate_to_tokens(text, remaining))
                used = 1
            break
        parts.append(text)
        remaining -= cost
        used += 1
    return "\n\n".join(parts), used


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
