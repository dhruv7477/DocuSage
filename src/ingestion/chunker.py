"""
src/ingestion/chunker.py
─────────────────────────
Semantic, token-based chunker.

Why a rewrite was needed
------------------------
The previous chunker was character-based, dropped the heading/section
context emitted by parsers, and never populated char offsets — so
embeddings missed structural signal and the UI could not point users
at the exact passage that produced an answer. It also approximated
tokens as `len(text) / 4` which over- and under-shot real tokenizer
output by 30–50 % on dense PDFs, leading to chunks that fell outside
the embedder's context window.

Algorithm
---------
1. Walk `ParsedDocument.blocks` in order.
2. Maintain a `current` accumulator of consecutive blocks whose
   combined `tiktoken` token count stays under `chunk_tokens`.
3. **Never cross a heading boundary.** When we hit a heading whose
   level is ≤ any heading active on the previous block in `current`,
   the accumulator flushes — we'd otherwise glue together two
   sibling sections, losing the boundary.
4. When flushing, emit a `TextChunk` carrying:
   * `text` (joined by blank lines)
   * `heading_path` from the *first* block in the chunk (its actual context)
   * `page_number` — the most-common page seen across the chunk's blocks
   * `char_start` / `char_end` — exact span across the source's `full_text`
   * `token_count` from the actual tokenizer
   * `prev_chunk_id` / `next_chunk_id` — populated by the orchestrator
     after IDs are assigned (left as `None` here)
5. Carry forward an overlap of the last ~`overlap_tokens` tokens of
   the just-flushed chunk so contiguous chunks share context — vital
   for answers that straddle a chunk boundary.

Tables get their own chunk (never combined with surrounding paragraphs)
because they're typically self-contained and slicing them mid-row
destroys meaning.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.core.logger import get_logger
from src.ingestion.parsers.base import ParsedBlock, ParsedDocument

if TYPE_CHECKING:  # pragma: no cover
    from tiktoken import Encoding

log = get_logger(__name__)


# Default tokenizer: tiktoken's cl100k_base is OpenAI/Anthropic-compatible
# and a reasonable proxy for BGE / Qwen tokenizers — within ~10% across
# typical English text. Using a real tokenizer (any tokenizer) is far
# better than the previous char-based heuristic.
_TOKENIZER_NAME = "cl100k_base"


@dataclass
class TextChunk:
    """A single chunk ready for embedding and vector storage."""

    text: str
    chunk_index: int
    page_number: int
    heading_path: list[str] = field(default_factory=list)
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0
    # Populated by the ingestion pipeline once per-chunk UUIDs exist.
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    # `text` without the headings that open it. The embedder gets the
    # heading path instead, so the heading lines are left out of its input.
    body_text: str | None = None

    def embedding_text(self) -> str:
        """
        Text actually fed to the embedder.

        Prepending the heading path produces noticeably better
        retrieval — a chunk that says "must present a passport" embeds
        far more usefully when the model also sees that this sits under
        "Section 2 → Acceptable Identity Documents".
        """
        body = self.text if self.body_text is None else self.body_text
        if self.heading_path:
            return " > ".join(self.heading_path) + "\n\n" + body
        return body


class SemanticChunker:
    """
    Heading-aware, token-budgeted chunker.

    Args:
        chunk_tokens:    Target chunk size in tokens.
        overlap_tokens:  Tokens of trailing-context overlap between
                         consecutive chunks (0 disables overlap).
        tokenizer_name:  tiktoken encoding name. cl100k_base is a sane
                         default; pass a model-specific encoding via
                         `encoding_for_model` if you need it.
    """

    def __init__(
        self,
        chunk_tokens: int = 500,
        overlap_tokens: int = 100,
        tokenizer_name: str = _TOKENIZER_NAME,
    ) -> None:
        if overlap_tokens >= chunk_tokens:
            raise ValueError("overlap_tokens must be less than chunk_tokens.")
        self._chunk_tokens = chunk_tokens
        self._overlap_tokens = overlap_tokens
        self._encoder = self._load_tokenizer(tokenizer_name)

    @staticmethod
    def _load_tokenizer(name: str) -> "Encoding":
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError(
                "tiktoken not installed. Run: pip install tiktoken"
            ) from exc
        return tiktoken.get_encoding(name)

    # ── Public API ────────────────────────────────────────────────

    def chunk_document(self, document: ParsedDocument) -> list[TextChunk]:
        """
        Walk the document's blocks and emit semantic chunks.

        Returns a flat list of `TextChunk` indexed from 0.
        """
        if not document.blocks:
            return []

        chunks: list[TextChunk] = []
        current: list[ParsedBlock] = []
        current_token_count = 0

        for block in document.blocks:
            if not block.text.strip():
                continue

            block_tokens = self._token_count(block.text)

            # Headings waiting for their section's first content. They are
            # never a chunk on their own: they open the next one.
            lead = current if not _has_body(current) else []

            # Tables are emitted as their own chunk regardless of size.
            if block.block_type == "table":
                if current and not lead:
                    chunks.append(self._flush(current, len(chunks)))
                    current = (
                        self._build_overlap(chunks[-1]) if self._overlap_tokens else []
                    )
                    current_token_count = sum(
                        self._token_count(b.text) for b in current
                    )
                    lead = []
                # If a table itself blows past the budget, we still emit
                # it as one chunk — slicing tables corrupts them.
                chunks.append(
                    self._flush(
                        [*lead, block],
                        len(chunks),
                        forced_token_count=block_tokens,
                    )
                )
                if lead:
                    current, current_token_count = [], 0
                continue

            # If this block alone exceeds the budget, hard-split it
            # at sentence boundaries; very rare, but defensive.
            if block_tokens > self._chunk_tokens:
                if current and not lead:
                    chunks.append(self._flush(current, len(chunks)))
                for i, sub in enumerate(self._split_oversized_block(block)):
                    chunks.append(
                        self._flush([*lead, sub] if i == 0 else [sub], len(chunks))
                    )
                current, current_token_count = [], 0
                continue

            # Heading boundary: flush whatever's in flight so the next
            # section starts fresh, and open the next chunk with the
            # heading. Its text is then part of the chunk the model may
            # cite and the page image highlights (a resume's name is its
            # largest heading). The embedder still gets it only once,
            # through the heading path (see `TextChunk.body_text`).
            # Headings do not count toward the budget, so chunks split where
            # they did before headings joined their text (and embed the same).
            if block.block_type == "heading":
                if current and not lead:
                    chunks.append(self._flush(current, len(chunks)))
                    current, current_token_count = [], 0
                current.append(block)
                continue

            # Budget exceeded → flush, then carry overlap into next chunk.
            # Leading headings stay with the content that follows them.
            if (
                current_token_count + block_tokens > self._chunk_tokens
                and current
                and not lead
            ):
                chunks.append(self._flush(current, len(chunks)))
                current = (
                    self._build_overlap(chunks[-1]) if self._overlap_tokens else []
                )
                current_token_count = sum(self._token_count(b.text) for b in current)

            current.append(block)
            current_token_count += block_tokens

        if current:
            chunks.append(self._flush(current, len(chunks)))

        # Wire prev/next pointers (using chunk_index — UUIDs are assigned
        # later by the ingestion pipeline; the pipeline copies these
        # indices into the actual ID strings).
        for i, c in enumerate(chunks):
            c.prev_chunk_id = str(chunks[i - 1].chunk_index) if i > 0 else None
            c.next_chunk_id = (
                str(chunks[i + 1].chunk_index) if i < len(chunks) - 1 else None
            )

        log.debug(
            "document_chunked",
            filename=document.filename,
            blocks=len(document.blocks),
            chunks=len(chunks),
            chunk_tokens=self._chunk_tokens,
            overlap_tokens=self._overlap_tokens,
        )
        return chunks

    # ── Private helpers ───────────────────────────────────────────

    def _token_count(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _flush(
        self,
        blocks: list[ParsedBlock],
        chunk_index: int,
        forced_token_count: int | None = None,
    ) -> TextChunk:
        """Build a `TextChunk` from a non-empty list of blocks."""
        text = "\n\n".join(b.text for b in blocks)
        body = [b for b in blocks if b.block_type != "heading"]
        body_text = "\n\n".join(b.text for b in body) if body else None
        # Page number: take the modal page across the contributing blocks
        # so a chunk that spans page boundaries reports its center.
        page_number = Counter(b.page_number for b in blocks).most_common(1)[0][0]

        # Heading context comes from the first content block, which sits
        # under every heading that opens the chunk. A chunk of headings
        # alone (a document that ends on one) uses its first heading.
        first = body[0] if body else blocks[0]
        heading_path = list(first.heading_path)
        if first.block_type == "heading" and (
            not heading_path or heading_path[-1] != first.text
        ):
            heading_path = heading_path + [first.text]

        token_count = (
            forced_token_count
            if forced_token_count is not None
            else self._token_count(text)
        )

        return TextChunk(
            text=text,
            chunk_index=chunk_index,
            page_number=page_number,
            heading_path=heading_path,
            char_start=blocks[0].char_start,
            char_end=blocks[-1].char_end,
            token_count=token_count,
            body_text=body_text if body_text != text else None,
        )

    def _build_overlap(self, prev_chunk: TextChunk) -> list[ParsedBlock]:
        """
        Trailing context for the next chunk. We build a synthetic
        single-block "overlap" containing the last ~`overlap_tokens`
        tokens of the previous chunk's text, sentence-aligned.
        """
        # From the body: a heading that opens the previous chunk is not
        # carried into the next one.
        body = prev_chunk.text if prev_chunk.body_text is None else prev_chunk.body_text
        if self._overlap_tokens == 0 or not body:
            return []
        # Encode → keep tail tokens → decode. This gives an exact token
        # count and survives multi-byte / non-Latin scripts cleanly.
        ids = self._encoder.encode(body)
        tail_ids = ids[-self._overlap_tokens :]
        tail_text = self._encoder.decode(tail_ids).strip()
        if not tail_text:
            return []
        return [
            ParsedBlock(
                text=tail_text,
                block_type="paragraph",
                heading_path=prev_chunk.heading_path,
                page_number=prev_chunk.page_number,
                char_start=prev_chunk.char_end,
                char_end=prev_chunk.char_end,
            )
        ]

    def _split_oversized_block(self, block: ParsedBlock) -> list[ParsedBlock]:
        """
        Hard-split a single block that exceeds the chunk budget. We
        slice on encoded-token windows and decode back to text so each
        sub-block is exactly within budget. Char offsets are
        approximate for these (rare) sub-blocks — we proportion the
        original block's char span.
        """
        ids = self._encoder.encode(block.text)
        window = self._chunk_tokens - 1
        sub_blocks: list[ParsedBlock] = []
        total_tokens = len(ids)
        char_span = max(1, block.char_end - block.char_start)
        for offset in range(0, total_tokens, window):
            window_ids = ids[offset : offset + window]
            text = self._encoder.decode(window_ids).strip()
            if not text:
                continue
            ratio_start = offset / total_tokens
            ratio_end = min(1.0, (offset + window) / total_tokens)
            sub_blocks.append(
                ParsedBlock(
                    text=text,
                    block_type=block.block_type,
                    heading_path=block.heading_path,
                    page_number=block.page_number,
                    char_start=block.char_start + int(char_span * ratio_start),
                    char_end=block.char_start + int(char_span * ratio_end),
                    heading_level=block.heading_level,
                )
            )
        return sub_blocks


def _has_body(blocks: list[ParsedBlock]) -> bool:
    """True when the blocks hold content, not only headings."""
    return any(b.block_type != "heading" for b in blocks)


# ── Backward-compat shim ──────────────────────────────────────────
# The legacy name `RecursiveChunker` is preserved so existing imports
# and tests keep working during the Phase 2 transition. New code should
# use `SemanticChunker` directly.


class RecursiveChunker(SemanticChunker):
    """Legacy alias. Char-based defaults are converted to token defaults."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        # Old defaults (800 chars / 150 overlap ≈ 200/40 tokens) were
        # too small for modern embedders. Map to sane token defaults
        # but let an explicit char-budget caller still influence size.
        approx_tokens = max(200, chunk_size // 4)
        approx_overlap = max(40, chunk_overlap // 4)
        if approx_overlap >= approx_tokens:
            approx_overlap = approx_tokens // 4
        super().__init__(
            chunk_tokens=approx_tokens,
            overlap_tokens=approx_overlap,
        )
