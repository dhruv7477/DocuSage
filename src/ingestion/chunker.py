"""
src/ingestion/chunker.py
─────────────────────────
Recursive character text splitter for RAG-optimised chunking.

Strategy:
    Recursively splits on paragraph → sentence → word boundaries.
    This preserves semantic units: a chunk never cuts mid-sentence
    unless the sentence itself exceeds chunk_size.

    Overlap ensures context continuity between adjacent chunks:
    the last `overlap` characters of chunk N appear at the start of chunk N+1.
    This prevents information loss at chunk boundaries.

Why recursive (not fixed-size):
    Fixed-size splitting can cut sentences mid-word, losing meaning.
    Recursive splitting respects natural text structure, producing
    chunks that make linguistic sense — critical for embedding quality.

Output:
    Each TextChunk carries its source page_number, chunk_index,
    and text. The ingestion pipeline uses these to build VectorPayload.
"""

from dataclasses import dataclass

from src.core.logger import get_logger
from src.ingestion.parsers.base import ParsedDocument

log = get_logger(__name__)

# Separator hierarchy: try to split at the most semantic boundary first
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""]


@dataclass
class TextChunk:
    """A single chunk ready for embedding and vector storage."""
    text: str
    chunk_index: int        # Position within this document's chunks
    page_number: int        # Source page (for citation)
    char_start: int = 0     # Character offset in the full document text


class RecursiveChunker:
    """
    Splits ParsedDocument into overlapping text chunks.

    Args:
        chunk_size:  Target chunk size in characters (~800 chars ≈ ~200 tokens)
        chunk_overlap: Characters of overlap between adjacent chunks
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size.")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk_document(self, document: ParsedDocument) -> list[TextChunk]:
        """
        Chunk all pages of a parsed document.

        Returns:
            Flat list of TextChunk objects, indexed from 0.
        """
        all_chunks: list[TextChunk] = []

        for page in document.pages:
            if not page.text.strip():
                continue
            raw_chunks = self._split_text(page.text)
            for text in raw_chunks:
                if text.strip():
                    all_chunks.append(
                        TextChunk(
                            text=text.strip(),
                            chunk_index=len(all_chunks),
                            page_number=page.page_number,
                        )
                    )

        log.debug(
            "document_chunked",
            filename=document.filename,
            pages=len(document.pages),
            chunks=len(all_chunks),
            chunk_size=self._chunk_size,
            overlap=self._chunk_overlap,
        )
        return all_chunks

    def _split_text(self, text: str) -> list[str]:
        """
        Recursively split text using the separator hierarchy.
        Returns a list of strings each <= chunk_size characters.
        """
        return self._recursive_split(text, _SEPARATORS)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """
        Try each separator in order. If a separator produces pieces
        that are still too large, recurse with the next separator.
        """
        if len(text) <= self._chunk_size:
            return [text]

        # Find the first separator that produces at least 2 pieces
        separator = separators[-1]   # fallback: split anywhere
        for sep in separators:
            if sep in text:
                separator = sep
                break

        splits = text.split(separator)
        return self._merge_splits(splits, separator, separators[1:] if separators else [])

    def _merge_splits(
        self,
        splits: list[str],
        separator: str,
        remaining_separators: list[str],
    ) -> list[str]:
        """
        Merge small splits into chunks up to chunk_size,
        then add overlap by carrying the tail of the previous chunk.
        """
        chunks: list[str] = []
        current_parts: list[str] = []
        current_len: int = 0

        for split in splits:
            split_len = len(split)

            # If a single split exceeds chunk_size, recurse with next separator
            if split_len > self._chunk_size:
                if current_parts:
                    merged = self._join(current_parts, separator)
                    chunks.append(merged)
                    current_parts = []
                    current_len = 0
                if remaining_separators:
                    sub_chunks = self._recursive_split(split, remaining_separators)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(split)
                continue

            sep_len = len(separator) if current_parts else 0
            if current_len + sep_len + split_len > self._chunk_size and current_parts:
                # Flush current chunk
                merged = self._join(current_parts, separator)
                chunks.append(merged)
                # Build overlap: keep tail parts until we have ~chunk_overlap chars
                overlap_parts: list[str] = []
                overlap_len = 0
                for part in reversed(current_parts):
                    part_len = len(part) + len(separator)
                    if overlap_len + part_len > self._chunk_overlap:
                        break
                    overlap_parts.insert(0, part)
                    overlap_len += part_len
                current_parts = overlap_parts
                current_len = overlap_len

            current_parts.append(split)
            current_len += split_len + (len(separator) if len(current_parts) > 1 else 0)

        if current_parts:
            chunks.append(self._join(current_parts, separator))

        return [c for c in chunks if c.strip()]

    @staticmethod
    def _join(parts: list[str], separator: str) -> str:
        return separator.join(parts)