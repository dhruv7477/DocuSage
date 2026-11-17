"""
tests/unit/test_chunker.py
───────────────────────────
Unit tests for the semantic, token-based chunker.

These tests target the new `SemanticChunker` (Phase 2) operating over
the new `ParsedBlock` / `ParsedDocument` shape. The legacy
`RecursiveChunker` alias still exists for back-compat callers and is
exercised here too.
"""

import pytest

from src.ingestion.chunker import RecursiveChunker, SemanticChunker, TextChunk
from src.ingestion.parsers.base import ParsedBlock, ParsedDocument


def _para(
    text: str, page: int = 1, char_start: int = 0, heading_path: list[str] | None = None
) -> ParsedBlock:
    return ParsedBlock(
        text=text,
        block_type="paragraph",
        heading_path=heading_path or [],
        page_number=page,
        char_start=char_start,
        char_end=char_start + len(text),
    )


def _heading(
    text: str,
    level: int,
    page: int = 1,
    char_start: int = 0,
    heading_path: list[str] | None = None,
) -> ParsedBlock:
    return ParsedBlock(
        text=text,
        block_type="heading",
        heading_path=(heading_path or []) + [text],
        page_number=page,
        char_start=char_start,
        char_end=char_start + len(text),
        heading_level=level,
    )


def _make_doc(text: str, filename: str = "test.txt") -> ParsedDocument:
    return ParsedDocument(
        filename=filename,
        blocks=[_para(text)],
        total_pages=1,
    )


class TestSemanticChunker:
    def test_short_text_is_single_chunk(self):
        chunker = SemanticChunker(chunk_tokens=200, overlap_tokens=40)
        doc = _make_doc("This is a short document.")
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "This is a short document."

    def test_long_text_produces_multiple_chunks(self):
        chunker = SemanticChunker(chunk_tokens=40, overlap_tokens=10)
        doc = _make_doc("word " * 400)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1

    def test_chunks_respect_token_budget(self):
        budget = 50
        chunker = SemanticChunker(chunk_tokens=budget, overlap_tokens=10)
        doc = _make_doc("The quick brown fox jumps over the lazy dog. " * 50)
        chunks = chunker.chunk_document(doc)
        # token_count is set by the chunker itself, so we check it. Allow
        # a small fudge factor — overlap pre-seeds the next chunk.
        for chunk in chunks:
            assert chunk.token_count <= budget * 2

    def test_chunk_indices_are_sequential(self):
        chunker = SemanticChunker(chunk_tokens=40, overlap_tokens=8)
        doc = _make_doc("sentence. " * 100)
        chunks = chunker.chunk_document(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_empty_document_produces_no_chunks(self):
        chunker = SemanticChunker(chunk_tokens=200, overlap_tokens=40)
        doc = ParsedDocument(filename="empty.txt", blocks=[])
        assert chunker.chunk_document(doc) == []

    def test_page_number_preserved(self):
        # Use enough text per page to force a flush before the next page
        # is appended; that proves both page numbers reach distinct chunks.
        chunker = SemanticChunker(chunk_tokens=80, overlap_tokens=10)
        doc = ParsedDocument(
            filename="multi.pdf",
            blocks=[
                _para("Page one content. " * 40, page=1, char_start=0),
                _para("Page two content. " * 40, page=2, char_start=2000),
            ],
            total_pages=2,
        )
        chunks = chunker.chunk_document(doc)
        page_numbers = {c.page_number for c in chunks}
        assert 1 in page_numbers and 2 in page_numbers

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            SemanticChunker(chunk_tokens=100, overlap_tokens=100)

    def test_heading_boundary_creates_new_chunk(self):
        """A heading must flush the in-flight chunk so siblings stay separate."""
        chunker = SemanticChunker(chunk_tokens=500, overlap_tokens=0)
        doc = ParsedDocument(
            filename="t.md",
            blocks=[
                _heading("Section A", level=1, page=1, char_start=0, heading_path=[]),
                _para(
                    "Body of section A.",
                    page=1,
                    char_start=10,
                    heading_path=["Section A"],
                ),
                _heading("Section B", level=1, page=2, char_start=30, heading_path=[]),
                _para(
                    "Body of section B.",
                    page=2,
                    char_start=40,
                    heading_path=["Section B"],
                ),
            ],
            total_pages=2,
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 2
        # heading_path is propagated onto the chunk
        assert chunks[0].heading_path == ["Section A"]
        assert chunks[1].heading_path == ["Section B"]
        # Heading text itself is not duplicated in the chunk content
        assert "Section A" not in chunks[0].text
        assert chunks[0].text == "Body of section A."

    def test_embedding_text_prepends_heading_path(self):
        chunker = SemanticChunker(chunk_tokens=100, overlap_tokens=0)
        doc = ParsedDocument(
            filename="t.md",
            blocks=[
                _heading("Refunds", level=1, page=1, char_start=0),
                _para(
                    "All refunds processed within 30 days.",
                    page=1,
                    char_start=10,
                    heading_path=["Refunds"],
                ),
            ],
            total_pages=1,
        )
        chunks = chunker.chunk_document(doc)
        assert chunks[0].embedding_text().startswith("Refunds\n\n")

    def test_chunk_carries_char_offsets(self):
        chunker = SemanticChunker(chunk_tokens=200, overlap_tokens=0)
        doc = _make_doc("The full text of the document.")
        doc.blocks[0].char_start = 100
        doc.blocks[0].char_end = 130
        chunks = chunker.chunk_document(doc)
        assert chunks[0].char_start == 100
        assert chunks[0].char_end == 130


class TestRecursiveChunkerCompat:
    """The legacy class name still works — it just delegates to SemanticChunker."""

    def test_legacy_alias_chunks_a_doc(self):
        chunker = RecursiveChunker(chunk_size=800, chunk_overlap=150)
        doc = _make_doc("This is a short document.")
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert isinstance(chunks[0], TextChunk)
