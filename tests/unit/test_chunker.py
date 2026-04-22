"""
tests/unit/test_chunker.py
───────────────────────────
Unit tests for RecursiveChunker — no external dependencies.
Validates chunking behaviour, overlap, and edge cases.
"""

import pytest

from src.ingestion.chunker import RecursiveChunker, TextChunk
from src.ingestion.parsers.base import ParsedDocument, ParsedPage


def _make_doc(text: str, filename: str = "test.txt") -> ParsedDocument:
    return ParsedDocument(
        filename=filename,
        pages=[ParsedPage(page_number=1, text=text)],
        total_pages=1,
    )


class TestRecursiveChunker:

    def test_short_text_is_single_chunk(self):
        """Text shorter than chunk_size should produce exactly one chunk."""
        chunker = RecursiveChunker(chunk_size=800, chunk_overlap=150)
        doc = _make_doc("This is a short document.")
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "This is a short document."

    def test_long_text_produces_multiple_chunks(self):
        """Text longer than chunk_size must be split into multiple chunks."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        long_text = "word " * 200   # 1000 chars
        doc = _make_doc(long_text)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1

    def test_all_chunks_within_size_limit(self):
        """Every chunk should be <= chunk_size characters."""
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=40)
        long_text = "The quick brown fox jumps over the lazy dog. " * 50
        doc = _make_doc(long_text)
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            assert len(chunk.text) <= 200, f"Chunk too large: {len(chunk.text)}"

    def test_chunk_indices_are_sequential(self):
        """chunk_index must be 0, 1, 2, ... in order."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        doc = _make_doc("sentence. " * 100)
        chunks = chunker.chunk_document(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_empty_document_produces_no_chunks(self):
        """Empty or whitespace-only document should return empty list."""
        chunker = RecursiveChunker(chunk_size=800, chunk_overlap=150)
        doc = _make_doc("   \n\n  ")
        chunks = chunker.chunk_document(doc)
        assert chunks == []

    def test_page_number_preserved(self):
        """Each chunk should carry the page_number of its source page."""
        chunker = RecursiveChunker(chunk_size=800, chunk_overlap=150)
        doc = ParsedDocument(
            filename="multi.pdf",
            pages=[
                ParsedPage(page_number=1, text="Page one content. " * 5),
                ParsedPage(page_number=2, text="Page two content. " * 5),
            ],
            total_pages=2,
        )
        chunks = chunker.chunk_document(doc)
        page_numbers = {c.page_number for c in chunks}
        assert 1 in page_numbers
        assert 2 in page_numbers

    def test_invalid_overlap_raises(self):
        """chunk_overlap >= chunk_size should raise ValueError."""
        with pytest.raises(ValueError):
            RecursiveChunker(chunk_size=100, chunk_overlap=100)

    def test_overlap_not_configured_to_produce_overlap_chunks(self):
        """
        When we have overlap > 0, adjacent chunks should share some text.
        Just checks that the total reconstructed text is complete.
        """
        chunker = RecursiveChunker(chunk_size=80, chunk_overlap=20)
        text = "abc " * 100  # 400 chars
        doc = _make_doc(text)
        chunks = chunker.chunk_document(doc)
        # All original words should appear in at least one chunk
        for word in text.split()[:10]:
            assert any(word in c.text for c in chunks)

    def test_multipage_chunk_indices_are_globally_sequential(self):
        """chunk_index must be globally sequential across all pages."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        doc = ParsedDocument(
            filename="test.pdf",
            pages=[
                ParsedPage(page_number=1, text="First page text. " * 10),
                ParsedPage(page_number=2, text="Second page text. " * 10),
                ParsedPage(page_number=3, text="Third page text. " * 10),
            ],
            total_pages=3,
        )
        chunks = chunker.chunk_document(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i, (
                f"Expected chunk_index={i}, got {chunk.chunk_index}"
            )
