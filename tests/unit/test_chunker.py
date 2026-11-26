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
        # The heading opens its chunk, so the chunk's own text states it.
        assert chunks[0].text == "Section A\n\nBody of section A."
        assert chunks[1].text == "Section B\n\nBody of section B."

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

    def test_a_document_title_heading_is_citable_text(self):
        """
        A resume's name is its largest font, so it parses as a heading. It
        must be in the chunk's own text: that text is what a citation is
        checked against and what the page image highlights.
        """
        chunker = SemanticChunker(chunk_tokens=500, overlap_tokens=0)
        doc = ParsedDocument(
            filename="resume.pdf",
            blocks=[
                _heading("Talia Brennerholt", level=1, page=1, char_start=0),
                _para(
                    "Portland, Oregon | talia.brennerholt@example.com",
                    page=1,
                    char_start=18,
                    heading_path=["Talia Brennerholt"],
                ),
            ],
            total_pages=1,
        )
        [chunk] = chunker.chunk_document(doc)
        assert chunk.text.startswith("Talia Brennerholt\n\n")
        assert chunk.heading_path == ["Talia Brennerholt"]
        assert chunk.char_start == 0

    def test_a_document_title_heads_only_the_chunk_that_prints_it(self):
        """
        A resume's name opens every heading path. Kept there, it is in
        every chunk's embedding and prompt header, so any passage looks
        like evidence for the name. Only the chunk whose text prints the
        name keeps it.
        """
        chunker = SemanticChunker(chunk_tokens=500, overlap_tokens=0)
        name = "Rosalind Kettleborne"
        doc = ParsedDocument(
            filename="resume.pdf",
            blocks=[
                _heading(name, level=1, page=1, char_start=0),
                _para("Austin, Texas", page=1, char_start=21, heading_path=[name]),
                _heading(
                    "Hackathons", level=2, page=2, char_start=40, heading_path=[name]
                ),
                _para(
                    "First place, Riverbend Data Jam 2024",
                    page=2,
                    char_start=52,
                    heading_path=[name, "Hackathons"],
                ),
            ],
            total_pages=2,
        )
        first, second = chunker.chunk_document(doc)
        assert first.heading_path == [name]
        assert first.embedding_text() == f"{name}\n\nAustin, Texas"
        assert second.heading_path == ["Hackathons"]
        assert name not in second.embedding_text()

    def test_a_heading_that_opens_only_some_paths_is_not_a_title(self):
        chunker = SemanticChunker(chunk_tokens=500, overlap_tokens=0)
        doc = ParsedDocument(
            filename="t.pdf",
            blocks=[
                _para("Front matter.", char_start=0),
                _heading("Guide", level=1, char_start=14),
                _para("Body.", char_start=20, heading_path=["Guide"]),
            ],
            total_pages=1,
        )
        first, second = chunker.chunk_document(doc)
        assert (first.heading_path, second.heading_path) == ([], ["Guide"])

    def test_consecutive_headings_open_the_next_chunk_together(self):
        chunker = SemanticChunker(chunk_tokens=500, overlap_tokens=0)
        doc = ParsedDocument(
            filename="resume.pdf",
            blocks=[
                _heading("Talia Brennerholt", level=1, char_start=0),
                _heading(
                    "Experience",
                    level=2,
                    char_start=18,
                    heading_path=["Talia Brennerholt"],
                ),
                _para(
                    "Senior Data Engineer, Pellwick Freight",
                    char_start=30,
                    heading_path=["Talia Brennerholt", "Experience"],
                ),
            ],
            total_pages=1,
        )
        [chunk] = chunker.chunk_document(doc)
        assert chunk.text == (
            "Talia Brennerholt\n\nExperience\n\nSenior Data Engineer, Pellwick Freight"
        )
        assert chunk.heading_path == ["Talia Brennerholt", "Experience"]

    def test_embedding_input_does_not_repeat_the_heading(self):
        # The embedder already gets the heading path; the heading line in the
        # text must not change what is embedded (or every cosine shifts).
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
        [chunk] = chunker.chunk_document(doc)
        assert (
            chunk.embedding_text() == "Refunds\n\nAll refunds processed within 30 days."
        )

    def test_headings_move_no_chunk_boundary(self):
        """
        Headings do not count toward the token budget, and the overlap is
        taken from the body: where chunks split, and so what is embedded,
        stays the same whatever the heading says.
        """

        def bodies(heading: str) -> list[str]:
            blocks = [_heading(heading, level=1, char_start=0)]
            blocks += [
                _para(
                    f"Sentence number {i} about pipelines and data quality.",
                    char_start=100 + 60 * i,
                    heading_path=[heading],
                )
                for i in range(30)
            ]
            doc = ParsedDocument(filename="t.pdf", blocks=blocks, total_pages=1)
            chunker = SemanticChunker(chunk_tokens=60, overlap_tokens=12)
            return [c.body_text or c.text for c in chunker.chunk_document(doc)]

        short = bodies("Name")
        long = bodies(
            "A much longer heading that runs to well over twenty tokens of text"
        )
        assert len(short) > 2
        assert short == long

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
