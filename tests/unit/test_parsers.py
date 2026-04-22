"""
tests/unit/test_parsers.py
───────────────────────────
Unit tests for parsers and ParserRegistry.
Uses synthetic bytes — no real files needed.
"""

import pytest

from src.core.exceptions import UnsupportedFileTypeError
from src.ingestion.parsers.registry import ParserRegistry
from src.ingestion.parsers.text_parser import HTMLParser, TextParser


class TestParserRegistry:

    def test_get_pdf_parser(self):
        parser = ParserRegistry.get("report.pdf")
        assert ".pdf" in parser.supported_extensions

    def test_get_docx_parser(self):
        parser = ParserRegistry.get("document.docx")
        assert ".docx" in parser.supported_extensions

    def test_get_txt_parser(self):
        parser = ParserRegistry.get("notes.txt")
        assert ".txt" in parser.supported_extensions

    def test_get_markdown_parser(self):
        parser = ParserRegistry.get("README.md")
        assert ".md" in parser.supported_extensions

    def test_get_html_parser(self):
        parser = ParserRegistry.get("page.html")
        assert ".html" in parser.supported_extensions

    def test_unsupported_extension_raises(self):
        with pytest.raises(UnsupportedFileTypeError):
            ParserRegistry.get("image.png")

    def test_case_insensitive_extension(self):
        """Uppercase extensions should work."""
        parser = ParserRegistry.get("REPORT.PDF")
        assert ".pdf" in parser.supported_extensions

    def test_is_supported_true(self):
        assert ParserRegistry.is_supported("file.pdf") is True

    def test_is_supported_false(self):
        assert ParserRegistry.is_supported("file.exe") is False

    def test_supported_extensions_not_empty(self):
        exts = ParserRegistry.supported_extensions()
        assert len(exts) >= 5


class TestTextParser:

    @pytest.mark.asyncio
    async def test_parse_plain_text(self):
        content = b"Hello world. This is a test document."
        parser = TextParser()
        doc = await parser.parse(content, "test.txt")
        assert not doc.is_empty
        assert "Hello world" in doc.full_text
        assert doc.filename == "test.txt"

    @pytest.mark.asyncio
    async def test_empty_text_is_empty(self):
        content = b"   \n\n   "
        parser = TextParser()
        doc = await parser.parse(content, "empty.txt")
        assert doc.is_empty

    @pytest.mark.asyncio
    async def test_long_text_splits_into_multiple_pages(self):
        # 10000 chars = should produce multiple logical pages
        content = ("word " * 2000).encode()
        parser = TextParser()
        doc = await parser.parse(content, "long.txt")
        assert doc.total_pages > 1

    @pytest.mark.asyncio
    async def test_page_numbers_sequential(self):
        content = ("word " * 2000).encode()
        parser = TextParser()
        doc = await parser.parse(content, "long.txt")
        for i, page in enumerate(doc.pages):
            assert page.page_number == i + 1


class TestHTMLParser:

    @pytest.mark.asyncio
    async def test_strips_html_tags(self):
        content = b"<html><body><h1>Title</h1><p>Body text here.</p></body></html>"
        parser = HTMLParser()
        doc = await parser.parse(content, "page.html")
        assert "Title" in doc.full_text
        assert "Body text here" in doc.full_text
        assert "<h1>" not in doc.full_text
        assert "<p>" not in doc.full_text

    @pytest.mark.asyncio
    async def test_removes_script_tags(self):
        content = b"<html><body><script>alert('xss')</script><p>Clean content</p></body></html>"
        parser = HTMLParser()
        doc = await parser.parse(content, "page.html")
        assert "alert" not in doc.full_text
        assert "Clean content" in doc.full_text
