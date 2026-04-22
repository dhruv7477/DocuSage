"""
src/ingestion/parsers/text_parser.py
──────────────────────────────────────
Parsers for plain text formats: TXT, Markdown, HTML.

All three produce ParsedDocument with logical page groupings.
HTML is stripped of tags to extract readable text only.
Markdown is treated as plain text — headings and bullets are preserved
as-is since they provide useful context for the LLM.
"""

import asyncio
import re

from src.core.exceptions import DocumentProcessingError
from src.core.logger import get_logger
from src.ingestion.parsers.base import BaseParser, ParsedDocument, ParsedPage

log = get_logger(__name__)

_CHARS_PER_PAGE = 3000   # ~750 words per logical page


class TextParser(BaseParser):
    """Handles .txt and .md files."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".txt", ".md", ".markdown"]

    async def parse(self, content: bytes, filename: str) -> ParsedDocument:
        try:
            text = content.decode("utf-8", errors="replace")
        except Exception as exc:
            raise DocumentProcessingError(
                f"Cannot decode text file '{filename}'", detail=str(exc)
            ) from exc

        text = self._clean_text(text)
        pages = self._split_into_pages(text)

        log.debug("text_parsed", filename=filename, pages=len(pages))
        return ParsedDocument(
            filename=filename,
            pages=pages,
            total_pages=len(pages),
            metadata={"parser": "text"},
        )

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\t", "    ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_into_pages(self, text: str) -> list[ParsedPage]:
        """Split long text into logical pages by character count."""
        if not text:
            return []
        pages = []
        for i in range(0, len(text), _CHARS_PER_PAGE):
            chunk = text[i : i + _CHARS_PER_PAGE].strip()
            if chunk:
                pages.append(ParsedPage(page_number=len(pages) + 1, text=chunk))
        return pages


class HTMLParser(BaseParser):
    """Strips HTML tags and extracts readable text."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".html", ".htm"]

    async def parse(self, content: bytes, filename: str) -> ParsedDocument:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, self._parse_sync, content, filename
            )
        except Exception as exc:
            raise DocumentProcessingError(
                f"Failed to parse HTML '{filename}'", detail=str(exc)
            ) from exc

    def _parse_sync(self, content: bytes, filename: str) -> ParsedDocument:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            # Fallback: strip tags with regex (less accurate but no dependency)
            text = re.sub(r"<[^>]+>", " ", content.decode("utf-8", errors="replace"))
            text = re.sub(r"\s+", " ", text).strip()
        else:
            soup = BeautifulSoup(content, "html.parser")
            # Remove script and style elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            text = re.sub(r"\n{3,}", "\n\n", text).strip()

        text_parser = TextParser()
        pages = text_parser._split_into_pages(text)

        log.debug("html_parsed", filename=filename, pages=len(pages))
        return ParsedDocument(
            filename=filename,
            pages=pages,
            total_pages=len(pages),
            metadata={"parser": "html"},
        )
