"""
src/ingestion/parsers/docx_parser.py
──────────────────────────────────────
DOCX parser using python-docx.

Extracts:
- Paragraph text (preserving heading hierarchy)
- Table contents (row by row, pipe-delimited)
- Skips headers/footers to avoid noise

DOCX files don't have "pages" in the same way PDFs do.
We create logical pages by grouping ~50 paragraphs each,
which preserves document structure for the chunker.
"""

import asyncio
import io

from src.core.exceptions import DocumentProcessingError
from src.core.logger import get_logger
from src.ingestion.parsers.base import BaseParser, ParsedDocument, ParsedPage

log = get_logger(__name__)

_PARAGRAPHS_PER_PAGE = 50   # Logical grouping for consistent page size


class DOCXParser(BaseParser):

    @property
    def supported_extensions(self) -> list[str]:
        return [".docx", ".doc"]

    async def parse(self, content: bytes, filename: str) -> ParsedDocument:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, self._parse_sync, content, filename
            )
        except Exception as exc:
            raise DocumentProcessingError(
                f"Failed to parse DOCX '{filename}'", detail=str(exc)
            ) from exc

    def _parse_sync(self, content: bytes, filename: str) -> ParsedDocument:
        try:
            from docx import Document
        except ImportError as exc:
            raise DocumentProcessingError(
                "python-docx not installed. Run: pip install python-docx"
            ) from exc

        doc = Document(io.BytesIO(content))
        text_blocks: list[str] = []

        # ── Paragraphs ────────────────────────────────────────────
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            # Add heading markers to preserve document structure
            if para.style.name.startswith("Heading"):
                level = para.style.name.split()[-1]
                text = f"{'#' * int(level) if level.isdigit() else '#'} {text}"
            text_blocks.append(text)

        # ── Tables ────────────────────────────────────────────────
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                row_text = " | ".join(c for c in cells if c)
                if row_text:
                    text_blocks.append(row_text)

        # Group into logical pages
        pages: list[ParsedPage] = []
        for i in range(0, max(len(text_blocks), 1), _PARAGRAPHS_PER_PAGE):
            chunk = text_blocks[i : i + _PARAGRAPHS_PER_PAGE]
            page_text = "\n".join(chunk).strip()
            if page_text:
                pages.append(
                    ParsedPage(
                        page_number=len(pages) + 1,
                        text=page_text,
                    )
                )

        log.debug(
            "docx_parsed",
            filename=filename,
            paragraphs=len(text_blocks),
            logical_pages=len(pages),
        )
        return ParsedDocument(
            filename=filename,
            pages=pages,
            total_pages=len(pages),
            metadata={"parser": "python-docx"},
        )
