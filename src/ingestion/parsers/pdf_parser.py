"""
src/ingestion/parsers/pdf_parser.py
─────────────────────────────────────
PDF parser using pdfplumber.

Why pdfplumber over PyMuPDF or pdfminer:
- Extracts text with accurate layout (respects column order)
- Handles tables — extracts them as structured text
- Pure Python, no binary dependencies
- Handles encrypted/password-protected PDFs gracefully

For scanned PDFs (image-only): pdfplumber returns empty text.
OCR support (pytesseract) can be added as a future enhancement.
"""

import asyncio
import io

from src.core.exceptions import DocumentProcessingError
from src.core.logger import get_logger
from src.ingestion.parsers.base import BaseParser, ParsedDocument, ParsedPage

log = get_logger(__name__)


class PDFParser(BaseParser):

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    async def parse(self, content: bytes, filename: str) -> ParsedDocument:
        """
        Extract text from PDF bytes, one ParsedPage per PDF page.
        Runs in executor to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, self._parse_sync, content, filename
            )
        except Exception as exc:
            raise DocumentProcessingError(
                f"Failed to parse PDF '{filename}'", detail=str(exc)
            ) from exc

    def _parse_sync(self, content: bytes, filename: str) -> ParsedDocument:
        try:
            import pdfplumber
        except ImportError as exc:
            raise DocumentProcessingError(
                "pdfplumber not installed.",
                detail="Run: pip install pdfplumber",
            ) from exc

        pages: list[ParsedPage] = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                # Extract plain text — preserves reading order
                text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

                # Extract tables and append as pipe-delimited text
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        if row:
                            cells = [str(c or "").strip() for c in row]
                            text += "\n" + " | ".join(cells)

                cleaned = self._clean_text(text)
                if cleaned:
                    pages.append(ParsedPage(page_number=i + 1, text=cleaned))

        log.debug(
            "pdf_parsed",
            filename=filename,
            total_pages=total,
            non_empty_pages=len(pages),
        )
        return ParsedDocument(
            filename=filename,
            pages=pages,
            total_pages=total,
            metadata={"parser": "pdfplumber"},
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove excessive whitespace while preserving paragraph breaks."""
        import re
        # Collapse multiple spaces into one
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse more than 2 consecutive newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
