"""
src/ingestion/parsers/pdf_parser.py
─────────────────────────────────────
Structure-aware PDF parser.

Pipeline per page:
    1. PyMuPDF (`fitz`) extracts blocks with font sizes — we cluster
       sizes to infer heading levels (top-3 sizes → H1/H2/H3).
    2. If the page yields fewer than `_OCR_FALLBACK_THRESHOLD` characters
       of text, we render the page to a PIL image and run pytesseract.
       Skipping this fallback was the single biggest cause of "scanned
       PDF uploads silently produced zero chunks" in the previous
       implementation.
    3. Tables are extracted via `pdfplumber.find_tables()` and serialized
       as Markdown tables — Markdown survives chunking better than
       pipe-separated rows and is also better understood by the LLM.
    4. Recurring headers/footers (lines that appear on ≥30% of pages)
       are stripped to reduce embedding noise.

Output: a `ParsedDocument` of `ParsedBlock`s with full heading_path,
char offsets, and page numbers.
"""

from __future__ import annotations

import asyncio
import io
import re
from collections import Counter
from dataclasses import dataclass

from src.core.exceptions import DocumentProcessingError
from src.core.logger import get_logger
from src.ingestion.parsers.base import (
    BaseParser,
    HeadingStack,
    ParsedBlock,
    ParsedDocument,
)

log = get_logger(__name__)


# Minimum characters a page must produce via PyMuPDF before we fall back
# to OCR. Tuned low because some pages legitimately contain little text
# (e.g. cover pages) and OCR is expensive.
_OCR_FALLBACK_THRESHOLD = 20

# Detect repeating header/footer lines that appear on most pages.
_HEADER_FOOTER_MIN_FRACTION = 0.3

# Page-number-only lines we always strip (e.g. "Page 3 of 50", "- 12 -").
_PAGE_NUMBER_PATTERNS = (
    re.compile(r"^\s*(?:page\s+)?\d+(?:\s*(?:of|/)\s*\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*[-–—]\s*\d+\s*[-–—]\s*$"),
)


@dataclass
class _PageRaw:
    page_number: int
    blocks: list[tuple[str, float]]  # (text, dominant font size)


class PDFParser(BaseParser):
    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    async def parse(self, content: bytes, filename: str) -> ParsedDocument:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, self._parse_sync, content, filename)
        except DocumentProcessingError:
            raise
        except Exception as exc:
            raise DocumentProcessingError(
                f"Failed to parse PDF '{filename}'", detail=str(exc)
            ) from exc

    # ── Sync implementation ────────────────────────────────────────

    def _parse_sync(self, content: bytes, filename: str) -> ParsedDocument:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise DocumentProcessingError(
                "pymupdf not installed.",
                detail="Run: pip install pymupdf",
            ) from exc

        # Pass 1: extract raw blocks per page (with font sizes)
        page_blobs: list[_PageRaw] = []
        with fitz.open(stream=content, filetype="pdf") as pdf:
            total = len(pdf)
            for i in range(total):
                page = pdf[i]
                raw = self._extract_blocks(page)
                if not raw or sum(len(t) for t, _ in raw) < _OCR_FALLBACK_THRESHOLD:
                    ocr_text = self._ocr_page(page)
                    if ocr_text:
                        raw = [(ocr_text, 12.0)]  # neutral font size
                page_blobs.append(_PageRaw(page_number=i + 1, blocks=raw))

        # Pass 2: identify recurring headers/footers across pages
        repeating = self._find_repeating_lines(page_blobs)

        # Pass 3: optionally extract Markdown tables via pdfplumber.
        # Tables are appended after the body blocks for each page so they
        # don't disrupt heading hierarchy.
        try:
            tables_per_page = self._extract_tables_markdown(content)
        except Exception as exc:
            log.warning("pdf_table_extraction_failed", error=str(exc))
            tables_per_page = {}

        # Pass 4: cluster all (non-table) font sizes to derive heading
        # thresholds for this specific document. PDFs vary wildly so a
        # global heuristic doesn't work — we adapt per file.
        heading_thresholds = self._derive_heading_thresholds(page_blobs)

        # Pass 5: emit ParsedBlocks with heading_path + char offsets
        blocks: list[ParsedBlock] = []
        stack = HeadingStack()
        cursor = 0  # running char offset across the full document text

        for page in page_blobs:
            for raw_text, font_size in page.blocks:
                cleaned = self._clean_line(raw_text, repeating)
                if not cleaned:
                    continue

                level = self._heading_level(font_size, heading_thresholds)
                start = cursor
                end = cursor + len(cleaned)
                cursor = end + 2  # account for the join "\n\n"

                if level is not None:
                    stack.push(level, cleaned)
                    blocks.append(
                        ParsedBlock(
                            text=cleaned,
                            block_type="heading",
                            heading_path=stack.path(),
                            page_number=page.page_number,
                            char_start=start,
                            char_end=end,
                            heading_level=level,
                        )
                    )
                else:
                    blocks.append(
                        ParsedBlock(
                            text=cleaned,
                            block_type="paragraph",
                            heading_path=stack.path(),
                            page_number=page.page_number,
                            char_start=start,
                            char_end=end,
                        )
                    )

            # Append any tables for this page
            for tbl_md in tables_per_page.get(page.page_number, []):
                start = cursor
                end = cursor + len(tbl_md)
                cursor = end + 2
                blocks.append(
                    ParsedBlock(
                        text=tbl_md,
                        block_type="table",
                        heading_path=stack.path(),
                        page_number=page.page_number,
                        char_start=start,
                        char_end=end,
                    )
                )

        log.debug(
            "pdf_parsed",
            filename=filename,
            total_pages=len(page_blobs),
            blocks=len(blocks),
            headings=sum(1 for b in blocks if b.block_type == "heading"),
            tables=sum(1 for b in blocks if b.block_type == "table"),
        )

        return ParsedDocument(
            filename=filename,
            blocks=blocks,
            total_pages=len(page_blobs),
            metadata={"parser": "pymupdf+ocr+pdfplumber"},
        )

    # ── Per-page extraction helpers ───────────────────────────────

    @staticmethod
    def _extract_blocks(page) -> list[tuple[str, float]]:
        """
        Group contiguous spans of the same font size into one block.
        Returns [(text, dominant_font_size)] in reading order.
        """
        out: list[tuple[str, float]] = []
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:  # 0 = text, 1 = image
                continue
            for line in block.get("lines", []):
                line_text = "".join(span["text"] for span in line.get("spans", []))
                if not line_text.strip():
                    continue
                # Use the line's largest span size — robust to mid-line
                # superscript/subscript noise.
                sizes = [
                    span["size"] for span in line.get("spans", []) if span.get("size")
                ]
                if not sizes:
                    continue
                out.append((line_text.strip(), max(sizes)))
        return out

    @staticmethod
    def _ocr_page(page) -> str:
        """
        Render the page to a 300 DPI image and OCR it via Tesseract.
        Returns "" if Tesseract or Pillow is missing — that scenario
        is logged but not fatal because most non-scanned PDFs don't
        need OCR at all.
        """
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            log.warning("ocr_skipped_missing_deps")
            return ""

        try:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes(
                "RGB" if pix.alpha == 0 else "RGBA",
                (pix.width, pix.height),
                pix.samples,
            )
            return pytesseract.image_to_string(img).strip()
        except Exception as exc:
            log.warning("ocr_failed", error=str(exc))
            return ""

    @staticmethod
    def _extract_tables_markdown(content: bytes) -> dict[int, list[str]]:
        """
        Run pdfplumber's table detector and serialize each table as
        Markdown. Markdown tables survive the chunker far better than
        the previous pipe-separated rows because RecursiveChunker treats
        a table as one paragraph.
        """
        try:
            import pdfplumber
        except ImportError:
            return {}

        out: dict[int, list[str]] = {}
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                tables = page.find_tables() or []
                page_tables: list[str] = []
                for tbl in tables:
                    rows = tbl.extract() or []
                    if not rows or not rows[0]:
                        continue
                    md = _rows_to_markdown(rows)
                    if md:
                        page_tables.append(md)
                if page_tables:
                    out[i] = page_tables
        return out

    # ── Document-level analysis ───────────────────────────────────

    @staticmethod
    def _find_repeating_lines(pages: list[_PageRaw]) -> set[str]:
        """
        Lines that appear on ≥`_HEADER_FOOTER_MIN_FRACTION` of pages are
        almost certainly running headers / footers / page numbers.
        """
        if len(pages) < 4:
            # On very short docs the heuristic is unreliable — bail out.
            return set()
        counts: Counter[str] = Counter()
        for p in pages:
            seen_on_page: set[str] = set()
            for text, _ in p.blocks:
                # Only consider short lines for header/footer detection.
                # A genuine recurring section header is fine to keep; we
                # specifically want to strip page chrome.
                if len(text) <= 80:
                    seen_on_page.add(text)
            counts.update(seen_on_page)
        threshold = max(2, int(len(pages) * _HEADER_FOOTER_MIN_FRACTION))
        return {text for text, n in counts.items() if n >= threshold}

    @staticmethod
    def _derive_heading_thresholds(pages: list[_PageRaw]) -> dict[int, float]:
        """
        Cluster the document's font sizes; sizes above the body-text
        median become H3+ (smallest jump), H2 (medium jump), H1 (largest
        jump). Returns {level → minimum size} so we can classify each
        line in one comparison.
        """
        sizes = [size for p in pages for _, size in p.blocks]
        if not sizes:
            return {}
        # Body size = mode (most common rounded size)
        rounded = [round(s) for s in sizes]
        body = Counter(rounded).most_common(1)[0][0]
        # Distinct heading-candidate sizes are anything notably above body
        candidates = sorted({s for s in sizes if s >= body + 1.5}, reverse=True)
        if not candidates:
            return {}
        # Top three biggest sizes become H1/H2/H3
        thresholds: dict[int, float] = {}
        for level, size in enumerate(candidates[:3], start=1):
            thresholds[level] = size
        return thresholds

    @staticmethod
    def _heading_level(font_size: float, thresholds: dict[int, float]) -> int | None:
        """Return the heading level for a given line, or None for body text."""
        # Iterate H1 → H3 in order, return the first level whose threshold
        # this line meets (or exceeds within 0.5pt of rounding noise).
        for level in (1, 2, 3):
            t = thresholds.get(level)
            if t is not None and font_size + 0.5 >= t:
                return level
        return None

    @staticmethod
    def _clean_line(text: str, repeating: set[str]) -> str:
        text = text.strip()
        if not text:
            return ""
        if text in repeating:
            return ""
        for pat in _PAGE_NUMBER_PATTERNS:
            if pat.match(text):
                return ""
        # Collapse internal runs of whitespace
        text = re.sub(r"[ \t]+", " ", text)
        return text


# ── Markdown table helper ─────────────────────────────────────────


def _rows_to_markdown(rows: list[list[str | None]]) -> str:
    """Render a list of rows as a Markdown table. Skips empty rows."""
    cleaned = [
        [(cell or "").strip().replace("\n", " ") for cell in row]
        for row in rows
        if any((cell or "").strip() for cell in row)
    ]
    if not cleaned:
        return ""
    n_cols = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (n_cols - len(r)) for r in cleaned]
    header = cleaned[0]
    body = cleaned[1:] if len(cleaned) > 1 else []
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * n_cols) + " |",
    ]
    lines.extend("| " + " | ".join(r) + " |" for r in body)
    return "\n".join(lines)
