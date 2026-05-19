"""
src/ingestion/parsers/docx_parser.py
──────────────────────────────────────
Structure-aware DOCX parser.

What changes vs. the legacy parser
----------------------------------
* Headings are emitted as `block_type="heading"` with the correct
  `heading_level` (1–6) inferred from `Heading 1` / `Heading 2` / …
  paragraph styles. The previous parser flattened them into `# heading`
  Markdown markers that the chunker then discarded.
* A `HeadingStack` tracks the active heading hierarchy so each
  paragraph carries `heading_path` describing where it sits.
* The synthetic "50 paragraphs per page" hack is gone. We open one
  logical page per top-level heading boundary, falling back to one
  page for the whole document if there are no H1 headings.
* Tables are emitted as Markdown tables (one block per table) so they
  survive chunking as a single unit.
"""

from __future__ import annotations

import asyncio
import io

from src.core.exceptions import DocumentProcessingError
from src.core.logger import get_logger
from src.ingestion.parsers.base import (
    BaseParser,
    HeadingStack,
    ParsedBlock,
    ParsedDocument,
)

log = get_logger(__name__)


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
        except DocumentProcessingError:
            raise
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

        blocks: list[ParsedBlock] = []
        stack = HeadingStack()
        page_number = 1
        cursor = 0

        # ── Body paragraphs ───────────────────────────────────────
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            level = _heading_level_from_style(para.style.name)

            start = cursor
            end = cursor + len(text)
            cursor = end + 2

            if level is not None:
                # New top-level heading → bump to a new logical page so
                # the chunker won't merge content across major sections.
                if level == 1 and blocks:
                    page_number += 1
                stack.push(level, text)
                blocks.append(
                    ParsedBlock(
                        text=text,
                        block_type="heading",
                        heading_path=stack.path(),
                        page_number=page_number,
                        char_start=start,
                        char_end=end,
                        heading_level=level,
                    )
                )
            else:
                blocks.append(
                    ParsedBlock(
                        text=text,
                        block_type=(
                            "list_item"
                            if para.style.name.startswith("List")
                            else "paragraph"
                        ),
                        heading_path=stack.path(),
                        page_number=page_number,
                        char_start=start,
                        char_end=end,
                    )
                )

        # ── Tables ────────────────────────────────────────────────
        for table in doc.tables:
            rows = [
                [cell.text.strip() for cell in row.cells]
                for row in table.rows
            ]
            md = _rows_to_markdown(rows)
            if not md:
                continue
            start = cursor
            end = cursor + len(md)
            cursor = end + 2
            blocks.append(
                ParsedBlock(
                    text=md,
                    block_type="table",
                    heading_path=stack.path(),
                    page_number=page_number,
                    char_start=start,
                    char_end=end,
                )
            )

        log.debug(
            "docx_parsed",
            filename=filename,
            blocks=len(blocks),
            headings=sum(1 for b in blocks if b.block_type == "heading"),
            logical_pages=page_number,
        )

        return ParsedDocument(
            filename=filename,
            blocks=blocks,
            total_pages=page_number,
            metadata={"parser": "python-docx"},
        )


def _heading_level_from_style(name: str) -> int | None:
    """
    Map a paragraph style name to a heading level (1–6) or None.
    Handles "Heading 1", "Heading 2", … "Heading 9", and the bare
    "Title" style which we treat as H1.
    """
    if not name:
        return None
    if name.lower() == "title":
        return 1
    if name.startswith("Heading "):
        suffix = name.split()[-1]
        if suffix.isdigit():
            level = int(suffix)
            if 1 <= level <= 6:
                return level
    return None


def _rows_to_markdown(rows: list[list[str]]) -> str:
    cleaned = [
        [c.strip().replace("\n", " ") for c in row]
        for row in rows
        if any(c.strip() for c in row)
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
