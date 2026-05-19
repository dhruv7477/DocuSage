"""
src/ingestion/parsers/text_parser.py
──────────────────────────────────────
Parsers for plain-text formats: TXT, Markdown, HTML.

* `TextParser` handles `.txt` (paragraph splits) and `.md / .markdown`
  (real Markdown via `markdown-it-py` so headings / lists / tables
  become first-class blocks).
* `HTMLParser` walks the BeautifulSoup tree, treating h1–h6 as headings
  and emitting paragraphs / list items / tables as separate blocks.

Both produce `ParsedBlock`s with `heading_path` populated so chunks
keep the surrounding section context when they hit the embedder.
"""

from __future__ import annotations

import asyncio
import re

from src.core.exceptions import DocumentProcessingError
from src.core.logger import get_logger
from src.ingestion.parsers.base import (
    BaseParser,
    HeadingStack,
    ParsedBlock,
    ParsedDocument,
)

log = get_logger(__name__)


# Plain-text paragraphs are split on blank lines.
_PARAGRAPH_SPLIT = re.compile(r"\n{2,}")


class TextParser(BaseParser):
    """Handles `.txt` (paragraph mode) and `.md / .markdown` (real parser)."""

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

        ext = filename.lower().rsplit(".", 1)[-1]
        if ext in ("md", "markdown"):
            blocks = self._parse_markdown(text)
            parser_name = "markdown-it-py"
        else:
            blocks = self._parse_plain_text(text)
            parser_name = "text"

        log.debug(
            "text_parsed",
            filename=filename,
            blocks=len(blocks),
            parser=parser_name,
        )
        return ParsedDocument(
            filename=filename,
            blocks=blocks,
            total_pages=max(b.page_number for b in blocks) if blocks else 0,
            metadata={"parser": parser_name},
        )

    # ── Markdown ──────────────────────────────────────────────────

    def _parse_markdown(self, text: str) -> list[ParsedBlock]:
        try:
            from markdown_it import MarkdownIt
        except ImportError:
            log.warning("markdown_it_missing_falling_back_to_plain")
            return self._parse_plain_text(text)

        md = MarkdownIt("commonmark").enable("table")
        tokens = md.parse(text)

        blocks: list[ParsedBlock] = []
        stack = HeadingStack()
        page_number = 1
        cursor = 0
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.type == "heading_open":
                level = int(tok.tag[1])  # h1 → 1, h2 → 2, …
                inline = tokens[i + 1]
                heading_text = inline.content.strip()
                if heading_text:
                    if level == 1 and blocks:
                        page_number += 1
                    stack.push(level, heading_text)
                    start, end, cursor = _advance_cursor(cursor, heading_text)
                    blocks.append(
                        ParsedBlock(
                            text=heading_text,
                            block_type="heading",
                            heading_path=stack.path(),
                            page_number=page_number,
                            char_start=start,
                            char_end=end,
                            heading_level=level,
                        )
                    )
                i += 3   # heading_open, inline, heading_close
                continue

            if tok.type == "paragraph_open":
                inline = tokens[i + 1]
                para_text = inline.content.strip()
                if para_text:
                    start, end, cursor = _advance_cursor(cursor, para_text)
                    blocks.append(
                        ParsedBlock(
                            text=para_text,
                            block_type="paragraph",
                            heading_path=stack.path(),
                            page_number=page_number,
                            char_start=start,
                            char_end=end,
                        )
                    )
                i += 3
                continue

            if tok.type == "bullet_list_open" or tok.type == "ordered_list_open":
                # Walk until the matching close, emitting one block per item
                depth = 1
                j = i + 1
                while j < len(tokens) and depth > 0:
                    if tokens[j].type in ("bullet_list_open", "ordered_list_open"):
                        depth += 1
                    elif tokens[j].type in ("bullet_list_close", "ordered_list_close"):
                        depth -= 1
                        if depth == 0:
                            break
                    elif tokens[j].type == "inline" and tokens[j].level == tok.level + 2:
                        item_text = tokens[j].content.strip()
                        if item_text:
                            start, end, cursor = _advance_cursor(cursor, item_text)
                            blocks.append(
                                ParsedBlock(
                                    text=item_text,
                                    block_type="list_item",
                                    heading_path=stack.path(),
                                    page_number=page_number,
                                    char_start=start,
                                    char_end=end,
                                )
                            )
                    j += 1
                i = j + 1
                continue

            if tok.type == "table_open":
                # Reconstruct the original Markdown table as one block —
                # markdown-it preserves source positions on tokens via .map.
                end_idx = i
                depth = 1
                while end_idx < len(tokens) - 1 and depth > 0:
                    end_idx += 1
                    if tokens[end_idx].type == "table_open":
                        depth += 1
                    elif tokens[end_idx].type == "table_close":
                        depth -= 1
                table_lines = self._reconstruct_table(text, tok, tokens[end_idx])
                if table_lines:
                    start, end, cursor = _advance_cursor(cursor, table_lines)
                    blocks.append(
                        ParsedBlock(
                            text=table_lines,
                            block_type="table",
                            heading_path=stack.path(),
                            page_number=page_number,
                            char_start=start,
                            char_end=end,
                        )
                    )
                i = end_idx + 1
                continue

            i += 1

        return blocks

    @staticmethod
    def _reconstruct_table(source: str, open_tok, close_tok) -> str:
        """Slice the original source between the table's source lines."""
        if not (open_tok.map and close_tok.map):
            return ""
        start_line, _ = open_tok.map
        _, end_line = close_tok.map
        lines = source.splitlines()
        return "\n".join(lines[start_line:end_line]).strip()

    # ── Plain text ────────────────────────────────────────────────

    def _parse_plain_text(self, text: str) -> list[ParsedBlock]:
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"\t", "    ", text)
        paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
        blocks: list[ParsedBlock] = []
        cursor = 0
        for para in paragraphs:
            start, end, cursor = _advance_cursor(cursor, para)
            blocks.append(
                ParsedBlock(
                    text=para,
                    block_type="paragraph",
                    heading_path=[],
                    page_number=1,
                    char_start=start,
                    char_end=end,
                )
            )
        return blocks


class HTMLParser(BaseParser):
    """Walk the DOM and emit headings, paragraphs, list items, and tables."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".html", ".htm"]

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
                f"Failed to parse HTML '{filename}'", detail=str(exc)
            ) from exc

    def _parse_sync(self, content: bytes, filename: str) -> ParsedDocument:
        try:
            from bs4 import BeautifulSoup, Tag
        except ImportError:
            # Fallback: strip tags with regex; we lose structure but at
            # least chunks aren't completely empty.
            text = re.sub(r"<[^>]+>", " ", content.decode("utf-8", errors="replace"))
            text = re.sub(r"\s+", " ", text).strip()
            blocks: list[ParsedBlock] = []
            cursor = 0
            if text:
                start, end, cursor = _advance_cursor(cursor, text)
                blocks.append(
                    ParsedBlock(
                        text=text,
                        block_type="paragraph",
                        heading_path=[],
                        page_number=1,
                        char_start=start,
                        char_end=end,
                    )
                )
            return ParsedDocument(
                filename=filename,
                blocks=blocks,
                total_pages=1 if blocks else 0,
                metadata={"parser": "html-regex-fallback"},
            )

        soup = BeautifulSoup(content, "html.parser")
        for tag_name in ("script", "style", "nav", "footer", "header", "aside"):
            for tag in soup(tag_name):
                tag.decompose()

        blocks: list[ParsedBlock] = []
        stack = HeadingStack()
        page_number = 1
        cursor = 0

        body = soup.body or soup
        for tag in body.descendants:
            if not isinstance(tag, Tag):
                continue
            name = tag.name.lower() if tag.name else ""

            if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(name[1])
                txt = tag.get_text(" ", strip=True)
                if not txt:
                    continue
                if level == 1 and blocks:
                    page_number += 1
                stack.push(level, txt)
                start, end, cursor = _advance_cursor(cursor, txt)
                blocks.append(
                    ParsedBlock(
                        text=txt,
                        block_type="heading",
                        heading_path=stack.path(),
                        page_number=page_number,
                        char_start=start,
                        char_end=end,
                        heading_level=level,
                    )
                )
            elif name == "p":
                txt = tag.get_text(" ", strip=True)
                if txt:
                    start, end, cursor = _advance_cursor(cursor, txt)
                    blocks.append(
                        ParsedBlock(
                            text=txt,
                            block_type="paragraph",
                            heading_path=stack.path(),
                            page_number=page_number,
                            char_start=start,
                            char_end=end,
                        )
                    )
            elif name == "li":
                txt = tag.get_text(" ", strip=True)
                if txt:
                    start, end, cursor = _advance_cursor(cursor, txt)
                    blocks.append(
                        ParsedBlock(
                            text=txt,
                            block_type="list_item",
                            heading_path=stack.path(),
                            page_number=page_number,
                            char_start=start,
                            char_end=end,
                        )
                    )
            elif name == "table":
                rows = []
                for tr in tag.find_all("tr"):
                    rows.append([
                        cell.get_text(" ", strip=True)
                        for cell in tr.find_all(["th", "td"])
                    ])
                md = _rows_to_markdown(rows)
                if md:
                    start, end, cursor = _advance_cursor(cursor, md)
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
            "html_parsed",
            filename=filename,
            blocks=len(blocks),
            headings=sum(1 for b in blocks if b.block_type == "heading"),
        )
        return ParsedDocument(
            filename=filename,
            blocks=blocks,
            total_pages=page_number if blocks else 0,
            metadata={"parser": "beautifulsoup4"},
        )


# ── Shared helpers ────────────────────────────────────────────────


def _advance_cursor(cursor: int, text: str) -> tuple[int, int, int]:
    """Return (start, end, new_cursor) for a block of `text`."""
    start = cursor
    end = cursor + len(text)
    new_cursor = end + 2  # match the "\n\n" join in `full_text`
    return start, end, new_cursor


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
