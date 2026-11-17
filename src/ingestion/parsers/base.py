"""
src/ingestion/parsers/base.py
──────────────────────────────
Abstract base for document parsers.

Strategy Pattern:
    Each file type (PDF, DOCX, TXT, HTML, MD) is its own class.
    The ingestion pipeline selects the correct parser by file extension.

Output model
------------
Every parser produces a `ParsedDocument` containing a flat list of
`ParsedBlock`s. Each block carries:

* `block_type`     — heading | paragraph | list_item | table | caption
* `heading_path`   — the heading stack at the time this block was
                     emitted, e.g. ["Chapter 3", "3.2 Refunds"].
                     The chunker uses this to keep semantically related
                     content together AND to enrich each chunk's
                     embedding text so retrieval is context-aware.
* `page_number`    — best-effort source page (1-indexed). For DOCX/MD
                     where there are no real pages we synthesize one
                     "section page" per top-level heading boundary.
* `char_start` / `char_end` — character offsets into `full_text` so
                     the UI can highlight the exact passage that
                     produced a given citation.

The legacy `ParsedPage` view is preserved as a derived property so
existing chunker / test code that reads `.pages` continues to work
during the Phase 2 transition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

BlockType = Literal["heading", "paragraph", "list_item", "table", "caption"]


@dataclass
class ParsedBlock:
    """A single semantically-meaningful unit of extracted text."""

    text: str
    block_type: BlockType
    heading_path: list[str] = field(default_factory=list)
    page_number: int = 1
    char_start: int = 0
    char_end: int = 0
    # Only populated for `block_type == "heading"`.
    heading_level: int | None = None


@dataclass
class ParsedPage:
    """Legacy view: blocks grouped by `page_number` for older chunker code."""

    page_number: int
    text: str


@dataclass
class ParsedDocument:
    """
    Normalised output from any parser.
    Block-first; the `pages` property is derived for backward compat.
    """

    filename: str
    blocks: list[ParsedBlock] = field(default_factory=list)
    total_pages: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Concatenated text of every block, separated by blank lines."""
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())

    @property
    def is_empty(self) -> bool:
        return not self.full_text.strip()

    @property
    def pages(self) -> list[ParsedPage]:
        """Group blocks by `page_number` — preserves the legacy interface."""
        if not self.blocks:
            return []
        by_page: dict[int, list[str]] = {}
        for b in self.blocks:
            by_page.setdefault(b.page_number, []).append(b.text)
        return [
            ParsedPage(page_number=p, text="\n\n".join(parts).strip())
            for p, parts in sorted(by_page.items())
        ]


class BaseParser(ABC):
    """
    Contract every document parser must satisfy.

    Usage (via ParserRegistry, not directly):
        parser = ParserRegistry.get("report.pdf")
        doc = await parser.parse(file_bytes, "report.pdf")
    """

    @abstractmethod
    async def parse(self, content: bytes, filename: str) -> ParsedDocument:
        """Parse raw file bytes into a `ParsedDocument` of `ParsedBlock`s."""
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """File extensions this parser handles, e.g. ['.pdf']"""
        ...


# ── Helper: heading-stack builder shared by parsers ──────────────────


class HeadingStack:
    """
    Maintains the current heading hierarchy as a parser walks a document.

    Calling `push(level, text)` updates the stack so that the path is
    always the most-recent heading at each level <= the new heading's
    level. `path()` returns the stack's text values for embedding into
    block.heading_path.

    Example trace:
        push(1, "Chapter 3")    → path == ["Chapter 3"]
        push(2, "3.1 Intro")    → path == ["Chapter 3", "3.1 Intro"]
        push(2, "3.2 Refunds")  → path == ["Chapter 3", "3.2 Refunds"]
        push(1, "Chapter 4")    → path == ["Chapter 4"]
    """

    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []

    def push(self, level: int, text: str) -> None:
        # Drop any deeper headings — entering a same-or-shallower heading
        # closes everything below it.
        self._stack = [(lvl, txt) for (lvl, txt) in self._stack if lvl < level]
        self._stack.append((level, text))

    def path(self) -> list[str]:
        return [txt for (_lvl, txt) in self._stack]
