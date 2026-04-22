"""
src/ingestion/parsers/base.py
──────────────────────────────
Abstract base for document parsers.

Strategy Pattern:
    Each file type (PDF, DOCX, TXT, HTML) is its own class.
    The ingestion pipeline selects the correct parser by file extension.
    Adding a new format = adding a new class. No if/elif chains.

ParsedDocument:
    The output of every parser. A flat list of pages, each with
    page number and raw text. Downstream chunker works on this
    normalised format, unaware of the source file type.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ParsedPage:
    """A single page of extracted text from a document."""
    page_number: int
    text: str


@dataclass
class ParsedDocument:
    """
    Normalised output from any parser.
    All parsers return this same structure regardless of source format.
    """
    filename: str
    pages: list[ParsedPage] = field(default_factory=list)
    total_pages: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Concatenated text of all pages, separated by newlines."""
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def is_empty(self) -> bool:
        return not self.full_text.strip()


class BaseParser(ABC):
    """
    Contract every document parser must satisfy.

    Usage (via ParserRegistry, not directly):
        parser = ParserRegistry.get_parser("report.pdf")
        doc = await parser.parse(file_bytes, "report.pdf")
    """

    @abstractmethod
    async def parse(self, content: bytes, filename: str) -> ParsedDocument:
        """
        Parse raw file bytes into a ParsedDocument.

        Args:
            content:  Raw bytes of the uploaded file.
            filename: Original filename (used for metadata and logging).

        Returns:
            ParsedDocument with one ParsedPage per logical page.
        """
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """File extensions this parser handles, e.g. ['.pdf']"""
        ...
