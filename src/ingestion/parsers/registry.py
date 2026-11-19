"""
src/ingestion/parsers/registry.py
───────────────────────────────────
Parser registry — maps file extensions to parser instances.

Design:
- Single registry dict built at module load time.
- ParserRegistry.get(filename) returns the correct parser in O(1).
- Adding a new format: create the class, register it here. Done.
- No if/elif chains anywhere in the ingestion pipeline.
"""

from src.core.exceptions import UnsupportedFileTypeError
from src.ingestion.parsers.base import BaseParser
from src.ingestion.parsers.docx_parser import DOCXParser
from src.ingestion.parsers.pdf_parser import PDFParser
from src.ingestion.parsers.text_parser import HTMLParser, TextParser

# ── Registry ──────────────────────────────────────────────────────
# Instances are singletons — parsers are stateless, safe to share.
_PARSERS: dict[str, BaseParser] = {}


def _register(parser: BaseParser) -> None:
    for ext in parser.supported_extensions:
        _PARSERS[ext.lower()] = parser


_register(PDFParser())
_register(DOCXParser())
_register(TextParser())
_register(HTMLParser())


class ParserRegistry:
    """Lookup table for file parsers by extension."""

    @staticmethod
    def get(filename: str) -> BaseParser:
        """
        Return the parser for the given filename's extension.

        Raises:
            UnsupportedFileTypeError: if no parser handles this extension.
        """
        ext = _get_extension(filename)
        parser = _PARSERS.get(ext)
        if not parser:
            supported = sorted(_PARSERS.keys())
            raise UnsupportedFileTypeError(
                f"File type '{ext}' is not supported.",
                detail=f"Supported types: {', '.join(supported)}",
            )
        return parser

    @staticmethod
    def supported_extensions() -> list[str]:
        return sorted(_PARSERS.keys())

    @staticmethod
    def is_supported(filename: str) -> bool:
        return _get_extension(filename) in _PARSERS


def _get_extension(filename: str) -> str:
    import os

    _, ext = os.path.splitext(filename.lower())
    return ext
