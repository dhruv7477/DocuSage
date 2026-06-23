"""
src/services/page_renderer.py
──────────────────────────────
Server-side PDF page rendering with citation highlighting.

Renders the cited page to a PNG with amber highlight annotations over
the exact text that grounded the answer — a "screenshot of the source".
Replaces the client-side pdf.js viewer: works identically for scanned
PDFs (the highlight just won't match → plain page + flag), needs no
500 KB frontend bundle, and the browser caches the PNG.

All functions here are synchronous and pure (bytes in → bytes out);
the endpoint runs them in a thread executor since fitz blocks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import fitz  # PyMuPDF

from src.core.logger import get_logger

log = get_logger(__name__)

# Amber highlight, matching the UI accent. RGB in 0-1.
_HIGHLIGHT_COLOR = (1.0, 0.84, 0.25)
_HIGHLIGHT_OPACITY = 0.45

_MIN_SENTENCE_CHARS = 20
_MAX_SENTENCES = 12


@dataclass
class RenderedPage:
    png: bytes
    page_count: int
    highlight_found: bool


# Padding (PDF points) around the highlighted passage when cropping.
_CROP_PAD_X = 14.0
_CROP_PAD_Y = 12.0


def render_pdf_page(
    pdf_bytes: bytes,
    page_number: int,          # 1-based (matches chunk payload page_number)
    highlight_text: str | None = None,
    scale: float = 2.0,
    crop: bool = False,
) -> RenderedPage:
    """
    Render one PDF page to PNG, highlighting `highlight_text` if found.

    When `crop` is True and the passage is located, the pixmap is clipped
    to the highlighted region (+ padding) — a tight "screenshot of just
    the cited paragraph" rather than the whole page. If the passage can't
    be located, crop is ignored and the full page is returned.

    Raises:
        ValueError: page_number out of range (endpoint maps to 422).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if not 1 <= page_number <= doc.page_count:
            raise ValueError(
                f"Page {page_number} out of range (document has {doc.page_count} pages)."
            )
        page = doc[page_number - 1]

        any_rects = False
        clip: fitz.Rect | None = None
        if highlight_text:
            rects = _find_highlight_rects(page, highlight_text)
            any_rects = bool(rects)
            for rect in rects:
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=_HIGHLIGHT_COLOR)
                annot.set_opacity(_HIGHLIGHT_OPACITY)
                annot.update()

            if crop and rects:
                union = fitz.Rect(rects[0])
                for rect in rects[1:]:
                    union |= rect
                clip = fitz.Rect(
                    union.x0 - _CROP_PAD_X,
                    union.y0 - _CROP_PAD_Y,
                    union.x1 + _CROP_PAD_X,
                    union.y1 + _CROP_PAD_Y,
                ) & page.rect

        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), annots=True, clip=clip)
        return RenderedPage(
            png=pix.tobytes("png"),
            page_count=doc.page_count,
            highlight_found=any_rects,
        )
    finally:
        doc.close()


# ── Highlight location ────────────────────────────────────────────


def _find_highlight_rects(page: "fitz.Page", text: str) -> list["fitz.Rect"]:
    """
    Locate the chunk text on the page with graceful degradation:

    1. Sentence-by-sentence `search_for` (PyMuPDF matches across line
       wraps). Long sentences are trimmed to dodge hyphenation/ligature
       mismatches at the tail.
    2. If nothing matched at all: 5-word shingles from the first two
       sentences — finds at least the opening of the passage.
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", normalized)
        if len(s.strip()) >= _MIN_SENTENCE_CHARS
    ][:_MAX_SENTENCES]
    if not sentences:
        sentences = [normalized[:120]]

    rects: list[fitz.Rect] = []
    for sentence in sentences:
        found = page.search_for(sentence[:120])
        if not found and len(sentence) > 60:
            found = page.search_for(sentence[:60])
        rects.extend(found)

    if rects:
        return rects

    # Last resort: 5-word shingles from the first two sentences.
    words = " ".join(sentences[:2]).split()
    for i in range(0, max(1, len(words) - 4), 5):
        shingle = " ".join(words[i : i + 5])
        found = page.search_for(shingle)
        if found:
            return list(found)

    log.debug("highlight_text_not_found_on_page", preview=normalized[:80])
    return []
