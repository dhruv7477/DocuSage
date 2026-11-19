"""
tests/unit/test_page_renderer.py
─────────────────────────────────
Page renderer tests using synthetic in-memory PDFs (no I/O).
"""

import struct

import fitz
import pytest

from src.services.page_renderer import (
    MAX_RENDER_PIXELS,
    RenderedPage,
    render_pdf_page,
)

pytestmark = pytest.mark.unit

SENTENCE = "The employee must present documents from List A within three business days."


def _make_pdf(pages: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def _png_dimensions(png: bytes) -> tuple[int, int]:
    # IHDR chunk: width/height are bytes 16-24 of a valid PNG.
    width, height = struct.unpack(">II", png[16:24])
    return width, height


class TestRenderPdfPage:
    def test_returns_png_bytes(self):
        pdf = _make_pdf(["Hello world page one."])
        rendered = render_pdf_page(pdf, 1)
        assert isinstance(rendered, RenderedPage)
        assert rendered.png[:8] == b"\x89PNG\r\n\x1a\n"
        assert rendered.page_count == 1

    def test_page_count_multi_page(self):
        pdf = _make_pdf(["one", "two", "three"])
        rendered = render_pdf_page(pdf, 2)
        assert rendered.page_count == 3

    def test_highlight_found_for_present_text(self):
        pdf = _make_pdf([SENTENCE])
        rendered = render_pdf_page(pdf, 1, highlight_text=SENTENCE)
        assert rendered.highlight_found

    def test_highlight_not_found_for_absent_text(self):
        pdf = _make_pdf(["Completely different content on this page."])
        rendered = render_pdf_page(
            pdf, 1, highlight_text="The capital of France is Paris and nothing else."
        )
        assert not rendered.highlight_found
        assert rendered.png[:8] == b"\x89PNG\r\n\x1a\n"  # plain page still rendered

    def test_page_zero_raises_value_error(self):
        pdf = _make_pdf(["only page"])
        with pytest.raises(ValueError):
            render_pdf_page(pdf, 0)

    def test_page_overflow_raises_value_error(self):
        pdf = _make_pdf(["only page"])
        with pytest.raises(ValueError):
            render_pdf_page(pdf, 5)

    def test_scale_changes_dimensions(self):
        pdf = _make_pdf(["scaled page"])
        small = render_pdf_page(pdf, 1, scale=1.0)
        large = render_pdf_page(pdf, 1, scale=2.0)
        sw, sh = _png_dimensions(small.png)
        lw, lh = _png_dimensions(large.png)
        assert lw == sw * 2
        assert lh == sh * 2

    def test_empty_highlight_text_is_safe(self):
        pdf = _make_pdf(["content"])
        rendered = render_pdf_page(pdf, 1, highlight_text="   ")
        assert not rendered.highlight_found


class TestRenderSizeCap:
    """A crafted PDF with a huge page must not render a gigapixel image."""

    @staticmethod
    def _huge_page_pdf() -> bytes:
        doc = fitz.open()
        page = doc.new_page(width=14400, height=14400)  # largest PDF page size
        page.insert_text((72, 72), "tiny text on a huge page", fontsize=11)
        data = doc.tobytes()
        doc.close()
        return data

    def test_huge_page_is_rendered_within_the_pixel_cap(self):
        rendered = render_pdf_page(self._huge_page_pdf(), 1, scale=3.0)
        width, height = _png_dimensions(rendered.png)
        assert width * height <= MAX_RENDER_PIXELS
        assert width > 1000  # still a usable image, just smaller

    def test_normal_page_keeps_the_requested_scale(self):
        pdf = _make_pdf(["A4 page"])
        w1, h1 = _png_dimensions(render_pdf_page(pdf, 1, scale=1.0).png)
        w3, h3 = _png_dimensions(render_pdf_page(pdf, 1, scale=3.0).png)
        assert (w3, h3) == (w1 * 3, h1 * 3)
