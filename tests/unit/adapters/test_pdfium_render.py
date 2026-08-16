"""The one place a PDF page becomes pixels.

`handlers/pdf.py` had this inline and `adapters/soffice_renderer.py` needs the
identical thing — but a handler may not be imported by an adapter, so sharing
it means it lives here and the handler delegates. Duplicating six lines of
pdfium driving would have meant a dpi-to-scale fix landing in one of two
places.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from readeverything.adapters.pdfium_render import (
    PIL_AVAILABLE,
    page_count,
    render_page_png,
)
from readeverything.domain.errors import InfrastructureError
from tests.fixtures_pdf import born_digital


def test_page_count_reports_what_the_document_holds() -> None:
    assert page_count(born_digital(["alpha", "beta", "gamma"])) == 3


def test_page_count_on_bytes_that_are_not_a_pdf_raises() -> None:
    """Loudly, because this is adapter-side. The handler's own `_open` catches
    and degrades; an adapter is allowed to fail."""
    with pytest.raises(InfrastructureError):
        page_count(b"not a pdf")


def test_a_rendered_page_is_a_png() -> None:
    png = render_page_png(born_digital(["alpha"]), 1, dpi=72)
    assert png.startswith(b"\x89PNG")


def test_dpi_scales_the_output() -> None:
    """The only dimension assertion this project makes about a rendering:
    doubling the dpi doubles the pixels. No golden image, no pixel comparison.
    """
    data = born_digital(["alpha"])
    small = Image.open(io.BytesIO(render_page_png(data, 1, dpi=72)))
    large = Image.open(io.BytesIO(render_page_png(data, 1, dpi=144)))
    assert large.width == pytest.approx(small.width * 2, abs=2)
    assert large.height == pytest.approx(small.height * 2, abs=2)


def test_a_page_past_the_end_raises_rather_than_rendering_the_last_one() -> None:
    """Silently clamping would hand a caller page 1 labelled page 9."""
    with pytest.raises(InfrastructureError):
        render_page_png(born_digital(["alpha"]), 9)


def test_pages_are_one_indexed_as_a_reader_counts() -> None:
    """pdfium counts from zero; every locator in this library counts from one.
    Off-by-one here would misattribute every page image in the library."""
    with pytest.raises(InfrastructureError):
        render_page_png(born_digital(["alpha", "beta"]), 0)


def test_pil_availability_is_reported_rather_than_assumed() -> None:
    """Pillow is optional (the `images` extra). Callers check this name rather
    than catching an ImportError three frames down."""
    assert PIL_AVAILABLE is True
