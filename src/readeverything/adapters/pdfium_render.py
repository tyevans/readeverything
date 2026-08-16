"""Turning a PDF page into pixels, in one place.

`handlers/pdf.py` grew this first, and it was right there: rendering a page is
what a PDF handler does. It stopped being only a handler's business when
`adapters/soffice_renderer.py` needed the identical thing — a converter that
produces a PDF and then hands back page images renders exactly the same way,
and the alternative to sharing was six duplicated lines of pdfium driving in
which a dpi-to-scale fix would land in one of two places.

An adapter cannot import a handler (the layering contract forbids it and the
direction is right), so the shared code lives here and `handlers/pdf.py`
delegates to it.

Two granularities, because there are two callers. The handler already holds an
open document and wants one page from it, so `render_pil`/`render_png` take a
`PdfPage`. The renderer holds converted BYTES and no document, so
`page_count`/`render_page_png` open and close one around the call.

These RAISE on bad input, matching `pdfium_probe.open_document` and unlike
`PdfHandler._open`. An adapter is allowed to fail loudly; a handler must
degrade instead, and it is the handler that catches. Do not unify the two.
"""

from __future__ import annotations

import importlib.util
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Annotation-only, exactly as `handlers/pdf.py` does it: `from __future__
    # import annotations` keeps this out of runtime evaluation, so it costs
    # nothing when Pillow is absent. `pdfium`'s own `to_pil()` is the only
    # thing that touches Pillow at runtime, and only once `PIL_AVAILABLE`.
    from PIL import Image

import pypdfium2 as pdfium  # type: ignore[import-untyped]  # pypdfium2 ships no py.typed

from readeverything.domain.errors import InfrastructureError

#: Pillow is optional here (unlike `handlers/image.py`, which requires it
#: outright): rendering degrades without it rather than the whole module
#: failing to import. Checked by name rather than imported, so this module
#: never itself depends on Pillow.
PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None

#: pdfium's unit is a scale factor against 72 points-per-inch, which is what
#: "dpi" means for a PDF. One named constant rather than a bare 72 in two
#: expressions.
POINTS_PER_INCH = 72


def render_pil(page: pdfium.PdfPage, dpi: int) -> Image.Image:
    """The page rendered, not yet encoded.

    `Image.Image` is a type-checking-only name here; see the `TYPE_CHECKING`
    import above.
    """
    bitmap = page.render(scale=dpi / POINTS_PER_INCH)
    try:
        return bitmap.to_pil()  # type: ignore[no-any-return]  # pypdfium2 ships no py.typed
    finally:
        bitmap.close()


def encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_png(page: pdfium.PdfPage, dpi: int) -> bytes:
    return encode_png(render_pil(page, dpi))


def _open(data: bytes) -> pdfium.PdfDocument:
    try:
        return pdfium.PdfDocument(data)
    except Exception as exc:
        raise InfrastructureError(f"could not open as a PDF: {exc}") from exc


def page_count(data: bytes) -> int:
    """How many pages these PDF bytes hold."""
    document = _open(data)
    try:
        return len(document)
    finally:
        document.close()


def _checked(document: pdfium.PdfDocument, page: int) -> pdfium.PdfPage:
    if page < 1 or page > len(document):
        raise InfrastructureError(
            f"page {page} does not exist; the document has {len(document)} page(s)"
        )
    return document[page - 1]


def page_text(data: bytes, page: int) -> str:
    """One page's text layer.

    This is how a *converted* document is read: the converter produces a PDF,
    and the PDF's own text layer is the text. No second extractor, no OCR. It
    is the same extraction `handlers/pdf.py` performs, at the bytes-in
    granularity `adapters/soffice_renderer.py` needs.
    """
    document = _open(data)
    try:
        textpage = _checked(document, page).get_textpage()
        try:
            return str(textpage.get_text_range(index=0, count=-1))
        finally:
            textpage.close()
    finally:
        document.close()


def render_page_png(data: bytes, page: int, *, dpi: int = 150) -> bytes:
    """One page of these PDF bytes as a PNG. `page` is 1-indexed.

    Out of range RAISES rather than clamping. Clamping would hand a caller
    page 1 labelled page 9, which is a false claim about the document rather
    than a failure to answer.
    """
    if not PIL_AVAILABLE:
        raise InfrastructureError("cannot render a page: Pillow is not installed")
    document = _open(data)
    try:
        return render_png(_checked(document, page), dpi)
    finally:
        document.close()
