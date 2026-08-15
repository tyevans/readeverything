"""PDFs generated at test time rather than committed as binaries.

Committed binary fixtures rot: nobody can read a diff of them, nobody can tell
what a change to one did, and a corrupted byte looks identical to an
intentional edit. Generating them keeps the input to every PDF test readable in
the same file as the test.

reportlab is a dev dependency only. Nothing under `src/` imports it.
"""

from __future__ import annotations

import io
from collections.abc import Sequence

from PIL import Image
from reportlab.lib.pagesizes import LETTER  # type: ignore[import-untyped]  # no stubs published
from reportlab.lib.utils import ImageReader  # type: ignore[import-untyped]
from reportlab.pdfgen import canvas  # type: ignore[import-untyped]


def born_digital(pages: Sequence[str]) -> bytes:
    """A PDF with a real text layer, one string per page."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    for text in pages:
        pdf.drawString(72, 720, text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def scanned_like(pages: int = 1) -> bytes:
    """Image content and NO text layer — what a scan looks like to an extractor.

    Indistinguishable from `blank()` through the text layer: both report zero
    characters and empty text. `page.get_objects()` is what tells them apart,
    and that difference is the whole point of the scanned-PDF handling.
    """
    image = Image.new("RGB", (400, 200), (30, 30, 30))
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    for _ in range(pages):
        pdf.drawImage(ImageReader(image), 72, 500, width=300, height=150)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def blank(pages: int = 1) -> bytes:
    """Genuinely empty pages: no text, no objects."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    for _ in range(pages):
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def many_pages(count: int) -> bytes:
    """For asserting the locator map's size and the barrier count."""
    return born_digital([f"This is page {i + 1}." for i in range(count)])


def mixed(pages: Sequence[str | None]) -> bytes:
    """Pages that alternate between real text and no text layer.

    `None` draws an image and no text — a scanned page. A string draws that
    string. This is the adversarial shape for the locator map: an empty page
    between two full ones must still own at least one character, or its
    CharSpan is zero-width and raises.
    """
    image = Image.new("RGB", (400, 200), (30, 30, 30))
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    for text in pages:
        if text is None:
            pdf.drawImage(ImageReader(image), 72, 500, width=300, height=150)
        else:
            pdf.drawString(72, 720, text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()
