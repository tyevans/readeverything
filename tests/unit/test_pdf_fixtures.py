"""Guards the PDF fixture module itself.

If `scanned_like` ever stops differing from `blank`, every later scanned-PDF
test silently becomes a blank-page test and still passes.
"""

from __future__ import annotations

import pypdfium2 as pdfium  # type: ignore[import-untyped]  # pypdfium2 ships no py.typed marker

from tests.fixtures_pdf import blank, scanned_like


def test_the_scanned_and_blank_fixtures_differ_only_in_page_objects() -> None:
    """Both report zero characters. That is the point: the text layer cannot
    tell them apart, so the handler must use something else."""
    scan = pdfium.PdfDocument(scanned_like())
    empty = pdfium.PdfDocument(blank())

    assert scan[0].get_textpage().count_chars() == 0
    assert empty[0].get_textpage().count_chars() == 0

    assert len(list(scan[0].get_objects())) > 0
    assert len(list(empty[0].get_objects())) == 0
