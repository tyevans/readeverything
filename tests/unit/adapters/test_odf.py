"""ODF read with lxml, because odfpy is unmaintained."""

from __future__ import annotations

from readeverything.adapters.odf import odf_blocks, odf_sheets, odf_slides
from tests.fixtures_office import odp_bytes, ods_bytes, odt_bytes


def test_headings_and_paragraphs_come_back_in_document_order() -> None:
    blocks = odf_blocks(odt_bytes())
    assert [(b.level, b.text) for b in blocks] == [
        (1, "Alpha"),
        (0, "The body of the alpha section."),
        (1, "Bravo"),
        (0, "The body of the bravo section."),
    ]


def test_a_heading_carries_its_outline_level() -> None:
    """A section outline is a tree. Flattening every heading to one level would
    make a sub-section look like a peer of the section containing it.
    """
    blocks = odf_blocks(odt_bytes(blocks=(("h", "Top"), ("p", "Body"))))
    assert blocks[0].level == 1
    assert blocks[1].level == 0


def test_bytes_that_are_not_an_odf_package_yield_nothing_rather_than_raising() -> None:
    """Every caller is a handler, and a handler degrades."""
    assert odf_blocks(b"not a zip") == ()
    assert odf_slides(b"not a zip") == ()
    assert odf_sheets(b"not a zip") == ()


def test_a_package_whose_content_is_not_xml_yields_nothing() -> None:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as package:
        package.writestr("content.xml", "this is not xml <<<")
    assert odf_blocks(buffer.getvalue()) == ()


def test_slides_come_back_one_tuple_per_page() -> None:
    slides = odf_slides(odp_bytes())
    assert len(slides) == 2
    assert slides[0][0] == "Opening position"
    assert "First point" in slides[0]


def test_sheets_come_back_named_and_in_row_order() -> None:
    sheets = odf_sheets(ods_bytes())
    assert [name for name, _rows in sheets] == ["Data", "Notes"]
    assert sheets[0][1][0] == ("region", "units")
    assert sheets[0][1][1] == ("north", "2")


def test_an_empty_document_yields_an_empty_tuple_not_a_fabricated_block() -> None:
    assert odf_blocks(odt_bytes(blocks=())) == ()


def test_text_split_across_styled_spans_is_joined_rather_than_truncated() -> None:
    """ODF splits a paragraph across `text:span` children the moment any of it
    is styled. Reading `.text` alone silently drops every bold word.
    """
    import io
    import zipfile

    content = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<office:document-content "
        'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
        "<office:body><office:text>"
        "<text:p>plain <text:span>and styled</text:span> together</text:p>"
        "</office:text></office:body></office:document-content>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as package:
        stored = zipfile.ZipInfo("mimetype")
        stored.compress_type = zipfile.ZIP_STORED
        package.writestr(stored, "application/vnd.oasis.opendocument.text")
        package.writestr("content.xml", content)

    blocks = odf_blocks(buffer.getvalue())
    assert [b.text for b in blocks] == ["plain and styled together"]
