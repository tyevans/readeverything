"""The container sniffer, which is what stops a .docx being a hex dump."""

from __future__ import annotations

import io
import zipfile

from readeverything.adapters.ooxml import (
    ODF_TEXT_MIME,
    SHEETS_MIME,
    SLIDES_MIME,
    WORD_MIME,
    office_mimetype,
    part_names,
    read_part,
    zip_part_names,
)
from tests.fixtures_office import docx_bytes, odt_bytes, pptx_bytes, xlsx_bytes


def test_a_docx_is_recognised_from_its_head() -> None:
    assert office_mimetype(docx_bytes()[:4096]) == WORD_MIME


def test_a_pptx_is_recognised_from_its_head() -> None:
    assert office_mimetype(pptx_bytes()[:4096]) == SLIDES_MIME


def test_an_xlsx_is_recognised_even_though_content_types_is_written_last() -> None:
    """The measurement this whole module exists for.

    openpyxl writes `[Content_Types].xml` as the LAST zip entry, so it is
    unreachable from any bounded head. Classifying by the `xl/` part-name
    prefix is what makes a spreadsheet detectable at all.
    """
    head = xlsx_bytes()[:4096]
    assert b"[Content_Types].xml" not in head
    assert office_mimetype(head) == SHEETS_MIME


def test_an_odt_is_recognised_from_its_stored_mimetype_entry() -> None:
    """ODF stores `mimetype` first and uncompressed precisely so it can be
    sniffed. Reading it is the whole rule.
    """
    assert office_mimetype(odt_bytes()[:4096]) == ODF_TEXT_MIME


def test_a_plain_zip_is_not_claimed_as_an_office_document() -> None:
    """A `.zip` and a `.jar` must stay archives; Spec 8 descends into those and
    must not descend into a `.docx`.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.txt", "hello")
        archive.writestr("data/rows.csv", "a,b\n1,2\n")
    assert office_mimetype(buffer.getvalue()[:4096]) is None


def test_bytes_that_are_not_a_zip_are_not_claimed() -> None:
    assert office_mimetype(b"%PDF-1.7\nnot a zip at all") is None
    assert office_mimetype(b"") is None


def test_a_truncated_head_yields_no_answer_rather_than_a_wrong_one() -> None:
    """Better `None` — which leaves the caller on its existing fallbacks — than
    a guess from a name that was cut in half.
    """
    assert office_mimetype(docx_bytes()[:35]) is None


def test_a_stored_mimetype_entry_that_is_cut_short_is_not_read() -> None:
    """Half a mimetype is not evidence of a whole one."""
    assert office_mimetype(odt_bytes()[:45]) is None


def test_part_names_walking_stops_at_the_end_of_the_head() -> None:
    """The walk must terminate on a partial header rather than reading past the
    buffer or looping.
    """
    data = xlsx_bytes()
    truncated = zip_part_names(data[:600])
    whole = part_names(data)

    # A proper prefix of the real listing: what the head could see, in order,
    # and nothing invented past where the bytes ran out.
    assert truncated
    assert len(truncated) < len(whole)
    assert list(truncated) == list(whole)[: len(truncated)]


def test_zip_part_names_of_something_that_is_not_a_zip_is_empty() -> None:
    assert zip_part_names(b"not a zip at all, not even close") == ()


def test_read_part_returns_a_named_member_of_the_whole_package() -> None:
    body = read_part(docx_bytes(), "word/document.xml")
    assert body is not None
    assert b"<w:document" in body


def test_read_part_returns_none_for_a_member_that_is_not_there() -> None:
    assert read_part(docx_bytes(), "word/nonexistent.xml") is None


def test_read_part_returns_none_for_bytes_that_are_not_a_zip() -> None:
    """No exception: every caller of this is on a handler's degrade path."""
    assert read_part(b"not a zip", "anything") is None


def test_part_names_lists_the_whole_package_not_just_the_head() -> None:
    names = part_names(xlsx_bytes())
    assert "[Content_Types].xml" in names
    assert "xl/workbook.xml" in names


def test_part_names_of_something_that_is_not_a_zip_is_empty() -> None:
    assert part_names(b"not a zip") == ()
