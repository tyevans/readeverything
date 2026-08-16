"""Reading an EPUB's structure: the spine, the metadata, and the refusals."""

import zipfile

import pytest

from readeverything.adapters.epub_book import read_epub
from readeverything.domain.errors import DomainError
from tests.fixtures_epub import build_epub


def test_reads_the_title_and_author_from_the_package_metadata() -> None:
    book = read_epub(build_epub(title="Deep Water", author="M. Diver"))
    assert book.title == "Deep Water"
    assert book.author == "M. Diver"


def test_parts_come_back_in_spine_order_not_zip_order() -> None:
    """A zip has no reading order; the spine is the only thing that does."""
    book = read_epub(build_epub())
    assert [part.name for part in book.parts] == ["OEBPS/ch0.xhtml", "OEBPS/ch1.xhtml"]


def test_a_part_is_read_as_prose() -> None:
    book = read_epub(build_epub())
    text = " ".join(block.text for block in book.parts[0].blocks)
    assert "It began quietly." in text
    assert "<p>" not in text


def test_a_chapter_is_named_by_the_table_of_contents() -> None:
    """The TOC is the author's own name for the chapter, so it wins."""
    book = read_epub(build_epub(toc_labels=["I. Arrival", "II. Departure"]))
    assert [part.title for part in book.parts] == ["I. Arrival", "II. Departure"]


def test_an_epub_2_ncx_table_of_contents_is_read_too() -> None:
    book = read_epub(build_epub(nav=False, toc_labels=["Older", "Format"]))
    assert [part.title for part in book.parts] == ["Older", "Format"]


def test_a_chapter_missing_from_the_toc_falls_back_to_its_own_title() -> None:
    book = read_epub(build_epub(toc_labels=[]))
    assert [part.title for part in book.parts] == ["Chapter One", "Chapter Two"]


def test_the_navigation_document_is_not_itself_a_chapter() -> None:
    """It is in the manifest but not the spine, and it is a table, not prose."""
    book = read_epub(build_epub())
    assert all("nav" not in part.name for part in book.parts)


def test_a_drm_protected_book_says_so_rather_than_returning_nothing() -> None:
    with pytest.raises(DomainError, match="encrypted"):
        read_epub(build_epub(encrypted=True))


def test_a_zip_with_no_container_is_refused() -> None:
    with pytest.raises(DomainError, match=r"container\.xml"):
        read_epub(build_epub(omit_container=True))


def test_something_that_is_not_a_zip_is_refused() -> None:
    with pytest.raises(DomainError, match="not a zip"):
        read_epub(b"this is not an epub")


def test_a_spine_item_missing_from_the_zip_is_skipped_rather_than_fatal() -> None:
    """One broken part must not cost a reader the other nine chapters."""
    data = build_epub()
    stripped = _without(data, "OEBPS/ch1.xhtml")
    book = read_epub(stripped)
    assert [part.name for part in book.parts] == ["OEBPS/ch0.xhtml"]


def _without(data: bytes, member: str) -> bytes:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(buffer, "w") as target:
        for info in source.infolist():
            if info.filename != member:
                target.writestr(info, source.read(info.filename))
    return buffer.getvalue()


BILLION_LAUGHS = b"""<?xml version="1.0"?>
<!DOCTYPE container [
  <!ENTITY a "aaaaaaaaaa">
  <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">
  <!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">
]>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">&c;</container>
"""


def test_an_entity_declaration_is_refused_rather_than_expanded() -> None:
    """A corpus is untrusted input, and expat expands internal entities.

    Nothing in a real epub declares an entity, so refusing the declaration
    costs no readable book and closes the amplification.
    """
    data = build_epub(container_xml=BILLION_LAUGHS.decode())
    with pytest.raises(DomainError, match="entity"):
        read_epub(data)
