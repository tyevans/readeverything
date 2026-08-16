"""ODF text extraction, walked directly rather than through a library.

odfpy is unmaintained, and the part of ODF this library needs — the text, in
document order — is a walk over one flat XML part. Taking on an unmaintained
dependency to avoid two hundred lines is the trade that ages worst.

Matched by fully-qualified namespace rather than by literal prefix: ODF files
in the wild bind the same namespaces to different prefixes, so matching on the
string `"text:p"` works right up until the first file written by something
other than the tool it was tested against.

No I/O here and no environment, exactly like `ooxml.py`: bytes in, data out.
"""

from __future__ import annotations

from dataclasses import dataclass

from lxml import etree

from readeverything.adapters.ooxml import read_part

#: The single part every ODF package keeps its body in.
_CONTENT_PART = "content.xml"

_TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
_TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_DRAW_NS = "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"

_OUTLINE_LEVEL = f"{{{_TEXT_NS}}}outline-level"
_TEXT_H = f"{{{_TEXT_NS}}}h"
_TEXT_P = f"{{{_TEXT_NS}}}p"
_DRAW_PAGE = f"{{{_DRAW_NS}}}page"
_TABLE_NAME = f"{{{_TABLE_NS}}}name"
_TABLE_TABLE = f"{{{_TABLE_NS}}}table"
_TABLE_ROW = f"{{{_TABLE_NS}}}table-row"
_TABLE_CELL = f"{{{_TABLE_NS}}}table-cell"

#: What an unlabelled heading level defaults to. ODF makes `outline-level`
#: optional, and a heading without one is still a heading.
_DEFAULT_HEADING_LEVEL = 1

#: A body paragraph's level. Zero rather than None so `OdfBlock.level` is one
#: comparable number and no caller has to branch on a missing value.
_BODY_LEVEL = 0


@dataclass(frozen=True, slots=True)
class OdfBlock:
    """One paragraph or heading. `level` is 0 for body text, 1-9 for a heading."""

    level: int
    text: str


def _content(data: bytes) -> etree._Element | None:
    """The parsed `content.xml`, or None for anything unreadable.

    Never raises. `resolve_entities=False` and `no_network=True` are not
    optional: this parses a file the caller found on disk, and an XML parser
    that resolves external entities on untrusted input is an exfiltration
    primitive rather than a parser.
    """
    part = read_part(data, _CONTENT_PART)
    if part is None:
        return None
    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
    try:
        return etree.fromstring(part, parser=parser)
    except etree.XMLSyntaxError:
        return None


def _text_of(element: etree._Element) -> str:
    """Every descendant's text, joined and whitespace-normalised.

    `itertext` rather than `.text`, because ODF splits a paragraph across
    `text:span` children the moment any of it is styled — reading `.text` alone
    silently drops every bold word in the document.
    """
    return " ".join("".join(element.itertext()).split())


def _heading_level(element: etree._Element) -> int:
    raw = element.get(_OUTLINE_LEVEL)
    if raw is None:
        return _DEFAULT_HEADING_LEVEL
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_HEADING_LEVEL


def odf_blocks(data: bytes) -> tuple[OdfBlock, ...]:
    """Headings and paragraphs of an `.odt`, in document order."""
    root = _content(data)
    if root is None:
        return ()
    return tuple(
        OdfBlock(
            level=_heading_level(element) if element.tag == _TEXT_H else _BODY_LEVEL,
            text=_text_of(element),
        )
        for element in root.iter(_TEXT_H, _TEXT_P)
    )


def odf_slides(data: bytes) -> tuple[tuple[str, ...], ...]:
    """Each `draw:page`'s text runs, in order.

    The first run is the title by convention, which is both how ODP stores it
    and how the slides handler reads it.
    """
    root = _content(data)
    if root is None:
        return ()
    slides: list[tuple[str, ...]] = []
    for page in root.iter(_DRAW_PAGE):
        slides.append(tuple(text for element in page.iter(_TEXT_P) if (text := _text_of(element))))
    return tuple(slides)


def odf_sheets(data: bytes) -> tuple[tuple[str, tuple[tuple[str, ...], ...]], ...]:
    """Each `table:table`'s name and its rows of cell text."""
    root = _content(data)
    if root is None:
        return ()
    sheets: list[tuple[str, tuple[tuple[str, ...], ...]]] = []
    for table in root.iter(_TABLE_TABLE):
        rows = tuple(
            tuple(_text_of(cell) for cell in row.iter(_TABLE_CELL))
            for row in table.iter(_TABLE_ROW)
        )
        sheets.append((table.get(_TABLE_NAME) or "", rows))
    return tuple(sheets)
