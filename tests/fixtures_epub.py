"""EPUBs built at test time, for the reason `tests/fixtures_office.py` gives.

An EPUB is a zip holding a `META-INF/container.xml` that points at an OPF
package document, which lists the reading order. There is no writer library in
this project's dependencies and no need for one: the format is small enough to
emit by hand, and emitting it here means every test can read exactly what it
fed the reader — including the malformed cases, which no writer would agree to
produce.

`build_epub` writes EPUB 3 with a nav document by default and can be asked for
an EPUB 2 NCX instead, because the handler has to read both and a fixture that
only ever produces one would hide half of that.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Sequence

CONTAINER_XML = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/book.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""


def chapter_xhtml(title: str, paragraphs: Sequence[str]) -> str:
    body = "".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml">'
        f"<head><title>{title}</title></head>"
        f"<body><h1>{title}</h1>{body}</body></html>"
    )


def _opf(*, title: str, author: str, parts: Sequence[str], nav: bool) -> str:
    items = "".join(
        f'<item id="c{index}" href="{name}" media-type="application/xhtml+xml"/>'
        for index, name in enumerate(parts)
    )
    if nav:
        items += '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" '
        items += 'properties="nav"/>'
        spine = "".join(f'<itemref idref="c{index}"/>' for index in range(len(parts)))
        toc = ""
    else:
        items += '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
        spine = "".join(f'<itemref idref="c{index}"/>' for index in range(len(parts)))
        toc = ' toc="ncx"'
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f"<dc:title>{title}</dc:title><dc:creator>{author}</dc:creator>"
        '<dc:identifier id="id">urn:uuid:test</dc:identifier>'
        "</metadata>"
        f"<manifest>{items}</manifest>"
        f"<spine{toc}>{spine}</spine>"
        "</package>"
    )


def _nav_xhtml(labels: Sequence[str], parts: Sequence[str]) -> str:
    links = "".join(
        f'<li><a href="{name}">{label}</a></li>' for label, name in zip(labels, parts, strict=False)
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">'
        "<head><title>Contents</title></head>"
        f'<body><nav epub:type="toc"><ol>{links}</ol></nav></body></html>'
    )


def _ncx(labels: Sequence[str], parts: Sequence[str]) -> str:
    points = "".join(
        f'<navPoint id="n{index}" playOrder="{index + 1}">'
        f"<navLabel><text>{label}</text></navLabel>"
        f'<content src="{name}"/></navPoint>'
        for index, (label, name) in enumerate(zip(labels, parts, strict=False))
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">'
        f"<navMap>{points}</navMap></ncx>"
    )


def build_epub(
    *,
    title: str = "A Short Book",
    author: str = "A. Writer",
    chapters: Sequence[tuple[str, Sequence[str]]] = (
        ("Chapter One", ("It began quietly.", "Then it did not.")),
        ("Chapter Two", ("It ended.",)),
    ),
    toc_labels: Sequence[str] | None = None,
    nav: bool = True,
    extra: dict[str, bytes] | None = None,
    container_xml: str = CONTAINER_XML,
    omit_container: bool = False,
    encrypted: bool = False,
) -> bytes:
    """A complete EPUB as bytes.

    `toc_labels` overrides the names the table of contents gives the chapters,
    which is how a test tells a TOC-derived title apart from one that fell back
    to the part's own `<title>`.
    """
    parts = [f"ch{index}.xhtml" for index in range(len(chapters))]
    labels = list(toc_labels) if toc_labels is not None else [name for name, _ in chapters]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as book:
        # First entry, stored, no extra field: this is what makes the file
        # sniffable as an EPUB rather than as a plain zip.
        book.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip", zipfile.ZIP_STORED)
        if not omit_container:
            book.writestr("META-INF/container.xml", container_xml)
        if encrypted:
            book.writestr("META-INF/encryption.xml", '<encryption xmlns="urn:x"/>')
        book.writestr("OEBPS/book.opf", _opf(title=title, author=author, parts=parts, nav=nav))
        for name, (chapter_title, paragraphs) in zip(parts, chapters, strict=True):
            book.writestr(f"OEBPS/{name}", chapter_xhtml(chapter_title, paragraphs))
        if nav:
            book.writestr("OEBPS/nav.xhtml", _nav_xhtml(labels, parts))
        else:
            book.writestr("OEBPS/toc.ncx", _ncx(labels, parts))
        for name, data in (extra or {}).items():
            book.writestr(name, data)
    return buffer.getvalue()
