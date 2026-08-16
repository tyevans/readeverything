"""An EPUB's structure: what the chapters are, and what order they are in.

An EPUB is a zip, and a zip has no reading order — `infolist()` returns
whatever order the writer happened to use. The reading order lives in the OPF
package document's spine, reached through `META-INF/container.xml`, and a
reader that walked the zip instead would hand back a book with its chapters
shuffled and call it a book.

Only the structure lives here. The prose comes from `adapters/html_text`, the
same reader the HTML handler uses, because every part of an EPUB is XHTML and
reading it a second way would mean two answers to "what does this page say".
That sharing is also what gives a chapter its offsets: a citation into an EPUB
is a span of one named part, and the part is a file a reader can open.

Stdlib only: `zipfile` and `xml.etree`. An EPUB needs no dependency, and one
taken here would be paid for by every installation that never opens a book.
`xml.etree` does not resolve external entities, which matters because these
files come from wherever the user's corpus came from.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from urllib.parse import unquote, urldefrag
from xml.etree import ElementTree  # nosec B405 -- entity declarations refused in `_parse`

from readeverything.adapters.html_text import HtmlBlock, html_blocks, html_title
from readeverything.domain.errors import DomainError

_CONTAINER = "META-INF/container.xml"

#: Present when the book is DRM-protected, or has obfuscated fonts. Either way
#: the parts are not readable as XHTML, and saying so beats a book of noise.
_ENCRYPTION = "META-INF/encryption.xml"

_NS = {
    "container": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
    "ncx": "http://www.daisy.org/z3986/2005/ncx/",
    "xhtml": "http://www.w3.org/1999/xhtml",
}


@dataclass(frozen=True, slots=True)
class EpubPart:
    """One spine item: a chapter, its name in the zip, and its prose.

    `blocks` carry spans into the part's own XHTML -- not into the EPUB, which
    is a zip and has no character offsets worth citing.
    """

    name: str
    title: str
    blocks: tuple[HtmlBlock, ...]


@dataclass(frozen=True, slots=True)
class Epub:
    title: str | None
    author: str | None
    parts: tuple[EpubPart, ...]


def read_epub(data: bytes) -> Epub:
    """Parse an EPUB's spine and every part in it.

    Raises `DomainError` rather than returning an empty book when the file is
    not one: a book that reads as zero chapters is indistinguishable from a
    book with nothing in it, and the caller deserves the difference.
    """
    try:
        book = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as error:
        raise DomainError(f"not a zip, so not an epub: {error}") from error
    with book:
        names = set(book.namelist())
        if _ENCRYPTION in names:
            raise DomainError(
                "this epub is encrypted (DRM or obfuscated fonts); its parts cannot be read"
            )
        if _CONTAINER not in names:
            raise DomainError(f"no {_CONTAINER}, so this zip is not an epub")
        opf_path = _opf_path(book.read(_CONTAINER))
        if opf_path not in names:
            raise DomainError(f"container.xml points at {opf_path!r}, which is not in the epub")
        package = _parse(book.read(opf_path), what="the opf package document")
        base = opf_path.rpartition("/")[0]
        spine = _spine(package, base)
        labels = _toc_labels(book, package, base, names)
        return Epub(
            title=_text(package, ".//dc:title"),
            author=_text(package, ".//dc:creator"),
            parts=tuple(_part(book, name, labels) for name in spine if name in names),
        )


def _parse(data: bytes, *, what: str) -> ElementTree.Element:
    """Parse with entity declarations refused.

    These files come from wherever the corpus came from. `xml.etree` does not
    fetch external entities, but expat does expand internally declared ones,
    which is the billion-laughs amplification: a few hundred bytes of DTD
    become gigabytes of string. Nothing in a real EPUB declares an entity, so
    refusing the declaration outright costs no readable book. This is what
    `defusedxml` would do, without adding a dependency for one hook.

    An entity can only be declared in the DOCTYPE's internal subset, so that is
    the only place worth looking, and looking is what `xml.etree` leaves to the
    caller: its C parser exposes no expat handle to hook. External DTDs are not
    a second hole -- expat does not fetch them.
    """
    _refuse_entity_declarations(data, what)
    try:
        return ElementTree.fromstring(data)  # nosec B314 -- hardened above
    except ElementTree.ParseError as error:
        raise DomainError(f"{what} is not well-formed xml: {error}") from error


def _refuse_entity_declarations(data: bytes, what: str) -> None:
    doctype = data.find(b"<!DOCTYPE")
    if doctype < 0:
        return
    closing = data.find(b"]>", doctype)
    if closing < 0:
        closing = data.find(b">", doctype)
    subset = data[doctype : closing if closing >= 0 else len(data)]
    if b"<!ENTITY" in subset:
        raise DomainError(f"{what} declares an xml entity, which is refused as unsafe")


def _opf_path(container: bytes) -> str:
    root = _parse(container, what="container.xml")
    rootfile = root.find(".//container:rootfile", _NS)
    full_path = None if rootfile is None else rootfile.get("full-path")
    if not full_path:
        raise DomainError("container.xml names no rootfile, so there is no package document")
    return full_path


def _resolve(base: str, href: str) -> str:
    """A manifest href is relative to the OPF, and percent-encoded like a url."""
    path = unquote(urldefrag(href).url)
    return f"{base}/{path}" if base else path


def _spine(package: ElementTree.Element, base: str) -> tuple[str, ...]:
    """The reading order, as zip member names.

    The manifest maps id to href and the spine names ids, so neither alone is
    the answer: the manifest is unordered and the spine holds no paths.
    """
    hrefs = {
        item.get("id"): item.get("href")
        for item in package.findall(".//opf:manifest/opf:item", _NS)
    }
    order = []
    for ref in package.findall(".//opf:spine/opf:itemref", _NS):
        href = hrefs.get(ref.get("idref"))
        if href:
            order.append(_resolve(base, href))
    return tuple(order)


def _toc_labels(
    book: zipfile.ZipFile,
    package: ElementTree.Element,
    base: str,
    names: set[str],
) -> dict[str, str]:
    """What the table of contents calls each part, if it has an opinion.

    EPUB 3 keeps this in a nav document and EPUB 2 in an NCX, and books in the
    wild carry either -- often both. Both are read; neither is required, since
    the spine alone is enough to have a readable book.
    """
    labels: dict[str, str] = {}
    for reader, path in (
        (_ncx_labels, _ncx_path(package, base, names)),
        (_nav_labels, _nav_path(package, base, names)),
    ):
        if path is not None and path in names:
            try:
                labels.update(reader(book.read(path), path.rpartition("/")[0]))
            except DomainError:
                # A malformed table of contents costs the book its chapter
                # names, not its chapters.
                continue
    return labels


def _nav_path(package: ElementTree.Element, base: str, names: set[str]) -> str | None:
    for item in package.findall(".//opf:manifest/opf:item", _NS):
        if "nav" in (item.get("properties") or "").split():
            href = item.get("href")
            if href:
                return _resolve(base, href)
    return None


def _ncx_path(package: ElementTree.Element, base: str, names: set[str]) -> str | None:
    spine = package.find(".//opf:spine", _NS)
    toc_id = None if spine is None else spine.get("toc")
    for item in package.findall(".//opf:manifest/opf:item", _NS):
        if item.get("id") == toc_id or item.get("media-type") == "application/x-dtbncx+xml":
            href = item.get("href")
            if href:
                return _resolve(base, href)
    return None


def _ncx_labels(data: bytes, base: str) -> dict[str, str]:
    root = _parse(data, what="the ncx table of contents")
    labels: dict[str, str] = {}
    for point in root.findall(".//ncx:navPoint", _NS):
        content = point.find("ncx:content", _NS)
        label = point.find("ncx:navLabel/ncx:text", _NS)
        if content is None or label is None or not label.text:
            continue
        src = content.get("src")
        if src:
            labels.setdefault(_resolve(base, src), label.text.strip())
    return labels


def _nav_labels(data: bytes, base: str) -> dict[str, str]:
    root = _parse(data, what="the nav table of contents")
    labels: dict[str, str] = {}
    for anchor in root.findall(".//xhtml:a", _NS):
        href = anchor.get("href")
        text = "".join(anchor.itertext()).strip()
        if href and text:
            labels.setdefault(_resolve(base, href), text)
    return labels


def _part(book: zipfile.ZipFile, name: str, labels: dict[str, str]) -> EpubPart:
    source = book.read(name).decode("utf-8", errors="replace")
    blocks = html_blocks(source)
    # The table of contents first, because it is the author's own name for the
    # chapter; the part's `<title>` is often boilerplate, and the filename is
    # the last thing anyone wants to read.
    title = labels.get(name) or html_title(source) or name.rpartition("/")[2]
    return EpubPart(name=name, title=title, blocks=blocks)


def _text(package: ElementTree.Element, path: str) -> str | None:
    element = package.find(path, _NS)
    if element is None or not element.text:
        return None
    return element.text.strip() or None
