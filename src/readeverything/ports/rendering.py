"""Turning a document into the pages a reader would see.

The library can already answer questions about pictures — `describe_image`,
`ocr` and `ask_about_image` all exist — and it can already turn a paginated
document into page images, because `handlers/pdf.py` does. What it cannot do is
*get* a faithful page image out of a format pypdfium2 will not open. That is
what this port is for, and it is deliberately the whole of it.

**Path in, bytes out.** A renderer is an external process, and external
processes take paths rather than streams. This is the same acknowledgement
`SourceReader.local_path` already makes, and the reason that method exists. It
also means rendering a deck *inside a tarball* works with no further
arrangement: `NestedSource.local_path` materialises the member, and the
renderer never learns it was nested.

`revision` is not decoration. It goes into the converted document's cache key,
so a converter upgrade does not silently serve page images produced by two
different versions of two different renderers; and it is what a composition
declares as `Capability.DOCUMENT_RENDER`'s revision when a caller injects a
renderer of their own instead of letting the binary probe answer.

A renderer's page images are a RENDERING, never the document itself — fonts
substitute, layout engines differ. Callers are expected to record that; see
`handlers/office_slides.py` for how it is worded.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from readeverything.domain.identity import MimeType


@runtime_checkable
class DocumentRenderer(Protocol):
    @property
    def revision(self) -> str:
        """What is behind this renderer, for the cache key and the capability."""
        ...

    def claims(self, mime: MimeType) -> bool:
        """Whether this renderer can convert that format at all.

        Synchronous and cheap: a handler asks this while assembling a card, and
        a question about a format must never cost a subprocess.
        """
        ...

    async def page_count(self, path: str) -> int:
        """How many pages the document renders to."""
        ...

    async def render_page(self, path: str, page: int, *, dpi: int = 150) -> bytes:
        """A PNG of `page`, 1-indexed as a reader would count.

        Raises `RenditionFailedError` rather than returning empty bytes: `b""`
        is a value a caller can pass along, and a blank image handed to a model
        as "page 4" is an observation nothing made.
        """
        ...
