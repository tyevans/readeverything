"""Legacy OLE2 documents: `.doc`, `.ppt`, `.xls`.

The family Spec 9 deliberately declined. Pure-Python support for the OLE2
compound-file formats is poor, and the honest answer was to read none of them
rather than to read them badly. This is the other honest answer: read them
through a converter, and only where a converter exists.

So unlike the three modern office handlers — which merely *gain* `page_image`
when a renderer is available — this handler **requires** `DOCUMENT_RENDER` from
`requires()`. Without it the registry drops it entirely and these files keep
falling through to `BinaryHandler`'s hex dump, which is exactly today's
behaviour and therefore no regression. There is no half-working version of this
handler to fall back to: the converter is not an enhancement here, it is the
only reader.

**One family, not three, and this is measured rather than assumed.** All three
formats share the OLE2 compound-file header, and `PuremagicDetector` reports
`application/msword` for a real `.doc`, `.ppt` and `.xls` alike — at 0.80
confidence, which is above `detection._SIGNATURE_FLOOR`, so the filename is
never consulted. Distinguishing them needs a real OLE2 directory walk with
sector chasing; the stream names are not reliably within the 4096-byte head the
pipeline reads (a `.ppt` produced here had none in its first 64KB, and a `.doc`
matched two).

It costs nothing, because the converter detects the real format itself: a
`.ppt` arriving labelled `application/msword` still converts, and LibreOffice
picks Impress. What it costs is the right to *say* which application made the
file — so this card does not. It reports the page count of the converted
document, which is a thing that was observed, rather than a format inferred
from a mimetype known to be wrong for two thirds of the family.

All three mimetypes are still claimed, so a caller who plugs in a detector with
a real OLE2 walker is served correctly with no change here.
"""

from __future__ import annotations

import time
from typing import ClassVar

from pydantic import BaseModel, Field

from readeverything.domain.affordance import Affordance, DetailLevel
from readeverything.domain.capability import Capability
from readeverything.domain.card import Card, Segment
from readeverything.domain.errors import UnknownAffordanceError
from readeverything.domain.identity import MediaKind, SourceRef
from readeverything.domain.locator_map import LocatorMap, LocatorSegment
from readeverything.domain.locators import ByteRange, CharSpan, PageRef
from readeverything.domain.observation import OperationFinished, OperationStarted
from readeverything.domain.rendition import (
    Budget,
    Degradation,
    Rendered,
    Rendition,
    TextContent,
)
from readeverything.handlers.page_images import (
    PageImageParams,
    page_image_affordance,
    render_page_image,
)
from readeverything.ports.observation import Observer, emit
from readeverything.ports.rendering import DocumentRenderer
from readeverything.ports.source import SourceReader

#: What `represent` calls itself when it narrates, matching every other handler.
_OPERATION = "represent"

#: What `page_image` renders one of. "Page" and not "slide": this handler
#: cannot tell a deck from a memo, and guessing would be the exact dishonesty
#: the module docstring is about.
_RENDER_UNIT = "page"

#: The three OLE2 mimetypes. See the module docstring for why only the first is
#: reachable through this library's own detector today.
LEGACY_MIMETYPES: tuple[str, ...] = (
    "application/msword",
    "application/vnd.ms-powerpoint",
    "application/vnd.ms-excel",
)

#: Every page's text ends with this, and the page's `LocatorSegment` INCLUDES
#: it. `LocatorMap` demands total, gapless, zero-start coverage and
#: `CharSpan.__post_init__` rejects `start >= end`, so a page that converted to
#: nothing would otherwise contribute a zero-width span and raise. Owning the
#: separator means every page owns at least one character. The same reasoning,
#: and the same trap, as `handlers/pdf.py`'s `PAGE_SEPARATOR`.
PAGE_SEPARATOR = "\n"


def _empty_page(number: int) -> str:
    """What stands in for a page the conversion produced no text from.

    Not silence. A blank stretch in the middle of a document would otherwise
    read as the document ending early, and "this page has no text" is a
    different and true statement.
    """
    return f"(page {number} converted to no text)"


class ReadPageParams(BaseModel):
    page: int = Field(default=1, ge=1, description="1-indexed page number to read.")


class OfficeLegacyHandler:
    """Reads a legacy OLE2 document by converting it, and says that it did."""

    mime_patterns: ClassVar[tuple[str, ...]] = LEGACY_MIMETYPES
    priority: ClassVar[int] = 0
    handler_id: ClassVar[str] = "office_legacy"
    handler_version: ClassVar[int] = 1

    def __init__(
        self,
        *,
        source: SourceReader,
        renderer: DocumentRenderer,
        observer: Observer | None = None,
    ) -> None:
        """`renderer` is required, not optional.

        Every other handler takes its collaborators as `| None` and degrades.
        This one cannot: with no converter it has no reader at all, and a
        constructor that accepted `None` would let a composition build a
        handler that answers nothing. `requires()` is what keeps it out of the
        registry; this is what keeps it out of existence.
        """
        self._source = source
        self._renderer = renderer
        self._observer = observer

    def requires(self) -> frozenset[Capability]:
        """A converter, outright. See the module docstring."""
        return frozenset({Capability.DOCUMENT_RENDER})

    def affordances(self) -> tuple[Affordance, ...]:
        return (
            Affordance(
                name="read_page",
                description=(
                    "Return the text of one page of the converted document. The text "
                    "is what the converter's importer made of the original file."
                ),
                params=ReadPageParams,
                requires=frozenset({Capability.DOCUMENT_RENDER}),
                level=DetailLevel.SEGMENT,
            ),
            page_image_affordance(_RENDER_UNIT),
        )

    # -- reading through the converter -----------------------------------

    async def _pages(self, ref: SourceRef) -> tuple[str, ...] | None:
        """Every page's text, or None if the document could not be converted.

        None rather than an exception: this handler never raises about its
        input, so an unreadable file becomes an honest card rather than a
        traceback.
        """
        try:
            path = await self._source.local_path(ref.uri)
            count = await self._renderer.page_count(path)
            # A list comprehension and not `tuple(... for ...)`: an `await`
            # inside a generator expression makes it an ASYNC generator, which
            # `tuple()` cannot consume, and the resulting TypeError would be
            # swallowed by the `except` below and reported as "this document
            # could not be converted". It was, until a test said otherwise.
            return tuple([await self._renderer.page_text(path, n) for n in range(1, count + 1)])
        except Exception:
            return None

    # -- the handler surface ---------------------------------------------

    async def describe(self, ref: SourceRef) -> Card:
        """Page count and nothing about which application made the file.

        `kind` is `BINARY`, matching `pdf.py` and the office handlers: these
        mimetypes reach this handler at the registry's exact-mimetype step,
        long before the kind step.

        The card costs a conversion, which is more than `pdf.py`'s probe. There
        is no cheaper honest answer: nothing about a legacy file's page count
        is knowable without opening it, and the conversion is cached, so the
        cost is paid once per document and every later read is free.
        """
        pages = await self._pages(ref)
        if pages is None:
            return Card(
                ref=ref,
                kind=MediaKind.BINARY,
                facts={"readable": "no", "size_bytes": ref.size_bytes},
                outline=(),
                excerpt=None,
                affordances=self.affordances(),
            )
        return Card(
            ref=ref,
            kind=MediaKind.BINARY,
            facts={
                "readable": "yes",
                "page_count": len(pages),
                "read_via": "conversion",
                "size_bytes": ref.size_bytes,
            },
            outline=tuple(
                Segment(PageRef(number), f"page {number}") for number in range(1, len(pages) + 1)
            ),
            excerpt=None,
            affordances=self.affordances(),
        )

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition:
        match name:
            case "read_page":
                if not isinstance(params, ReadPageParams):
                    raise TypeError(f"expected ReadPageParams, got {type(params).__name__}")
                return await self._read_page(ref, params.page)
            case "page_image":
                if not isinstance(params, PageImageParams):
                    raise TypeError(f"expected PageImageParams, got {type(params).__name__}")
                return await render_page_image(
                    renderer=self._renderer,
                    source=self._source,
                    ref=ref,
                    params=params,
                    unit=_RENDER_UNIT,
                )
            case _:
                raise UnknownAffordanceError(name, (a.name for a in self.affordances()))

    def _degraded(self, ref: SourceRef, detail: str) -> Rendition:
        return Rendition(
            locator=ByteRange(0, max(1, ref.size_bytes)),
            content=TextContent(detail),
            degraded=True,
        )

    async def _read_page(self, ref: SourceRef, number: int) -> Rendition:
        pages = await self._pages(ref)
        if pages is None:
            return self._degraded(ref, f"{ref.uri} could not be converted and so not read")
        if number > len(pages):
            return self._degraded(
                ref,
                f"page {number} does not exist; the converted document has {len(pages)} page(s)",
            )
        body = pages[number - 1] or _empty_page(number)
        return Rendition(
            locator=PageRef(number),
            content=TextContent(body),
            degradations=(_conversion_provenance(self._renderer),),
        )

    async def represent(self, ref: SourceRef, budget: Budget) -> Rendered:
        """Narrated start to finish, matching every other handler."""
        emit(self._observer, OperationStarted(operation=_OPERATION, ref=ref))
        started = time.perf_counter()
        try:
            return await self._represent(ref, budget)
        finally:
            emit(
                self._observer,
                OperationFinished(
                    operation=_OPERATION, ref=ref, elapsed_s=time.perf_counter() - started
                ),
            )

    async def _represent(self, ref: SourceRef, budget: Budget) -> Rendered:
        pages = await self._pages(ref)
        if pages is None:
            return self._nothing_to_read(
                ref,
                budget,
                summary=f"Unconvertible legacy document {ref.uri}, {ref.size_bytes} bytes.",
                what="conversion failed",
                detail=(
                    "the file could not be converted, so no text was extracted; it may "
                    "not be a document the converter can open"
                ),
            )
        if not pages:
            return self._nothing_to_read(
                ref,
                budget,
                summary=f"Legacy document {ref.uri} converted to no pages.",
                what="converted document has no pages",
                detail="the file converted but produced no pages; no text was extracted",
            )

        chunks: list[str] = []
        segments: list[LocatorSegment] = []
        barriers: list[int] = []
        cursor = 0
        for index, body in enumerate(pages):
            number = index + 1  # `PageRef` counts as a reader does.
            chunk = (body or _empty_page(number)) + PAGE_SEPARATOR
            if index:
                barriers.append(cursor)
            segments.append(LocatorSegment(CharSpan(cursor, cursor + len(chunk)), PageRef(number)))
            cursor += len(chunk)
            chunks.append(chunk)
        return self._fit(
            "".join(chunks),
            tuple(segments),
            tuple(barriers),
            budget,
            (_conversion_provenance(self._renderer),),
        )

    def _nothing_to_read(
        self, ref: SourceRef, budget: Budget, *, summary: str, what: str, detail: str
    ) -> Rendered:
        """Located by `ByteRange`: no page was ever observed."""
        segments = (
            LocatorSegment(CharSpan(0, len(summary)), ByteRange(0, max(1, ref.size_bytes))),
        )
        return self._fit(summary, segments, (), budget, (Degradation(what=what, detail=detail),))

    def _fit(
        self,
        full: str,
        segments: tuple[LocatorSegment, ...],
        barriers: tuple[int, ...],
        budget: Budget,
        degradations: tuple[Degradation, ...],
    ) -> Rendered:
        """Apply the budget, pruning the map and the barriers with the text.

        Identical in shape to `pdf.py._fit` and the office handlers': `Rendered`
        rejects a map that does not cover its text, so truncation cannot touch
        the text alone.
        """
        if budget.max_chars is None or len(full) <= budget.max_chars:
            return Rendered(
                text=full,
                locator_map=LocatorMap.build(segments),
                barriers=barriers,
                degradations=degradations,
            )
        keep = max(1, budget.max_chars)
        text = full[:keep]
        kept = tuple(
            LocatorSegment(CharSpan(s.span.start, min(s.span.end, keep)), s.locator)
            for s in segments
            if s.span.start < keep
        )
        return Rendered(
            text=text,
            locator_map=LocatorMap.build(kept),
            barriers=tuple(barrier for barrier in barriers if barrier < keep),
            degradations=(
                *degradations,
                Degradation(
                    what="text truncated",
                    detail=f"kept {len(text)} of {len(full)} characters",
                ),
            ),
        )


def _conversion_provenance(renderer: DocumentRenderer) -> Degradation:
    """That these words came through an importer, not out of the file.

    The image path says the same thing in `page_images.rendering_provenance`,
    and it matters at least as much here. A legacy binary format's text is
    reconstructed by the converter's importer: ordering can differ from what
    the original application would show, and a table's cells may arrive in
    reading order rather than the author's. Quoting this as the document's own
    text, verbatim, is a stronger claim than anything established.
    """
    return Degradation(
        what="read by converting the document",
        detail=(
            f"this text was produced by converting the file with {renderer.revision} "
            f"rather than read from the file's own format; wording is the original's "
            f"but ordering and layout are the converter's"
        ),
    )
