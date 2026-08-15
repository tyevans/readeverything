"""PDFs.

The first handler that produces a `PageRef`, and the first that produces a
`Rendered.barriers` that is not empty. Both fields have existed since Spec 1
with no producer; a paginated document is what they were designed for, and
this is where they become real.

The card costs a probe and no text extraction — page count is exactly the fact
that shapes an agent's next move, and paying a full extraction to learn it
would defeat progressive disclosure. `represent` is where the text is read.

pypdfium2 is imported directly here, guarded exactly as `image.py` guards
Pillow. The injected `MediaProbe` remains the card path's source of facts; the
import exists because extraction is not a probe's job and no handler reaches
into an adapter.
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar

try:
    import pypdfium2 as pdfium  # type: ignore[import-untyped]  # pypdfium2 ships no py.typed
except ImportError as exc:  # pragma: no cover - exercised via a patched sys.modules
    raise ImportError(
        "readeverything's PDF support needs pypdfium2, which ships in the "
        "'documents' extra: pip install 'deepagents-read-everything[documents]'. "
        "The composition root omits PDF handling when pypdfium2 is absent, so "
        "reaching this means the handler was imported directly."
    ) from exc
from pydantic import BaseModel

from readeverything.domain.affordance import Affordance
from readeverything.domain.capability import Capability
from readeverything.domain.card import Card, Segment
from readeverything.domain.errors import UnknownAffordanceError
from readeverything.domain.identity import MediaKind, SourceRef
from readeverything.domain.locator_map import LocatorMap, LocatorSegment
from readeverything.domain.locators import ByteRange, CharSpan, PageRef
from readeverything.domain.rendition import Budget, Degradation, Rendered, Rendition
from readeverything.ports.probe_media import MediaProbe
from readeverything.ports.source import SourceReader

#: Every page's text ends with this, and the page's `LocatorSegment` INCLUDES
#: it. `LocatorMap` demands total, gapless, zero-start coverage and
#: `CharSpan.__post_init__` rejects `start >= end`, so a page whose text layer
#: is empty would otherwise contribute a zero-width span and raise. Owning the
#: separator means every page owns at least one character no matter what it
#: extracted. Do not "simplify" this away: an empty page in the middle of a
#: document is what breaks the map.
PAGE_SEPARATOR = "\n"


class _PageState(Enum):
    """What a page turned out to be, once its text layer came back empty."""

    EXTRACTED = "extracted"
    SCANNED = "scanned"
    BLANK = "blank"


def _page_text(page: pdfium.PdfPage) -> str:
    textpage = page.get_textpage()
    try:
        return str(textpage.get_text_range(index=0, count=-1))
    finally:
        textpage.close()


def _page_state(page: pdfium.PdfPage, text: str) -> _PageState:
    """Extracted, scanned, or genuinely blank.

    Measured on real files: a scanned page and a blank page are IDENTICAL
    through the text layer — both report zero characters and empty text. The
    page's object list is what distinguishes them, a scan carrying at least one
    image object where a blank page carries none.

    Without this, a scanned contract is reported as an empty document, which is
    a false claim about the file rather than an honest report about the text
    layer.
    """
    if text.strip():
        return _PageState.EXTRACTED
    if any(True for _ in page.get_objects()):
        return _PageState.SCANNED
    return _PageState.BLANK


def _placeholder(state: _PageState, number: int) -> str:
    """What stands in the flattened text for a page that extracted nothing.

    A scan and a blank page must not read the same. The scan's line describes
    where its content actually is; the blank page's says it is blank. Neither
    says the document is empty, which is a claim about the document rather
    than about its text layer.
    """
    if state is _PageState.SCANNED:
        return f"(page {number} has no text layer; its content is in images and was not read)"
    return f"(page {number} is blank)"


def _listed(numbers: list[int], limit: int = 10) -> str:
    head = ", ".join(str(number) for number in numbers[:limit])
    if len(numbers) <= limit:
        return head
    return f"{head} and {len(numbers) - limit} more"


class PdfHandler:
    """Reads a PDF's text layer, and maps every character to the page it came from."""

    mime_patterns: ClassVar[tuple[str, ...]] = ("application/pdf",)
    priority: ClassVar[int] = 0
    handler_id: ClassVar[str] = "pdf"
    handler_version: ClassVar[int] = 1

    def __init__(
        self,
        *,
        source: SourceReader,
        probe: MediaProbe,
        # Typed `object | None` only until the `TextRecognizer` port lands with
        # OCR over a rendered page; the handler stores it and does not call it
        # yet, so a narrower type here would be a promise nothing keeps.
        recognizer: object | None = None,
    ) -> None:
        self._source = source
        self._probe = probe
        self._recognizer = recognizer

    def requires(self) -> frozenset[Capability]:
        """Nothing. Reading a born-digital PDF needs no model and no binary."""
        return frozenset()

    def affordances(self) -> tuple[Affordance, ...]:
        return ()

    def _open(self, data: bytes) -> pdfium.PdfDocument | None:
        """The opened document, or None if these bytes are not a readable PDF.

        Encrypted, truncated and malformed files all land here. None rather
        than an exception, because this handler never raises about its input.
        """
        try:
            return pdfium.PdfDocument(data)
        except Exception:
            return None

    async def describe(self, ref: SourceRef) -> Card:
        """Page count and page geometry, from the probe. No text is extracted.

        The card's `kind` is `BINARY`, not some new `DOCUMENT` member:
        `application/pdf` reaches this handler at the registry's exact-mimetype
        step, long before the kind step, so what identifies a PDF card is its
        mime, its page count and its affordances.
        """
        data = await self._source.read_bytes(ref.uri)
        try:
            facts = await self._probe.probe(data)
        except Exception:
            return Card(
                ref=ref,
                kind=MediaKind.BINARY,
                facts={"readable": "no", "size_bytes": ref.size_bytes},
                outline=(),
                excerpt=None,
                affordances=self.affordances(),
            )
        width, height = facts.page_sizes[0] if facts.page_sizes else (0.0, 0.0)
        return Card(
            ref=ref,
            kind=MediaKind.BINARY,
            facts={
                "readable": "yes",
                "page_count": str(facts.page_count),
                "first_page_points": f"{width:g}x{height:g}",
                "size_bytes": ref.size_bytes,
                **{f"meta.{k}": v for k, v in sorted(facts.metadata.items())},
            },
            outline=tuple(
                Segment(PageRef(number + 1), f"page {number + 1}")
                for number in range(facts.page_count)
            ),
            excerpt=None,
            affordances=self.affordances(),
        )

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition:
        raise UnknownAffordanceError(name, (a.name for a in self.affordances()))

    async def represent(self, ref: SourceRef, budget: Budget) -> Rendered:
        data = await self._source.read_bytes(ref.uri)
        document = self._open(data)
        if document is None:
            return self._unreadable(ref, budget)
        try:
            pages: list[tuple[str, _PageState]] = []
            for index in range(len(document)):
                page = document[index]
                body = _page_text(page)
                pages.append((body, _page_state(page, body)))
        finally:
            document.close()
        if not pages:
            # Opened fine and carries no pages. Saying "could not be opened"
            # here would be a false report about a file this handler read.
            return self._nothing_to_read(
                ref,
                budget,
                summary=f"PDF {ref.uri} has no pages.",
                what="pdf has no pages",
                detail="the document opened but contains no pages; no text was extracted",
            )

        chunks: list[str] = []
        segments: list[LocatorSegment] = []
        barriers: list[int] = []
        scanned: list[int] = []
        blanks: list[int] = []
        cursor = 0
        for index, (body, state) in enumerate(pages):
            number = index + 1  # `PageRef` counts as a reader does; pdfium does not.
            if state is _PageState.SCANNED:
                scanned.append(number)
            elif state is _PageState.BLANK:
                blanks.append(number)
            content = body if state is _PageState.EXTRACTED else _placeholder(state, number)
            chunk = content + PAGE_SEPARATOR
            if index:
                # A barrier is where a new page's first character begins, so
                # there is exactly one per page break: page count minus one.
                barriers.append(cursor)
            segments.append(LocatorSegment(CharSpan(cursor, cursor + len(chunk)), PageRef(number)))
            cursor += len(chunk)
            chunks.append(chunk)
        return self._fit(
            "".join(chunks),
            tuple(segments),
            tuple(barriers),
            budget,
            self._page_degradations(scanned, blanks),
        )

    def _page_degradations(self, scanned: list[int], blanks: list[int]) -> tuple[Degradation, ...]:
        """One report per state, not one per page — a 400-page scan is one fact.

        The scanned report is the whole point of telling the two apart: an
        agent that knows a page's text is in an image knows to look harder,
        where "empty" would have told it to stop.
        """
        degradations: list[Degradation] = []
        if scanned:
            attempted = (
                "OCR was not attempted, as no recogniser is configured"
                if self._recognizer is None
                else "OCR was not attempted"
            )
            degradations.append(
                Degradation(
                    what="scanned pages: image content, no text layer",
                    detail=(
                        f"{len(scanned)} page(s) carry image content and no text layer "
                        f"(page {_listed(scanned)}); no text could be extracted from them "
                        f"and {attempted}"
                    ),
                )
            )
        if blanks:
            degradations.append(
                Degradation(
                    what="blank pages",
                    detail=(
                        f"{len(blanks)} page(s) carry no text and no page objects "
                        f"(page {_listed(blanks)}); they are blank"
                    ),
                )
            )
        return tuple(degradations)

    def _unreadable(self, ref: SourceRef, budget: Budget) -> Rendered:
        return self._nothing_to_read(
            ref,
            budget,
            summary=f"Unreadable PDF {ref.uri}, {ref.size_bytes} bytes.",
            what="pdf unopenable",
            detail="the file could not be opened as a PDF; no text was extracted",
        )

    def _nothing_to_read(
        self, ref: SourceRef, budget: Budget, *, summary: str, what: str, detail: str
    ) -> Rendered:
        """A rendition for a file with no page to point at.

        Located by `ByteRange` rather than `PageRef`: no page was ever
        observed, and claiming page 1 would be a claim about a document this
        handler never read a page of.
        """
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
        """Apply the budget, pruning the map and the barriers along with the text.

        `Rendered` rejects a map that does not cover its text exactly and a
        barrier past the end, so truncation cannot touch the text alone. A
        budget of zero still keeps one character, because `CharSpan(0, 0)`
        raises, and the degradation reports the character kept rather than the
        budget asked for.
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
