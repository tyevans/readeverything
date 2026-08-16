"""EPUBs, read as a book rather than as a folder of XML.

An EPUB is a zip, and `NOT_A_FOLDER_MIMES` has always said so -- descending
into one would bury the book under a manifest, a stylesheet and a dozen
numbered parts, and the agent would have to reassemble the novel itself. Until
now that decision left the format with nowhere to go: no handler claimed it, so
a book fell through to the hex dump. This is the handler that was missing from
that sentence.

Chapters are the spine's items, in the spine's order, because that is the only
place in the file that says what order the book is read in. Their prose comes
from `adapters/html_text`, the same reader `handlers/html.py` uses, so a
chapter of a book and a saved web page are read by identical code -- and a
chapter inherits the citation property that reader exists for.

The locator is `PartSpan`, which this format is the reason for. A chapter lives
in its own file inside the zip, so a citation has to name two things: which
part, and where in it. Extract `OEBPS/ch3.xhtml`, slice it at the offsets, and
the quoted sentence is there. A `CharSpan` would have named offsets with no
file; a `ByteRange` into the epub would have addressed compressed bytes.
"""

from __future__ import annotations

import time
from typing import ClassVar

from pydantic import BaseModel, Field

from readeverything.adapters.epub_book import Epub, EpubPart, read_epub
from readeverything.domain.affordance import Affordance, DetailLevel
from readeverything.domain.capability import Capability
from readeverything.domain.card import Card, Segment
from readeverything.domain.errors import DomainError, UnknownAffordanceError
from readeverything.domain.identity import MediaKind, SourceRef
from readeverything.domain.locator_map import LocatorMap, LocatorSegment
from readeverything.domain.locators import ByteRange, CharSpan, PartSpan
from readeverything.domain.observation import OperationFinished, OperationStarted
from readeverything.domain.rendition import (
    Budget,
    Degradation,
    Rendered,
    Rendition,
    TextContent,
)
from readeverything.ports.observation import Observer, emit
from readeverything.ports.source import SourceReader

_EXCERPT_CHARS = 1000

_OPERATION = "represent"

#: Ends every block, and the block's `LocatorSegment` INCLUDES it: `LocatorMap`
#: demands gapless coverage and `CharSpan` rejects a zero-width span, so every
#: block has to own at least one character. Same reasoning as `handlers/html`.
BLOCK_SEPARATOR = "\n"

#: Sits between chapters, on top of the last block's separator.
CHAPTER_SEPARATOR = "\n"

#: What a book with no `dc:title` is called, rather than leaving the fact absent.
UNTITLED_BOOK = "(untitled)"


class ReadChapterParams(BaseModel):
    index: int = Field(default=0, ge=0, description="0-indexed chapter, in reading order.")


class ReadRangeParams(BaseModel):
    start: int = Field(default=0, ge=0, description="First character of prose to return.")
    end: int = Field(default=4096, gt=0, description="One past the last character to return.")


class EpubHandler:
    """Reads an EPUB's chapters, and cites the part each sentence came from."""

    mime_patterns: ClassVar[tuple[str, ...]] = ("application/epub+zip",)
    priority: ClassVar[int] = 0
    handler_id: ClassVar[str] = "epub"
    handler_version: ClassVar[int] = 1

    def __init__(self, *, source: SourceReader, observer: Observer | None = None) -> None:
        self._source = source
        self._observer = observer

    def requires(self) -> frozenset[Capability]:
        """Nothing. A book is a zip of XHTML, and stdlib reads both."""
        return frozenset()

    def affordances(self) -> tuple[Affordance, ...]:
        return (
            Affordance(
                name="read_chapter",
                description="Return one chapter of the book, in the spine's reading order.",
                params=ReadChapterParams,
                requires=frozenset(),
                level=DetailLevel.SEGMENT,
            ),
            Affordance(
                name="read_range",
                description=(
                    "Read a character range of the book's prose. Offsets index the "
                    "whole book as it reads; the locator returned names the part the "
                    "range came from."
                ),
                params=ReadRangeParams,
                requires=frozenset(),
                level=DetailLevel.SEGMENT,
            ),
        )

    async def _book(self, ref: SourceRef) -> Epub:
        return read_epub(await self._source.read_bytes(ref.uri))

    def _flatten(
        self, parts: tuple[EpubPart, ...]
    ) -> tuple[str, tuple[LocatorSegment, ...], tuple[int, ...], tuple[Segment, ...]]:
        """The book's prose, its map, its chapter barriers, and its outline.

        Built once and shared, so the outline addresses exactly the text
        `represent` returns rather than a second flattening that drifted.
        """
        chunks: list[str] = []
        segments: list[LocatorSegment] = []
        barriers: list[int] = []
        outline: list[Segment] = []
        cursor = 0
        for index, part in enumerate(parts):
            if index and cursor:
                # A chapter boundary is the one place a chunker should always
                # be willing to split, so it is named rather than inferred.
                barriers.append(cursor)
            chapter_start = cursor
            for position, block in enumerate(part.blocks):
                tail = BLOCK_SEPARATOR
                if position == len(part.blocks) - 1:
                    tail += CHAPTER_SEPARATOR
                chunk = block.text + tail
                segments.append(
                    LocatorSegment(
                        CharSpan(cursor, cursor + len(chunk)),
                        # Prose coordinates on the left, the part on the right.
                        PartSpan(part.name, block.span.start, block.span.end),
                    )
                )
                chunks.append(chunk)
                cursor += len(chunk)
            if cursor > chapter_start:
                outline.append(Segment(CharSpan(chapter_start, cursor), part.title))
        return "".join(chunks), tuple(segments), tuple(barriers), tuple(outline)

    async def describe(self, ref: SourceRef) -> Card:
        book = await self._book(ref)
        text, _segments, _barriers, outline = self._flatten(book.parts)
        blocks = [block for part in book.parts for block in part.blocks]
        return Card(
            ref=ref,
            kind=MediaKind.BINARY,
            facts={
                "title": book.title or UNTITLED_BOOK,
                "author": book.author or "(unknown)",
                "chapter_count": len(book.parts),
                "word_count": sum(len(block.text.split()) for block in blocks),
                "characters": len(text),
                "size_bytes": ref.size_bytes,
            },
            outline=outline,
            excerpt=text[:_EXCERPT_CHARS] if text else None,
            affordances=self.affordances(),
        )

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition:
        if name == "read_chapter":
            if not isinstance(params, ReadChapterParams):
                raise TypeError(f"expected ReadChapterParams, got {type(params).__name__}")
            return await self._read_chapter(ref, params)
        if name == "read_range":
            if not isinstance(params, ReadRangeParams):
                raise TypeError(f"expected ReadRangeParams, got {type(params).__name__}")
            return await self._read_range(ref, params)
        raise UnknownAffordanceError(name, (a.name for a in self.affordances()))

    async def _read_chapter(self, ref: SourceRef, params: ReadChapterParams) -> Rendition:
        book = await self._book(ref)
        if params.index >= len(book.parts):
            raise DomainError(
                f"{ref.uri} has {len(book.parts)} chapters; chapter {params.index} does not exist"
            )
        part = book.parts[params.index]
        if not part.blocks:
            raise DomainError(f"chapter {params.index} of {ref.uri} has no readable text")
        text = "\n".join(block.text for block in part.blocks)
        locator = PartSpan(part.name, part.blocks[0].span.start, part.blocks[-1].span.end)
        return Rendition(locator=locator, content=TextContent(text))

    async def _read_range(self, ref: SourceRef, params: ReadRangeParams) -> Rendition:
        book = await self._book(ref)
        text, segments, _barriers, _outline = self._flatten(book.parts)
        if not text:
            raise DomainError(f"{ref.uri} has no readable text; there is no range to read")
        length = len(text)
        start = max(0, min(params.start, length - 1))
        end = max(start + 1, min(params.end, length))
        # The caller asked in book coordinates and is answered in part ones. A
        # range spanning two chapters cites the first part it touched rather
        # than inventing a span across two files, which nothing could resolve.
        touched = [
            s.locator
            for s in segments
            if isinstance(s.locator, PartSpan) and s.span.start < end and s.span.end > start
        ]
        within = [locator for locator in touched if locator.part == touched[0].part]
        located = PartSpan(
            touched[0].part,
            min(locator.start for locator in within),
            max(locator.end for locator in within),
        )
        return Rendition(locator=located, content=TextContent(text[start:end]))

    async def represent(self, ref: SourceRef, budget: Budget) -> Rendered:
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
        book = await self._book(ref)
        text, segments, barriers, _outline = self._flatten(book.parts)
        if not text:
            # Located by `ByteRange`, because citing a part here would be a
            # claim about text this handler did not find.
            summary = f"EPUB {ref.uri} has {len(book.parts)} parts and no readable text."
            empty: tuple[LocatorSegment, ...] = (
                LocatorSegment(CharSpan(0, len(summary)), ByteRange(0, max(1, ref.size_bytes))),
            )
            return self._fit(
                summary,
                empty,
                (),
                budget,
                (
                    Degradation(
                        what="book has no text",
                        detail="the spine parsed but none of its parts contain prose",
                    ),
                ),
            )
        return self._fit(text, segments, barriers, budget, ())

    def _fit(
        self,
        full: str,
        segments: tuple[LocatorSegment, ...],
        barriers: tuple[int, ...],
        budget: Budget,
        degradations: tuple[Degradation, ...],
    ) -> Rendered:
        """Apply the budget, pruning the map and the barriers with the text.

        `Rendered` rejects a map that does not cover its text exactly, so
        truncation cannot touch the text alone. A budget of zero still keeps
        one character, because `CharSpan(0, 0)` raises.
        """
        if budget.max_chars is None or len(full) <= budget.max_chars:
            return Rendered(
                text=full,
                locator_map=LocatorMap.build(segments),
                barriers=barriers,
                degradations=degradations,
            )
        keep = max(1, budget.max_chars)
        kept = tuple(
            LocatorSegment(CharSpan(s.span.start, min(s.span.end, keep)), s.locator)
            for s in segments
            if s.span.start < keep
        )
        return Rendered(
            text=full[:keep],
            locator_map=LocatorMap.build(kept),
            barriers=tuple(barrier for barrier in barriers if barrier < keep),
            degradations=(
                *degradations,
                Degradation(
                    what="text truncated",
                    detail=f"kept {keep} of {len(full)} characters",
                ),
            ),
        )
