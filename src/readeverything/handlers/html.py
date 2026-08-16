"""HTML pages, read as prose but cited as markup.

An HTML file already matches `kind:text`, so `TextHandler` would take it and
hand back the raw source: tags, inline scripts, stylesheets. That is not wrong
so much as useless — for a saved article the markup is most of the bytes and
none of the answer.

The thing that makes this a handler rather than a filter is the citation. Every
other handler's `LocatorSegment` pairs a span of flattened text with a locator
that points back at the artifact, and stripping tags is exactly the operation
that breaks that correspondence: character 400 of the prose is nowhere near
character 400 of the file. So `adapters/html_text` tracks the source offset of
every block, and the map built here carries those offsets as locators. You read
the prose; you cite the html. Slicing the file at a citation and collapsing its
whitespace reproduces the quoted text, which is the property the adapter's
tests pin down.

`kind` is `TEXT`, unlike the office handlers' `BINARY`: an HTML file IS text,
its mimetype says so, and the only reason this handler sees it before
`TextHandler` is that `text/html` matches at `MatchRank.EXACT` while
`kind:text` matches at `MatchRank.KIND`. No priority tie-break is involved.

Sections mirror `office_word.py` deliberately, down to `UNTITLED_SECTION` and
the separator that keeps the map gapless: a document with headings reads the
same way whether it arrived as a `.docx` or a `.html`, and an agent that
learned one surface has learned both.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel, Field

from readeverything.adapters.html_text import HtmlBlock, html_blocks, html_title
from readeverything.domain.affordance import Affordance, DetailLevel
from readeverything.domain.capability import Capability
from readeverything.domain.card import Card, Segment
from readeverything.domain.errors import DomainError, UnknownAffordanceError
from readeverything.domain.identity import MediaKind, SourceRef
from readeverything.domain.locator_map import LocatorMap, LocatorSegment
from readeverything.domain.locators import ByteRange, CharSpan
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

#: What `represent` calls itself when it narrates, matching every other handler.
_OPERATION = "represent"

#: Ends every block, and the block's `LocatorSegment` INCLUDES it, for the
#: reason `office_word.SECTION_SEPARATOR` spells out: `LocatorMap` demands
#: gapless coverage and `CharSpan` rejects a zero-width span, so every block
#: must own at least one character of the flattened text.
BLOCK_SEPARATOR = "\n"

#: Sits between sections, on top of the last block's separator.
SECTION_SEPARATOR = "\n"

#: What the section holding any text before the first heading is called.
#: The same string `office_word.py` uses, so the two read alike.
UNTITLED_SECTION = "(untitled)"


@dataclass(frozen=True, slots=True)
class _Section:
    """One heading and the blocks beneath it, each still carrying its source span."""

    title: str
    is_titled: bool
    blocks: tuple[HtmlBlock, ...]


class ReadSectionParams(BaseModel):
    index: int = Field(default=0, ge=0, description="0-indexed section number to read.")


class ReadRangeParams(BaseModel):
    start: int = Field(default=0, ge=0, description="First character of prose to return.")
    end: int = Field(default=4096, gt=0, description="One past the last character to return.")


def _sections(blocks: tuple[HtmlBlock, ...]) -> tuple[_Section, ...]:
    """Group blocks under their headings, in document order."""
    sections: list[_Section] = []
    title: HtmlBlock | None = None
    pending: list[HtmlBlock] = []

    def close() -> None:
        nonlocal pending, title
        if title is not None:
            sections.append(_Section(title.text, True, (title, *pending)))
        elif pending:
            sections.append(_Section(UNTITLED_SECTION, False, tuple(pending)))
        pending = []
        title = None

    for block in blocks:
        if block.level:
            close()
            title = block
        else:
            pending.append(block)
    close()
    return tuple(sections)


class HtmlHandler:
    """Reads a web page's prose, and maps every character back to its markup."""

    mime_patterns: ClassVar[tuple[str, ...]] = ("text/html", "application/xhtml+xml")
    priority: ClassVar[int] = 0
    handler_id: ClassVar[str] = "html"
    handler_version: ClassVar[int] = 1

    def __init__(self, *, source: SourceReader, observer: Observer | None = None) -> None:
        self._source = source
        self._observer = observer

    def requires(self) -> frozenset[Capability]:
        """Nothing. Reading a page needs no model and no binary."""
        return frozenset()

    def affordances(self) -> tuple[Affordance, ...]:
        return (
            Affordance(
                name="read_section",
                description="Return one section of the page: a heading and the text under it.",
                params=ReadSectionParams,
                requires=frozenset(),
                level=DetailLevel.SEGMENT,
            ),
            Affordance(
                name="read_range",
                description=(
                    "Read a character range of the page's prose, with the tags removed. "
                    "Offsets index the stripped text; the locator returned points into "
                    "the original html."
                ),
                params=ReadRangeParams,
                requires=frozenset(),
                level=DetailLevel.SEGMENT,
            ),
        )

    async def _source_text(self, ref: SourceRef) -> str:
        data = await self._source.read_bytes(ref.uri)
        # `errors="replace"` rather than a charset sniff: a page that is not
        # valid UTF-8 should read imperfectly rather than not at all, and the
        # offsets must stay a function of the string actually parsed.
        return data.decode("utf-8", errors="replace")

    def _flatten(
        self, sections: tuple[_Section, ...]
    ) -> tuple[str, tuple[LocatorSegment, ...], tuple[int, ...], tuple[Segment, ...]]:
        """The prose, its map, its section barriers, and the card's outline.

        Built once and shared so the outline addresses exactly the text
        `represent` returns — computing them separately is how a table of
        contents acquires the wrong offsets.
        """
        chunks: list[str] = []
        segments: list[LocatorSegment] = []
        barriers: list[int] = []
        outline: list[Segment] = []
        cursor = 0
        for index, section in enumerate(sections):
            if index:
                barriers.append(cursor)
            section_start = cursor
            for position, block in enumerate(section.blocks):
                tail = BLOCK_SEPARATOR
                if position == len(section.blocks) - 1:
                    # The section separator belongs to the last block rather
                    # than to a gap, because a gap is what `LocatorMap` refuses.
                    tail += SECTION_SEPARATOR
                chunk = block.text + tail
                # The span is prose coordinates; the locator is the file.
                # That asymmetry is the entire point of this handler.
                segments.append(LocatorSegment(CharSpan(cursor, cursor + len(chunk)), block.span))
                chunks.append(chunk)
                cursor += len(chunk)
            outline.append(Segment(CharSpan(section_start, cursor), section.title))
        return "".join(chunks), tuple(segments), tuple(barriers), tuple(outline)

    async def _parsed(self, ref: SourceRef) -> tuple[str, tuple[_Section, ...]]:
        source = await self._source_text(ref)
        return source, _sections(html_blocks(source))

    async def describe(self, ref: SourceRef) -> Card:
        source, sections = await self._parsed(ref)
        blocks = [block for section in sections for block in section.blocks]
        text, _segments, _barriers, outline = (
            self._flatten(sections) if sections else ("", (), (), ())
        )
        return Card(
            ref=ref,
            kind=MediaKind.TEXT,
            facts={
                "title": html_title(source) or "(untitled)",
                "heading_count": sum(1 for section in sections if section.is_titled),
                "block_count": len(blocks),
                "word_count": sum(len(block.text.split()) for block in blocks),
                "characters": len(text),
                "size_bytes": ref.size_bytes,
            },
            outline=outline,
            excerpt=text[:_EXCERPT_CHARS] if text else None,
            affordances=self.affordances(),
        )

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition:
        if name == "read_section":
            if not isinstance(params, ReadSectionParams):
                raise TypeError(f"expected ReadSectionParams, got {type(params).__name__}")
            return await self._read_section(ref, params)
        if name == "read_range":
            if not isinstance(params, ReadRangeParams):
                raise TypeError(f"expected ReadRangeParams, got {type(params).__name__}")
            return await self._read_range(ref, params)
        raise UnknownAffordanceError(name, (a.name for a in self.affordances()))

    async def _read_section(self, ref: SourceRef, params: ReadSectionParams) -> Rendition:
        _source, sections = await self._parsed(ref)
        if not sections:
            raise DomainError(f"{ref.uri} has no text; there is no section to read")
        if params.index >= len(sections):
            raise DomainError(
                f"{ref.uri} has {len(sections)} sections; section {params.index} does not exist"
            )
        section = sections[params.index]
        text = "\n".join(block.text for block in section.blocks)
        # One span covering the section's markup, first block to last. The
        # blocks are contiguous in the source, so this cites no more than it read.
        locator = CharSpan(section.blocks[0].span.start, section.blocks[-1].span.end)
        return Rendition(locator=locator, content=TextContent(text))

    async def _read_range(self, ref: SourceRef, params: ReadRangeParams) -> Rendition:
        _source, sections = await self._parsed(ref)
        if not sections:
            raise DomainError(f"{ref.uri} has no text; there is no character range to read")
        text, segments, _barriers, _outline = self._flatten(sections)
        length = len(text)
        start = max(0, min(params.start, length - 1))
        end = max(start + 1, min(params.end, length))
        # The caller asked in prose coordinates and must be answered in source
        # ones, so the locator is the union of every block the range touched.
        # Every locator `_flatten` builds is a `CharSpan` into the source, but
        # `LocatorSegment.locator` is the whole `Locator` union, so narrow it
        # here rather than asserting: the union is what the domain promises.
        touched = [
            s.locator
            for s in segments
            if isinstance(s.locator, CharSpan) and s.span.start < end and s.span.end > start
        ]
        located = CharSpan(
            min(locator.start for locator in touched),
            max(locator.end for locator in touched),
        )
        return Rendition(locator=located, content=TextContent(text[start:end]))

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
        _source, sections = await self._parsed(ref)
        if not sections:
            # Located by `ByteRange` rather than a `CharSpan` into prose that
            # does not exist: citing a block here would be a claim about text
            # this handler never found.
            summary = f"HTML page {ref.uri} has no readable text."
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
                        what="page has no text",
                        detail="the file parsed as html but contains no prose outside script or style",
                    ),
                ),
            )
        text, segments, barriers, _outline = self._flatten(sections)
        return self._fit(text, segments, barriers, budget, ())

    def _fit(
        self,
        full: str,
        segments: tuple[LocatorSegment, ...],
        barriers: tuple[int, ...],
        budget: Budget,
        degradations: tuple[Degradation, ...],
    ) -> Rendered:
        """Apply the budget, pruning the map and the barriers along with the text.

        Truncation cannot touch the text alone: `Rendered` rejects a map that
        does not cover its text exactly. A budget of zero still keeps one
        character, because `CharSpan(0, 0)` raises.
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
