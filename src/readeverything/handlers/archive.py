"""Containers, as things in their own right.

An agent that lists a directory should learn what is in `release.tar.gz`
without descending into it, so the container gets a card of its own even
though `adapters/nested_source.py` is what makes its members readable.

`kind` is `BINARY` for the reason the README already gives for PDF:
`MediaKind` names how bytes are SHAPED, and a container's shape is binary.
What it *is* is carried by its facts and its affordances.

There is exactly one affordance and it is paged. Paging because a
40,000-entry tarball is not one response. One, because reading a member is
spelled `inspect("a.zip!inner.txt")` -- adding a `read_entry` here would give
one sequence of bytes two ways to be reached, and therefore two provenance
stories for one citation, which is the failure this library exists to prevent.

Card cost stays inside the contract: reading a zip's central directory or
walking tar headers is a probe, not a decompression.
"""

from __future__ import annotations

import time
from typing import ClassVar

from pydantic import BaseModel, Field

from readeverything.domain.affordance import Affordance, DetailLevel
from readeverything.domain.capability import Capability
from readeverything.domain.card import Card, Segment
from readeverything.domain.errors import SourceUnreadableError, UnknownAffordanceError
from readeverything.domain.identity import MediaKind, SourceRef
from readeverything.domain.locator_map import LocatorMap, LocatorSegment
from readeverything.domain.locators import ByteRange, CharSpan, Locator
from readeverything.domain.observation import OperationFinished, OperationStarted
from readeverything.domain.rendition import (
    Budget,
    Degradation,
    Rendered,
    Rendition,
    TextContent,
)
from readeverything.ports.containers import ARCHIVE_MIMES, ArchiveEntry, ArchiveOpener
from readeverything.ports.observation import Observer, emit
from readeverything.ports.source import SourceReader

#: What `represent` calls itself when it narrates. Matches the name every
#: other handler uses, per `video.py`.
_OPERATION = "represent"

#: How many member paths the card shows. A card is what a human skims, and a
#: skim of a 40,000-entry tarball is its first few names, not all of them.
_EXCERPT_ENTRIES = 20


class ListEntriesParams(BaseModel):
    offset: int = Field(default=0, ge=0, description="0-indexed entry to start from.")
    limit: int = Field(default=200, ge=1, le=2000, description="How many entries to return.")


def _line(entry: ArchiveEntry) -> str:
    kind = "dir " if entry.is_dir else ("link" if entry.is_symlink else "file")
    return f"{kind} {entry.size_bytes:>12}  {entry.path}"


class ArchiveHandler:
    """Describes a container without descending into it."""

    mime_patterns: ClassVar[tuple[str, ...]] = tuple(sorted(ARCHIVE_MIMES))
    priority: ClassVar[int] = 0
    handler_id: ClassVar[str] = "archive"
    handler_version: ClassVar[int] = 1

    def __init__(
        self,
        *,
        source: SourceReader,
        archives: ArchiveOpener,
        observer: Observer | None = None,
    ) -> None:
        self._source = source
        self._archives = archives
        self._observer = observer

    def requires(self) -> frozenset[Capability]:
        """Nothing. zipfile and tarfile are stdlib; there is no binary here."""
        return frozenset()

    def affordances(self) -> tuple[Affordance, ...]:
        return (
            Affordance(
                name="list_entries",
                description=(
                    "List the members of this container, a page at a time. "
                    "To READ a member, inspect it directly at "
                    "'<this uri>!<member path>' — that is the only way to reach "
                    "its bytes, and it gives you the member's own card and "
                    "affordances rather than a blob."
                ),
                params=ListEntriesParams,
                requires=frozenset(),
                level=DetailLevel.SEGMENT,
            ),
        )

    def _opener(self, mime_holder: SourceRef) -> ArchiveOpener | None:
        """The opener for this ref's mimetype, when `archives` is a composite.

        A composite has no one format and refuses to answer `entries` itself,
        so it must be asked which of its openers applies. A caller who wired a
        single opener has nothing to dispatch and gets it back unchanged.
        """
        chooser = getattr(self._archives, "opener_for", None)
        if chooser is None:
            return self._archives
        opener: ArchiveOpener | None = chooser(mime_holder.mime)
        return opener

    async def _entries(self, ref: SourceRef) -> tuple[ArchiveEntry, ...] | None:
        """The container's directory, or None when it could not be opened.

        None rather than an exception: this handler never raises about its
        input, matching `PdfHandler._open`. The source layer is where an
        unreadable container raises; a handler reports.
        """
        opener = self._opener(ref)
        if opener is None:
            return None
        try:
            path = await self._source.local_path(ref.uri)
            return tuple(await opener.entries(path))
        except (SourceUnreadableError, OSError, NotImplementedError):
            return None

    async def describe(self, ref: SourceRef) -> Card:
        entries = await self._entries(ref)
        if entries is None:
            return Card(
                ref=ref,
                kind=MediaKind.BINARY,
                facts={"readable": "no", "size_bytes": ref.size_bytes},
                outline=(),
                excerpt=None,
                affordances=self.affordances(),
            )
        uncompressed = sum(entry.size_bytes for entry in entries)
        compressed = sum(entry.compressed_bytes for entry in entries)
        # A solid container is one whose members have no seekable place in the
        # file. That is the fact a caller needs to predict what reading three
        # members will cost, so it is on the card rather than left to be
        # inferred from a surprise.
        solid = bool(entries) and all(entry.byte_offset is None for entry in entries)
        return Card(
            ref=ref,
            kind=MediaKind.BINARY,
            facts={
                "readable": "yes",
                "format": str(ref.mime),
                "entry_count": len(entries),
                "uncompressed_bytes": uncompressed,
                "compressed_bytes": compressed,
                "expansion_ratio": round(uncompressed / compressed, 2) if compressed else 0.0,
                "solid": "yes" if solid else "no",
                "size_bytes": ref.size_bytes,
            },
            outline=tuple(
                Segment(_locator(entry, index), entry.path) for index, entry in enumerate(entries)
            ),
            excerpt="\n".join(entry.path for entry in entries[:_EXCERPT_ENTRIES]) or None,
            affordances=self.affordances(),
        )

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition:
        if name != "list_entries":
            raise UnknownAffordanceError(name, (a.name for a in self.affordances()))
        if not isinstance(params, ListEntriesParams):
            raise TypeError(f"expected ListEntriesParams, got {type(params).__name__}")
        entries = await self._entries(ref)
        if entries is None:
            return self._degraded(ref, f"{ref.uri} could not be opened as an archive")
        page = entries[params.offset : params.offset + params.limit]
        if not page:
            return self._degraded(
                ref,
                f"offset {params.offset} is past the end; "
                f"this container has {len(entries)} entry(ies)",
            )
        header = f"{len(page)} of {len(entries)} entries, from offset {params.offset}\n"
        body = header + "\n".join(_line(entry) for entry in page)
        return Rendition(locator=ByteRange(0, max(1, ref.size_bytes)), content=TextContent(body))

    def _degraded(self, ref: SourceRef, detail: str) -> Rendition:
        """Never an exception: an agent guessing an offset gets a readable answer."""
        return Rendition(
            locator=ByteRange(0, max(1, ref.size_bytes)),
            content=TextContent(detail),
            degraded=True,
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
        """The entry listing as text, with a locator over every line.

        `barriers` stays empty: an entry listing has no natural chunk
        boundary, and inventing one would tell a chunker something about this
        text that is not true of it.
        """
        entries = await self._entries(ref)
        if not entries:
            # An archive that opened and holds nothing is NOT unreadable, and
            # conflating the two would report a false failure about a file
            # this handler successfully read.
            unreadable = entries is None
            summary = (
                f"Unreadable archive {ref.uri}, {ref.size_bytes} bytes."
                if unreadable
                else f"Archive {ref.uri} opened and has no entries."
            )
            detail = (
                "the file could not be opened as an archive; no entries were listed"
                if unreadable
                else "the archive opened and declares no members"
            )
            return self._fit(
                summary,
                (LocatorSegment(CharSpan(0, len(summary)), ByteRange(0, max(1, ref.size_bytes))),),
                budget,
                (
                    Degradation(
                        what="archive unlistable" if unreadable else "archive is empty",
                        detail=detail,
                    ),
                ),
            )
        chunks: list[str] = []
        segments: list[LocatorSegment] = []
        cursor = 0
        for index, entry in enumerate(entries):
            # The trailing newline is INSIDE the segment, exactly as
            # `pdf.PAGE_SEPARATOR` is inside a page's: `LocatorMap` demands
            # gapless zero-start coverage and `CharSpan` rejects a zero-width
            # span, so a separator owned by nobody is what breaks the map.
            chunk = _line(entry) + "\n"
            segments.append(
                LocatorSegment(CharSpan(cursor, cursor + len(chunk)), _locator(entry, index))
            )
            cursor += len(chunk)
            chunks.append(chunk)
        return self._fit("".join(chunks), tuple(segments), budget, ())

    def _fit(
        self,
        full: str,
        segments: tuple[LocatorSegment, ...],
        budget: Budget,
        degradations: tuple[Degradation, ...],
    ) -> Rendered:
        """Apply the budget, pruning the map along with the text.

        `Rendered` rejects a map that does not cover its text exactly, so
        truncation cannot touch the text alone. A budget of zero still keeps
        one character, because `CharSpan(0, 0)` raises.
        """
        if budget.max_chars is None or len(full) <= budget.max_chars:
            return Rendered(
                text=full,
                locator_map=LocatorMap.build(segments),
                barriers=(),
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
            barriers=(),
            degradations=(
                *degradations,
                Degradation(
                    what="text truncated",
                    detail=f"kept {keep} of {len(full)} characters",
                ),
            ),
        )


def _locator(entry: ArchiveEntry, index: int) -> Locator:
    """Where an entry is, in whatever terms the format actually supports.

    A `ByteRange` when the container gives an offset. A solid container gives
    none -- an offset into a gzip stream is not somewhere anyone can seek to
    -- so those fall back to the entry's position in the listing this handler
    itself produces, which is a place that genuinely exists rather than an
    invented byte range into a file nobody can seek.
    """
    if entry.byte_offset is None:
        return CharSpan(index, index + 1)
    return ByteRange(entry.byte_offset, entry.byte_offset + max(1, entry.compressed_bytes))
