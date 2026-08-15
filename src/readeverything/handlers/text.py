"""Text and source code.

The simplest possible handler that is still a real one: it needs no
capabilities, so it exercises the registry's satisfied path, and it produces a
genuine `LocatorMap` over `CharSpan`s, so it exercises the citation path
end to end without any model or binary being involved.
"""

from __future__ import annotations

import time
from typing import ClassVar

from charset_normalizer import from_bytes
from pydantic import BaseModel, Field

from readeverything.domain.affordance import Affordance, DetailLevel
from readeverything.domain.capability import Capability
from readeverything.domain.card import Card, Segment
from readeverything.domain.errors import DomainError, UnknownAffordanceError
from readeverything.domain.identity import MediaKind, SourceRef
from readeverything.domain.locator_map import LocatorMap, LocatorSegment
from readeverything.domain.locators import CharSpan
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

#: What `represent` calls itself when it narrates. Matches the name every
#: other handler uses, per `video.py`.
_OPERATION = "represent"

#: `Degradation.what` for text the handler wrote about a file rather than
#: extracted from it. See `binary.SYNTHESIZED` — the same string, deliberately,
#: so a consumer matches one value across every handler.
SYNTHESIZED = "synthesized description"


class ReadRangeParams(BaseModel):
    start: int = Field(default=0, ge=0)
    end: int = Field(default=_EXCERPT_CHARS, ge=1)


class TextHandler:
    """Reads decodable text."""

    mime_patterns: ClassVar[tuple[str, ...]] = ("kind:text", "application/json", "application/xml")
    priority: ClassVar[int] = 0
    handler_id: ClassVar[str] = "text"
    handler_version: ClassVar[int] = 1

    def __init__(self, *, source: SourceReader, observer: Observer | None = None) -> None:
        self._source = source
        self._observer = observer

    def requires(self) -> frozenset[Capability]:
        return frozenset()

    def affordances(self) -> tuple[Affordance, ...]:
        return (
            Affordance(
                name="read_range",
                description=(
                    "Read a character range of a text file. "
                    "Offsets are characters, not bytes, and the end is clamped to the file."
                ),
                params=ReadRangeParams,
                requires=frozenset(),
                level=DetailLevel.SEGMENT,
            ),
        )

    async def _text(self, ref: SourceRef) -> tuple[str, str]:
        """The decoded text and the encoding it was decoded with."""
        data = await self._source.read_bytes(ref.uri)
        if not data:
            return "", "utf-8"
        try:
            return data.decode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            best = from_bytes(data).best()
            if best is None:
                return data.decode("utf-8", errors="replace"), "utf-8/replace"
            return str(best), best.encoding

    async def describe(self, ref: SourceRef) -> Card:
        text, encoding = await self._text(ref)
        lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        outline = (Segment(CharSpan(0, len(text)), "whole file"),) if text else ()
        return Card(
            ref=ref,
            kind=MediaKind.TEXT,
            facts={"lines": lines, "characters": len(text), "encoding": encoding},
            outline=outline,
            excerpt=text[:_EXCERPT_CHARS] if text else None,
            affordances=self.affordances(),
        )

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition:
        if name != "read_range":
            raise UnknownAffordanceError(name, (a.name for a in self.affordances()))
        if not isinstance(params, ReadRangeParams):
            raise TypeError(f"expected ReadRangeParams, got {type(params).__name__}")
        text, _ = await self._text(ref)
        if not text:
            raise DomainError(f"{ref.uri} is empty; there is no character range to read")
        # Clamp both ends against the text, not just `start`. Clamping `start`
        # alone silently discarded the caller's `end` whenever `start` was at
        # or past the last character, always returning exactly one character —
        # a rendition whose locator did not describe the text beside it.
        length = len(text)
        start = max(0, min(params.start, length - 1))
        end = max(start + 1, min(params.end, length))
        body = text[start:end]
        return Rendition(locator=CharSpan(start, end), content=TextContent(body))

    async def represent(self, ref: SourceRef, budget: Budget) -> Rendered:
        """Narrated start to finish, matching `AudioHandler`/`VideoHandler`.

        One step, so only `OperationStarted`/`OperationFinished` fire — there
        is no per-unit loop here for `OperationProgressed` to describe.
        """
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
        full, _ = await self._text(ref)
        degradations: tuple[Degradation, ...] = ()
        if not full:
            # Only a genuinely empty source earns this. A truncated one is not
            # empty, and saying so would index a false claim about the file.
            text = f"[empty text file: {ref.uri}]"
            degradations = (
                Degradation(
                    what=SYNTHESIZED,
                    detail="the file is empty; this text describes it",
                ),
            )
        elif budget.max_chars is not None and len(full) > budget.max_chars:
            # A zero-width rendition is inexpressible — `CharSpan(0, 0)` raises
            # and `Rendered` requires `locator_map.length == len(text)` — so a
            # budget of zero still keeps one character, and the degradation
            # reports that character rather than the budget it was asked for.
            text = full[: budget.max_chars] or full[:1]
            degradations = (
                Degradation(
                    what="text truncated",
                    detail=f"kept {len(text)} of {len(full)} characters",
                ),
            )
        else:
            text = full
        return Rendered(
            text=text,
            locator_map=LocatorMap.build(
                (LocatorSegment(CharSpan(0, len(text)), CharSpan(0, len(text))),)
            ),
            barriers=(),
            degradations=degradations,
        )
