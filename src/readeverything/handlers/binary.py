"""The fallback that always succeeds.

Its existence is what removes the "unsupported file" error path: the registry's
last dispatch step always finds this, so the worst outcome of pointing an agent
at an unknown file is a thin, honest card rather than an exception.

`represent` deliberately does **not** emit a hexdump. Feeding hex to an
extractor produces noise that looks like claims. It emits one sentence
describing what the file is, which is the true and useful thing to index.
"""

from __future__ import annotations

import time
from typing import ClassVar

from pydantic import BaseModel, Field

from readeverything.domain.affordance import Affordance, DetailLevel
from readeverything.domain.capability import Capability
from readeverything.domain.card import Card
from readeverything.domain.errors import UnknownAffordanceError
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

#: What `represent` calls itself when it narrates. Matches the name every
#: other handler uses, per `video.py`.
_OPERATION = "represent"

_EXCERPT_BYTES = 64
_BYTES_PER_LINE = 16

#: `Degradation.what` for text the handler wrote about a file rather than
#: extracted from it. A consumer that indexes renditions must be able to tell
#: the difference, because attributing synthesized text to file content is a
#: false claim about the file.
SYNTHESIZED = "synthesized description"


class HexdumpParams(BaseModel):
    start: int = Field(default=0, ge=0)
    length: int = Field(default=_EXCERPT_BYTES, ge=1, le=4096)


def _hexdump(data: bytes, offset: int) -> str:
    lines: list[str] = []
    for index in range(0, len(data), _BYTES_PER_LINE):
        row = data[index : index + _BYTES_PER_LINE]
        hex_part = " ".join(f"{byte:02x}" for byte in row)
        text_part = "".join(chr(b) if 32 <= b < 127 else "." for b in row)
        lines.append(f"{offset + index:08x}  {hex_part:<47}  |{text_part}|")
    return "\n".join(lines)


class BinaryHandler:
    """Describes anything, by describing as little as is honest."""

    mime_patterns: ClassVar[tuple[str, ...]] = ("*",)
    priority: ClassVar[int] = 0
    handler_id: ClassVar[str] = "binary"
    handler_version: ClassVar[int] = 1

    def __init__(self, *, source: SourceReader, observer: Observer | None = None) -> None:
        self._source = source
        self._observer = observer

    def requires(self) -> frozenset[Capability]:
        return frozenset()

    def affordances(self) -> tuple[Affordance, ...]:
        return (
            Affordance(
                name="hexdump",
                description=(
                    "Dump a window of raw bytes as hex and printable ASCII. "
                    "Use only to identify an unknown format; it is not readable content."
                ),
                params=HexdumpParams,
                requires=frozenset(),
                level=DetailLevel.SEGMENT,
            ),
        )

    async def describe(self, ref: SourceRef) -> Card:
        head = await self._source.read_range(ref.uri, 0, _EXCERPT_BYTES)
        return Card(
            ref=ref,
            kind=MediaKind.BINARY,
            facts={"size_bytes": ref.size_bytes, "mime": str(ref.mime)},
            outline=(),
            excerpt=_hexdump(head, 0) if head else None,
            affordances=self.affordances(),
        )

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition:
        if name != "hexdump":
            raise UnknownAffordanceError(name, (a.name for a in self.affordances()))
        if not isinstance(params, HexdumpParams):
            raise TypeError(f"expected HexdumpParams, got {type(params).__name__}")
        end = params.start + params.length
        data = await self._source.read_range(ref.uri, params.start, end)
        actual_end = params.start + len(data)
        if actual_end <= params.start:
            actual_end = params.start + 1
        return Rendition(
            locator=ByteRange(params.start, actual_end),
            content=TextContent(_hexdump(data, params.start)),
        )

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
        full = (
            f"Binary file {ref.uri} of type {ref.mime}, {ref.size_bytes} bytes. "
            f"No textual content could be extracted."
        )
        text = full
        # Every byte of this rendition was written here, not read from the
        # file. The locator below points into the file because `Rendered`
        # requires a map whose length matches the text; this degradation is
        # what stops that from reading as a claim about the file's contents.
        degradations: tuple[Degradation, ...] = (
            Degradation(
                what=SYNTHESIZED,
                detail="no content was extracted; this text describes the file",
            ),
        )
        if budget.max_chars is not None and len(full) > budget.max_chars:
            # A zero-width rendition is inexpressible: `CharSpan(0, 0)` raises,
            # and `Rendered` requires `locator_map.length == len(text)`. So a
            # budget of zero still keeps one character, and the degradation must
            # report the character it kept rather than the budget it was asked
            # for — otherwise the rendition contradicts its own degradation.
            text = full[: budget.max_chars] or full[:1]
            degradations = (
                Degradation(
                    what="text truncated",
                    detail=f"kept {len(text)} of {len(full)} characters",
                ),
            )
        return Rendered(
            text=text,
            locator_map=LocatorMap.build(
                (LocatorSegment(CharSpan(0, len(text)), ByteRange(0, max(1, ref.size_bytes))),)
            ),
            barriers=(),
            degradations=degradations,
        )
