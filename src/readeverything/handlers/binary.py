"""The fallback that always succeeds.

Its existence is what removes the "unsupported file" error path: the registry's
last dispatch step always finds this, so the worst outcome of pointing an agent
at an unknown file is a thin, honest card rather than an exception.

`represent` deliberately does **not** emit a hexdump. Feeding hex to an
extractor produces noise that looks like claims. It emits one sentence
describing what the file is, which is the true and useful thing to index.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from readeverything.domain.affordance import Affordance, DetailLevel
from readeverything.domain.capability import Capability
from readeverything.domain.card import Card
from readeverything.domain.errors import UnknownAffordanceError
from readeverything.domain.identity import MediaKind, SourceRef
from readeverything.domain.locator_map import LocatorMap, LocatorSegment
from readeverything.domain.locators import ByteRange, CharSpan
from readeverything.domain.rendition import Budget, Rendered, Rendition, TextContent
from readeverything.ports.source import SourceReader

_EXCERPT_BYTES = 64
_BYTES_PER_LINE = 16


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

    def __init__(self, *, source: SourceReader) -> None:
        self._source = source

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
        text = (
            f"Binary file {ref.uri} of type {ref.mime}, {ref.size_bytes} bytes. "
            f"No textual content could be extracted."
        )
        if budget.max_chars is not None:
            text = text[: budget.max_chars] or text[:1]
        return Rendered(
            text=text,
            locator_map=LocatorMap.build(
                (LocatorSegment(CharSpan(0, len(text)), ByteRange(0, max(1, ref.size_bytes))),)
            ),
            barriers=(),
            degradations=(),
        )
