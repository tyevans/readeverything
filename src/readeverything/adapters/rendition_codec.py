"""Serializing a `Rendition` for the artifact cache, without guessing at unions.

`pydantic.TypeAdapter(Rendition)` looks like the obvious way to do this. It is
not safe here, and the unsafety is silent:

- `Locator` is a non-discriminated union (`TimeSpan | PageRef | BBox |
  CharSpan | ByteRange | CellRange`). `CharSpan` and `ByteRange` have identical field
  shapes (`start`, `end`), so pydantic resolves a serialized `ByteRange` to
  whichever union member comes first in the annotation, which is `CharSpan`.
  A cached hexdump — whose locator is a `ByteRange` — would come back
  claiming character offsets. Nothing raises; a caller just resolves the span
  against the wrong axis and gets a confidently wrong answer.
- `ImageContent.data` is raw `bytes`, and JSON has no bytes type. Pydantic's
  default JSON encoding tries utf-8 and raises on the first non-text byte,
  which is essentially every real image.

The fix is a tagged envelope: every locator and content object is encoded
with a `__type__` field naming its concrete class, and decoding looks that
name up in an explicit table rather than letting a union guess by shape. An
unrecognised tag raises `DomainError` — a cache miss costs a recomputation; a
guess costs a wrong answer. Do not "simplify" this back into
`TypeAdapter(Rendition)`; that reintroduces the corruption above.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from readeverything.domain.errors import DomainError
from readeverything.domain.locators import (
    BBox,
    ByteRange,
    CellRange,
    CharSpan,
    Locator,
    PageRef,
    TimeSpan,
)
from readeverything.domain.rendition import (
    ImageContent,
    Rendition,
    RenditionContent,
    StructuredContent,
    TextContent,
)


def _encode_locator(locator: Locator) -> dict[str, Any]:
    match locator:
        case CharSpan(start=start, end=end):
            return {"__type__": "CharSpan", "start": start, "end": end}
        case ByteRange(start=start, end=end):
            return {"__type__": "ByteRange", "start": start, "end": end}
        case TimeSpan(start_s=start_s, end_s=end_s):
            return {"__type__": "TimeSpan", "start_s": start_s, "end_s": end_s}
        case PageRef(page=page):
            return {"__type__": "PageRef", "page": page}
        case BBox(page=page, x=x, y=y, w=w, h=h):
            return {"__type__": "BBox", "page": page, "x": x, "y": y, "w": w, "h": h}
        case CellRange(sheet=sheet, row=row, col=col, rows=rows, cols=cols):
            return {
                "__type__": "CellRange",
                "sheet": sheet,
                "row": row,
                "col": col,
                "rows": rows,
                "cols": cols,
            }
    raise DomainError(f"unencodable locator type: {type(locator).__name__}")


def _decode_locator(raw: dict[str, Any]) -> Locator:
    tag = raw.get("__type__")
    match tag:
        case "CharSpan":
            return CharSpan(raw["start"], raw["end"])
        case "ByteRange":
            return ByteRange(raw["start"], raw["end"])
        case "TimeSpan":
            return TimeSpan(raw["start_s"], raw["end_s"])
        case "PageRef":
            return PageRef(raw["page"])
        case "BBox":
            return BBox(raw["page"], raw["x"], raw["y"], raw["w"], raw["h"])
        case "CellRange":
            return CellRange(
                sheet=raw["sheet"],
                row=raw["row"],
                col=raw["col"],
                rows=raw["rows"],
                cols=raw["cols"],
            )
        case _:
            raise DomainError(f"unknown locator __type__ tag: {tag!r}")


def _encode_content(content: RenditionContent) -> dict[str, Any]:
    match content:
        case TextContent(text=text):
            return {"__type__": "TextContent", "text": text}
        case ImageContent(data=data, mime=mime):
            return {
                "__type__": "ImageContent",
                "data": base64.b64encode(data).decode("ascii"),
                "mime": mime,
            }
        case StructuredContent(rows=rows):
            return {"__type__": "StructuredContent", "rows": list(rows)}
    raise DomainError(f"unencodable content type: {type(content).__name__}")


def _decode_content(raw: dict[str, Any]) -> RenditionContent:
    tag = raw.get("__type__")
    match tag:
        case "TextContent":
            return TextContent(raw["text"])
        case "ImageContent":
            return ImageContent(base64.b64decode(raw["data"]), raw["mime"])
        case "StructuredContent":
            # JSON gives back lists of dicts; StructuredContent.rows is a
            # tuple of dicts, and the round-trip fails on type without this.
            return StructuredContent(tuple(raw["rows"]))
        case _:
            raise DomainError(f"unknown content __type__ tag: {tag!r}")


def encode_rendition(rendition: Rendition) -> bytes:
    """Serialize `rendition` to bytes suitable for an `ArtifactStore`."""
    payload = {
        "locator": _encode_locator(rendition.locator),
        "content": _encode_content(rendition.content),
        "degraded": rendition.degraded,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def decode_rendition(raw: bytes) -> Rendition:
    """The inverse of `encode_rendition`. Raises `DomainError` on an unknown tag."""
    payload = json.loads(raw)
    return Rendition(
        locator=_decode_locator(payload["locator"]),
        content=_decode_content(payload["content"]),
        degraded=payload["degraded"],
    )
