"""What a source is, and how it is named.

`SourceRef` is the only handle a handler ever gets. It carries no filesystem
path semantics on purpose: `uri` is opaque to the domain, so an archive member
addressed as `/a.zip!inner.txt` and an object-store key are the same kind of
thing. Bytes are reached through the `FileSource` port, never from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType, Self

#: A blake2b hex digest of a source's bytes. See `adapters.hashing`.
ContentHash = NewType("ContentHash", str)


@dataclass(frozen=True, slots=True)
class MimeType:
    """A parsed mimetype, without parameters.

    Parameters are dropped rather than kept: `text/plain; charset=utf-8` and
    `text/plain` must dispatch to the same handler, and keeping the parameter
    would make the registry's exact-match step depend on whether the detector
    happened to report an encoding. Encoding is a handler's concern and is
    re-derived from content.
    """

    type: str
    subtype: str
    suffix: str | None = None

    @classmethod
    def parse(cls, raw: str) -> Self:
        value = raw.split(";", 1)[0].strip().lower()
        if "/" not in value:
            raise ValueError(f"not a mimetype: {raw!r}")
        type_, subtype = value.split("/", 1)
        if not type_ or not subtype:
            raise ValueError(f"not a mimetype: {raw!r}")
        suffix = subtype.rsplit("+", 1)[1] if "+" in subtype else None
        return cls(type=type_, subtype=subtype, suffix=suffix)

    def __str__(self) -> str:
        return f"{self.type}/{self.subtype}"


class MediaKind(StrEnum):
    """The coarse family a mimetype belongs to.

    This is the registry's fourth dispatch step and nothing else. It is
    deliberately coarser than the handler families: `application/pdf` is
    `BINARY` here and still reaches a PDF handler, because that match happens
    at the exact-mimetype step long before this one.
    """

    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"
    BINARY = "binary"

    @classmethod
    def for_mime(cls, mime: MimeType) -> MediaKind:
        match mime.type:
            case "video":
                return cls.VIDEO
            case "audio":
                return cls.AUDIO
            case "image":
                return cls.IMAGE
            case "text":
                return cls.TEXT
            case _:
                return cls.BINARY


@dataclass(frozen=True, slots=True)
class SourceRef:
    """A specific sequence of bytes, and what is known about it cheaply."""

    uri: str
    mime: MimeType
    content_hash: ContentHash
    size_bytes: int

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise ValueError(f"size_bytes must not be negative, got {self.size_bytes}")
        if not self.uri:
            raise ValueError("uri must not be empty")
