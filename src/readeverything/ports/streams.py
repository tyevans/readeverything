"""What can be said about a media file's shape, without decoding a frame.

`MediaProbe` (`ports/probe_media.py`) answers paginated documents and returns
`DocumentFacts` from `bytes`. This is a different question about a different
kind of source — a media file's stream structure, read from a `path` because
ffprobe needs to seek a container header rather than consume a buffer — so it
is a different protocol. A protocol whose return type depends on its input has
two jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class StreamInfo:
    """One stream within a media container."""

    kind: Literal["video", "audio"]
    codec: str
    width: int | None
    height: int | None
    frame_rate: float | None
    sample_rate: int | None
    channels: int | None


@dataclass(frozen=True, slots=True)
class MediaFacts:
    """Cheap structural facts about a media file."""

    duration_s: float
    container: str
    streams: tuple[StreamInfo, ...]

    def __post_init__(self) -> None:
        if self.duration_s < 0:
            raise ValueError(f"duration_s must not be negative, got {self.duration_s}")

    @property
    def video_streams(self) -> tuple[StreamInfo, ...]:
        return tuple(s for s in self.streams if s.kind == "video")

    @property
    def audio_streams(self) -> tuple[StreamInfo, ...]:
        return tuple(s for s in self.streams if s.kind == "audio")


@runtime_checkable
class StreamProbe(Protocol):
    """Cheap structural facts about a media file's streams, without decoding."""

    async def probe(self, path: str) -> MediaFacts: ...
