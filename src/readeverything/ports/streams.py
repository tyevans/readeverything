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


#: Subtitle codecs that carry characters. Everything else a container calls a
#: subtitle carries pixels — `dvd_subtitle`, `hdmv_pgs_subtitle` — and reaching
#: its words means OCR, which costs about what reading a page of a scanned PDF
#: costs and nothing like what reading text costs. The reference file carries
#: one of each, so this is not a hypothetical distinction: presenting them as
#: one kind of thing would send an agent down the expensive path while the
#: cheap one sat beside it.
TEXT_SUBTITLE_CODECS = frozenset({"mov_text", "subrip", "srt", "ass", "ssa", "webvtt", "text"})


@dataclass(frozen=True, slots=True)
class StreamInfo:
    """One stream within a media container."""

    kind: Literal["video", "audio", "subtitle"]
    codec: str
    width: int | None
    height: int | None
    frame_rate: float | None
    sample_rate: int | None
    channels: int | None
    #: From the container's `tags.language`, `None` when it does not say.
    #: Which track to read is a real choice on a multi-track file, and a
    #: caller cannot make it blind.
    language: str | None = None
    #: Whether this stream's words can be read as characters. Always False for
    #: video and audio: the field is a fact about subtitle codecs, and a video
    #: stream answering True would make `text_subtitle_streams` a lie.
    is_text: bool = False


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

    @property
    def subtitle_streams(self) -> tuple[StreamInfo, ...]:
        return tuple(s for s in self.streams if s.kind == "subtitle")

    @property
    def text_subtitle_streams(self) -> tuple[StreamInfo, ...]:
        """Subtitle streams whose words extract to characters, not pixels."""
        return tuple(s for s in self.subtitle_streams if s.is_text)


@runtime_checkable
class StreamProbe(Protocol):
    """Cheap structural facts about a media file's streams, without decoding."""

    async def probe(self, path: str) -> MediaFacts: ...
