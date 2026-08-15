from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from readeverything.adapters.ffmpeg_frames import FfmpegFrames
from readeverything.adapters.ffprobe_streams import FfprobeStreams
from readeverything.domain.capability import Capability
from readeverything.domain.errors import UnknownAffordanceError
from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
from readeverything.domain.locators import ByteRange, TimeSpan
from readeverything.domain.rendition import Budget, Rendered
from readeverything.handlers.video import VideoHandler
from readeverything.ports.frames import FrameExtractor
from readeverything.ports.streams import MediaFacts, StreamInfo, StreamProbe
from readeverything.testing.fakes import FakeVision
from readeverything.testing.handler_compliance import MediaHandlerCompliance


class _PathSource:
    """Serves one real filesystem path under any uri.

    The handler reads a path, not bytes — ffprobe seeks a container header —
    so `FakeSource`, whose `local_path` raises by design, cannot back it.
    """

    def __init__(self, path: str) -> None:
        self._path = path

    async def read_bytes(self, uri: str) -> bytes:
        return Path(self._path).read_bytes()

    async def read_range(self, uri: str, start: int, end: int) -> bytes:
        return Path(self._path).read_bytes()[start:end]

    def stream(self, uri: str, *, chunk_size: int = 1 << 20):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def local_path(self, uri: str) -> str:
        return self._path


class _NoPathSource:
    """A source that cannot materialise a path at all."""

    async def read_bytes(self, uri: str) -> bytes:
        return b""

    async def read_range(self, uri: str, start: int, end: int) -> bytes:
        return b""

    def stream(self, uri: str, *, chunk_size: int = 1 << 20):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def local_path(self, uri: str) -> str:
        raise OSError("no local path")


class _RaisingProbe:
    async def probe(self, path: str) -> MediaFacts:
        raise RuntimeError("ffprobe exploded")


class _StubProbe:
    def __init__(self, facts: MediaFacts) -> None:
        self._facts = facts

    async def probe(self, path: str) -> MediaFacts:
        return self._facts


class _RaisingFrames:
    async def frame_at(self, path: str, seconds: float) -> bytes | None:
        raise RuntimeError("ffmpeg exploded")


class _NoFrames:
    async def frame_at(self, path: str, seconds: float) -> bytes | None:
        return None


class _RefusingVision:
    model_id = "refusing@1"

    async def describe(self, data: bytes, mime: str, prompt: str) -> str:
        return "   "


def _time_spans(rendered: Rendered) -> list[TimeSpan]:
    """The map's locators, narrowed. Every one must be a `TimeSpan`."""
    spans = [s.locator for s in rendered.locator_map.segments]
    assert all(isinstance(s, TimeSpan) for s in spans)
    return [s for s in spans if isinstance(s, TimeSpan)]


def _ref(uri: str = "a.mp4", size_bytes: int = 4096) -> SourceRef:
    return SourceRef(
        uri=uri,
        mime=MimeType.parse("video/mp4"),
        content_hash=ContentHash("f" * 64),
        size_bytes=size_bytes,
    )


def _handler(
    path: str,
    *,
    vision: object | None = None,
    interval: float = 2.0,
    frames: object | None = None,
) -> VideoHandler:
    return VideoHandler(
        source=_PathSource(path),
        probe=FfprobeStreams(),
        frames=frames or FfmpegFrames(),  # type: ignore[arg-type]  # structural stub in tests
        vision=vision,  # type: ignore[arg-type]  # structural stub in tests
        sample_interval_s=interval,
    )


def _facts(duration_s: float = 5.0, *, frame_rate: float | None = 10.0) -> MediaFacts:
    return MediaFacts(
        duration_s=duration_s,
        container="mov,mp4",
        streams=(
            StreamInfo(
                kind="video",
                codec="h264",
                width=320,
                height=240,
                frame_rate=frame_rate,
                sample_rate=None,
                channels=None,
            ),
        ),
    )


def _stub_handler(facts: MediaFacts, **kwargs: object) -> VideoHandler:
    return VideoHandler(
        source=_PathSource("/nowhere.mp4"),
        probe=_StubProbe(facts),
        frames=kwargs.get("frames") or _NoFrames(),  # type: ignore[arg-type]
        vision=kwargs.get("vision"),  # type: ignore[arg-type]
        sample_interval_s=float(kwargs.get("interval", 2.0)),  # type: ignore[arg-type]
    )


# --- the card -----------------------------------------------------------------


def test_the_ports_are_satisfied_by_the_real_adapters() -> None:
    assert isinstance(FfprobeStreams(), StreamProbe)
    assert isinstance(FfmpegFrames(), FrameExtractor)


def test_the_handler_declares_ffmpeg() -> None:
    """Without ffmpeg nothing can be learned about a container at all, so the
    registry drops the handler entirely rather than dropping affordances."""
    assert _stub_handler(_facts()).requires() == frozenset({Capability.FFMPEG})


async def test_the_card_reports_duration_and_resolution_without_decoding(
    sample_video: str,
) -> None:
    """The card costs a probe. No frame is extracted — an extractor that raises
    on every call proves it, since a card that decoded would fail."""
    handler = _handler(sample_video, frames=_RaisingFrames())
    card = await handler.describe(_ref())
    assert card.kind is MediaKind.VIDEO
    assert card.facts["duration_s"] == pytest.approx(5.0, abs=0.2)
    assert card.facts["width"] == 320
    assert card.facts["height"] == 240
    assert card.facts["audio_streams"] == 1


async def test_the_card_of_a_silent_video_reports_no_audio_stream(
    silent_video: str,
) -> None:
    card = await _handler(silent_video).describe(_ref())
    assert card.facts["audio_streams"] == 0
    assert card.facts["video_streams"] == 1


async def test_the_card_outline_is_the_sampling_grid(sample_video: str) -> None:
    card = await _handler(sample_video, interval=2.0).describe(_ref())
    assert next(s.locator for s in card.outline) == TimeSpan(0.0, 2.0)
    assert all(isinstance(s.locator, TimeSpan) for s in card.outline)


async def test_an_unprobeable_file_still_produces_a_card(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    card = await _handler(str(junk)).describe(_ref())
    assert card.facts["readable"] == "no"
    assert card.outline == ()


async def test_a_probe_that_raises_degrades_the_card_rather_than_raising() -> None:
    handler = VideoHandler(
        source=_PathSource("/nowhere.mp4"),
        probe=_RaisingProbe(),
        frames=_NoFrames(),
    )
    assert (await handler.describe(_ref())).facts["readable"] == "no"


async def test_a_source_with_no_local_path_degrades_rather_than_raising() -> None:
    handler = VideoHandler(
        source=_NoPathSource(),
        probe=_StubProbe(_facts()),
        frames=_NoFrames(),
    )
    assert (await handler.describe(_ref())).facts["readable"] == "no"
    rendered = await handler.represent(_ref(), Budget(max_chars=None))
    assert rendered.degradations


# --- the timeline -------------------------------------------------------------


async def test_every_character_resolves_to_the_moment_it_describes(
    sample_video: str,
) -> None:
    """`TimeSpan`'s first producer. The property the cycle exists for."""
    rendered = await _handler(sample_video, vision=FakeVision()).represent(
        _ref(), Budget(max_chars=None)
    )
    first = rendered.locator_map.resolve(0)
    last = rendered.locator_map.resolve(len(rendered.text) - 1)
    assert isinstance(first, TimeSpan) and isinstance(last, TimeSpan)
    assert first.start_s == 0.0
    assert last.end_s == pytest.approx(5.0, abs=0.5)


async def test_the_timeline_is_gapless_and_covers_the_whole_duration(
    sample_video: str,
) -> None:
    """`LocatorMap` requires total gapless coverage, so the stretches between
    sampled frames belong to the sample that starts them. A timeline with holes
    is a timeline that cannot answer "what was on screen at 3.1 seconds".

    Sampled every 2 s over a 5 s video: the duration is NOT a multiple of the
    interval, so the final `[4.0, duration)` segment is genuinely exercised and
    a tail gap cannot ship unnoticed.
    """
    rendered = await _handler(sample_video, vision=FakeVision(), interval=2.0).represent(
        _ref(), Budget(max_chars=None)
    )
    spans = _time_spans(rendered)
    assert len(spans) >= 3, "the tail case needs a duration that is not a whole interval"
    assert spans[0].start_s == 0.0
    for earlier, later in pairwise(spans):
        assert earlier.end_s == pytest.approx(later.start_s)
    assert spans[-1].end_s == pytest.approx(5.0, abs=0.5)


async def test_a_frame_span_is_never_zero_width(sample_video: str) -> None:
    """`TimeSpan.__post_init__` rejects start >= end, so a frame — a point in
    time — is inexpressible as a point. Its span is one frame's duration, taken
    from r_frame_rate: the honest width, not an arbitrary epsilon."""
    rendered = await _handler(sample_video, vision=FakeVision(), interval=2.0).represent(
        _ref(), Budget(max_chars=None)
    )
    for span in _time_spans(rendered):
        assert span.end_s > span.start_s


async def test_a_duration_landing_exactly_on_a_sample_still_yields_a_real_span() -> None:
    """The tail case at its sharpest: a reported duration equal to the last
    sample's timestamp would give a zero-width `TimeSpan` and raise."""
    rendered = await _stub_handler(_facts(duration_s=4.0), interval=2.0).represent(
        _ref(), Budget(max_chars=None)
    )
    last = rendered.locator_map.segments[-1].locator
    assert isinstance(last, TimeSpan)
    assert last.start_s == 2.0
    assert last.end_s == pytest.approx(4.0)


async def test_a_missing_frame_rate_falls_back_rather_than_dividing_by_none() -> None:
    """ffprobe returns `None` for a zero denominator in r_frame_rate."""
    rendered = await _stub_handler(_facts(duration_s=2.0, frame_rate=None)).represent(
        _ref(), Budget(max_chars=None)
    )
    last = rendered.locator_map.segments[-1].locator
    assert isinstance(last, TimeSpan)
    assert last.end_s > last.start_s


async def test_a_barrier_sits_at_every_moment_boundary(sample_video: str) -> None:
    rendered = await _handler(sample_video, vision=FakeVision(), interval=2.0).represent(
        _ref(), Budget(max_chars=None)
    )
    starts = [s.span.start for s in rendered.locator_map.segments][1:]
    assert list(rendered.barriers) == starts


async def test_without_vision_the_timeline_still_reports_its_structure(
    sample_video: str,
) -> None:
    """A video is not empty because nothing looked at it. The scanned-PDF lesson
    at a new site: report what is there and say what was not done."""
    rendered = await _handler(sample_video, vision=None, interval=2.0).represent(
        _ref(), Budget(max_chars=None)
    )
    assert rendered.text.strip()
    assert rendered.locator_map.length == len(rendered.text)
    assert any(
        "vision" in d.what.lower() or "describe" in d.what.lower() for d in rendered.degradations
    )


async def test_undecodable_frames_are_reported_rather_than_left_blank() -> None:
    rendered = await _stub_handler(_facts(duration_s=5.0), vision=FakeVision()).represent(
        _ref(), Budget(max_chars=None)
    )
    assert rendered.text.strip()
    assert any("undecodable" in d.what for d in rendered.degradations)


async def test_an_extractor_that_raises_is_treated_as_no_frame() -> None:
    rendered = await _stub_handler(
        _facts(duration_s=5.0), vision=FakeVision(), frames=_RaisingFrames()
    ).represent(_ref(), Budget(max_chars=None))
    assert any("undecodable" in d.what for d in rendered.degradations)


async def test_a_model_that_answers_with_nothing_is_reported_not_indexed(
    sample_video: str,
) -> None:
    rendered = await _handler(sample_video, vision=_RefusingVision(), interval=2.0).represent(
        _ref(), Budget(max_chars=None)
    )
    assert any("descriptions failed" in d.what for d in rendered.degradations)
    assert rendered.locator_map.length == len(rendered.text)


async def test_a_zero_duration_video_degrades_rather_than_raising() -> None:
    rendered = await _stub_handler(_facts(duration_s=0.0)).represent(_ref(), Budget(max_chars=None))
    assert isinstance(rendered.locator_map.resolve(0), ByteRange)
    assert any(d.what == "no timeline" for d in rendered.degradations)


async def test_an_unreadable_video_degrades_rather_than_raising(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    rendered = await _handler(str(junk)).represent(_ref(), Budget(max_chars=None))
    assert rendered.degradations
    assert isinstance(rendered.locator_map.resolve(0), ByteRange)


async def test_truncation_reports_the_characters_it_kept(sample_video: str) -> None:
    unbounded = await _handler(sample_video, vision=FakeVision(), interval=2.0).represent(
        _ref(), Budget(max_chars=None)
    )
    rendered = await _handler(sample_video, vision=FakeVision(), interval=2.0).represent(
        _ref(), Budget(max_chars=20)
    )
    assert len(rendered.text) == 20
    assert not rendered.barriers or max(rendered.barriers) < 20
    assert any(
        d.detail == f"kept 20 of {len(unbounded.text)} characters" for d in rendered.degradations
    )


async def test_a_budget_of_zero_still_keeps_one_character(sample_video: str) -> None:
    rendered = await _handler(sample_video, vision=FakeVision()).represent(
        _ref(), Budget(max_chars=0)
    )
    assert len(rendered.text) == 1


# --- affordances --------------------------------------------------------------


async def test_no_affordances_are_offered_yet() -> None:
    handler = _stub_handler(_facts())
    assert handler.affordances() == ()
    with pytest.raises(UnknownAffordanceError):
        await handler.invoke(_ref(), "frame_at", None)  # type: ignore[arg-type]


def test_a_non_positive_sample_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="sample_interval_s"):
        VideoHandler(
            source=_PathSource("/nowhere.mp4"),
            probe=_StubProbe(_facts()),
            frames=_NoFrames(),
            sample_interval_s=0.0,
        )


class TestVideoHandlerCompliance(MediaHandlerCompliance):
    @pytest.fixture
    def handler(self, sample_video: str) -> VideoHandler:
        return _handler(sample_video, vision=FakeVision(), interval=2.0)

    @pytest.fixture
    def content(self, sample_video: str) -> bytes:
        return Path(sample_video).read_bytes()
