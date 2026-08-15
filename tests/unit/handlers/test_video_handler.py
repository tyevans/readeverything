from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from itertools import pairwise
from pathlib import Path

import pytest

from readeverything.adapters.ffmpeg_audio import FfmpegAudio
from readeverything.adapters.ffmpeg_frames import FfmpegFrames
from readeverything.adapters.ffprobe_streams import FfprobeStreams
from readeverything.adapters.semaphore_limiter import SemaphoreLimiter
from readeverything.domain.capability import Capability
from readeverything.domain.errors import UnknownAffordanceError
from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
from readeverything.domain.locators import ByteRange, TimeSpan
from readeverything.domain.observation import OperationProgressed
from readeverything.domain.rendition import (
    Budget,
    ImageContent,
    Rendered,
    SpeakerId,
    TextContent,
    TranscriptCue,
)
from readeverything.handlers.video import (
    MOMENT_SEPARATOR,
    SPEECH_MARKER,
    DescribeFrameParams,
    FrameAtParams,
    VideoHandler,
)
from readeverything.ports.frames import FrameExtractor
from readeverything.ports.streams import MediaFacts, StreamInfo, StreamProbe
from readeverything.testing.fakes import (
    FakeTranscriber,
    FakeVision,
    RaisingObserver,
    RecordingObserver,
)
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

    async def scene_cuts(self, path: str, threshold: float = 0.4) -> tuple[float, ...]:
        raise RuntimeError("ffmpeg exploded")


class _CancellingFrames:
    """A frame extractor whose fetch is cancelled from within, as a limiter or
    a per-call timeout would cancel it — not an extractor that raises an
    ordinary error."""

    async def frame_at(self, path: str, seconds: float) -> bytes | None:
        raise asyncio.CancelledError

    async def scene_cuts(self, path: str, threshold: float = 0.4) -> tuple[float, ...]:
        return ()


class _NoFrames:
    async def frame_at(self, path: str, seconds: float) -> bytes | None:
        return None

    async def scene_cuts(self, path: str, threshold: float = 0.4) -> tuple[float, ...]:
        return ()


class _StubAudio:
    """An extractor that always yields bytes, so a stubbed transcriber runs
    without a real container behind it."""

    async def extract(self, path: str) -> bytes | None:
        return b"RIFF...."


class _RecordingFrames:
    """Deterministic frame bytes, recording the order of request and completion.

    The bytes' LENGTH varies with the timestamp, and `FakeVision` describes an
    image by its length, so a timeline assembled in completion order rather
    than in entry order would say visibly different things at each moment
    instead of merely reordering identical lines.
    """

    def __init__(self, delay_s: float = 0.0) -> None:
        self.requested: list[float] = []
        self.completed: list[float] = []
        self._delay_s = delay_s

    async def _wait(self, seconds: float) -> None:
        await asyncio.sleep(self._delay_s)

    async def frame_at(self, path: str, seconds: float) -> bytes | None:
        self.requested.append(seconds)
        await self._wait(seconds)
        self.completed.append(seconds)
        return b"frame".ljust(64 + int(seconds * 10), b"\0")

    async def scene_cuts(self, path: str, threshold: float = 0.4) -> tuple[float, ...]:
        return ()


#: The longest a `_ReverseOrderFrames` extraction waits, at t=0. Every later
#: timestamp waits strictly less, so completion order is exactly reversed.
_REVERSE_UNIT_S = 0.05


class _ReverseOrderFrames(_RecordingFrames):
    """An extractor whose completions arrive in exactly the reverse of the
    order they were requested in.

    A fake that returns in call order cannot fail the test it exists for: it
    would pass against an implementation that appended results in completion
    order, because the two orders would be the same. The wait shrinks strictly
    as the timestamp grows, so the LAST moment requested is the first to
    finish, and `test_the_timeline_is_identical_when_moments_complete_out_of_order`
    asserts that inversion actually happened rather than assuming it.
    """

    async def _wait(self, seconds: float) -> None:
        await asyncio.sleep(_REVERSE_UNIT_S / (1.0 + seconds))


class _BoundedCountingLimiter:
    """A real bound, and the peak in-flight count it actually saw.

    `CountingLimiter` records without bounding and `SemaphoreLimiter` bounds
    without recording; asserting that a bound HELD needs both at once, and a
    measured peak is an assertion about concurrency rather than about timing.
    """

    def __init__(self, limits: Mapping[Capability, int]) -> None:
        self._inner = SemaphoreLimiter(limits)
        self.in_flight: dict[Capability, int] = {}
        self.peak: dict[Capability, int] = {}

    @asynccontextmanager
    async def limit(self, capability: Capability) -> AsyncIterator[None]:
        async with self._inner.limit(capability):
            self.in_flight[capability] = self.in_flight.get(capability, 0) + 1
            self.peak[capability] = max(self.peak.get(capability, 0), self.in_flight[capability])
            try:
                yield
            finally:
                self.in_flight[capability] -= 1


class _SlowVision:
    """A vision model slow enough for concurrent calls to overlap.

    An instantaneous fake would never show a peak above one, and a test that
    measured one would prove nothing about the bound.
    """

    model_id = "slow-vision@1"

    async def describe(self, data: bytes, mime: str, prompt: str) -> str:
        await asyncio.sleep(0.01)
        return f"[{len(data)} bytes]"


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
    audio: object | None = None,
    transcriber: object | None = None,
    observer: object | None = None,
    limiter: object | None = None,
) -> VideoHandler:
    return VideoHandler(
        source=_PathSource(path),
        probe=FfprobeStreams(),
        frames=frames or FfmpegFrames(),  # type: ignore[arg-type]  # structural stub in tests
        vision=vision,  # type: ignore[arg-type]  # structural stub in tests
        audio=audio,  # type: ignore[arg-type]  # structural stub in tests
        transcriber=transcriber,  # type: ignore[arg-type]  # structural stub in tests
        sample_interval_s=interval,
        observer=observer,  # type: ignore[arg-type]  # structural stub in tests
        limiter=limiter,  # type: ignore[arg-type]  # structural stub in tests
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
        audio=kwargs.get("audio"),  # type: ignore[arg-type]
        transcriber=kwargs.get("transcriber"),  # type: ignore[arg-type]
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
    """`sample_video` is uniform testsrc content: no real scene cut exists, so
    detection finding none falls back to the every-moment status quo."""
    rendered = await _handler(sample_video, vision=FakeVision(), interval=2.0).represent(
        _ref(), Budget(max_chars=None)
    )
    starts = [s.span.start for s in rendered.locator_map.segments][1:]
    assert list(rendered.barriers) == starts


async def test_barriers_land_at_cuts(scene_cut_video: str) -> None:
    rendered = await _handler(scene_cut_video, vision=FakeVision()).represent(
        _ref(), Budget(max_chars=None)
    )
    assert rendered.barriers
    for barrier in rendered.barriers:
        assert 0 < barrier < len(rendered.text)


async def test_scene_detection_failing_is_reported_and_falls_back() -> None:
    """Detection failing and detection finding nothing must be distinguishable:
    the failure case records a degradation, the empty-result case does not."""

    class _FailingSceneDetection:
        async def frame_at(self, path: str, seconds: float) -> bytes | None:
            return None

        async def scene_cuts(self, path: str, threshold: float = 0.4) -> tuple[float, ...]:
            raise RuntimeError("ffmpeg exploded")

    rendered = await _stub_handler(
        _facts(duration_s=5.0), vision=FakeVision(), frames=_FailingSceneDetection()
    ).represent(_ref(), Budget(max_chars=None))
    assert any("scene detection failed" in d.what for d in rendered.degradations)


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


async def test_a_cancelled_fetch_propagates_rather_than_becoming_a_degraded_frame() -> None:
    """Cancellation is a request to stop, not a frame we failed to read.

    `gather(return_exceptions=True)` returns a child's own CancelledError as a
    result rather than propagating it, and CancelledError is a BaseException —
    so a blanket `isinstance(outcome, BaseException)` mapping would render a
    cancelled fetch as a degraded moment, putting a claim about the file into
    the timeline that nothing established.
    """
    rendered = _stub_handler(
        _facts(duration_s=5.0), vision=FakeVision(), frames=_CancellingFrames()
    ).represent(_ref(), Budget(max_chars=None))
    with pytest.raises(asyncio.CancelledError):
        await rendered


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


# --- the transcript on the timeline -------------------------------------------


async def test_video_without_a_transcriber_is_unchanged(sample_video: str) -> None:
    """The regression guard. Existing behaviour must be byte-identical, because
    every video test in the suite depends on it, and if the merge changed the
    formatting even when no cue exists they would all move at once."""
    before = await _handler(sample_video, vision=FakeVision()).represent(
        _ref(), Budget(max_chars=None)
    )
    after = await _handler(sample_video, vision=FakeVision(), transcriber=None).represent(
        _ref(), Budget(max_chars=None)
    )
    assert before.text == after.text


async def test_the_card_transcribes_nothing(sample_video: str) -> None:
    """`describe()` stays probe-only and cheap. An extractor and a transcriber
    that both explode prove it: a card that reached for the audio would fail."""

    class _ExplodingAudio:
        async def extract(self, path: str) -> bytes | None:
            raise AssertionError("describe() must not extract audio")

    class _ExplodingTranscriber:
        model_id = "exploding@1"

        async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]:
            raise AssertionError("describe() must not transcribe")

    card = await _handler(
        sample_video, audio=_ExplodingAudio(), transcriber=_ExplodingTranscriber()
    ).describe(_ref())
    assert card.facts["readable"] == "yes"


async def test_cues_and_frames_interleave_in_timestamp_order(sample_video: str) -> None:
    rendered = await _handler(
        sample_video,
        vision=FakeVision(),
        audio=FfmpegAudio(),
        transcriber=FakeTranscriber(duration_s=5.0),
    ).represent(_ref(), Budget(max_chars=None))
    spans = _time_spans(rendered)
    assert spans == sorted(spans, key=lambda s: s.start_s)
    assert spans[0].start_s == 0.0
    for earlier, later in pairwise(spans):
        assert earlier.end_s == pytest.approx(later.start_s)


async def test_a_transcript_adds_entries_the_frames_alone_did_not(sample_video: str) -> None:
    """The point of the merge: with a transcriber the timeline carries strictly
    more, and the extra entries are the cues."""
    without = await _handler(sample_video, vision=FakeVision()).represent(
        _ref(), Budget(max_chars=None)
    )
    with_speech = await _handler(
        sample_video,
        vision=FakeVision(),
        audio=FfmpegAudio(),
        transcriber=FakeTranscriber(duration_s=5.0),
    ).represent(_ref(), Budget(max_chars=None))
    assert len(with_speech.locator_map.segments) > len(without.locator_map.segments)
    assert SPEECH_MARKER in with_speech.text
    assert SPEECH_MARKER not in without.text


async def test_a_silent_video_degrades_rather_than_raising(silent_video: str) -> None:
    """`AudioExtractor.extract` returning `None` is a normal answer, not an
    error: a silent video still has a visual timeline."""
    rendered = await _handler(
        silent_video,
        vision=FakeVision(),
        audio=FfmpegAudio(),
        transcriber=FakeTranscriber(duration_s=5.0),
    ).represent(_ref(), Budget(max_chars=None))
    assert rendered.text.strip()
    assert rendered.locator_map.length == len(rendered.text)
    assert any("audio track" in d.what for d in rendered.degradations)


async def test_a_transcriber_that_raises_degrades_rather_than_raising(sample_video: str) -> None:
    class _RaisingTranscriber:
        model_id = "raising@1"

        async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]:
            raise RuntimeError("whisper exploded")

    rendered = await _handler(
        sample_video, vision=FakeVision(), audio=FfmpegAudio(), transcriber=_RaisingTranscriber()
    ).represent(_ref(), Budget(max_chars=None))
    assert rendered.text.strip()
    assert any("transcription failed" in d.what for d in rendered.degradations)


async def test_an_extractor_that_raises_is_treated_as_no_audio_track() -> None:
    class _RaisingAudio:
        async def extract(self, path: str) -> bytes | None:
            raise RuntimeError("ffmpeg exploded")

    rendered = await _stub_handler(
        _facts(duration_s=5.0),
        vision=FakeVision(),
        audio=_RaisingAudio(),
        transcriber=FakeTranscriber(duration_s=5.0),
    ).represent(_ref(), Budget(max_chars=None))
    assert any("audio track" in d.what for d in rendered.degradations)


async def test_cues_past_the_probed_duration_are_dropped_and_reported() -> None:
    """A locator on a moment the file does not contain is the defect this
    project keeps finding. The transcriber and the probe disagreeing is
    reported rather than silently resolved in the transcriber's favour."""

    class _OverrunningTranscriber:
        model_id = "overrun@1"

        async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]:
            return (
                TranscriptCue(
                    span=TimeSpan(1.0, 9.0), text="overhangs", speaker=None, confidence=None
                ),
                TranscriptCue(
                    span=TimeSpan(20.0, 21.0), text="past the end", speaker=None, confidence=None
                ),
            )

    rendered = await _stub_handler(
        _facts(duration_s=5.0),
        vision=FakeVision(),
        audio=_StubAudio(),
        transcriber=_OverrunningTranscriber(),
    ).represent(_ref(), Budget(max_chars=None))
    assert "past the end" not in rendered.text
    assert "overhangs" in rendered.text
    assert any("cues outside the file" in d.what for d in rendered.degradations)
    for span in _time_spans(rendered):
        assert span.start_s < 5.0


async def test_a_cue_and_a_frame_at_the_same_instant_are_two_entries() -> None:
    """Not a conflict: they say different things about the same moment, and a
    zero-width `TimeSpan` is what the domain forbids, not a collision."""

    class _CueAtZero:
        model_id = "at-zero@1"

        async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]:
            return (
                TranscriptCue(
                    span=TimeSpan(0.0, 1.0), text="spoken", speaker=None, confidence=None
                ),
            )

    rendered = await _stub_handler(
        _facts(duration_s=5.0),
        vision=FakeVision(),
        audio=_StubAudio(),
        transcriber=_CueAtZero(),
        interval=2.0,
    ).represent(_ref(), Budget(max_chars=None))
    spans = _time_spans(rendered)
    assert spans[0].start_s == 0.0
    for span in spans:
        assert span.end_s > span.start_s
    for earlier, later in pairwise(spans):
        assert earlier.end_s == pytest.approx(later.start_s)


async def test_a_speaker_is_named_when_the_transcriber_reports_one() -> None:
    class _DiarizingTranscriber:
        model_id = "diarizing@1"

        async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]:
            return (
                TranscriptCue(
                    span=TimeSpan(1.0, 2.0),
                    text="hello",
                    speaker=SpeakerId("SPEAKER_01"),
                    confidence=None,
                ),
            )

    rendered = await _stub_handler(
        _facts(duration_s=5.0),
        vision=FakeVision(),
        audio=_StubAudio(),
        transcriber=_DiarizingTranscriber(),
    ).represent(_ref(), Budget(max_chars=None))
    assert "SPEAKER_01" in rendered.text


async def test_a_transcriber_that_hears_nothing_leaves_the_timeline_alone() -> None:
    class _SilentTranscriber:
        model_id = "silent@1"

        async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]:
            return ()

    rendered = await _stub_handler(
        _facts(duration_s=5.0),
        vision=FakeVision(),
        audio=_StubAudio(),
        transcriber=_SilentTranscriber(),
    ).represent(_ref(), Budget(max_chars=None))
    assert rendered.text.strip()
    assert any("no speech detected" in d.what for d in rendered.degradations)


async def test_a_transcriber_without_an_extractor_reports_rather_than_guessing() -> None:
    """Nothing can reach the audio track without an extractor, so the handler
    says so instead of quietly rendering a visual-only timeline."""
    rendered = await _stub_handler(
        _facts(duration_s=5.0),
        vision=FakeVision(),
        audio=None,
        transcriber=FakeTranscriber(duration_s=5.0),
    ).represent(_ref(), Budget(max_chars=None))
    assert any("audio track" in d.what for d in rendered.degradations)


async def test_truncation_with_a_transcript_reports_the_characters_it_kept(
    sample_video: str,
) -> None:
    def build() -> VideoHandler:
        return _handler(
            sample_video,
            vision=FakeVision(),
            audio=FfmpegAudio(),
            transcriber=FakeTranscriber(duration_s=5.0),
        )

    unbounded = await build().represent(_ref(), Budget(max_chars=None))
    rendered = await build().represent(_ref(), Budget(max_chars=30))
    assert len(rendered.text) == 30
    assert any(
        d.detail == f"kept 30 of {len(unbounded.text)} characters" for d in rendered.degradations
    )


# --- affordances --------------------------------------------------------------


async def test_frame_at_returns_an_image_located_in_time(sample_video: str) -> None:
    rendition = await _handler(sample_video).invoke(_ref(), "frame_at", FrameAtParams(seconds=2.5))
    assert isinstance(rendition.content, ImageContent)
    assert rendition.content.data.startswith(b"\x89PNG")
    assert isinstance(rendition.locator, TimeSpan)


async def test_a_frame_past_the_end_says_how_long_the_video_actually_is(
    sample_video: str,
) -> None:
    """The name of this test used to promise "says why" while asserting only
    that it degraded. An agent told "no frame could be decoded at 0:16:39.0"
    cannot tell whether to retry elsewhere or give up; told the video is five
    seconds long, it knows. The message is the useful part, so the test checks
    the message.
    """
    rendition = await _handler(sample_video).invoke(
        _ref(), "frame_at", FrameAtParams(seconds=999.0)
    )
    assert rendition.degraded
    assert not isinstance(rendition.content, ImageContent)
    assert isinstance(rendition.content, TextContent)
    assert "5" in rendition.content.text  # the real duration appears
    assert "long" in rendition.content.text


async def test_a_negative_frame_request_says_so() -> None:
    """`FrameAtParams` forbids a negative value at the boundary, but the
    handler's own message-selection logic must still name the cause rather
    than fall through to the generic "no frame" text if it ever sees one."""
    handler = _stub_handler(_facts(duration_s=5.0))
    rendition = await handler.invoke(
        _ref(), "frame_at", FrameAtParams.model_construct(seconds=-1.0)
    )
    assert rendition.degraded
    assert isinstance(rendition.content, TextContent)
    assert "negative" in rendition.content.text


async def test_a_decode_failure_within_the_duration_keeps_the_generic_message() -> None:
    """Within the duration, a `None` frame is a genuine decode failure, not a
    past-the-end request — the probe knows there SHOULD be a frame there, so
    claiming otherwise would be the same defect wearing different clothes."""
    handler = _stub_handler(_facts(duration_s=5.0), frames=_NoFrames())
    rendition = await handler.invoke(_ref(), "frame_at", FrameAtParams(seconds=2.0))
    assert rendition.degraded
    assert isinstance(rendition.content, TextContent)
    assert "no frame could be decoded" in rendition.content.text
    assert "long" not in rendition.content.text


async def test_an_undeterminable_duration_falls_back_to_the_generic_message() -> None:
    """When the probe itself fails, the handler does not know where the end
    is, so it must not guess that the request was past it."""
    handler = VideoHandler(
        source=_PathSource("/nowhere.mp4"),
        probe=_RaisingProbe(),
        frames=_NoFrames(),
    )
    rendition = await handler.invoke(_ref(), "frame_at", FrameAtParams(seconds=999.0))
    assert rendition.degraded
    assert isinstance(rendition.content, TextContent)
    assert "no frame could be decoded" in rendition.content.text


async def test_frame_at_is_offered_without_vision() -> None:
    handler = _stub_handler(_facts())
    names = {a.name for a in handler.affordances()}
    assert "frame_at" in names
    assert "describe_frame" not in names


async def test_describe_frame_requires_vision_to_be_offered() -> None:
    handler = _stub_handler(_facts(), vision=FakeVision())
    names = {a.name for a in handler.affordances()}
    assert "describe_frame" in names


async def test_describe_frame_returns_a_description_located_in_time(sample_video: str) -> None:
    rendition = await _handler(sample_video, vision=FakeVision()).invoke(
        _ref(), "describe_frame", DescribeFrameParams(seconds=2.5)
    )
    assert isinstance(rendition.content, TextContent)
    assert isinstance(rendition.locator, TimeSpan)
    assert not rendition.degraded


async def test_describe_frame_is_unknown_without_vision(sample_video: str) -> None:
    with pytest.raises(UnknownAffordanceError):
        await _handler(sample_video, vision=None).invoke(
            _ref(), "describe_frame", DescribeFrameParams(seconds=2.5)
        )


async def test_an_unknown_affordance_raises() -> None:
    handler = _stub_handler(_facts())
    with pytest.raises(UnknownAffordanceError):
        await handler.invoke(_ref(), "nonexistent", None)  # type: ignore[arg-type]


def test_a_non_positive_sample_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="sample_interval_s"):
        VideoHandler(
            source=_PathSource("/nowhere.mp4"),
            probe=_StubProbe(_facts()),
            frames=_NoFrames(),
            sample_interval_s=0.0,
        )


# --- fetching the moments concurrently ----------------------------------------


async def test_each_moment_carries_the_description_of_its_own_frame() -> None:
    """THE ABSOLUTE ASSERTION the test below does not make.

    `test_the_timeline_is_identical_when_moments_complete_out_of_order` compares
    two runs of the SAME code, so a mapping bug that shuffles `fetched`
    identically in both runs is invisible to it — both runs would still agree
    with each other. This asserts against an independently computable fact
    instead: the moment rendered at timestamp T must describe the frame that
    was FETCHED AT T, not at some other entry's timestamp.

    `_RecordingFrames.frame_at` returns bytes whose LENGTH varies with the
    timestamp requested, and `FakeVision` describes an image by its length, so
    the expected description at each moment is computable from that moment's
    own timestamp alone — no comparison to a second run is needed.

    Cues are interleaved between the sampled frames on purpose: with a
    transcriber configured, `sampled` here is `(0, 2, 4)` over five entries,
    not `range(5)`. That gap between "index into `entries`" and "position
    among sampled moments" is exactly what separates a correct
    `zip(sampled, outcomes)` from a mis-mapping `enumerate(outcomes)` — with no
    cues the two sequences are identical and this test would prove nothing.
    """

    class _TwoCues:
        model_id = "two-cues@1"

        async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]:
            return (
                TranscriptCue(span=TimeSpan(1.0, 1.2), text="one", speaker=None, confidence=None),
                TranscriptCue(span=TimeSpan(3.0, 3.2), text="three", speaker=None, confidence=None),
            )

    frames = _RecordingFrames()
    rendered = await _stub_handler(
        _facts(duration_s=5.0),
        vision=FakeVision(),
        audio=_StubAudio(),
        transcriber=_TwoCues(),
        interval=2.0,
        frames=frames,
    ).represent(_ref(), Budget(max_chars=None))

    lines = [line for line in rendered.text.split(MOMENT_SEPARATOR) if line]
    frame_lines = [line for line in lines if SPEECH_MARKER not in line]
    assert len(frame_lines) == 3, "expected exactly the three sampled frames, in order"
    for seconds, line in zip((0.0, 2.0, 4.0), frame_lines, strict=True):
        expected_length = 64 + int(seconds * 10)
        assert f"of {expected_length} bytes" in line, (
            f"the moment at {seconds}s must describe the frame fetched at {seconds}s, "
            f"but got: {line!r}"
        )


async def test_the_timeline_is_identical_when_moments_complete_out_of_order(
    sample_video: str,
) -> None:
    """Completion order does not change the output — NOT the mapping guarantee.

    This proves that a naive `gather` appending results in completion order
    would scramble the timeline. It does NOT prove that entries map to the
    RIGHT moments: a mapping bug that shuffles `fetched` the same way in both
    runs compared here would still pass, because the two runs would still
    agree with each other. `test_each_moment_carries_the_description_of_its_own_frame`
    is the absolute check for that; keep both.

    The fake extractor here returns each frame after a delay inversely
    proportional to its timestamp, so completion order is exactly reversed —
    and this asserts that it WAS, because a fake that quietly returned in call
    order would make the rest of this test vacuous.
    """
    in_order = _RecordingFrames()
    reversed_order = _ReverseOrderFrames()
    ordered = await _handler(
        sample_video, vision=FakeVision(), frames=in_order, interval=1.0
    ).represent(_ref(), Budget(max_chars=None))
    scrambled = await _handler(
        sample_video, vision=FakeVision(), frames=reversed_order, interval=1.0
    ).represent(_ref(), Budget(max_chars=None))

    assert len(reversed_order.requested) > 1
    assert reversed_order.completed == list(reversed(reversed_order.requested))
    assert in_order.completed == in_order.requested

    assert ordered.text == scrambled.text
    assert [s.locator for s in ordered.locator_map.segments] == [
        s.locator for s in scrambled.locator_map.segments
    ]
    assert [s.span for s in ordered.locator_map.segments] == [
        s.span for s in scrambled.locator_map.segments
    ]


async def test_the_moments_are_fetched_concurrently(sample_video: str) -> None:
    """Every request is out before the first completion — which is what makes
    the ordering above a real risk rather than a hypothetical one."""
    frames = _ReverseOrderFrames()
    await _handler(sample_video, vision=FakeVision(), frames=frames, interval=1.0).represent(
        _ref(), Budget(max_chars=None)
    )
    assert len(frames.requested) > 1
    # The last moment requested is the first to finish, which is only possible
    # if every request was in flight before any of them returned.
    assert frames.completed[0] == frames.requested[-1]


async def test_frame_work_is_bounded_by_the_configured_limits(sample_video: str) -> None:
    """Asserted by peak in-flight count against a counting limiter, not timing.

    The peak is asserted EQUAL to the bound, not merely under it: a handler
    that had quietly gone back to sequential fetching would also never exceed
    two, and this test must fail for that.
    """
    limiter = _BoundedCountingLimiter({Capability.VISION: 2, Capability.FFMPEG: 3})
    await _handler(
        sample_video,
        vision=_SlowVision(),
        frames=_RecordingFrames(delay_s=0.01),
        limiter=limiter,
        interval=1.0,
    ).represent(_ref(), Budget(max_chars=None))
    assert limiter.peak[Capability.VISION] == 2
    assert limiter.peak[Capability.FFMPEG] == 3


async def test_without_a_limiter_behaviour_is_unchanged(sample_video: str) -> None:
    """Every existing video test rests on this."""
    before = await _handler(sample_video, vision=FakeVision()).represent(
        _ref(), Budget(max_chars=None)
    )
    after = await _handler(sample_video, vision=FakeVision(), limiter=None).represent(
        _ref(), Budget(max_chars=None)
    )
    assert before.text == after.text


async def test_an_observer_that_raises_does_not_change_the_result(
    sample_video: str,
) -> None:
    """A read must not fail — or differ — because progress reporting failed."""
    quiet = await _handler(sample_video, vision=FakeVision()).represent(
        _ref(), Budget(max_chars=None)
    )
    noisy = await _handler(sample_video, vision=FakeVision(), observer=RaisingObserver()).represent(
        _ref(), Budget(max_chars=None)
    )
    assert quiet.text == noisy.text
    assert quiet.locator_map.length == noisy.locator_map.length


async def test_progress_reaches_the_observer_in_order(sample_video: str) -> None:
    recorder = RecordingObserver()
    await _handler(
        sample_video,
        vision=FakeVision(),
        frames=_ReverseOrderFrames(),
        interval=1.0,
        observer=recorder,
    ).represent(_ref(), Budget(max_chars=None))
    kinds = [type(e).__name__ for e in recorder.events]
    assert kinds[0] == "OperationStarted"
    assert kinds[-1] == "OperationFinished"
    progress = [e for e in recorder.events if isinstance(e, OperationProgressed)]
    dones = [e.done for e in progress]
    # Monotonic AND complete: the moments finished in reverse, and the count
    # still counted up from one to the total exactly once each.
    assert dones == sorted(dones)
    assert dones == list(range(1, len(progress) + 1))
    assert {e.total for e in progress} == {len(progress)}


async def test_a_read_that_never_reaches_a_timeline_is_still_narrated(
    tmp_path: Path,
) -> None:
    """A start with no end is the hang this narration exists to make visible,
    so the degraded paths report one too."""
    path = tmp_path / "not.mp4"
    path.write_bytes(b"not a video")
    recorder = RecordingObserver()
    await _handler(str(path), observer=recorder).represent(_ref(), Budget(max_chars=None))
    kinds = [type(e).__name__ for e in recorder.events]
    assert kinds == ["OperationStarted", "OperationFinished"]


class TestVideoHandlerCompliance(MediaHandlerCompliance):
    @pytest.fixture
    def handler(self, sample_video: str) -> VideoHandler:
        return _handler(sample_video, vision=FakeVision(), interval=2.0)

    @pytest.fixture
    def content(self, sample_video: str) -> bytes:
        return Path(sample_video).read_bytes()
