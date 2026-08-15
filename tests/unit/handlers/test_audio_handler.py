from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from readeverything.adapters.ffmpeg_audio import FfmpegAudio
from readeverything.adapters.ffprobe_streams import FfprobeStreams
from readeverything.domain.capability import Capability
from readeverything.domain.errors import InfrastructureError, UnknownAffordanceError
from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
from readeverything.domain.locators import ByteRange, TimeSpan
from readeverything.domain.rendition import Budget, TextContent, TranscriptCue
from readeverything.handlers.audio import AudioHandler, ReadSpanParams
from readeverything.ports.audio import AudioExtractor
from readeverything.ports.streams import MediaFacts, StreamInfo, StreamProbe
from readeverything.ports.transcription import Transcriber
from readeverything.testing.fakes import FakeTranscriber
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
    async def read_bytes(self, uri: str) -> bytes:
        return b""

    async def read_range(self, uri: str, start: int, end: int) -> bytes:
        return b""

    def stream(self, uri: str, *, chunk_size: int = 1 << 20):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def local_path(self, uri: str) -> str:
        raise OSError("no local path")


class _StubProbe:
    def __init__(self, facts: MediaFacts) -> None:
        self._facts = facts

    async def probe(self, path: str) -> MediaFacts:
        return self._facts


class _RaisingProbe:
    async def probe(self, path: str) -> MediaFacts:
        raise RuntimeError("ffprobe exploded")


class _NoAudio:
    async def extract(self, path: str) -> bytes | None:
        return None


class _StubAudio:
    def __init__(self, data: bytes = b"x" * 3000) -> None:
        self._data = data

    async def extract(self, path: str) -> bytes | None:
        return self._data


class _EmptyTranscriber:
    """Ran, listened to the whole track, and heard nothing."""

    model_id = "empty-asr@1"

    async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]:
        return ()


class _RaisingTranscriber:
    model_id = "raising-asr@1"

    async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]:
        raise InfrastructureError("the model fell over")


class _LateFirstCue:
    """A first cue that starts well after zero, as whisper's routinely does."""

    model_id = "late-asr@1"

    async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]:
        return (
            TranscriptCue(span=TimeSpan(1.7, 2.4), text="first", speaker=None, confidence=None),
            TranscriptCue(span=TimeSpan(2.4, 2.9), text="second", speaker=None, confidence=None),
        )


class _CueBeyondTheDuration:
    """A transcriber that disagrees with the probe about the file's length.

    The first cue is inside the file; the second starts a full minute after a
    three-second file ends. Real transcribers do this when handed audio whose
    container header lies about its duration.
    """

    model_id = "overrunning-asr@1"

    async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]:
        return (
            TranscriptCue(span=TimeSpan(0.5, 1.5), text="inside", speaker=None, confidence=None),
            TranscriptCue(span=TimeSpan(60.0, 61.0), text="beyond", speaker=None, confidence=None),
        )


class _OverhangingCue:
    """One cue, straddling the end of the file. The utterance happened; only
    its tail is in dispute."""

    model_id = "overhanging-asr@1"

    async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]:
        return (
            TranscriptCue(span=TimeSpan(0.5, 1.0), text="early", speaker=None, confidence=None),
            TranscriptCue(span=TimeSpan(2.5, 8.0), text="overhang", speaker=None, confidence=None),
        )


class _EntirelyBeyondTheDuration:
    """Every cue after the end of the file. Nothing survives the clamp."""

    model_id = "wholly-overrunning-asr@1"

    async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]:
        return (
            TranscriptCue(span=TimeSpan(30.0, 31.0), text="far", speaker=None, confidence=None),
        )


class _CoincidentCues:
    """Two cues at the same timestamp — legal, and a zero-width span is not."""

    model_id = "coincident-asr@1"

    async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]:
        return (
            TranscriptCue(span=TimeSpan(1.0, 2.0), text="alpha", speaker=None, confidence=None),
            TranscriptCue(span=TimeSpan(1.0, 2.0), text="beta", speaker=None, confidence=None),
        )


class _SilentCue:
    """A cue whose text is empty — `CharSpan` forbids a zero-width segment."""

    model_id = "blank-asr@1"

    async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]:
        return (TranscriptCue(span=TimeSpan(0.5, 1.0), text="", speaker=None, confidence=None),)


def _ref(uri: str = "a.wav", size_bytes: int = 4096) -> SourceRef:
    return SourceRef(
        uri=uri,
        mime=MimeType.parse("audio/wav"),
        content_hash=ContentHash("f" * 64),
        size_bytes=size_bytes,
    )


def _handler(
    path: str,
    *,
    transcriber: object | None = None,
    audio: object | None = None,
) -> AudioHandler:
    return AudioHandler(
        source=_PathSource(path),
        probe=FfprobeStreams(),
        audio=audio or FfmpegAudio(),  # type: ignore[arg-type]  # structural stub in tests
        transcriber=transcriber,  # type: ignore[arg-type]  # structural stub in tests
    )


def _facts(duration_s: float = 3.0, *, sample_rate: int | None = 44100) -> MediaFacts:
    return MediaFacts(
        duration_s=duration_s,
        container="wav",
        streams=(
            StreamInfo(
                kind="audio",
                codec="pcm_s16le",
                width=None,
                height=None,
                frame_rate=None,
                sample_rate=sample_rate,
                channels=1,
            ),
        ),
    )


def _stub_handler(
    facts: MediaFacts,
    *,
    transcriber: object | None = None,
    audio: object | None = None,
    source: object | None = None,
) -> AudioHandler:
    return AudioHandler(
        source=source or _PathSource("/nowhere.wav"),  # type: ignore[arg-type]
        probe=_StubProbe(facts),
        audio=audio or _StubAudio(),  # type: ignore[arg-type]
        transcriber=transcriber,  # type: ignore[arg-type]
    )


def _spans(rendered_segments: object) -> list[TimeSpan]:
    spans = [s.locator for s in rendered_segments]  # type: ignore[attr-defined]
    assert all(isinstance(s, TimeSpan) for s in spans)
    return [s for s in spans if isinstance(s, TimeSpan)]


# --- ports and capabilities ---------------------------------------------------


def test_the_ports_are_satisfied_by_the_real_adapters() -> None:
    assert isinstance(FfprobeStreams(), StreamProbe)
    assert isinstance(FfmpegAudio(), AudioExtractor)
    assert isinstance(FakeTranscriber(), Transcriber)


def test_the_handler_declares_ffmpeg() -> None:
    """Without ffmpeg the track cannot be extracted at all, so the registry
    drops the handler entirely rather than dropping affordances."""
    assert _stub_handler(_facts()).requires() == frozenset({Capability.FFMPEG})


def test_read_span_declares_ffmpeg_and_asr() -> None:
    affordance = _stub_handler(_facts(), transcriber=FakeTranscriber()).affordances()[0]
    assert affordance.name == "read_span"
    assert affordance.requires == frozenset({Capability.FFMPEG, Capability.ASR})


def test_read_span_is_not_offered_without_a_transcriber() -> None:
    assert _stub_handler(_facts()).affordances() == ()


# --- the card -----------------------------------------------------------------


async def test_the_card_reports_codec_and_sample_rate_without_decoding(
    audio_path: str,
) -> None:
    card = await _handler(audio_path).describe(_ref())
    assert card.facts["audio_codec"]
    assert card.facts["sample_rate"] == 44100


async def test_the_card_costs_a_probe_and_no_extraction(audio_path: str) -> None:
    """An extractor that would fail the test if called proves the card is cheap."""

    class _Exploding:
        async def extract(self, path: str) -> bytes | None:
            raise AssertionError("describe must not extract audio")

    card = await _handler(audio_path, audio=_Exploding()).describe(_ref())
    assert card.kind is MediaKind.AUDIO
    assert card.facts["duration_s"] == pytest.approx(3.0, abs=0.2)


async def test_an_unprobeable_card_says_so_rather_than_raising() -> None:
    card = await AudioHandler(
        source=_NoPathSource(),
        probe=_RaisingProbe(),
        audio=_NoAudio(),
    ).describe(_ref())
    assert card.facts["readable"] == "no"


async def test_a_probe_that_raises_over_a_readable_path_still_degrades() -> None:
    """The path resolved fine; ffprobe is what fell over. Still not an exception."""
    card = await AudioHandler(
        source=_PathSource("/nowhere.wav"),
        probe=_RaisingProbe(),
        audio=_NoAudio(),
    ).describe(_ref())
    assert card.facts["readable"] == "no"


async def test_the_card_omits_a_sample_rate_the_probe_could_not_report() -> None:
    """An absent key admits ignorance; a zero would be a measurement nothing made."""
    card = await _stub_handler(_facts(sample_rate=None)).describe(_ref())
    assert "sample_rate" not in card.facts


# --- the transcript -----------------------------------------------------------


async def test_every_character_resolves_to_the_moment_it_was_said(
    audio_path: str,
) -> None:
    """`TranscriptCue`'s first producer, and the property the cycle exists for."""
    rendered = await _handler(audio_path, transcriber=FakeTranscriber()).represent(
        _ref(), Budget(max_chars=None)
    )
    assert isinstance(rendered.locator_map.resolve(0), TimeSpan)
    assert isinstance(rendered.locator_map.resolve(len(rendered.text) - 1), TimeSpan)


async def test_the_timeline_covers_the_whole_duration_with_no_gaps(
    audio_path: str,
) -> None:
    """Cues have silence between them, and `LocatorMap` must be total. Each
    cue's span extends to the next cue's start; the last extends to the file
    duration — exactly `VideoHandler._bounds`, one medium over.
    """
    rendered = await _handler(audio_path, transcriber=FakeTranscriber()).represent(
        _ref(), Budget(max_chars=None)
    )
    spans = _spans(rendered.locator_map.segments)
    assert spans[0].start_s == 0.0
    for earlier, later in pairwise(spans):
        assert earlier.end_s == pytest.approx(later.start_s)


async def test_the_last_cue_extends_to_the_file_duration(audio_path: str) -> None:
    """The rule the spec states, which no test exercised.

    The fake's cues used to run 47.9x past the file's duration, so the last cue
    always got a floored minimum width and the extend-to-duration branch never
    ran. A timeline is of the FILE, so its final moment ends where the file does.
    """
    handler = _handler(audio_path, transcriber=FakeTranscriber())
    card = await handler.describe(_ref())
    duration_s = card.facts["duration_s"]
    assert isinstance(duration_s, float)
    rendered = await handler.represent(_ref(), Budget(max_chars=None))
    spans = _spans(rendered.locator_map.segments)
    assert len(spans) > 1
    assert spans[-1].end_s == pytest.approx(duration_s)


async def test_a_cue_beyond_the_duration_is_dropped_and_reported() -> None:
    """A transcriber that disagrees with the probe about the file's length is
    reporting something worth knowing. Dropping it silently would hide the
    disagreement; keeping it would put a locator on a moment the file does not
    have."""
    rendered = await _stub_handler(
        _facts(duration_s=3.0), transcriber=_CueBeyondTheDuration()
    ).represent(_ref(), Budget(max_chars=None))
    spans = _spans(rendered.locator_map.segments)
    assert len(spans) == 1
    assert spans[0] == TimeSpan(0.0, 3.0)
    assert "beyond" not in rendered.text
    assert any("outside the file" in d.what for d in rendered.degradations)


async def test_a_cue_overhanging_the_end_is_truncated_not_dropped() -> None:
    """The utterance did happen; only its tail is in dispute."""
    rendered = await _stub_handler(_facts(duration_s=3.0), transcriber=_OverhangingCue()).represent(
        _ref(), Budget(max_chars=None)
    )
    spans = _spans(rendered.locator_map.segments)
    assert "overhang" in rendered.text
    assert spans[-1].end_s == pytest.approx(3.0)
    assert rendered.degradations == ()


async def test_a_transcript_entirely_outside_the_file_says_so() -> None:
    rendered = await _stub_handler(
        _facts(duration_s=3.0), transcriber=_EntirelyBeyondTheDuration()
    ).represent(_ref(), Budget(max_chars=None))
    assert rendered.degradations[0].what == "transcript outside the file"
    assert rendered.text.strip()
    assert isinstance(rendered.locator_map.resolve(0), TimeSpan)


async def test_the_first_span_starts_at_zero_even_when_speech_does_not() -> None:
    """The map is of the FILE, not of the speech: the opening span owns the
    lead-in silence before whisper's first cue."""
    rendered = await _stub_handler(_facts(), transcriber=_LateFirstCue()).represent(
        _ref(), Budget(max_chars=None)
    )
    spans = _spans(rendered.locator_map.segments)
    assert spans[0].start_s == 0.0
    assert spans[0].end_s == pytest.approx(2.4)


async def test_the_last_span_runs_to_the_file_duration() -> None:
    rendered = await _stub_handler(_facts(duration_s=9.0), transcriber=_LateFirstCue()).represent(
        _ref(), Budget(max_chars=None)
    )
    assert _spans(rendered.locator_map.segments)[-1].end_s == pytest.approx(9.0)


async def test_no_silence_segment_is_invented_between_cues() -> None:
    """A gap belongs to the cue that most recently spoke. Emitting "(silence)"
    would put text in an index describing something nothing measured."""
    rendered = await _stub_handler(_facts(), transcriber=_LateFirstCue()).represent(
        _ref(), Budget(max_chars=None)
    )
    assert "silence" not in rendered.text.lower()
    assert len(rendered.locator_map.segments) == 2


async def test_cues_at_the_same_timestamp_still_produce_a_legal_map() -> None:
    rendered = await _stub_handler(_facts(), transcriber=_CoincidentCues()).represent(
        _ref(), Budget(max_chars=None)
    )
    spans = _spans(rendered.locator_map.segments)
    assert spans[0].end_s == pytest.approx(spans[1].start_s)
    assert spans[0].start_s < spans[0].end_s


async def test_a_cue_with_no_text_still_holds_its_span() -> None:
    rendered = await _stub_handler(_facts(), transcriber=_SilentCue()).represent(
        _ref(), Budget(max_chars=None)
    )
    assert rendered.locator_map.length == len(rendered.text)


async def test_there_are_no_barriers(audio_path: str) -> None:
    """A speaker turn is the audio analogue of a scene cut, and it needs
    diarization. Every cue boundary is a pause, not a hard chunk break."""
    rendered = await _handler(audio_path, transcriber=FakeTranscriber()).represent(
        _ref(), Budget(max_chars=None)
    )
    assert rendered.barriers == ()


# --- the four ways there is no transcript -------------------------------------


async def test_without_a_transcriber_it_says_so_and_is_not_empty(
    audio_path: str,
) -> None:
    """An audio file has no cheaper layer — without transcription it has only a
    duration and a codec. An empty string would claim there was nothing to hear."""
    rendered = await _handler(audio_path, transcriber=None).represent(
        _ref(), Budget(max_chars=None)
    )
    assert rendered.text.strip()
    assert any("transcri" in d.what.lower() for d in rendered.degradations)


async def test_silence_is_distinguishable_from_no_transcriber(audio_path: str) -> None:
    """Two different facts. A transcriber that ran and heard nothing is not the
    same as no transcriber, and reporting them identically would lose the
    difference between "this file is silent" and "nobody listened"."""
    silent = await _handler(audio_path, transcriber=_EmptyTranscriber()).represent(
        _ref(), Budget(max_chars=None)
    )
    absent = await _handler(audio_path, transcriber=None).represent(_ref(), Budget(max_chars=None))
    assert silent.text != absent.text
    assert silent.degradations[0].what != absent.degradations[0].what


async def test_a_file_with_no_audio_track_degrades_and_says_which(
    video_only_path: str,
) -> None:
    rendered = await _handler(video_only_path).represent(_ref(), Budget(max_chars=None))
    assert rendered.degradations
    assert "audio" in rendered.degradations[0].detail.lower()


async def test_an_extractor_that_finds_no_track_degrades_rather_than_raising() -> None:
    rendered = await _stub_handler(
        _facts(), transcriber=FakeTranscriber(), audio=_NoAudio()
    ).represent(_ref(), Budget(max_chars=None))
    assert "audio track" in rendered.degradations[0].detail.lower()


async def test_a_transcriber_that_raises_degrades_rather_than_raising() -> None:
    rendered = await _stub_handler(_facts(), transcriber=_RaisingTranscriber()).represent(
        _ref(), Budget(max_chars=None)
    )
    assert rendered.degradations[0].what == "transcription failed"
    assert rendered.text.strip()


async def test_an_unprobeable_file_is_located_by_bytes_not_by_time() -> None:
    """No timeline was ever observed, so claiming one would be a claim about a
    file this handler never established exists."""
    rendered = await AudioHandler(
        source=_NoPathSource(),
        probe=_RaisingProbe(),
        audio=_NoAudio(),
    ).represent(_ref(), Budget(max_chars=None))
    assert isinstance(rendered.locator_map.resolve(0), ByteRange)
    assert rendered.degradations[0].what == "audio unprobeable"


# --- the budget ---------------------------------------------------------------


async def test_truncation_reports_the_characters_actually_kept(audio_path: str) -> None:
    handler = _handler(audio_path, transcriber=FakeTranscriber())
    full = await handler.represent(_ref(), Budget(max_chars=None))
    rendered = await handler.represent(_ref(), Budget(max_chars=12))
    assert len(rendered.text) == 12
    detail = next(d.detail for d in rendered.degradations if d.what == "text truncated")
    assert detail == f"kept 12 of {len(full.text)} characters"


async def test_a_zero_budget_keeps_one_character_because_a_span_cannot_be_empty(
    audio_path: str,
) -> None:
    rendered = await _handler(audio_path, transcriber=FakeTranscriber()).represent(
        _ref(), Budget(max_chars=0)
    )
    assert len(rendered.text) == 1
    assert rendered.locator_map.length == 1


async def test_a_budget_never_transcribes_less_audio(audio_path: str) -> None:
    """The whole track is transcribed and the flattened text is cut afterwards,
    so the tail is reported missing rather than silently never heard."""
    handler = _handler(audio_path, transcriber=FakeTranscriber())
    full = await handler.represent(_ref(), Budget(max_chars=None))
    clipped = await handler.represent(_ref(), Budget(max_chars=12))
    assert clipped.text == full.text[:12]


# --- read_span ----------------------------------------------------------------


async def test_read_span_returns_the_cues_inside_the_window() -> None:
    rendition = await _stub_handler(_facts(), transcriber=_LateFirstCue()).invoke(
        _ref(), "read_span", ReadSpanParams(start_s=1.0, end_s=2.0)
    )
    assert isinstance(rendition.content, TextContent)
    assert "first" in rendition.content.text
    assert "second" not in rendition.content.text
    assert rendition.locator == TimeSpan(1.0, 2.0)


async def test_read_span_degrades_when_the_window_holds_no_speech() -> None:
    rendition = await _stub_handler(_facts(), transcriber=_LateFirstCue()).invoke(
        _ref(), "read_span", ReadSpanParams(start_s=10.0, end_s=20.0)
    )
    assert rendition.degraded
    assert isinstance(rendition.locator, ByteRange)


async def test_read_span_degrades_on_an_empty_window() -> None:
    rendition = await _stub_handler(_facts(), transcriber=FakeTranscriber()).invoke(
        _ref(), "read_span", ReadSpanParams(start_s=5.0, end_s=5.0)
    )
    assert rendition.degraded
    assert isinstance(rendition.content, TextContent)
    assert "empty" in rendition.content.text


async def test_read_span_degrades_when_there_is_no_audio_track() -> None:
    rendition = await _stub_handler(
        _facts(), transcriber=FakeTranscriber(), audio=_NoAudio()
    ).invoke(_ref(), "read_span", ReadSpanParams())
    assert rendition.degraded
    assert isinstance(rendition.content, TextContent)
    assert "audio track" in rendition.content.text


async def test_read_span_degrades_when_the_transcriber_raises() -> None:
    rendition = await _stub_handler(_facts(), transcriber=_RaisingTranscriber()).invoke(
        _ref(), "read_span", ReadSpanParams()
    )
    assert rendition.degraded


async def test_read_span_degrades_when_the_file_cannot_be_read() -> None:
    rendition = await _stub_handler(
        _facts(), transcriber=FakeTranscriber(), source=_NoPathSource()
    ).invoke(_ref(), "read_span", ReadSpanParams())
    assert rendition.degraded


async def test_read_span_without_a_transcriber_is_an_unknown_affordance() -> None:
    with pytest.raises(UnknownAffordanceError):
        await _stub_handler(_facts()).invoke(_ref(), "read_span", ReadSpanParams())


async def test_read_span_rejects_the_wrong_params_type() -> None:
    with pytest.raises(TypeError):
        await _stub_handler(_facts(), transcriber=FakeTranscriber()).invoke(
            _ref(),
            "read_span",
            _facts,  # type: ignore[arg-type]
        )


async def test_an_unknown_affordance_raises() -> None:
    with pytest.raises(UnknownAffordanceError):
        await _stub_handler(_facts(), transcriber=FakeTranscriber()).invoke(
            _ref(), "definitely_not_an_affordance", ReadSpanParams()
        )


# --- the laws -----------------------------------------------------------------


class TestAudioHandlerCompliance(MediaHandlerCompliance):
    @pytest.fixture
    def content(self, audio_path: str) -> bytes:
        return Path(audio_path).read_bytes()

    @pytest.fixture
    def handler(self, audio_path: str) -> AudioHandler:
        return _handler(audio_path, transcriber=FakeTranscriber())
