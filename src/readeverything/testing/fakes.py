"""Deterministic stand-ins for everything expensive or nondeterministic.

Unit tests never assert on model text. These fakes produce output derived
mechanically from their input, so a test can assert on *structure and
locators* — the things that must be right — without depending on what a model
happened to say. Model quality is a bench concern, not a test concern.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager

from readeverything.domain.capability import Capability
from readeverything.domain.errors import InfrastructureError
from readeverything.domain.locators import TimeSpan
from readeverything.domain.observation import Event
from readeverything.domain.rendition import SpeakerId, TranscriptCue


class FakeSource:
    """An in-memory `FileSource` over a dict of uri to bytes."""

    def __init__(self, files: Mapping[str, bytes]) -> None:
        self._files = dict(files)

    def _get(self, uri: str) -> bytes:
        if uri not in self._files:
            raise KeyError(uri)
        return self._files[uri]

    async def exists(self, uri: str) -> bool:
        return uri in self._files

    async def size(self, uri: str) -> int:
        return len(self._get(uri))

    async def read_bytes(self, uri: str) -> bytes:
        return self._get(uri)

    async def read_range(self, uri: str, start: int, end: int) -> bytes:
        return self._get(uri)[start:end]

    async def stream(self, uri: str, *, chunk_size: int = 1 << 20) -> AsyncIterator[bytes]:
        data = self._get(uri)
        for offset in range(0, len(data), chunk_size):
            yield data[offset : offset + chunk_size]

    async def local_path(self, uri: str) -> str:
        raise NotImplementedError("FakeSource has no local path; use a tmp_path fixture")

    async def walk(self, uri: str) -> Sequence[str]:
        prefix = "" if uri in (".", "") else uri.rstrip("/") + "/"
        return sorted(path for path in self._files if path.startswith(prefix))


class FakeVision:
    """Describes an image by its size, deterministically.

    Counts its calls, so a test can assert a handler declined to invoke the
    model at all — an assertion no return value can express.
    """

    model_id: str = "fake-vision@1"

    def __init__(self) -> None:
        self.calls = 0

    async def describe(self, data: bytes, mime: str, prompt: str) -> str:
        self.calls += 1
        return f"[{mime} image of {len(data)} bytes] {prompt}"


class FakeVisionRefusing:
    """A vision model that answers with nothing.

    Not a hypothetical: reasoning models split their output into a reasoning
    channel and a content channel, and a model that spends its whole budget
    reasoning returns empty content. A handler must degrade rather than emit an
    empty description as if it were an observation.
    """

    model_id: str = "fake-vision-refusing@1"

    async def describe(self, data: bytes, mime: str, prompt: str) -> str:
        raise InfrastructureError("the model returned an empty completion")


#: How long a cue is, and how much silence follows it, in seconds.
_FAKE_CUE_S = 1.0
_FAKE_GAP_S = 0.5


class FakeTranscriber:
    """Cues derived mechanically from a stated duration, with silence between
    them.

    Contiguous, gap-free cues would hide the gap-filling problem a handler
    has to solve, so each cue is one second wide followed by a half-second of
    silence before the next — the shape a real transcriber produces, since
    speech is not continuous. `confidence` is `None`, matching
    `WhisperTranscriber`'s choice: this fake never measured anything, so it
    has nothing to report.

    THE CUES FIT INSIDE `duration_s`, and that is the point. An earlier
    version derived its cue count from the audio's BYTE length, so 160 KB of
    WAV became 160 cues spanning four minutes of a five-second file — a shape
    no transcriber can produce, and one under which a handler's
    "the last cue extends to the file duration" rule could never fire. The
    duration is stated rather than measured because this fake decodes nothing;
    tests that care pass the duration of their fixture.
    """

    model_id = "fake-asr@1"

    def __init__(self, duration_s: float = 3.0) -> None:
        self._duration_s = duration_s

    async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]:
        cues = []
        start = 0.0
        index = 0
        while start + _FAKE_CUE_S <= self._duration_s:
            cues.append(
                TranscriptCue(
                    span=TimeSpan(start, start + _FAKE_CUE_S),
                    text=f"cue {index}",
                    speaker=None,
                    confidence=None,
                )
            )
            start += _FAKE_CUE_S + _FAKE_GAP_S  # silence between cues
            index += 1
        return tuple(cues)


class FakeDiarizer:
    """Alternates two speakers, so speaker-turn barriers are exercised."""

    model_id = "fake-diarizer@1"

    async def diarize(self, path: str) -> tuple[tuple[TimeSpan, SpeakerId], ...]:
        return (
            (TimeSpan(0.0, 1.0), SpeakerId("SPEAKER_00")),
            (TimeSpan(1.0, 2.0), SpeakerId("SPEAKER_01")),
            (TimeSpan(2.0, 3.0), SpeakerId("SPEAKER_00")),
        )


class RecordingObserver:
    """Keeps every event it is handed, in order, for assertion."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def observe(self, event: Event) -> None:
        self.events.append(event)


class RaisingObserver:
    """An observer that always fails, to prove `emit` contains it."""

    def observe(self, event: Event) -> None:
        raise RuntimeError("this observer always fails")


class CountingLimiter:
    """Tracks peak in-flight count per capability, without bounding anything.

    A handler test that wants to assert "at most N of this ran at once"
    without pulling in `asyncio.Semaphore` semantics uses this instead: it
    records concurrency rather than enforcing it.
    """

    def __init__(self) -> None:
        self.in_flight: dict[Capability, int] = {}
        self.peak: dict[Capability, int] = {}

    @asynccontextmanager
    async def limit(self, capability: Capability) -> AsyncIterator[None]:
        self.in_flight[capability] = self.in_flight.get(capability, 0) + 1
        self.peak[capability] = max(self.peak.get(capability, 0), self.in_flight[capability])
        try:
            yield
        finally:
            self.in_flight[capability] -= 1
