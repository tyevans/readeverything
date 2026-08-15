"""Deterministic stand-ins for everything expensive or nondeterministic.

Unit tests never assert on model text. These fakes produce output derived
mechanically from their input, so a test can assert on *structure and
locators* — the things that must be right — without depending on what a model
happened to say. Model quality is a bench concern, not a test concern.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence

from readeverything.domain.errors import InfrastructureError
from readeverything.domain.locators import TimeSpan
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


class FakeTranscriber:
    """One cue per second, text derived from the index."""

    model_id = "fake-asr@1"

    def __init__(self, *, cues: int = 3) -> None:
        self._cues = cues

    async def transcribe(self, path: str) -> tuple[TranscriptCue, ...]:
        return tuple(
            TranscriptCue(
                span=TimeSpan(float(i), float(i) + 1.0),
                text=f"cue {i}",
                speaker=None,
                confidence=1.0,
            )
            for i in range(self._cues)
        )


class FakeDiarizer:
    """Alternates two speakers, so speaker-turn barriers are exercised."""

    model_id = "fake-diarizer@1"

    async def diarize(self, path: str) -> tuple[tuple[TimeSpan, SpeakerId], ...]:
        return (
            (TimeSpan(0.0, 1.0), SpeakerId("SPEAKER_00")),
            (TimeSpan(1.0, 2.0), SpeakerId("SPEAKER_01")),
            (TimeSpan(2.0, 3.0), SpeakerId("SPEAKER_00")),
        )
