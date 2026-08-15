from __future__ import annotations

from pathlib import Path

import pytest

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.caching_words import (
    CachingCaptionExtractor,
    CachingTranscriber,
    decode_cues,
    encode_cues,
)
from readeverything.domain.locators import TimeSpan
from readeverything.domain.rendition import CueSource, SpeakerId, TranscriptCue
from readeverything.ports.captions import CaptionExtractor
from readeverything.ports.transcription import Transcriber

CUES = (
    TranscriptCue(
        span=TimeSpan(0.0, 1.0),
        text="first",
        speaker=SpeakerId("SPEAKER_00"),
        confidence=None,
        source=CueSource.CAPTIONED,
    ),
    TranscriptCue(span=TimeSpan(1.0, 2.0), text="second", speaker=None, confidence=0.5),
)


class _CountingTranscriber:
    model_id = "counting-asr@1"

    def __init__(self, cues: tuple[TranscriptCue, ...] = CUES) -> None:
        self.calls = 0
        self._cues = cues

    async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]:
        self.calls += 1
        return self._cues


class _CountingCaptions:
    def __init__(self, cues: tuple[TranscriptCue, ...] | None = CUES) -> None:
        self.calls = 0
        self._cues = cues

    async def extract(
        self, path: str, track: int | None = None
    ) -> tuple[TranscriptCue, ...] | None:
        self.calls += 1
        return self._cues


def test_the_wrappers_still_satisfy_their_ports() -> None:
    store = InMemoryArtifactStore()
    assert isinstance(CachingTranscriber(inner=_CountingTranscriber(), store=store), Transcriber)
    assert isinstance(
        CachingCaptionExtractor(inner=_CountingCaptions(), store=store), CaptionExtractor
    )


def test_a_round_trip_preserves_every_field() -> None:
    """`source` above all: a codec that dropped it would silently turn every
    cached caption back into speech, which is the one claim this feature
    exists to keep honest."""
    assert decode_cues(encode_cues(CUES)) == CUES


def test_an_unreadable_entry_raises_rather_than_guessing() -> None:
    with pytest.raises(ValueError):
        decode_cues(b'[{"start_s": 0, "end_s": 1, "text": "x", "source": "said"}]')


async def test_the_same_audio_is_transcribed_once(tmp_path: Path) -> None:
    """The measurement that motivated this: ~100 seconds of Whisper per
    question about one file, because each window was a different cache key at
    the rendition layer."""
    inner = _CountingTranscriber()
    caching = CachingTranscriber(inner=inner, store=InMemoryArtifactStore())
    first = await caching.transcribe(b"audio-bytes", "audio/wav")
    second = await caching.transcribe(b"audio-bytes", "audio/wav")
    assert first == second == CUES
    assert inner.calls == 1


async def test_different_audio_is_transcribed_again() -> None:
    inner = _CountingTranscriber()
    caching = CachingTranscriber(inner=inner, store=InMemoryArtifactStore())
    await caching.transcribe(b"one", "audio/wav")
    await caching.transcribe(b"two", "audio/wav")
    assert inner.calls == 2


async def test_a_different_model_does_not_share_an_entry() -> None:
    """The same reason `artifact_key` carries the capability fingerprint:
    serving one model's transcript as another's is invisible until someone
    compares two answers and cannot explain the difference."""
    store = InMemoryArtifactStore()
    first = _CountingTranscriber()
    second = _CountingTranscriber()
    second.model_id = "counting-asr@2"
    await CachingTranscriber(inner=first, store=store).transcribe(b"same", "audio/wav")
    await CachingTranscriber(inner=second, store=store).transcribe(b"same", "audio/wav")
    assert first.calls == 1
    assert second.calls == 1


async def test_an_empty_result_is_not_cached() -> None:
    """A transcriber that heard nothing may have failed transiently. Caching
    the silence would make one bad run permanent."""
    inner = _CountingTranscriber(cues=())
    caching = CachingTranscriber(inner=inner, store=InMemoryArtifactStore())
    await caching.transcribe(b"quiet", "audio/wav")
    await caching.transcribe(b"quiet", "audio/wav")
    assert inner.calls == 2


async def test_captions_are_extracted_once_per_file(tmp_path: Path) -> None:
    video = tmp_path / "a.mp4"
    video.write_bytes(b"pretend container")
    inner = _CountingCaptions()
    caching = CachingCaptionExtractor(inner=inner, store=InMemoryArtifactStore())
    assert await caching.extract(str(video), 1) == CUES
    assert await caching.extract(str(video), 1) == CUES
    assert inner.calls == 1


async def test_a_different_track_is_a_different_entry(tmp_path: Path) -> None:
    """The reference file's two subtitle tracks say different things — one is
    English text and the other is bitmaps."""
    video = tmp_path / "a.mp4"
    video.write_bytes(b"pretend container")
    inner = _CountingCaptions()
    caching = CachingCaptionExtractor(inner=inner, store=InMemoryArtifactStore())
    await caching.extract(str(video), 0)
    await caching.extract(str(video), 1)
    assert inner.calls == 2


async def test_an_edited_file_is_read_again(tmp_path: Path) -> None:
    video = tmp_path / "a.mp4"
    video.write_bytes(b"pretend container")
    inner = _CountingCaptions()
    caching = CachingCaptionExtractor(inner=inner, store=InMemoryArtifactStore())
    await caching.extract(str(video), 1)
    video.write_bytes(b"a longer pretend container")
    await caching.extract(str(video), 1)
    assert inner.calls == 2


async def test_a_missing_file_is_passed_through_uncached(tmp_path: Path) -> None:
    """No file means no key to cache under, and the inner extractor already
    answers `None` for exactly this case."""
    inner = _CountingCaptions(cues=None)
    caching = CachingCaptionExtractor(inner=inner, store=InMemoryArtifactStore())
    assert await caching.extract(str(tmp_path / "nope.mp4")) is None
    assert inner.calls == 1
