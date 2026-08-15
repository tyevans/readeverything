from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from readeverything.adapters.whisper_transcriber import WhisperTranscriber
from readeverything.domain.errors import InfrastructureError
from readeverything.ports.transcription import Transcriber
from readeverything.testing.fakes import FakeTranscriber


def test_the_model_directory_is_required() -> None:
    """No default, so a caller who has not configured ASR finds out at the
    composition root rather than when a first `represent()` quietly pulls
    several hundred megabytes from Hugging Face."""
    with pytest.raises(TypeError):
        WhisperTranscriber()  # type: ignore[call-arg]  # the point of the test


def test_construction_never_reaches_the_network() -> None:
    """`local_files_only=True` is what makes "downloads nothing implicitly"
    enforced rather than merely conventional. A local path alone does not
    prevent a lookup; this flag does.
    """
    source = Path("src/readeverything/adapters/whisper_transcriber.py").read_text()
    assert "local_files_only=True" in source


def test_a_missing_model_directory_fails_loudly(tmp_path: Path) -> None:
    """Loud here, because this is the composition root's job to surface. The
    HANDLER is what must never raise, and it catches this."""
    with pytest.raises(InfrastructureError):
        WhisperTranscriber(model_dir=str(tmp_path / "nope"))


async def test_the_fake_satisfies_the_port() -> None:
    assert isinstance(FakeTranscriber(), Transcriber)


async def test_the_fake_produces_ordered_cues_with_silence_between_them() -> None:
    """The fake stands in for a real transcriber in every unit test, so its
    shape has to be the shape the handler must cope with — cues in order, with
    silence between them."""
    cues = await FakeTranscriber().transcribe(b"x" * 1000, "audio/wav")
    assert cues
    assert all(c.span.start_s < c.span.end_s for c in cues)
    assert all(a.span.end_s < b.span.start_s for a, b in pairwise(cues))


async def test_the_fake_never_places_a_cue_past_the_duration_it_was_given() -> None:
    """No real transcriber returns cues beyond the audio's own length. An
    earlier version derived its cue count from the audio's BYTE length, so
    160 KB of WAV became cues spanning 239 seconds of a five-second file — a
    shape that cannot occur in production, and one under which the handler's
    extend-to-the-duration rule could never fire."""
    cues = await FakeTranscriber(duration_s=5.0).transcribe(b"x" * 160_000, "audio/wav")
    assert cues
    assert all(c.span.end_s <= 5.0 for c in cues)
