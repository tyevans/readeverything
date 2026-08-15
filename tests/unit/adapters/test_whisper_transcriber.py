from __future__ import annotations

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


async def test_the_fake_produces_cues_that_tile_without_gaps() -> None:
    """The fake stands in for a real transcriber in every unit test, so its
    shape has to be the shape the handler must cope with — cues in order, with
    silence between them."""
    cues = await FakeTranscriber().transcribe(b"x" * 1000, "audio/wav")
    assert cues
    assert all(c.span.start_s < c.span.end_s for c in cues)
    assert all(a.span.end_s <= b.span.start_s for a, b in zip(cues, cues[1:], strict=False))
