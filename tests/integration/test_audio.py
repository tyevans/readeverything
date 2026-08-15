"""`AudioHandler` reached through `build_perception`, not constructed by hand."""

from __future__ import annotations

from pathlib import Path

import pytest

from readeverything.composition import build_perception
from readeverything.domain.identity import MediaKind
from tests.fixtures_media import audio_only_m4a, ffmpeg_available

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not ffmpeg_available(), reason="requires ffmpeg")
async def test_read_span_is_not_offered_without_a_transcriber(media_root: Path) -> None:
    """Negotiation, not a runtime apology."""
    perception = await build_perception(media_root)
    card = await perception.inspect("clip.wav")
    assert "read_span" not in {a.name for a in card.affordances}


@pytest.mark.skipif(not ffmpeg_available(), reason="requires ffmpeg")
async def test_an_m4a_file_dispatches_to_the_audio_handler(media_root: Path) -> None:
    """`.m4a` and `.mp4` share a header; only the extension tells them apart.

    Without the detection fix, this audio-only file would be misdetected as
    `video/mp4` and dispatch to `VideoHandler`, whose `represent()` degrades to
    an opaque byte range instead of reaching the handler built to read audio.
    """
    (media_root / "voice-memo.m4a").write_bytes(audio_only_m4a())
    perception = await build_perception(media_root)
    card = await perception.inspect("voice-memo.m4a")
    assert card.kind == MediaKind.AUDIO
