"""Captions through the whole stack, on a container ffmpeg actually built.

The unit tests parse SRT text and drive the precedence rule with fakes. What
neither can prove is that a real container's real subtitle track survives
probing, track selection, extraction, tiling and rendering — five components
whose seams are exactly where a caption track goes missing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from readeverything import Budget, build_perception
from readeverything.handlers.video import CAPTION_MARKER, SPEECH_MARKER

SRT = """1
00:00:00,500 --> 00:00:02,000
first thing said

2
00:00:03,000 --> 00:00:04,500
second thing said
"""


@pytest.fixture
def captioned_video(tmp_path: Path) -> Path:
    """A real mp4 with a real mov_text track — the reference file's codec."""
    srt = tmp_path / "s.srt"
    srt.write_text(SRT)
    out = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=size=64x48:rate=5:duration=5",
            "-i", str(srt),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:s", "mov_text",
            str(out),
        ],
        check=True,
    )
    return out


async def test_the_card_announces_the_caption_track(captioned_video: Path) -> None:
    """The step that was missing: an agent cannot choose the cheap path if
    nothing tells it the cheap path exists."""
    perception = await build_perception(captioned_video.parent)
    card = await perception.inspect(captioned_video.name)
    assert card.facts["text_caption_tracks"] == 1


async def test_the_timeline_reads_the_captions(captioned_video: Path) -> None:
    perception = await build_perception(captioned_video.parent)
    rendered = await perception.represent(captioned_video.name, Budget(max_chars=None))
    assert f"{CAPTION_MARKER} first thing said" in rendered.text
    assert f"{CAPTION_MARKER} second thing said" in rendered.text


async def test_no_model_of_any_kind_was_needed(captioned_video: Path) -> None:
    """No vision model, no transcriber, no endpoint — and the words are there.
    This is the whole argument for reading captions first."""
    perception = await build_perception(captioned_video.parent)
    rendered = await perception.represent(captioned_video.name, Budget(max_chars=None))
    assert "first thing said" in rendered.text
    assert SPEECH_MARKER not in rendered.text


async def test_every_character_still_maps_to_a_span(captioned_video: Path) -> None:
    """A hit that cannot be cited is the failure this library exists to avoid,
    and a new producer of text is a new way to break the map."""
    perception = await build_perception(captioned_video.parent)
    rendered = await perception.represent(captioned_video.name, Budget(max_chars=None))
    assert rendered.locator_map.length == len(rendered.text)
