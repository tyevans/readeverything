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
from readeverything.testing.fakes import FakeTranscriber
from tests.assertions import text_of

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
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=5:duration=5",
            "-i",
            str(srt),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:s",
            "mov_text",
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


async def test_an_agent_can_reach_the_words_without_a_model(captioned_video: Path) -> None:
    """The affordance whose absence made the whole caption path invisible.

    `represent()` had the words from the first day captions were wired, but an
    agent holds inspect_path/list_paths/invoke_affordance and has no route to
    `represent()`. So the card announced a readable caption track and then
    offered nothing but pixels, and a real agent run did the only thing it
    could: sampled frames, at ~65s and one model call each.
    """
    perception = await build_perception(captioned_video.parent)
    card = await perception.inspect(captioned_video.name)
    assert "read_transcript" in {a.name for a in card.affordances}

    result = await perception.invoke(captioned_video.name, "read_transcript", {})
    assert "first thing said" in text_of(result)
    assert "second thing said" in text_of(result)
    assert result.degraded is False


async def test_a_window_reads_only_what_is_inside_it(captioned_video: Path) -> None:
    perception = await build_perception(captioned_video.parent)
    result = await perception.invoke(
        captioned_video.name, "read_transcript", {"start_s": 2.5, "end_s": 5.0}
    )
    assert "second thing said" in text_of(result)
    assert "first thing said" not in text_of(result)


async def test_a_file_without_captions_says_so_rather_than_failing(tmp_path: Path) -> None:
    out = tmp_path / "silent.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=5:duration=2",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
    )
    perception = await build_perception(tmp_path)
    result = await perception.invoke(out.name, "read_transcript", {})
    assert result.degraded is True
    assert "no readable caption" in text_of(result)


async def test_read_transcript_falls_back_to_asr_when_there_are_no_captions(
    tmp_path: Path,
) -> None:
    """The second time in this feature that words existed and nothing could
    reach them. `read_transcript` read captions only, so on a caption-less
    file it told an agent holding a working transcriber that there were no
    words to read — and the agent went back to sampling frames.

    It now uses the same precedence `represent()` does.
    """
    out = tmp_path / "spoken.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=5:duration=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(out),
        ],
        check=True,
    )
    perception = await build_perception(tmp_path, transcriber=FakeTranscriber(duration_s=3.0))
    result = await perception.invoke(out.name, "read_transcript", {})
    assert result.degraded is False
    assert "cue 0" in text_of(result)


async def test_read_transcript_is_offered_with_only_a_transcriber(tmp_path: Path) -> None:
    out = tmp_path / "any.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=5:duration=1",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
    )
    perception = await build_perception(tmp_path, transcriber=FakeTranscriber(duration_s=1.0))
    card = await perception.inspect(out.name)
    assert "read_transcript" in {a.name for a in card.affordances}
