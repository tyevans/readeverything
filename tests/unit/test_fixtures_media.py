"""Guards on the generated video fixtures themselves.

These are load-bearing, not decorative: later tasks test specific behaviour
(no-audio-stream handling, scene-cut detection) against these exact fixtures.
If a fixture's content ever drifts, those tests would silently start testing
something else while still passing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from tests.fixtures_media import ffmpeg_available, scene_cuts, video_only

from readeverything.adapters.ffprobe_streams import FfprobeStreams

pytestmark = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not available")


def _mean_pixel(video_path: Path, at_second: float, tmp_path: Path) -> tuple[float, ...]:
    """The mean RGB of the frame nearest `at_second`, via ffmpeg -> a raw PPM."""
    frame_path = tmp_path / f"frame_{at_second}.ppm"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            str(at_second),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(frame_path),
        ],
        check=True,
        capture_output=True,
    )
    from PIL import Image

    # Downsampling to 1x1 with box resampling averages every pixel — a cheap
    # mean colour without touching the deprecated `getdata()`.
    image = Image.open(frame_path).convert("RGB").resize((1, 1), Image.Resampling.BOX)
    r, g, b = image.getpixel((0, 0))  # type: ignore[misc]  # PIL's getpixel is typed as untyped tuple
    return (float(r), float(g), float(b))


async def test_the_video_only_fixture_really_has_no_audio_stream(tmp_path: Path) -> None:
    """Task 5's "no audio stream" path is tested against this fixture. If it
    ever gains an audio track, that test silently becomes a different test."""
    path = tmp_path / "video_only.mp4"
    path.write_bytes(video_only())
    facts = await FfprobeStreams().probe(str(path))
    assert facts.video_streams
    assert not facts.audio_streams


async def test_the_scene_cut_fixture_really_contains_a_cut(tmp_path: Path) -> None:
    """Barrier tests depend on a detectable cut existing. A fixture of uniform
    content would make "no barriers found" look correct."""
    path = tmp_path / "scene_cuts.mp4"
    path.write_bytes(scene_cuts())
    facts = await FfprobeStreams().probe(str(path))
    assert facts.video_streams
    # Two 2s segments concatenated: the container duration should reflect
    # both, not just one segment silently dropped by the concat filter.
    assert facts.duration_s == pytest.approx(4.0, abs=0.3)

    # And the segments must actually differ in content: a frame from the
    # first half (testsrc pattern) vs. the second half (solid red) should
    # have distinctly different mean colour, proving a real cut exists rather
    # than uniform content that would make "no barriers found" pass emptily.
    before = _mean_pixel(path, at_second=0.5, tmp_path=tmp_path)
    after = _mean_pixel(path, at_second=3.5, tmp_path=tmp_path)
    distance = sum(abs(b - a) for b, a in zip(before, after, strict=True))
    assert distance > 60
