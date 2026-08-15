"""Video fixtures generated at test time rather than committed as binaries.

Same rationale as `fixtures_pdf.py`: a committed binary rots silently, so
these are generated fresh with `ffmpeg` into a temp directory and read back as
bytes. `ffmpeg` is a dev-time-only dependency here — nothing under `src/`
imports it directly (the library shells out to `ffprobe`/`ffmpeg` as external
executables, never as a Python import).

`ffmpeg_available()` lets media tests skip cleanly on a machine without the
binary. It must NOT be used to skip the library's own no-ffmpeg behaviour
(Task 6) — that path is what proves degradation works, and skipping it would
hide the one test that exercises it.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _run_ffmpeg(args: list[str], out_path: Path) -> bytes:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *args, str(out_path)],
        check=True,
        capture_output=True,
    )
    return out_path.read_bytes()


def video_with_audio(seconds: int = 5, size: str = "320x240", rate: int = 10) -> bytes:
    """An h264+aac mp4 with one video stream and one audio stream."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "sample.mp4"
        return _run_ffmpeg(
            [
                "-f",
                "lavfi",
                "-i",
                f"testsrc=duration={seconds}:size={size}:rate={rate}",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:duration={seconds}",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-shortest",
            ],
            out_path,
        )


def video_only(seconds: int = 2) -> bytes:
    """An h264 mp4 with a video stream and deliberately NO audio stream.

    Task 5's "no audio stream" path is tested against this fixture. Do not add
    an audio track to it — a later test's premise depends on this fixture
    staying audio-free.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "video_only.mp4"
        return _run_ffmpeg(
            [
                "-f",
                "lavfi",
                "-i",
                f"testsrc=duration={seconds}:size=320x240:rate=10",
                "-c:v",
                "libx264",
                "-an",
            ],
            out_path,
        )


def scene_cuts(seconds_each: int = 2) -> bytes:
    """Two visually distinct segments concatenated, so a cut is detectable.

    `testsrc` and `color=c=red` are unrelated content, so the join between
    them is a real discontinuity — not the uniform content that would make
    "no barriers found" pass for the wrong reason.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "scene_cuts.mp4"
        return _run_ffmpeg(
            [
                "-f",
                "lavfi",
                "-i",
                f"testsrc=duration={seconds_each}:size=320x240:rate=10",
                "-f",
                "lavfi",
                "-i",
                f"color=c=red:duration={seconds_each}:size=320x240:rate=10",
                "-filter_complex",
                "[0:v][1:v]concat=n=2:v=1:a=0[v]",
                "-map",
                "[v]",
                "-c:v",
                "libx264",
            ],
            out_path,
        )
