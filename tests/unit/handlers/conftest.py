from __future__ import annotations

from pathlib import Path

import pytest
from tests.fixtures_media import (
    audio_only,
    ffmpeg_available,
    scene_cuts,
    video_only,
    video_with_audio,
)


@pytest.fixture
def audio_path(tmp_path: Path) -> str:
    """Three seconds of a sine tone in a WAV, with no video stream at all."""
    if not ffmpeg_available():
        pytest.skip("ffmpeg not available")
    path = tmp_path / "audio_only.wav"
    path.write_bytes(audio_only())
    return str(path)


@pytest.fixture
def video_only_path(tmp_path: Path) -> str:
    """A file with a video stream and deliberately no audio track."""
    if not ffmpeg_available():
        pytest.skip("ffmpeg not available")
    path = tmp_path / "no_audio.mp4"
    path.write_bytes(video_only())
    return str(path)


@pytest.fixture
def sample_video(tmp_path: Path) -> str:
    """Five seconds of testsrc with an aac track, at 10 fps."""
    if not ffmpeg_available():
        pytest.skip("ffmpeg not available")
    path = tmp_path / "sample.mp4"
    path.write_bytes(video_with_audio())
    return str(path)


@pytest.fixture
def silent_video(tmp_path: Path) -> str:
    if not ffmpeg_available():
        pytest.skip("ffmpeg not available")
    path = tmp_path / "video_only.mp4"
    path.write_bytes(video_only())
    return str(path)


@pytest.fixture
def scene_cut_video(tmp_path: Path) -> str:
    """Two visually distinct segments, so a real scene cut exists to find."""
    if not ffmpeg_available():
        pytest.skip("ffmpeg not available")
    path = tmp_path / "scene_cuts.mp4"
    path.write_bytes(scene_cuts())
    return str(path)
