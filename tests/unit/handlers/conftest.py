from __future__ import annotations

from pathlib import Path

import pytest
from tests.fixtures_media import ffmpeg_available, video_only, video_with_audio


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
