from __future__ import annotations

from pathlib import Path

import pytest
from tests.fixtures_media import ffmpeg_available, video_with_audio
from tests.fixtures_pdf import born_digital


@pytest.fixture
def three_page_pdf() -> bytes:
    return born_digital(["alpha", "beta", "gamma"])


@pytest.fixture
def sample_video(tmp_path: Path) -> str:
    if not ffmpeg_available():
        pytest.skip("ffmpeg not available")
    path = tmp_path / "sample.mp4"
    path.write_bytes(video_with_audio())
    return str(path)
