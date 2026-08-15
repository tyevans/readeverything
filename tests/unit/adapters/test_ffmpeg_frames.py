from __future__ import annotations

from pathlib import Path

import pytest

from readeverything.adapters.ffmpeg_frames import FfmpegFrames
from readeverything.domain.errors import InfrastructureError
from readeverything.ports.frames import FrameExtractor


async def test_a_real_adapter_satisfies_the_port() -> None:
    assert isinstance(FfmpegFrames(), FrameExtractor)


async def test_a_frame_in_range_comes_back_as_a_png(sample_video: str) -> None:
    png = await FfmpegFrames().frame_at(sample_video, 2.5)
    assert png is not None
    assert png.startswith(b"\x89PNG")


async def test_seeking_past_the_end_returns_none_rather_than_empty_bytes(
    sample_video: str,
) -> None:
    """The measured trap. ffmpeg exits 0 with zero bytes and no stderr when the
    seek is past the end — it does not error:

        -ss 2.5  -> exit 0, 16942 bytes, no stderr
        -ss 999  -> exit 0,     0 bytes, no stderr

    An adapter checking `returncode` would return b"" and a handler would hand
    an empty PNG to a model as "the frame at t=999". Output length is the only
    thing that distinguishes success here.
    """
    assert await FfmpegFrames().frame_at(sample_video, 999.0) is None


async def test_a_negative_time_returns_none(sample_video: str) -> None:
    assert await FfmpegFrames().frame_at(sample_video, -1.0) is None


async def test_a_file_that_is_not_media_returns_none(tmp_path: Path) -> None:
    """The extractor answers "no frame", never raises. A handler must be able to
    call it about anything."""
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    assert await FfmpegFrames().frame_at(str(junk), 0.0) is None


async def test_a_missing_file_returns_none() -> None:
    assert await FfmpegFrames().frame_at("/no/such/file.mp4", 0.0) is None


# --- scene cuts -----------------------------------------------------------------


async def test_a_video_with_a_cut_reports_a_cut(scene_cut_video: str) -> None:
    cuts = await FfmpegFrames().scene_cuts(scene_cut_video)
    assert cuts


async def test_uniform_content_reports_no_cuts(sample_video: str) -> None:
    """ "No cuts found" must be a real answer, not a stand-in for "detection
    failed" — otherwise a caller cannot tell an unedited video from a broken
    detector."""
    assert await FfmpegFrames().scene_cuts(sample_video) == ()


async def test_a_missing_file_raises_rather_than_returning_no_cuts() -> None:
    """Detection failing and detection finding nothing must be distinguishable
    outcomes: a missing file is a failure, not "no cuts"."""
    with pytest.raises(InfrastructureError):
        await FfmpegFrames().scene_cuts("/no/such/file.mp4")
