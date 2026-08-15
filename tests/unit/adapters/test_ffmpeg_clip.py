from __future__ import annotations

from pathlib import Path

from readeverything.adapters.ffmpeg_clip import CLIP_MIME, FfmpegClip
from readeverything.adapters.ffprobe_streams import FfprobeStreams
from readeverything.ports.clip_source import ClipExtractor


def test_a_real_adapter_satisfies_the_port() -> None:
    assert isinstance(FfmpegClip(), ClipExtractor)


async def test_a_range_becomes_a_playable_container(sample_video: str, tmp_path: Path) -> None:
    """Asserted by probing the output rather than by looking for magic bytes:
    what matters is that a decoder can open it and find a video stream, which
    is exactly what the model on the other end will try to do."""
    data = await FfmpegClip().clip(sample_video, 1.0, 3.0)
    assert data is not None
    out = tmp_path / "clip.mp4"
    out.write_bytes(data)
    facts = await FfprobeStreams().probe(str(out))
    assert facts.video_streams
    assert facts.duration_s > 0


async def test_the_clip_covers_roughly_the_requested_span(
    sample_video: str, tmp_path: Path
) -> None:
    """`-ss` before `-i` seeks by keyframe, so the duration is approximate.
    Pinned loosely on purpose: tightening this would be asserting ffmpeg's
    keyframe placement, which is not this adapter's promise. The handler
    reports the REQUESTED span as its locator for the same reason."""
    data = await FfmpegClip().clip(sample_video, 1.0, 3.0)
    assert data is not None
    out = tmp_path / "clip.mp4"
    out.write_bytes(data)
    facts = await FfprobeStreams().probe(str(out))
    assert 1.0 < facts.duration_s < 3.5


async def test_audio_is_dropped(sample_video: str, tmp_path: Path) -> None:
    """The clip goes to a model that cannot hear it, and the transcript path
    already answers what was said. Carrying the audio would be bytes paid for
    and discarded."""
    data = await FfmpegClip().clip(sample_video, 0.0, 2.0)
    assert data is not None
    out = tmp_path / "clip.mp4"
    out.write_bytes(data)
    assert (await FfprobeStreams().probe(str(out))).audio_streams == ()


async def test_a_range_past_the_end_is_none(sample_video: str) -> None:
    assert await FfmpegClip().clip(sample_video, 300.0, 302.0) is None


async def test_an_empty_or_backwards_range_is_none(sample_video: str) -> None:
    """Refused before spawning anything: there is no process worth starting to
    learn that 3 is not after 4."""
    assert await FfmpegClip().clip(sample_video, 3.0, 3.0) is None
    assert await FfmpegClip().clip(sample_video, 4.0, 3.0) is None


async def test_a_negative_start_is_none(sample_video: str) -> None:
    assert await FfmpegClip().clip(sample_video, -1.0, 2.0) is None


async def test_a_missing_file_is_none_not_an_exception(tmp_path: Path) -> None:
    assert await FfmpegClip().clip(str(tmp_path / "nope.mp4"), 0.0, 1.0) is None


async def test_a_file_that_is_not_media_is_none(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    assert await FfmpegClip().clip(str(junk), 0.0, 1.0) is None


def test_the_mime_matches_what_the_muxer_produces() -> None:
    assert CLIP_MIME == "video/mp4"
