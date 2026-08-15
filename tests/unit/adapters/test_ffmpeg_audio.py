from __future__ import annotations

from pathlib import Path

from readeverything.adapters.ffmpeg_audio import FfmpegAudio
from readeverything.ports.audio import AudioExtractor


async def test_a_real_adapter_satisfies_the_port() -> None:
    assert isinstance(FfmpegAudio(), AudioExtractor)


async def test_a_video_with_sound_yields_wav_bytes(sample_video: str) -> None:
    wav = await FfmpegAudio().extract(sample_video)
    assert wav is not None
    assert wav.startswith(b"RIFF")


async def test_a_file_with_no_audio_stream_returns_none(video_only_path: str) -> None:
    """The opposite convention from frame extraction, and both are measured.

    ffmpeg exits 234 with "Output file does not contain any stream" here, but
    exits 0 with empty output when a frame seek is past the end. Streams are
    checked by exit status; frames by output length. An adapter that checked the
    wrong one for either would be silently wrong in one direction and noisily
    wrong in the other.
    """
    assert await FfmpegAudio().extract(video_only_path) is None


async def test_an_audio_only_file_also_works(audio_only_path: str) -> None:
    wav = await FfmpegAudio().extract(audio_only_path)
    assert wav is not None
    assert wav.startswith(b"RIFF")


async def test_a_file_that_is_not_media_returns_none(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not media at all")
    assert await FfmpegAudio().extract(str(junk)) is None


async def test_the_extractor_never_raises(tmp_path: Path) -> None:
    """A handler must be able to ask about anything."""
    assert await FfmpegAudio().extract(str(tmp_path / "absent.mp4")) is None
