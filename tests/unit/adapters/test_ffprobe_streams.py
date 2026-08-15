from __future__ import annotations

import time
from pathlib import Path

import pytest

from readeverything.adapters.ffprobe_streams import FfprobeStreams, _parse_facts
from readeverything.domain.errors import InfrastructureError
from readeverything.ports.streams import StreamProbe


async def test_a_real_adapter_satisfies_the_port() -> None:
    assert isinstance(FfprobeStreams(), StreamProbe)


async def test_a_probe_reports_duration_and_both_streams(sample_video: str) -> None:
    facts = await FfprobeStreams().probe(sample_video)
    assert facts.duration_s == pytest.approx(5.0, abs=0.2)
    assert len(facts.video_streams) == 1
    assert len(facts.audio_streams) == 1


async def test_duration_is_a_float_though_ffprobe_returns_a_string(sample_video: str) -> None:
    """`format.duration` comes back as the string "5.000000". A caller doing
    arithmetic on a string gets a confusing failure a long way from here."""
    facts = await FfprobeStreams().probe(sample_video)
    assert isinstance(facts.duration_s, float)


async def test_the_frame_rate_rational_is_parsed(sample_video: str) -> None:
    """`r_frame_rate` is "10/1", not 10. And for an audio stream it is "0/0",
    which is a division by zero waiting for whoever forgets to guard it."""
    facts = await FfprobeStreams().probe(sample_video)
    assert facts.video_streams[0].frame_rate == pytest.approx(10.0)
    assert facts.audio_streams[0].frame_rate is None


async def test_the_container_keeps_the_whole_candidate_list(sample_video: str) -> None:
    """`format_name` is a comma-joined list of candidates ffprobe declined to
    narrow down. Picking one would assert an identification it did not make."""
    facts = await FfprobeStreams().probe(sample_video)
    assert "," in facts.container
    assert "mp4" in facts.container


async def test_a_file_that_is_not_media_raises_infrastructure_error(tmp_path: Path) -> None:
    """The probe may raise; the HANDLER must not. Keeping it loud here lets the
    handler decide how to degrade rather than receive fabricated facts."""
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    with pytest.raises(InfrastructureError):
        await FfprobeStreams().probe(str(junk))


async def test_a_missing_file_raises_infrastructure_error(tmp_path: Path) -> None:
    with pytest.raises(InfrastructureError):
        await FfprobeStreams().probe(str(tmp_path / "does-not-exist.mp4"))


async def test_probing_does_not_decode(sample_video: str) -> None:
    """The card must stay cheap. Asserted as a bound on wall time rather than
    left as a claim — a decode of even this tiny file is orders of magnitude
    slower than a header read."""
    start = time.monotonic()
    await FfprobeStreams().probe(sample_video)
    assert time.monotonic() - start < 5.0


async def test_subtitle_streams_are_kept_and_classified() -> None:
    """A card that is silent about captions sends an agent to a vision model
    to learn what one ffmpeg call would have told it for free. Classification
    matters as much as presence: a `mov_text` track extracts to characters in
    a second, a `dvd_subtitle` track is pixels and needs OCR, and the
    reference file carries one of each."""
    facts = _parse_facts(
        {
            "format": {"duration": "10.0", "format_name": "mov,mp4"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "r_frame_rate": "30/1"},
                {"codec_type": "subtitle", "codec_name": "mov_text", "tags": {"language": "eng"}},
                {"codec_type": "subtitle", "codec_name": "dvd_subtitle"},
            ],
        }
    )
    subtitles = facts.subtitle_streams
    assert [s.codec for s in subtitles] == ["mov_text", "dvd_subtitle"]
    assert subtitles[0].is_text is True
    assert subtitles[0].language == "eng"
    assert subtitles[1].is_text is False
    assert subtitles[1].language is None
    assert facts.text_subtitle_streams == (subtitles[0],)


async def test_streams_that_are_neither_media_nor_subtitles_are_still_dropped() -> None:
    """The reference file also carries a `bin_data` stream. Nothing can be
    asked of it, so it has no business on a card."""
    facts = _parse_facts(
        {
            "format": {"duration": "10.0", "format_name": "mov,mp4"},
            "streams": [{"codec_type": "data", "codec_name": "bin_data"}],
        }
    )
    assert facts.streams == ()


async def test_video_and_audio_streams_are_never_marked_as_text() -> None:
    """`is_text` is a fact about subtitle codecs. A video stream answering
    True would make `text_subtitle_streams` a lie."""
    facts = _parse_facts(
        {
            "format": {"duration": "10.0", "format_name": "mov,mp4"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "r_frame_rate": "30/1"},
                {"codec_type": "audio", "codec_name": "aac"},
            ],
        }
    )
    assert all(s.is_text is False for s in facts.streams)
    assert facts.text_subtitle_streams == ()
