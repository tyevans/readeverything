from __future__ import annotations

import time
from pathlib import Path

import pytest

from readeverything.adapters.ffprobe_streams import FfprobeStreams
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
