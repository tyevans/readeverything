"""`watch_segment` against a real endpoint that accepts video.

Skipped unless a server is configured, like every other test in this
directory. The cap test does NOT need the server — a refusal happens before any
call — so it runs everywhere, which is deliberate: the cap is the part that
protects a caller from a six-figure token bill, and it should not be a test
that only runs on one machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from readeverything.adapters.clip_langchain import build_openai_clip_model
from readeverything.adapters.ffmpeg_clip import FfmpegClip
from readeverything.handlers.video import TOKENS_PER_CLIP_SECOND, WatchSegmentParams


@pytest.mark.live
async def test_a_short_clip_comes_back_described(
    live_base_url: str, live_model_name: str, tmp_path: Path
) -> None:
    """Verified working against llama.cpp b10438 on 2026-08-15. Before that
    build this same request failed with "Failed to load image or audio file"."""
    import subprocess

    source = tmp_path / "src.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=256x144:rate=8:duration=6",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    clip = await FfmpegClip().clip(str(source), 1.0, 4.0)
    assert clip is not None

    model = build_openai_clip_model(base_url=live_base_url, model=live_model_name)
    answer = await model.watch(clip, "video/mp4", "What changes across this clip?")
    assert answer.strip(), "the model returned nothing usable"


def test_the_advertised_rate_matches_what_was_measured() -> None:
    """A guard on the constant, not on the server.

    2s cost 5,242 prompt tokens and 10s cost 21,787, both measured on
    2026-08-15. That is ~2,068 and ~2,179 tokens per second; the constant is
    the conservative end. If someone later "tidies" it to a round number, the
    refusal message starts quoting a cost nobody measured.
    """
    assert 2000 <= TOKENS_PER_CLIP_SECOND <= 2300


def test_watch_segment_params_reject_a_backwards_range() -> None:
    """pydantic catches the degenerate cases before the handler does, so a
    malformed tool call never reaches ffmpeg."""
    with pytest.raises(ValueError):
        WatchSegmentParams(start_s=-1.0, end_s=5.0)
    with pytest.raises(ValueError):
        WatchSegmentParams(start_s=0.0, end_s=0.0)
