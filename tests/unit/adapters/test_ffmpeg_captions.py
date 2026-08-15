from __future__ import annotations

from pathlib import Path

import pytest

from readeverything.adapters.ffmpeg_captions import FfmpegCaptions, parse_srt
from readeverything.domain.rendition import CueSource
from readeverything.ports.captions import CaptionExtractor

#: Two cues copied verbatim from `media/mystery_subject.mp4`'s mov_text track,
#: markup included. A hand-written sample would have omitted the `<font>` tags,
#: and the tags are the thing most likely to reach an index unnoticed.
REAL_SAMPLE = """1
00:00:00,868 --> 00:00:03,202
<font size="24">[music playing]</font>

2
00:00:16,484 --> 00:00:18,617
<font size="24">When I say computer,
you probably</font>
"""


def test_a_real_adapter_satisfies_the_port() -> None:
    assert isinstance(FfmpegCaptions(), CaptionExtractor)


def test_timings_become_spans() -> None:
    cues = parse_srt(REAL_SAMPLE)
    assert len(cues) == 2
    assert cues[0].span.start_s == pytest.approx(0.868)
    assert cues[0].span.end_s == pytest.approx(3.202)
    assert cues[1].span.start_s == pytest.approx(16.484)


def test_markup_is_stripped() -> None:
    """Every cue in the reference file is wrapped in `<font size="24">`.
    Indexed unstripped, that tag becomes content, and a citation returns it."""
    cues = parse_srt(REAL_SAMPLE)
    assert cues[0].text == "[music playing]"
    assert "<font" not in cues[1].text


def test_a_wrapped_cue_becomes_one_line() -> None:
    """SRT wraps a sentence across lines for display width. That is a
    rendering decision about a player's screen, not a sentence boundary."""
    assert parse_srt(REAL_SAMPLE)[1].text == "When I say computer, you probably"


def test_every_cue_says_it_was_captioned() -> None:
    assert all(cue.source is CueSource.CAPTIONED for cue in parse_srt(REAL_SAMPLE))


def test_a_zero_width_cue_is_dropped() -> None:
    """`TimeSpan` forbids `start >= end`. A malformed block must be dropped
    here rather than raised out of an adapter that promised never to."""
    assert parse_srt("1\n00:00:05,000 --> 00:00:05,000\nzero width\n") == ()


def test_one_malformed_block_does_not_cost_the_others() -> None:
    """A caption track is a best-effort artifact of whoever authored the disc.
    Refusing the whole file over one bad block would throw away 847 good
    cues to punish one."""
    cues = parse_srt("1\nnot a timing line\nbody\n\n2\n00:00:01,000 --> 00:00:02,000\nkept\n")
    assert [c.text for c in cues] == ["kept"]


def test_a_cue_with_no_words_is_dropped() -> None:
    """A block whose body is only markup would otherwise contribute an empty
    cue, and an empty cue in a timeline is a moment that says nothing while
    claiming a span."""
    assert parse_srt('1\n00:00:01,000 --> 00:00:02,000\n<font size="24"></font>\n') == ()


def test_empty_input_is_no_cues_rather_than_an_error() -> None:
    assert parse_srt("") == ()
    assert parse_srt("   \n\n  ") == ()


def test_a_dot_separated_timestamp_parses() -> None:
    """WebVTT uses a dot where SRT uses a comma, and ffmpeg's `srt` muxer is
    not the only thing that ever reaches this parser."""
    cues = parse_srt("1\n00:00:01.500 --> 00:00:02.500\nhi\n")
    assert cues[0].span.start_s == pytest.approx(1.5)


async def test_a_file_with_no_caption_track_is_none(sample_video: str) -> None:
    """`None` means "nothing to read" — a normal answer, the same convention
    as `FrameExtractor.frame_at` and `AudioExtractor.extract`."""
    assert await FfmpegCaptions().extract(sample_video) is None


async def test_a_missing_file_is_none_not_an_exception(tmp_path: Path) -> None:
    assert await FfmpegCaptions().extract(str(tmp_path / "nope.mp4")) is None


async def test_a_file_that_is_not_media_is_none(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    assert await FfmpegCaptions().extract(str(junk)) is None
