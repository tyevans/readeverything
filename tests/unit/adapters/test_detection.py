import pytest
from tests.fixtures_media import audio_only_m4a, ffmpeg_available, video_with_audio

from readeverything.adapters.detection import PuremagicDetector
from readeverything.domain.identity import MimeType

PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24


async def test_content_beats_a_lying_extension() -> None:
    """An extension is a claim; the bytes are a fact."""
    detected = await PuremagicDetector().detect("photo.txt", PNG_HEADER)
    assert detected.type == "image"


async def test_the_filename_is_used_when_content_is_inconclusive() -> None:
    detected = await PuremagicDetector().detect("notes.md", b"# heading\n")
    assert detected == MimeType.parse("text/markdown")


async def test_utf8_text_without_an_extension_is_plain_text() -> None:
    detected = await PuremagicDetector().detect("notes", b"just some words\n")
    assert detected == MimeType.parse("text/plain")


async def test_undecodable_bytes_without_a_signature_are_octet_stream() -> None:
    detected = await PuremagicDetector().detect("blob", b"\x00\x01\x02\xff\xfe")
    assert detected == MimeType.parse("application/octet-stream")


async def test_empty_content_is_octet_stream() -> None:
    assert await PuremagicDetector().detect("empty", b"") == MimeType.parse(
        "application/octet-stream"
    )


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
async def test_an_audio_only_m4a_is_detected_as_audio() -> None:
    """m4a and mp4 are the same container, byte-identical at the header, so
    magic reports video/mp4 for both. Only the extension distinguishes them, and
    an audio file that dispatches to the video handler never reaches the handler
    built to read it.
    """
    mime = await PuremagicDetector().detect("voice-memo.m4a", audio_only_m4a())
    assert str(mime).startswith("audio/")


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
async def test_an_mp4_video_is_still_detected_as_video() -> None:
    """The narrow rule must not swallow the common case: a real .mp4 with a
    video stream, whose extension guesses video/mp4 too, stays video."""
    mime = await PuremagicDetector().detect("clip.mp4", video_with_audio())
    assert str(mime) == "video/mp4"


async def test_a_png_named_txt_is_still_a_png() -> None:
    """The reasoning the module docstring already gives, guarded. Content beats
    filename in general; this fix carves out one container ambiguity, not a
    reordering."""
    detected = await PuremagicDetector().detect("photo.txt", PNG_HEADER)
    assert detected.type == "image"


async def test_a_detector_whose_library_raises_still_returns_a_mimetype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """puremagic raising must degrade to the fallback, not propagate.

    Detection sits in front of every read. A library that raises on a malformed
    header would take down every call for that file, so the except exists — it
    had simply never run.
    """
    import puremagic

    def _boom(*args: object, **kwargs: object) -> object:
        raise ValueError("malformed header")

    monkeypatch.setattr(puremagic, "magic_string", _boom)
    mime = await PuremagicDetector().detect("f.bin", b"\x00\x01")
    assert str(mime) == "application/octet-stream"


#: The first 32 bytes of a real DASH-derived mp4 — `media/myster_subject_3.mp4`,
#: a 44-minute video. Its `ftyp` brand is `iso5`, which puremagic has no
#: signature for, so the only thing it matches is noise.
ISO5_HEADER = b"\x00\x00\x00$ftypiso5\x00\x00\x00\x01avc1iso5dsmsmsixdash\x00\x00\x00:free"


async def test_a_low_confidence_signature_does_not_beat_the_filename() -> None:
    """The bug this test was written for: a 44-minute video was detected as
    `audio/x-sndr` and dispatched to the audio handler, which silently cost it
    every visual affordance — no frame extraction, no frame description, no way
    to see anything at all.

    puremagic returns "Macintosh SNDR Resource" at 0.20 confidence for EVERY
    mp4 in the corpus. It normally loses to a real `video/mp4` match at 0.80.
    On a brand puremagic has no signature for, it was the only match, and a
    0.20 guess was treated as an authoritative signature.

    A signature that weak is the bytes being SILENT, which is the case the
    filename exists to answer.
    """
    detected = await PuremagicDetector().detect("documentary.mp4", ISO5_HEADER)
    assert detected == MimeType.parse("video/mp4")


async def test_an_iso_bmff_container_is_recognised_by_its_ftyp_box() -> None:
    """Content, not filename, is what fixes this properly.

    Every ISO BMFF file carries a `ftyp` box at offset 4, whatever its brand.
    puremagic enumerates brands and therefore misses new ones; the box itself
    is the stable fact, so this does not depend on the extension being present
    or honest.
    """
    detected = await PuremagicDetector().detect("no-extension", ISO5_HEADER)
    assert detected == MimeType.parse("video/mp4")


async def test_an_unrecognised_brand_still_honours_the_audio_extension() -> None:
    """The `.m4a`/`.mp4` ambiguity is unchanged by the new rule: the container
    is one family and only the extension separates audio-only from video."""
    detected = await PuremagicDetector().detect("podcast.m4a", ISO5_HEADER)
    assert detected.type == "audio"


async def test_a_weak_signature_still_wins_over_nothing_at_all() -> None:
    """The floor must not throw away a weak match when there is no filename
    claim and no better evidence — that would trade one silent misdetection for
    another. A weak signature beats octet-stream; it just does not beat a
    filename."""
    sndr_only = b"\x00\x00\x00\x08sndr" + b"\xff" * 24
    detected = await PuremagicDetector().detect("mystery", sndr_only)
    assert detected != MimeType.parse("text/plain")
