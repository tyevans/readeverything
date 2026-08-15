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
