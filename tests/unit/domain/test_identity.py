import pytest

from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef


def test_mimetype_parses_type_and_subtype() -> None:
    mime = MimeType.parse("video/mp4")
    assert mime.type == "video"
    assert mime.subtype == "mp4"
    assert str(mime) == "video/mp4"


def test_mimetype_extracts_structured_suffix() -> None:
    assert MimeType.parse("application/epub+zip").suffix == "zip"
    assert MimeType.parse("video/mp4").suffix is None


def test_mimetype_lowercases_and_drops_parameters() -> None:
    mime = MimeType.parse("TEXT/Plain; charset=utf-8")
    assert str(mime) == "text/plain"


def test_mimetype_rejects_a_string_without_a_slash() -> None:
    with pytest.raises(ValueError, match="not a mimetype"):
        MimeType.parse("video")


def test_media_kind_is_derived_from_the_type() -> None:
    assert MediaKind.for_mime(MimeType.parse("video/mp4")) is MediaKind.VIDEO
    assert MediaKind.for_mime(MimeType.parse("audio/flac")) is MediaKind.AUDIO
    assert MediaKind.for_mime(MimeType.parse("image/png")) is MediaKind.IMAGE
    assert MediaKind.for_mime(MimeType.parse("text/plain")) is MediaKind.TEXT
    assert MediaKind.for_mime(MimeType.parse("application/pdf")) is MediaKind.BINARY


def test_source_ref_rejects_a_negative_size() -> None:
    with pytest.raises(ValueError, match="size_bytes"):
        SourceRef(
            uri="/a.txt",
            mime=MimeType.parse("text/plain"),
            content_hash=ContentHash("abc"),
            size_bytes=-1,
        )
