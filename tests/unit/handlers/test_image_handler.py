import io

from PIL import Image

from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
from readeverything.handlers.image import ImageHandler
from readeverything.testing.fakes import FakeSource


def _png(width: int = 8, height: int = 4, colour: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


PNG = _png()


def _ref(size: int = len(PNG)) -> SourceRef:
    return SourceRef(
        uri="a.png",
        mime=MimeType.parse("image/png"),
        content_hash=ContentHash("e" * 64),
        size_bytes=size,
    )


def _handler() -> ImageHandler:
    return ImageHandler(source=FakeSource({"a.png": PNG, "somewhere/else": PNG}))


async def test_the_card_reports_dimensions_and_format() -> None:
    card = await _handler().describe(_ref())
    assert card.kind is MediaKind.IMAGE
    assert card.facts["width"] == 8
    assert card.facts["height"] == 4
    assert card.facts["format"] == "PNG"
    assert card.facts["mode"] == "RGB"


async def test_the_card_has_no_excerpt() -> None:
    """An image has no cheap textual excerpt; describing it costs a model call."""
    assert (await _handler().describe(_ref())).excerpt is None


async def test_the_card_outlines_the_whole_image() -> None:
    card = await _handler().describe(_ref())
    assert len(card.outline) == 1
    assert card.outline[0].label == "whole image"


async def test_a_handler_without_vision_requires_nothing() -> None:
    """The handler stays usable for metadata even with no model configured."""
    assert _handler().requires() == frozenset()


async def test_an_undecodable_image_still_produces_a_card() -> None:
    """There is no unsupported-file path; a corrupt image is a thin card."""
    handler = ImageHandler(source=FakeSource({"a.png": b"not an image at all"}))
    card = await handler.describe(_ref(size=19))
    assert card.kind is MediaKind.IMAGE
    assert card.facts["decodable"] is False
    assert card.outline == ()


async def test_a_decodable_image_says_so() -> None:
    card = await _handler().describe(_ref())
    assert card.facts["decodable"] is True
