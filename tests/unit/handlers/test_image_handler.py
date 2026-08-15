import io

import pytest
from PIL import Image
from pydantic import ValidationError

from readeverything.domain.capability import Capability
from readeverything.domain.errors import UnknownAffordanceError
from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
from readeverything.domain.locators import BBox
from readeverything.domain.rendition import Budget, ImageContent, TextContent
from readeverything.handlers.image import (
    CropParams,
    DescribeImageParams,
    ImageHandler,
    OcrParams,
)
from readeverything.testing.fakes import FakeSource, FakeVision, FakeVisionRefusing
from readeverything.testing.handler_compliance import MediaHandlerCompliance


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


def _seeing() -> ImageHandler:
    return ImageHandler(
        source=FakeSource({"a.png": PNG, "somewhere/else": PNG}), vision=FakeVision()
    )


def test_without_vision_only_crop_is_offered() -> None:
    """Cropping is pure Pillow; describing and OCR are not."""
    names = tuple(a.name for a in _handler().affordances())
    assert names == ("crop_region",)


def test_with_vision_all_three_are_offered() -> None:
    names = tuple(a.name for a in _seeing().affordances())
    assert set(names) == {"crop_region", "describe_image", "ocr"}


def test_the_model_backed_affordances_declare_the_vision_capability() -> None:
    """The registry filters on this; a wrong declaration makes negotiation a lie."""
    by_name = {a.name: a for a in _seeing().affordances()}
    assert by_name["describe_image"].requires == frozenset({Capability.VISION})
    assert by_name["ocr"].requires == frozenset({Capability.VISION})
    assert by_name["crop_region"].requires == frozenset()


async def test_describe_image_returns_text_located_at_the_whole_frame() -> None:
    rendition = await _seeing().invoke(_ref(), "describe_image", DescribeImageParams())
    assert isinstance(rendition.content, TextContent)
    assert rendition.locator == BBox(page=None, x=0.0, y=0.0, w=1.0, h=1.0)


async def test_describe_image_passes_the_prompt_through() -> None:
    rendition = await _seeing().invoke(
        _ref(), "describe_image", DescribeImageParams(prompt="count the squares")
    )
    assert isinstance(rendition.content, TextContent)
    assert "count the squares" in rendition.content.text


async def test_ocr_locates_its_text_at_the_whole_frame() -> None:
    rendition = await _seeing().invoke(_ref(), "ocr", OcrParams())
    assert isinstance(rendition.content, TextContent)
    assert rendition.locator == BBox(page=None, x=0.0, y=0.0, w=1.0, h=1.0)


async def test_crop_region_returns_image_bytes_located_at_the_crop() -> None:
    params = CropParams(x=0.0, y=0.0, w=0.5, h=1.0)
    rendition = await _seeing().invoke(_ref(), "crop_region", params)
    assert isinstance(rendition.content, ImageContent)
    assert rendition.content.mime == "image/png"
    assert rendition.locator == BBox(page=None, x=0.0, y=0.0, w=0.5, h=1.0)


async def test_a_crop_is_actually_cropped() -> None:
    """Verified by decoding the result, not by trusting the locator."""
    params = CropParams(x=0.0, y=0.0, w=0.5, h=1.0)
    rendition = await _seeing().invoke(_ref(), "crop_region", params)
    assert isinstance(rendition.content, ImageContent)
    cropped = Image.open(io.BytesIO(rendition.content.data))
    assert cropped.size == (4, 4)


async def test_a_crop_of_an_undecodable_image_raises_a_domain_error() -> None:
    from readeverything.domain.errors import DomainError

    handler = ImageHandler(source=FakeSource({"a.png": b"nonsense"}), vision=FakeVision())
    with pytest.raises(DomainError, match="not a readable image"):
        await handler.invoke(_ref(size=8), "crop_region", CropParams(x=0.0, y=0.0, w=1.0, h=1.0))


async def test_invoking_a_vision_affordance_without_vision_raises() -> None:
    """Handler-level guard; the registry normally prevents this from being reachable."""
    with pytest.raises(UnknownAffordanceError, match="describe_image"):
        await _handler().invoke(_ref(), "describe_image", DescribeImageParams())


async def test_represent_without_vision_states_the_facts_and_degrades() -> None:
    """A card's worth of truth, plus an honest note that description was unavailable."""
    rendered = await _handler().represent(_ref(), Budget(max_chars=None))
    assert "8x4" in rendered.text
    assert rendered.degradations
    assert "vision" in rendered.degradations[0].what


async def test_represent_with_vision_describes_the_image() -> None:
    rendered = await _seeing().represent(_ref(), Budget(max_chars=None))
    assert "image/png" in rendered.text
    assert rendered.degradations == ()
    assert rendered.locator_map.length == len(rendered.text)


async def test_represent_degrades_when_the_model_returns_nothing() -> None:
    """An empty completion must not enter an index as an observation."""
    handler = ImageHandler(source=FakeSource({"a.png": PNG}), vision=FakeVisionRefusing())
    rendered = await handler.represent(_ref(), Budget(max_chars=None))
    assert "8x4" in rendered.text
    assert rendered.degradations
    assert "vision" in rendered.degradations[0].what


async def test_represent_degrades_when_the_model_returns_an_empty_string() -> None:
    """A `str`-returning model that answers with nothing is legal and must degrade."""

    class _Silent:
        model_id: str = "silent@1"

        async def describe(self, data: bytes, mime: str, prompt: str) -> str:
            return "   "

    handler = ImageHandler(source=FakeSource({"a.png": PNG}), vision=_Silent())
    rendered = await handler.represent(_ref(), Budget(max_chars=None))
    assert "8x4" in rendered.text
    assert rendered.degradations
    assert "vision" in rendered.degradations[0].what


async def test_invoking_ocr_without_vision_names_ocr_and_not_describe_image() -> None:
    """The name was hardcoded in `_see`, so `ocr` reported `describe_image` unknown.

    Asserted on the attribute rather than the message: a caller needs the name
    as data to react to it, not only as prose it would have to parse.
    """
    with pytest.raises(UnknownAffordanceError) as caught:
        await _handler().invoke(_ref(), "ocr", OcrParams())
    assert caught.value.affordance == "ocr"


async def test_a_crop_may_not_run_off_the_right_edge_of_the_image() -> None:
    """Rejected at parameter parsing, where a parameter error belongs.

    `x + w > 1` used to construct fine and then surface a bare `ValueError`
    from `BBox.__post_init__` — not a `DomainError` — from deep inside the
    domain, long after the caller's mistake.
    """
    with pytest.raises(ValidationError):
        CropParams(x=0.5, w=0.8)


async def test_a_crop_may_not_run_off_the_bottom_edge_of_the_image() -> None:
    with pytest.raises(ValidationError):
        CropParams(y=0.5, h=0.8)


def test_the_whole_image_defaults_and_in_bounds_crops_still_construct() -> None:
    """`test_declared_affordances_are_invocable` requires no-argument params."""
    assert CropParams().w == 1.0
    assert CropParams(x=0.25, y=0.25, w=0.5, h=0.5).w == 0.5


async def test_an_undecodable_image_degrades_as_undecodable_not_as_missing_vision() -> None:
    """The image was the problem, and the degradation must say which thing failed.

    `vision unavailable` is load-bearing: it means no model configured or the
    model call failed. Reusing it for bytes Pillow cannot open sent operators
    looking at the wrong component.
    """
    vision = FakeVision()
    handler = ImageHandler(source=FakeSource({"a.png": b"not an image at all"}), vision=vision)
    rendered = await handler.represent(_ref(size=19), Budget(max_chars=None))
    assert [d.what for d in rendered.degradations] == ["image undecodable"]
    assert "a.png" in rendered.degradations[0].detail


async def test_an_undecodable_image_is_never_shown_to_the_vision_model() -> None:
    """Describing bytes that are not a decodable image indexes invention as fact."""
    vision = FakeVision()
    handler = ImageHandler(source=FakeSource({"a.png": b"not an image at all"}), vision=vision)
    await handler.represent(_ref(size=19), Budget(max_chars=None))
    assert vision.calls == 0


async def test_a_decodable_image_with_vision_still_reports_no_degradations() -> None:
    """Guard against closing the undecodable hole by degrading the healthy path too."""
    rendered = await _seeing().represent(_ref(), Budget(max_chars=None))
    assert rendered.degradations == ()


@pytest.mark.parametrize("max_chars", [0, 1, 5, 20])
async def test_the_truncation_degradation_reports_the_characters_actually_kept(
    max_chars: int,
) -> None:
    """The rendition and its own degradation must not contradict each other.

    A zero-width rendition is inexpressible — `CharSpan(0, 0)` raises — so a
    budget of zero still keeps one character. The degradation used to report the
    budget rather than that character, claiming "kept 0" of text one character
    long.
    """
    rendered = await _handler().represent(_ref(), Budget(max_chars=max_chars))
    truncations = [d for d in rendered.degradations if d.what == "text truncated"]
    assert truncations
    assert truncations[0].detail.startswith(f"kept {len(rendered.text)} of ")


class TestImageHandlerCompliance(MediaHandlerCompliance):
    @pytest.fixture
    def handler(self) -> ImageHandler:
        return _seeing()

    @pytest.fixture
    def content(self) -> bytes:
        return PNG

    @pytest.fixture
    def ref(self, content: bytes) -> SourceRef:
        return _ref()
