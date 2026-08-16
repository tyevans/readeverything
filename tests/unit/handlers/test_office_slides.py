"""The slides handler: the most natural barrier in any format the library reads."""

from __future__ import annotations

import pytest

from readeverything.adapters.ooxml import SLIDES_MIME
from readeverything.domain.capability import Capability
from readeverything.domain.identity import ContentHash, MimeType, SourceRef
from readeverything.domain.locators import ByteRange, PageRef
from readeverything.domain.rendition import Budget, TextContent
from readeverything.handlers.office_slides import (
    NOTES_HEADING,
    DescribeSlideImageParams,
    ListMediaParams,
    OfficeSlidesHandler,
    ReadSlideParams,
)
from readeverything.ports.vision import VisionModel
from readeverything.testing.fakes import FakeSource, FakeVision, FakeVisionRefusing
from readeverything.testing.handler_compliance import MediaHandlerCompliance
from tests.fixtures_office import odp_bytes, pptx_bytes

URI = "deck.pptx"


def _handler(content: bytes, *, vision: VisionModel | None = None) -> OfficeSlidesHandler:
    return OfficeSlidesHandler(
        source=FakeSource({URI: content, "somewhere/else": content}), vision=vision
    )


def _ref(content: bytes) -> SourceRef:
    return SourceRef(
        uri=URI,
        mime=MimeType.parse(SLIDES_MIME),
        content_hash=ContentHash("0" * 64),
        size_bytes=len(content),
    )


def _text(rendition: object) -> str:
    content = rendition.content  # type: ignore[attr-defined]
    assert isinstance(content, TextContent)
    return content.text


class TestSlidesCompliance(MediaHandlerCompliance):
    @pytest.fixture
    def content(self) -> bytes:
        return pptx_bytes(picture_on=(2,))

    @pytest.fixture
    def handler(self, content: bytes) -> OfficeSlidesHandler:
        return OfficeSlidesHandler(
            source=FakeSource({"compliance-subject": content, "somewhere/else": content}),
            vision=FakeVision(),
        )


async def test_the_outline_is_one_segment_per_slide_labelled_by_title() -> None:
    content = pptx_bytes()
    card = await _handler(content).describe(_ref(content))
    assert [s.label for s in card.outline] == [
        "Opening position",
        "The numbers",
        "What we decided",
    ]
    assert [s.locator for s in card.outline] == [PageRef(1), PageRef(2), PageRef(3)]


async def test_the_card_reports_slide_count_notes_and_media() -> None:
    content = pptx_bytes(picture_on=(2,))
    card = await _handler(content).describe(_ref(content))
    assert card.facts["slide_count"] == 3
    assert card.facts["notes_present"] == "yes"
    assert card.facts["media_count"] == 1


async def test_a_deck_with_no_notes_says_no_rather_than_omitting_the_fact() -> None:
    """A missing fact and a "no" read the same to a model that only sees the
    card. Only one of them is a claim the handler actually checked.
    """
    content = pptx_bytes(notes=(None, None, None))
    card = await _handler(content).describe(_ref(content))
    assert card.facts["notes_present"] == "no"


async def test_describing_a_deck_does_not_create_notes_slides() -> None:
    """`slide.notes_slide` CREATES one as a side effect. Touching it unguarded
    makes the notes fact depend on whether anything looked at the deck first —
    a bug that only ever shows on the SECOND call.
    """
    content = pptx_bytes(notes=(None, None, None))
    handler = _handler(content)
    await handler.describe(_ref(content))
    card = await handler.describe(_ref(content))
    assert card.facts["notes_present"] == "no"


async def test_every_character_resolves_to_the_slide_it_came_from() -> None:
    content = pptx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    for number, word in ((1, "Opening position"), (2, "The numbers"), (3, "What we decided")):
        offset = rendered.text.index(word)
        assert rendered.locator_map.resolve(offset) == PageRef(number)
        assert rendered.locator_map.resolve(offset + len(word) - 1) == PageRef(number)


async def test_there_is_one_barrier_per_slide_break() -> None:
    content = pptx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert len(rendered.barriers) == 2
    for barrier in rendered.barriers:
        assert rendered.locator_map.resolve(barrier) != rendered.locator_map.resolve(barrier - 1)


async def test_speaker_notes_are_included_and_labelled() -> None:
    """They routinely hold the reasoning the slide only asserts. Labelling is
    what stops a model attributing a presenter's aside to the slide itself.
    """
    content = pptx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert "The number is soft" in rendered.text
    marker = rendered.text.index(NOTES_HEADING)
    assert rendered.text.index("The number is soft") > marker


async def test_a_note_resolves_to_the_slide_that_carries_it() -> None:
    """A note attributed to the wrong slide is worse than a missing note."""
    content = pptx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    offset = rendered.text.index("The number is soft")
    assert rendered.locator_map.resolve(offset) == PageRef(2)


async def test_read_slide_returns_that_slide_including_its_notes() -> None:
    content = pptx_bytes()
    rendition = await _handler(content).invoke(_ref(content), "read_slide", ReadSlideParams(page=2))
    assert rendition.locator == PageRef(2)
    assert "The numbers" in _text(rendition)
    assert "The number is soft" in _text(rendition)


async def test_asking_for_a_slide_past_the_end_degrades_rather_than_raising() -> None:
    content = pptx_bytes()
    rendition = await _handler(content).invoke(
        _ref(content), "read_slide", ReadSlideParams(page=99)
    )
    assert rendition.degraded
    assert isinstance(rendition.locator, ByteRange)


async def test_list_media_reports_each_embedded_image_with_its_slide() -> None:
    content = pptx_bytes(picture_on=(2,))
    rendition = await _handler(content).invoke(_ref(content), "list_media", ListMediaParams())
    assert "image/png" in _text(rendition)
    assert "slide 2" in _text(rendition)


async def test_a_deck_with_no_media_says_so_rather_than_returning_nothing() -> None:
    content = pptx_bytes()
    rendition = await _handler(content).invoke(_ref(content), "list_media", ListMediaParams())
    assert _text(rendition).strip()


async def test_describe_slide_image_is_absent_without_a_vision_model() -> None:
    """Negotiation, not a runtime apology: the affordance must not appear."""
    names = {a.name for a in _handler(pptx_bytes()).affordances()}
    assert "describe_slide_image" not in names


async def test_describe_slide_image_appears_with_a_vision_model() -> None:
    handler = _handler(pptx_bytes(), vision=FakeVision())
    affordance = next(a for a in handler.affordances() if a.name == "describe_slide_image")
    assert affordance.requires == frozenset({Capability.VISION})


async def test_describe_slide_image_reaches_the_embedded_picture() -> None:
    content = pptx_bytes(picture_on=(2,))
    rendition = await _handler(content, vision=FakeVision()).invoke(
        _ref(content),
        "describe_slide_image",
        DescribeSlideImageParams(page=2, index=0, question="What is shown?"),
    )
    assert rendition.locator == PageRef(2)
    assert _text(rendition)
    assert not rendition.degraded


async def test_asking_about_an_image_that_is_not_there_degrades() -> None:
    content = pptx_bytes()
    rendition = await _handler(content, vision=FakeVision()).invoke(
        _ref(content),
        "describe_slide_image",
        DescribeSlideImageParams(page=1, index=0, question="What is shown?"),
    )
    assert rendition.degraded


async def test_a_vision_model_that_fails_degrades_rather_than_raising() -> None:
    content = pptx_bytes(picture_on=(2,))
    rendition = await _handler(content, vision=FakeVisionRefusing()).invoke(
        _ref(content),
        "describe_slide_image",
        DescribeSlideImageParams(page=2, index=0, question="What is shown?"),
    )
    assert rendition.degraded


async def test_describing_an_image_does_not_read_bytes_for_the_card() -> None:
    """The card path must not touch a vision model at all."""
    vision = FakeVision()
    content = pptx_bytes(picture_on=(2,))
    await _handler(content, vision=vision).describe(_ref(content))
    assert vision.calls == 0


async def test_an_odp_reads_through_the_same_handler() -> None:
    content = odp_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert "Opening position" in rendered.text
    assert len(rendered.barriers) == 1
    card = await _handler(content).describe(_ref(content))
    assert [s.label for s in card.outline] == ["Opening position", "What we decided"]


async def test_an_unreadable_deck_degrades_rather_than_raising() -> None:
    content = b"not a presentation"
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert rendered.degradations
    assert rendered.text
    assert isinstance(rendered.locator_map.resolve(0), ByteRange)


async def test_a_slide_with_no_text_still_owns_a_character() -> None:
    """`CharSpan` rejects a zero-width span, so an empty slide between two full
    ones is what breaks the map.
    """
    content = pptx_bytes(titles=("Alpha", "", "Charlie"), body="", notes=(None, None, None))
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert rendered.locator_map.length == len(rendered.text)
    assert len(rendered.locator_map.segments) == 3


async def test_a_budget_truncates_and_says_so() -> None:
    content = pptx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=20))
    assert len(rendered.text) <= 20
    assert any("truncated" in d.what for d in rendered.degradations)


async def test_reading_a_deck_needs_no_capability() -> None:
    assert _handler(pptx_bytes()).requires() == frozenset()


async def test_an_unopenable_deck_is_never_reported_as_embedding_no_pictures() -> None:
    """ "Embeds no pictures" is a claim about a deck that was READ. Saying it
    about a file that could not be opened tells an agent to stop looking for
    something nothing ever checked.
    """
    content = b"not a presentation"
    rendition = await _handler(content).invoke(_ref(content), "list_media", ListMediaParams())
    assert rendition.degraded
    assert "no pictures" not in _text(rendition)


@pytest.mark.parametrize(
    ("affordance", "params"),
    [
        ("read_slide", ReadSlideParams()),
        ("list_media", ListMediaParams()),
        ("describe_slide_image", DescribeSlideImageParams(question="What is shown?")),
    ],
)
async def test_every_affordance_degrades_on_an_unreadable_deck(
    affordance: str, params: object
) -> None:
    content = b"not a presentation"
    rendition = await _handler(content, vision=FakeVision()).invoke(
        _ref(content),
        affordance,
        params,  # type: ignore[arg-type]
    )
    assert rendition.degraded


@pytest.mark.parametrize(
    ("affordance", "wrong"),
    [
        ("read_slide", ListMediaParams()),
        ("list_media", ReadSlideParams()),
        ("describe_slide_image", ReadSlideParams()),
    ],
)
async def test_the_wrong_params_model_is_refused_rather_than_coerced(
    affordance: str, wrong: object
) -> None:
    content = pptx_bytes()
    with pytest.raises(TypeError):
        await _handler(content, vision=FakeVision()).invoke(
            _ref(content),
            affordance,
            wrong,  # type: ignore[arg-type]
        )


async def test_asking_about_an_image_without_a_vision_model_raises_unknown_affordance() -> None:
    """The affordance is not declared, so invoking it by name is an unknown
    affordance rather than a degraded answer — negotiation, not an apology.
    """
    from readeverything.domain.errors import UnknownAffordanceError

    content = pptx_bytes(picture_on=(2,))
    with pytest.raises(UnknownAffordanceError):
        await _handler(content).invoke(
            _ref(content),
            "describe_slide_image",
            DescribeSlideImageParams(page=2, index=0, question="What is shown?"),
        )


async def test_a_deck_that_opens_with_no_slides_says_so_rather_than_claiming_it_is_unreadable() -> (
    None
):
    content = pptx_bytes(titles=(), notes=())
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert any("no slides" in d.what for d in rendered.degradations)
    assert "could not be opened" not in rendered.text


async def test_a_slide_holding_only_a_picture_is_not_reported_as_empty() -> None:
    """A slide with one diagram is not an empty slide, and saying nothing about
    it would report the deck as shorter than it is.
    """
    content = pptx_bytes(titles=("",), body="", notes=(None,), picture_on=(1,))
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert rendered.text.strip()
    assert "empty" not in rendered.text.lower()
