import pytest

from readeverything.domain.errors import DomainError, UnknownAffordanceError
from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
from readeverything.domain.locators import CharSpan
from readeverything.domain.observation import OperationFinished, OperationStarted
from readeverything.domain.rendition import Budget, TextContent
from readeverything.handlers.html import (
    UNTITLED_SECTION,
    HtmlHandler,
    ReadRangeParams,
    ReadSectionParams,
)
from readeverything.testing.fakes import FakeSource, RecordingObserver
from readeverything.testing.handler_compliance import MediaHandlerCompliance

PAGE = b"""<html>
<head><title>The Report</title></head>
<body>
<p>Front matter, before any heading.</p>
<h1>Findings</h1>
<p>The first finding.</p>
<p>The second finding.</p>
<h1>Method</h1>
<p>We looked.</p>
<script>var tracking = 1;</script>
</body>
</html>
"""


def _ref(*, uri: str = "a.html", size_bytes: int = len(PAGE)) -> SourceRef:
    return SourceRef(
        uri=uri,
        mime=MimeType.parse("text/html"),
        content_hash=ContentHash("c" * 64),
        size_bytes=size_bytes,
    )


def _handler(*, observer: object | None = None) -> HtmlHandler:
    return HtmlHandler(
        source=FakeSource({"a.html": PAGE, "somewhere/else": PAGE, "compliance-subject": PAGE}),
        observer=observer,  # type: ignore[arg-type]  # structural stub in tests
    )


class TestCompliance(MediaHandlerCompliance):
    @pytest.fixture
    def handler(self) -> HtmlHandler:
        return _handler()

    @pytest.fixture
    def content(self) -> bytes:
        return PAGE


async def test_the_card_reports_the_documents_title() -> None:
    card = await _handler().describe(_ref())
    assert card.facts["title"] == "The Report"


async def test_the_card_counts_headings_and_words_rather_than_markup() -> None:
    card = await _handler().describe(_ref())
    assert card.facts["heading_count"] == 2
    # "var tracking = 1" is not prose and must not be counted.
    assert "tracking" not in str(card.facts)


async def test_the_outline_is_one_segment_per_heading() -> None:
    card = await _handler().describe(_ref())
    assert [segment.label for segment in card.outline] == [
        UNTITLED_SECTION,
        "Findings",
        "Method",
    ]


async def test_the_excerpt_is_prose_and_carries_no_tags() -> None:
    card = await _handler().describe(_ref())
    assert card.excerpt is not None
    assert "<p>" not in card.excerpt
    assert "Front matter" in card.excerpt


async def test_the_card_kind_is_text() -> None:
    card = await _handler().describe(_ref())
    assert card.kind is MediaKind.TEXT


async def test_read_section_returns_the_heading_and_the_text_under_it() -> None:
    rendition = await _handler().invoke(_ref(), "read_section", ReadSectionParams(index=1))
    assert isinstance(rendition.content, TextContent)
    assert "Findings" in rendition.content.text
    assert "The first finding." in rendition.content.text
    assert "We looked." not in rendition.content.text


async def test_read_section_cites_a_span_of_the_original_html() -> None:
    """The citation promise: the locator indexes the file, not the prose."""
    rendition = await _handler().invoke(_ref(), "read_section", ReadSectionParams(index=1))
    assert isinstance(rendition.locator, CharSpan)
    source = PAGE.decode()[rendition.locator.start : rendition.locator.end]
    assert "The first finding." in source


async def test_reading_past_the_last_section_is_an_error() -> None:
    with pytest.raises(DomainError):
        await _handler().invoke(_ref(), "read_section", ReadSectionParams(index=99))


async def test_read_range_returns_prose_characters() -> None:
    rendition = await _handler().invoke(_ref(), "read_range", ReadRangeParams(start=0, end=12))
    assert isinstance(rendition.content, TextContent)
    assert "<" not in rendition.content.text
    assert len(rendition.content.text) == 12


async def test_an_unknown_affordance_is_rejected() -> None:
    with pytest.raises(UnknownAffordanceError):
        await _handler().invoke(_ref(), "translate", ReadRangeParams())


async def test_represent_maps_every_character_back_to_the_source_html() -> None:
    """`LocatorMap` is total, and every locator points into the file itself."""
    rendered = await _handler().represent(_ref(), Budget(max_chars=None))
    source = PAGE.decode()
    for offset in range(len(rendered.text)):
        locator = rendered.locator_map.resolve(offset)
        assert isinstance(locator, CharSpan)
        assert 0 <= locator.start < locator.end <= len(source)


async def test_represent_returns_prose_rather_than_markup() -> None:
    rendered = await _handler().represent(_ref(), Budget(max_chars=None))
    assert "<p>" not in rendered.text
    assert "var tracking" not in rendered.text
    assert "The first finding." in rendered.text


async def test_represent_honours_a_character_budget() -> None:
    rendered = await _handler().represent(_ref(), Budget(max_chars=20))
    assert len(rendered.text) == 20
    assert any(d.what == "text truncated" for d in rendered.degradations)


async def test_a_page_with_no_text_still_produces_a_located_rendition() -> None:
    empty = b"<html><head><title>Nothing</title></head><body></body></html>"
    handler = HtmlHandler(source=FakeSource({"a.html": empty}))
    rendered = await handler.represent(_ref(size_bytes=len(empty)), Budget(max_chars=None))
    assert rendered.text
    assert rendered.degradations


async def test_represent_narrates_itself() -> None:
    observer = RecordingObserver()
    await _handler(observer=observer).represent(_ref(), Budget(max_chars=None))
    kinds = [type(event) for event in observer.events]
    assert OperationStarted in kinds
    assert OperationFinished in kinds
