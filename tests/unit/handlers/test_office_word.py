"""The Word handler: a table of contents, and text that knows its section."""

from __future__ import annotations

import pytest

from readeverything.adapters.ooxml import WORD_MIME
from readeverything.domain.identity import ContentHash, MimeType, SourceRef
from readeverything.domain.locators import ByteRange, CharSpan
from readeverything.domain.rendition import Budget, TextContent
from readeverything.handlers.office_word import (
    ListCommentsParams,
    OfficeWordHandler,
    ReadRangeParams,
    ReadSectionParams,
    ReadTableParams,
)
from readeverything.testing.fakes import FakeSource
from readeverything.testing.handler_compliance import MediaHandlerCompliance
from tests.fixtures_office import docx_bytes, odt_bytes

URI = "policy.docx"


def _handler(content: bytes) -> OfficeWordHandler:
    return OfficeWordHandler(source=FakeSource({URI: content, "somewhere/else": content}))


def _ref(content: bytes) -> SourceRef:
    return SourceRef(
        uri=URI,
        mime=MimeType.parse(WORD_MIME),
        content_hash=ContentHash("0" * 64),
        size_bytes=len(content),
    )


def _text(rendition: object) -> str:
    content = rendition.content  # type: ignore[attr-defined]
    assert isinstance(content, TextContent)
    return content.text


class TestWordCompliance(MediaHandlerCompliance):
    @pytest.fixture
    def content(self) -> bytes:
        return docx_bytes(comment="Check this number.")

    @pytest.fixture
    def handler(self, content: bytes) -> OfficeWordHandler:
        return OfficeWordHandler(
            source=FakeSource({"compliance-subject": content, "somewhere/else": content})
        )


async def test_the_outline_is_a_table_of_contents() -> None:
    """What an agent needs to decide where to look, without reading the body."""
    content = docx_bytes()
    card = await _handler(content).describe(_ref(content))
    assert [segment.label for segment in card.outline] == ["Alpha", "Bravo", "Charlie"]


async def test_every_outline_segment_points_into_the_represented_text() -> None:
    """An outline whose locator does not address the text it summarises is a
    table of contents with the wrong page numbers.
    """
    content = docx_bytes()
    handler = _handler(content)
    card = await handler.describe(_ref(content))
    rendered = await handler.represent(_ref(content), Budget(max_chars=None))
    for segment in card.outline:
        assert isinstance(segment.locator, CharSpan)
        assert segment.label in rendered.text[segment.locator.start : segment.locator.end]


async def test_the_card_counts_paragraphs_headings_and_words() -> None:
    content = docx_bytes()
    card = await _handler(content).describe(_ref(content))
    assert card.facts["heading_count"] == 3
    assert card.facts["paragraph_count"] == 3
    words = card.facts["word_count"]
    assert isinstance(words, int)
    assert words > 0


async def test_tracked_changes_are_reported_as_a_fact_both_ways() -> None:
    """Both directions, because a fact that is always "no" tests nothing."""
    plain = docx_bytes()
    tracked = docx_bytes(tracked=True)
    assert (await _handler(plain).describe(_ref(plain))).facts["tracked_changes"] == "no"
    assert (await _handler(tracked).describe(_ref(tracked))).facts["tracked_changes"] == "yes"


async def test_the_card_counts_comments() -> None:
    content = docx_bytes(comment="Check this number.")
    card = await _handler(content).describe(_ref(content))
    assert card.facts["comment_count"] == 1


async def test_every_character_resolves_to_the_section_it_came_from() -> None:
    """The property the handler exists to provide, asserted across a boundary —
    an off-by-one in the segment starts passes any test that samples one
    section's middle.
    """
    content = docx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    first = rendered.text.index("alpha section")
    second = rendered.text.index("bravo section")
    assert rendered.locator_map.resolve(first) != rendered.locator_map.resolve(second)


async def test_barriers_sit_at_heading_boundaries() -> None:
    """One barrier per heading after the first: a chunker must not merge the
    end of one section with the start of the next.
    """
    content = docx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert len(rendered.barriers) == 2
    for barrier in rendered.barriers:
        assert rendered.locator_map.resolve(barrier) != rendered.locator_map.resolve(barrier - 1)


async def test_a_table_is_rendered_in_document_order_not_appended() -> None:
    """A table is frequently the answer, and where it sat is part of what it
    means. The fixture puts it after the first heading, so a handler that
    appends tables puts `north` after `charlie` and fails here.
    """
    content = docx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert rendered.text.index("north") < rendered.text.index("charlie section")


async def test_a_table_renders_as_pipe_delimited_rows() -> None:
    content = docx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert "region | total" in rendered.text


async def test_read_section_returns_that_section_located_at_its_span() -> None:
    content = docx_bytes()
    rendition = await _handler(content).invoke(
        _ref(content), "read_section", ReadSectionParams(index=1)
    )
    assert "Bravo" in _text(rendition)
    assert isinstance(rendition.locator, CharSpan)


async def test_asking_for_a_section_past_the_end_degrades_rather_than_raising() -> None:
    content = docx_bytes()
    rendition = await _handler(content).invoke(
        _ref(content), "read_section", ReadSectionParams(index=99)
    )
    assert rendition.degraded


async def test_read_range_returns_the_characters_it_was_asked_for() -> None:
    content = docx_bytes()
    handler = _handler(content)
    rendered = await handler.represent(_ref(content), Budget(max_chars=None))
    rendition = await handler.invoke(_ref(content), "read_range", ReadRangeParams(start=0, end=10))
    assert _text(rendition) == rendered.text[:10]
    assert rendition.locator == CharSpan(0, 10)


async def test_a_range_past_the_end_is_clamped_rather_than_raising() -> None:
    content = docx_bytes()
    rendition = await _handler(content).invoke(
        _ref(content), "read_range", ReadRangeParams(start=0, end=10_000_000)
    )
    assert _text(rendition)


async def test_list_comments_returns_the_comment_text_and_its_author() -> None:
    content = docx_bytes(comment="Check this number.")
    rendition = await _handler(content).invoke(_ref(content), "list_comments", ListCommentsParams())
    assert "Check this number." in _text(rendition)
    assert "Reviewer" in _text(rendition)


async def test_a_document_with_no_comments_says_so_rather_than_returning_nothing() -> None:
    """Empty output and "there are none" are different answers, and only one of
    them tells an agent to stop looking.
    """
    content = docx_bytes()
    rendition = await _handler(content).invoke(_ref(content), "list_comments", ListCommentsParams())
    assert _text(rendition).strip()


async def test_read_table_returns_one_table_by_index() -> None:
    content = docx_bytes()
    rendition = await _handler(content).invoke(
        _ref(content), "read_table", ReadTableParams(index=0)
    )
    assert "north" in _text(rendition)


async def test_asking_for_a_table_that_is_not_there_degrades() -> None:
    content = docx_bytes(table=None)
    rendition = await _handler(content).invoke(
        _ref(content), "read_table", ReadTableParams(index=0)
    )
    assert rendition.degraded


async def test_an_odt_reads_through_the_same_handler() -> None:
    content = odt_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert "alpha section" in rendered.text
    card = await _handler(content).describe(_ref(content))
    assert [segment.label for segment in card.outline] == ["Alpha", "Bravo"]


async def test_an_unreadable_document_degrades_rather_than_raising() -> None:
    content = b"this is not a word document at all"
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert rendered.degradations
    assert rendered.text
    assert isinstance(rendered.locator_map.resolve(0), ByteRange)


async def test_an_unreadable_document_still_produces_a_card() -> None:
    content = b"this is not a word document at all"
    card = await _handler(content).describe(_ref(content))
    assert card.facts["readable"] == "no"


async def test_a_document_with_no_headings_still_maps_every_character() -> None:
    """`LocatorMap` demands total, gapless, zero-start coverage. A document with
    no heading has no section to attribute text to, and a handler that emits no
    segment produces a `Rendered` that will not construct.
    """
    content = docx_bytes(headings=(), table=None, preamble="Just body text, no headings at all.")
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert rendered.locator_map.length == len(rendered.text)
    assert "Just body text" in rendered.text


async def test_text_before_the_first_heading_belongs_to_a_section_of_its_own() -> None:
    """A `LocatorMap` must start at offset 0, so there is no such thing as text
    belonging to no section.
    """
    content = docx_bytes(preamble="A preamble before any heading.")
    handler = _handler(content)
    rendered = await handler.represent(_ref(content), Budget(max_chars=None))
    offset = rendered.text.index("A preamble")
    assert rendered.locator_map.resolve(offset) != rendered.locator_map.resolve(
        rendered.text.index("alpha section")
    )


async def test_a_heading_with_no_body_still_owns_a_character() -> None:
    """`CharSpan` rejects a zero-width span, so an empty section between two
    full ones is what breaks the map.
    """
    content = docx_bytes(headings=((1, "Alpha", ""), (1, "Bravo", "")), table=None)
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert rendered.locator_map.length == len(rendered.text)
    assert len(rendered.locator_map.segments) == 2


async def test_a_budget_truncates_and_says_so() -> None:
    content = docx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=20))
    assert len(rendered.text) <= 20
    assert any("truncated" in d.what for d in rendered.degradations)


async def test_reading_a_word_document_needs_no_capability() -> None:
    assert _handler(docx_bytes()).requires() == frozenset()
