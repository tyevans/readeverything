import pytest

from readeverything.domain.errors import DomainError, UnknownAffordanceError
from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
from readeverything.domain.locators import PartSpan
from readeverything.domain.observation import OperationFinished, OperationStarted
from readeverything.domain.rendition import Budget, TextContent
from readeverything.handlers.epub import EpubHandler, ReadChapterParams, ReadRangeParams
from readeverything.testing.fakes import FakeSource, RecordingObserver
from readeverything.testing.handler_compliance import MediaHandlerCompliance
from tests.fixtures_epub import build_epub

BOOK = build_epub()


def _ref(*, uri: str = "book.epub", size_bytes: int = len(BOOK)) -> SourceRef:
    return SourceRef(
        uri=uri,
        mime=MimeType.parse("application/epub+zip"),
        content_hash=ContentHash("e" * 64),
        size_bytes=size_bytes,
    )


def _handler(*, data: bytes = BOOK, observer: object | None = None) -> EpubHandler:
    return EpubHandler(
        source=FakeSource({"book.epub": data, "somewhere/else": data, "compliance-subject": data}),
        observer=observer,  # type: ignore[arg-type]  # structural stub in tests
    )


class TestCompliance(MediaHandlerCompliance):
    @pytest.fixture
    def handler(self) -> EpubHandler:
        return _handler()

    @pytest.fixture
    def content(self) -> bytes:
        return BOOK


async def test_the_card_reports_the_title_and_author() -> None:
    card = await _handler().describe(_ref())
    assert card.facts["title"] == "A Short Book"
    assert card.facts["author"] == "A. Writer"


async def test_the_card_counts_chapters_and_words() -> None:
    card = await _handler().describe(_ref())
    assert card.facts["chapter_count"] == 2
    assert card.facts["word_count"] == 13


async def test_the_card_kind_is_binary() -> None:
    """An epub is a zip. Its text is reachable, but its bytes are not text."""
    card = await _handler().describe(_ref())
    assert card.kind is MediaKind.BINARY


async def test_the_outline_is_one_segment_per_chapter_in_reading_order() -> None:
    card = await _handler().describe(_ref())
    assert [segment.label for segment in card.outline] == ["Chapter One", "Chapter Two"]


async def test_read_chapter_returns_that_chapter_and_no_other() -> None:
    rendition = await _handler().invoke(_ref(), "read_chapter", ReadChapterParams(index=0))
    assert isinstance(rendition.content, TextContent)
    assert "It began quietly." in rendition.content.text
    assert "It ended." not in rendition.content.text


async def test_a_chapter_cites_the_part_it_came_from() -> None:
    """The citation promise, in a container: which file, and where in it."""
    rendition = await _handler().invoke(_ref(), "read_chapter", ReadChapterParams(index=1))
    assert isinstance(rendition.locator, PartSpan)
    assert rendition.locator.part == "OEBPS/ch1.xhtml"


async def test_reading_past_the_last_chapter_is_an_error() -> None:
    with pytest.raises(DomainError, match="chapter"):
        await _handler().invoke(_ref(), "read_chapter", ReadChapterParams(index=99))


async def test_read_range_returns_prose_characters() -> None:
    rendition = await _handler().invoke(_ref(), "read_range", ReadRangeParams(start=0, end=10))
    assert isinstance(rendition.content, TextContent)
    assert "<" not in rendition.content.text
    assert len(rendition.content.text) == 10


async def test_an_unknown_affordance_is_rejected() -> None:
    with pytest.raises(UnknownAffordanceError):
        await _handler().invoke(_ref(), "translate", ReadRangeParams())


async def test_represent_reads_the_whole_book_in_order() -> None:
    rendered = await _handler().represent(_ref(), Budget(max_chars=None))
    assert rendered.text.index("It began quietly.") < rendered.text.index("It ended.")
    assert "<p>" not in rendered.text


async def test_every_character_maps_back_to_a_part_a_reader_can_open() -> None:
    rendered = await _handler().represent(_ref(), Budget(max_chars=None))
    parts = {"OEBPS/ch0.xhtml", "OEBPS/ch1.xhtml"}
    for offset in range(len(rendered.text)):
        locator = rendered.locator_map.resolve(offset)
        assert isinstance(locator, PartSpan)
        assert locator.part in parts


async def test_a_chapter_boundary_is_a_barrier() -> None:
    """Chunking a book mid-chapter is worse than chunking it at one."""
    rendered = await _handler().represent(_ref(), Budget(max_chars=None))
    assert rendered.barriers


async def test_represent_honours_a_character_budget() -> None:
    rendered = await _handler().represent(_ref(), Budget(max_chars=20))
    assert len(rendered.text) == 20
    assert any(d.what == "text truncated" for d in rendered.degradations)


async def test_a_book_with_no_readable_text_still_produces_a_located_rendition() -> None:
    empty = build_epub(chapters=[("Blank", ())])
    handler = _handler(data=empty)
    rendered = await handler.represent(_ref(size_bytes=len(empty)), Budget(max_chars=None))
    assert rendered.text


async def test_a_drm_protected_book_reports_why_rather_than_hex_dumping() -> None:
    handler = _handler(data=build_epub(encrypted=True))
    with pytest.raises(DomainError, match="encrypted"):
        await handler.describe(_ref())


async def test_represent_narrates_itself() -> None:
    observer = RecordingObserver()
    await _handler(observer=observer).represent(_ref(), Budget(max_chars=None))
    kinds = [type(event) for event in observer.events]
    assert OperationStarted in kinds
    assert OperationFinished in kinds
