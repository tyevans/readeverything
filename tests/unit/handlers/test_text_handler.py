import pytest

from readeverything.domain.errors import DomainError, UnknownAffordanceError
from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
from readeverything.domain.locators import CharSpan
from readeverything.domain.rendition import Budget, TextContent
from readeverything.handlers.text import ReadRangeParams, TextHandler
from readeverything.testing.fakes import FakeSource
from readeverything.testing.handler_compliance import MediaHandlerCompliance

CONTENT = b"alpha\nbeta\ngamma\n"


def _ref(*, uri: str = "a.txt", size_bytes: int = len(CONTENT)) -> SourceRef:
    return SourceRef(
        uri=uri,
        mime=MimeType.parse("text/plain"),
        content_hash=ContentHash("a" * 64),
        size_bytes=size_bytes,
    )


def _handler() -> TextHandler:
    return TextHandler(source=FakeSource({"a.txt": CONTENT, "somewhere/else": CONTENT}))


async def test_the_card_reports_line_and_character_counts() -> None:
    card = await _handler().describe(_ref())
    assert card.kind is MediaKind.TEXT
    assert card.facts["lines"] == 3
    assert card.facts["characters"] == len(CONTENT.decode())
    assert card.facts["encoding"] == "utf-8"


async def test_the_card_excerpt_is_bounded() -> None:
    long = ("x" * 5000).encode()
    handler = TextHandler(source=FakeSource({"a.txt": long}))
    ref = SourceRef(
        uri="a.txt",
        mime=MimeType.parse("text/plain"),
        content_hash=ContentHash("b" * 64),
        size_bytes=len(long),
    )
    card = await handler.describe(ref)
    assert card.excerpt is not None
    assert len(card.excerpt) <= 1000


async def test_read_range_returns_the_requested_characters_and_its_locator() -> None:
    rendition = await _handler().invoke(_ref(), "read_range", ReadRangeParams(start=6, end=10))
    assert isinstance(rendition.content, TextContent)
    assert rendition.content.text == "beta"
    assert isinstance(rendition.locator, CharSpan)
    assert rendition.locator.start == 6
    assert rendition.locator.end == 10


async def test_read_range_clamps_to_the_end_of_the_text() -> None:
    rendition = await _handler().invoke(_ref(), "read_range", ReadRangeParams(start=12, end=9999))
    assert isinstance(rendition.content, TextContent)
    assert rendition.content.text == "amma\n"


async def test_read_range_on_an_empty_file_raises_a_domain_error() -> None:
    """An empty file has no character range; say so rather than invent one."""
    handler = TextHandler(source=FakeSource({"empty.txt": b""}))
    ref = SourceRef(
        uri="empty.txt",
        mime=MimeType.parse("text/plain"),
        content_hash=ContentHash("d" * 64),
        size_bytes=0,
    )
    with pytest.raises(DomainError, match="empty"):
        await handler.invoke(ref, "read_range", ReadRangeParams(start=0, end=5))


async def test_an_unknown_affordance_raises() -> None:
    with pytest.raises(UnknownAffordanceError, match="read_range"):
        await _handler().invoke(_ref(), "nope", ReadRangeParams(start=0, end=1))


async def test_represent_maps_the_whole_text_with_line_barriers() -> None:
    rendered = await _handler().represent(_ref(), Budget(max_chars=None))
    assert rendered.text == CONTENT.decode()
    assert rendered.locator_map.length == len(rendered.text)
    assert rendered.barriers == ()


async def test_represent_truncates_and_says_so() -> None:
    rendered = await _handler().represent(_ref(), Budget(max_chars=5))
    assert len(rendered.text) == 5
    assert rendered.degradations
    assert "truncated" in rendered.degradations[0].what


def _empty_handler() -> TextHandler:
    return TextHandler(source=FakeSource({"empty.txt": b""}))


def _empty_ref() -> SourceRef:
    return SourceRef(
        uri="empty.txt",
        mime=MimeType.parse("text/plain"),
        content_hash=ContentHash("d" * 64),
        size_bytes=0,
    )


@pytest.mark.parametrize("max_chars", [0, 1, 5, 10])
async def test_the_truncation_degradation_reports_the_characters_actually_kept(
    max_chars: int,
) -> None:
    """The rendition and its own degradation must not contradict each other.

    A zero-width rendition is inexpressible — `CharSpan(0, 0)` raises — so a
    budget of zero still keeps one character. The degradation used to report the
    budget instead, claiming "kept 0" of text that was not zero characters long.
    """
    rendered = await _handler().represent(_ref(), Budget(max_chars=max_chars))
    assert rendered.degradations[0].detail.startswith(f"kept {len(rendered.text)} of ")


async def test_a_truncated_file_is_never_indexed_as_an_empty_one() -> None:
    """The placeholder asserted the source was empty when it merely did not fit.

    The empty-source fallback fired whenever the text was empty AFTER
    truncation, so a 16-character file under a zero budget was indexed as
    `[empty text file: a.txt]` — longer than the budget, and false about the
    source. Anything reading that index learned something untrue.
    """
    rendered = await _handler().represent(_ref(), Budget(max_chars=0))
    assert "empty text file" not in rendered.text
    assert rendered.text == CONTENT.decode()[:1]


async def test_a_genuinely_empty_file_keeps_its_placeholder() -> None:
    """Guard against fixing the false-empty claim by deleting the true one."""
    rendered = await _empty_handler().represent(_empty_ref(), Budget(max_chars=None))
    assert rendered.text == "[empty text file: empty.txt]"
    assert any(d.what == "synthesized description" for d in rendered.degradations)


async def test_the_empty_file_placeholder_announces_that_it_is_synthesized() -> None:
    """`[empty text file: x]` is 24 characters mapped over a file of zero.

    The placeholder is correct behaviour and must stay. What must not stay is
    an indexer being unable to tell that those characters are not in the file.
    """
    handler = TextHandler(source=FakeSource({"empty.txt": b""}))
    rendered = await handler.represent(_ref(uri="empty.txt", size_bytes=0), Budget(max_chars=None))
    assert rendered.text.startswith("[empty text file:")
    assert any(d.what == "synthesized description" for d in rendered.degradations)


async def test_extracted_text_is_not_announced_as_synthesized() -> None:
    """The marker must distinguish. A marker on everything distinguishes nothing."""
    handler = TextHandler(source=FakeSource({"real.txt": b"actual file content"}))
    rendered = await handler.represent(_ref(uri="real.txt", size_bytes=19), Budget(max_chars=None))
    assert not any(d.what == "synthesized description" for d in rendered.degradations)


class TestTextHandlerCompliance(MediaHandlerCompliance):
    @pytest.fixture
    def handler(self) -> TextHandler:
        return _handler()

    @pytest.fixture
    def content(self) -> bytes:
        return CONTENT

    @pytest.fixture
    def ref(self, content: bytes) -> SourceRef:
        return _ref()
