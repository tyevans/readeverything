import pytest

from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
from readeverything.domain.rendition import Budget, TextContent
from readeverything.handlers.binary import BinaryHandler, HexdumpParams
from readeverything.testing.fakes import FakeSource
from readeverything.testing.handler_compliance import MediaHandlerCompliance

CONTENT = bytes(range(64))


def _ref() -> SourceRef:
    return SourceRef(
        uri="blob.bin",
        mime=MimeType.parse("application/octet-stream"),
        content_hash=ContentHash("c" * 64),
        size_bytes=len(CONTENT),
    )


def _handler() -> BinaryHandler:
    return BinaryHandler(source=FakeSource({"blob.bin": CONTENT, "somewhere/else": CONTENT}))


async def test_the_fallback_always_produces_a_card() -> None:
    """There is no unsupported-file error path; the worst case is a thin card."""
    card = await _handler().describe(_ref())
    assert card.kind is MediaKind.BINARY
    assert card.facts["size_bytes"] == 64
    assert card.excerpt is not None


async def test_hexdump_returns_the_requested_window() -> None:
    rendition = await _handler().invoke(_ref(), "hexdump", HexdumpParams(start=0, length=4))
    assert isinstance(rendition.content, TextContent)
    assert rendition.content.text.startswith("00000000")
    assert "00 01 02 03" in rendition.content.text


async def test_represent_describes_rather_than_dumping() -> None:
    """Feeding a hexdump to an extractor produces noise, not claims."""
    rendered = await _handler().represent(_ref(), Budget(max_chars=None))
    assert "application/octet-stream" in rendered.text
    assert rendered.locator_map.length == len(rendered.text)


@pytest.mark.parametrize("max_chars", [0, 1, 5, 10])
async def test_the_truncation_degradation_reports_the_characters_actually_kept(
    max_chars: int,
) -> None:
    """The rendition and its own degradation must not contradict each other.

    The third handler with this shape. A zero-width rendition is inexpressible
    — `CharSpan(0, 0)` raises — so a budget of zero still keeps one character.
    The degradation reported the budget instead, claiming "kept 0" of text that
    was one character long. The spec recorded this case as correct by trace;
    the trace was right that a character comes back and wrong that the count
    announcing it was true.
    """
    rendered = await _handler().represent(_ref(), Budget(max_chars=max_chars))
    assert rendered.degradations[0].detail.startswith(f"kept {len(rendered.text)} of ")


class TestBinaryHandlerCompliance(MediaHandlerCompliance):
    @pytest.fixture
    def handler(self) -> BinaryHandler:
        return _handler()

    @pytest.fixture
    def content(self) -> bytes:
        return CONTENT

    @pytest.fixture
    def ref(self, content: bytes) -> SourceRef:
        return _ref()
