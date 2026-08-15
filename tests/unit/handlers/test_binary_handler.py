import pytest

from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
from readeverything.domain.observation import OperationFinished, OperationStarted
from readeverything.domain.rendition import Budget, TextContent
from readeverything.handlers.binary import BinaryHandler, HexdumpParams
from readeverything.testing.fakes import FakeSource, RaisingObserver, RecordingObserver
from readeverything.testing.handler_compliance import MediaHandlerCompliance

CONTENT = bytes(range(64))


def _ref(*, uri: str = "blob.bin", size_bytes: int = len(CONTENT)) -> SourceRef:
    return SourceRef(
        uri=uri,
        mime=MimeType.parse("application/octet-stream"),
        content_hash=ContentHash("c" * 64),
        size_bytes=size_bytes,
    )


def _handler(*, observer: object | None = None) -> BinaryHandler:
    return BinaryHandler(
        source=FakeSource({"blob.bin": CONTENT, "somewhere/else": CONTENT}),
        observer=observer,  # type: ignore[arg-type]  # structural stub in tests
    )


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


async def test_a_binary_representation_announces_that_it_is_synthesized() -> None:
    """The text describes the file; it was not extracted from it.

    Nothing in the bytes says "No textual content could be extracted" — the
    handler wrote that. A consumer indexing this must be able to tell it apart
    from text that actually came out of the file, because attributing it to the
    file's content is a false claim about the file.
    """
    rendered = await _handler().represent(_ref(), Budget(max_chars=None))
    assert any(d.what == "synthesized description" for d in rendered.degradations)


async def test_an_empty_binary_file_does_not_claim_a_byte_that_is_not_there() -> None:
    """`ByteRange(0, max(1, size))` invents byte 0 for a 0-byte file.

    The `max(1, ...)` exists because ByteRange rejects start >= end, so the
    fabrication is structural rather than careless — which is exactly why it
    needs announcing rather than silently patching.
    """
    handler = BinaryHandler(source=FakeSource({"empty.bin": b""}))
    ref = _ref(uri="empty.bin", size_bytes=0)
    rendered = await handler.represent(ref, Budget(max_chars=None))
    assert any(d.what == "synthesized description" for d in rendered.degradations)


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


async def test_represent_reports_started_and_finished_with_the_ref() -> None:
    recorder = RecordingObserver()
    ref = _ref()
    await _handler(observer=recorder).represent(ref, Budget(max_chars=None))
    kinds = [type(e).__name__ for e in recorder.events]
    assert kinds == ["OperationStarted", "OperationFinished"]
    started = recorder.events[0]
    finished = recorder.events[1]
    assert isinstance(started, OperationStarted)
    assert isinstance(finished, OperationFinished)
    assert started.ref == ref
    assert finished.ref == ref
    assert finished.elapsed_s >= 0.0


async def test_without_an_observer_behaviour_is_unchanged() -> None:
    before = await _handler().represent(_ref(), Budget(max_chars=None))
    after = await _handler(observer=None).represent(_ref(), Budget(max_chars=None))
    assert before == after


async def test_an_observer_that_raises_does_not_change_the_result() -> None:
    """A read must not fail — or differ — because progress reporting failed."""
    quiet = await _handler().represent(_ref(), Budget(max_chars=None))
    noisy = await _handler(observer=RaisingObserver()).represent(_ref(), Budget(max_chars=None))
    assert quiet == noisy


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
