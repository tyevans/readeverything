import pytest
from tests.fixtures_pdf import blank, born_digital, many_pages, mixed, scanned_like

from readeverything.adapters.pdfium_probe import PdfiumProbe
from readeverything.adapters.vision_recognizer import VisionTextRecognizer
from readeverything.domain.capability import CapabilitySet
from readeverything.domain.identity import ContentHash, MimeType, SourceRef
from readeverything.domain.locators import BBox, CharSpan, PageRef
from readeverything.domain.observation import OperationFinished, OperationStarted
from readeverything.domain.rendition import Budget, ImageContent, TextContent
from readeverything.handlers.pdf import (
    OcrPageParams,
    PageImageParams,
    PageRegionParams,
    PdfHandler,
    ReadPageParams,
)
from readeverything.ports.recognition import TextRecognizer
from readeverything.registry.registry import MimeTypeRegistry
from readeverything.testing.fakes import FakeSource, FakeVision, RaisingObserver, RecordingObserver
from readeverything.testing.handler_compliance import MediaHandlerCompliance

COMPLIANCE_PDF = born_digital(["Alpha.", "Bravo."])


def _ref(uri: str = "a.pdf", size_bytes: int = 1024) -> SourceRef:
    return SourceRef(
        uri=uri,
        mime=MimeType.parse("application/pdf"),
        content_hash=ContentHash("d" * 64),
        size_bytes=size_bytes,
    )


def _handler(
    data: bytes,
    *,
    recognizer: TextRecognizer | None = None,
    observer: object | None = None,
) -> PdfHandler:
    return PdfHandler(
        source=FakeSource({"a.pdf": data, "somewhere/else": data, "compliance-subject": data}),
        probe=PdfiumProbe(),
        recognizer=recognizer,
        observer=observer,  # type: ignore[arg-type]  # structural stub in tests
    )


def _pdf_handler(*, recognizer: TextRecognizer | None = None) -> PdfHandler:
    return _handler(COMPLIANCE_PDF, recognizer=recognizer)


async def test_every_character_resolves_to_the_page_it_came_from() -> None:
    """The property the whole handler exists to provide.

    Asserted across a page boundary, not just inside page one — an off-by-one
    in the segment starts passes any test that only samples the middle.
    """
    handler = _handler(born_digital(["Alpha.", "Bravo.", "Charlie."]))
    rendered = await handler.represent(_ref(), Budget(max_chars=None))

    for page_number, word in ((1, "Alpha"), (2, "Bravo"), (3, "Charlie")):
        offset = rendered.text.index(word)
        assert rendered.locator_map.resolve(offset) == PageRef(page_number)
        # and the LAST character of that word, which is where an off-by-one shows
        assert rendered.locator_map.resolve(offset + len(word) - 1) == PageRef(page_number)


async def test_a_document_with_an_empty_middle_page_still_maps_every_character() -> None:
    """The case the page separator exists for.

    A page whose text layer is empty owns no characters of its own. Without the
    trailing separator its CharSpan would be zero-width, which CharSpan rejects,
    and the whole document would fail to render because of one blank page in
    the middle.
    """
    handler = _handler(mixed(["Alpha.", None, "Charlie."]))
    rendered = await handler.represent(_ref(), Budget(max_chars=None))

    assert rendered.locator_map.length == len(rendered.text)
    assert len(rendered.locator_map.segments) == 3
    assert rendered.locator_map.resolve(rendered.text.index("Alpha")) == PageRef(1)
    assert rendered.locator_map.resolve(rendered.text.index("Charlie")) == PageRef(3)
    # the middle page still resolves — to page 2, over its separator alone
    middle = rendered.barriers[0]
    assert rendered.locator_map.resolve(middle) == PageRef(2)


async def test_an_empty_first_page_and_an_empty_last_page_both_map() -> None:
    """The other two positions a naive implementation gets wrong."""
    for pages in (["Alpha.", None], [None, "Charlie."]):
        rendered = await _handler(mixed(pages)).represent(_ref(), Budget(max_chars=None))
        assert rendered.locator_map.length == len(rendered.text)
        assert len(rendered.locator_map.segments) == 2


async def test_barriers_sit_at_page_breaks() -> None:
    """`Rendered.barriers` has never had a producer until now.

    A barrier marks a point a chunker must not casually split across, because
    text either side belongs to different pages. So there is one barrier per
    page break — page count minus one — and each sits exactly where a new
    page's first character begins.
    """
    handler = _handler(born_digital(["Alpha.", "Bravo.", "Charlie."]))
    rendered = await handler.represent(_ref(), Budget(max_chars=None))

    assert len(rendered.barriers) == 2
    for barrier in rendered.barriers:
        # the character at a barrier is the first of its page, so the character
        # before it belongs to the previous page
        assert rendered.locator_map.resolve(barrier) != rendered.locator_map.resolve(barrier - 1)


async def test_a_chunk_spanning_a_page_break_cites_both_pages() -> None:
    """Why `resolve_span` returns a tuple, demonstrated on a real document."""
    handler = _handler(born_digital(["Alpha.", "Bravo."]))
    rendered = await handler.represent(_ref(), Budget(max_chars=None))
    barrier = rendered.barriers[0]

    pages = rendered.locator_map.resolve_span(CharSpan(barrier - 2, barrier + 2))
    assert pages == (PageRef(1), PageRef(2))


async def test_the_card_reports_the_page_count_without_extracting_text() -> None:
    handler = _handler(many_pages(12))
    card = await handler.describe(_ref())
    assert card.facts["page_count"] == 12
    assert "This is page" not in (card.excerpt or "")


async def test_the_card_is_a_pdf_by_its_mime_and_facts_not_by_a_new_media_kind() -> None:
    """`application/pdf` is `MediaKind.BINARY` — see `MediaKind`'s own docstring.

    Dispatch to this handler happens at the exact-mimetype step, long before
    the kind step, so adding a `DOCUMENT` member would change what `kind:`
    patterns mean without buying anything here.
    """
    card = await _handler(born_digital(["Alpha."])).describe(_ref())
    assert str(card.ref.mime) == "application/pdf"
    assert card.facts["page_count"] == 1
    assert card.facts["readable"] == "yes"


async def test_a_four_hundred_page_map_has_four_hundred_segments() -> None:
    """Page granularity, not character granularity — spec §5.2.

    A per-character map over a long document would be enormous and would have
    to fabricate a rectangle for every space and newline.
    """
    handler = _handler(many_pages(400))
    rendered = await handler.represent(_ref(), Budget(max_chars=None))
    assert len(rendered.locator_map.segments) == 400


async def test_an_unopenable_pdf_degrades_rather_than_raising() -> None:
    """Same law as an undecodable image: the handler reports, it never raises."""
    handler = _handler(b"%PDF-1.4 and then garbage")
    rendered = await handler.represent(_ref(), Budget(max_chars=None))
    assert rendered.text
    assert any("could not be opened" in d.detail for d in rendered.degradations)


async def test_an_unopenable_pdf_still_produces_a_card() -> None:
    card = await _handler(b"%PDF-1.4 and then garbage").describe(_ref())
    assert card.facts["readable"] == "no"
    assert card.outline == ()


async def test_truncation_reports_what_it_kept_and_drops_the_barriers_it_cut() -> None:
    """A barrier beyond the kept text is rejected by `Rendered.__post_init__`.

    So truncation has to prune barriers as well as text — and the map with
    them, since the map must cover the text exactly.
    """
    handler = _handler(born_digital(["Alpha.", "Bravo.", "Charlie."]))
    rendered = await handler.represent(_ref(), Budget(max_chars=5))

    assert len(rendered.text) == 5
    assert rendered.barriers == ()
    assert len(rendered.locator_map.segments) == 1
    assert any(
        d.detail.startswith("kept 5 of ") and d.detail.endswith(" characters")
        for d in rendered.degradations
    )


async def test_a_budget_of_zero_still_keeps_one_character() -> None:
    """A zero-width rendition is inexpressible: `CharSpan(0, 0)` raises."""
    rendered = await _handler(born_digital(["Alpha."])).represent(_ref(), Budget(max_chars=0))
    assert len(rendered.text) == 1


async def test_a_scanned_page_is_never_reported_as_empty() -> None:
    """The project's recurring defect at a new site.

    "This document is empty" is a claim about the document. What was observed
    is "the text layer is empty" — for a scan those are different, and the
    difference is whether an agent concludes a contract says nothing or knows
    to look harder.
    """
    handler = _handler(scanned_like())
    rendered = await handler.represent(_ref(), Budget(max_chars=None))

    assert "empty" not in rendered.text.lower()
    assert any("scan" in d.what.lower() or "image" in d.what.lower() for d in rendered.degradations)


async def test_a_blank_page_and_a_scanned_page_do_not_produce_the_same_text() -> None:
    """Both have zero characters in the text layer. If the handler cannot tell
    them apart, it is guessing about one of them."""
    scanned = await _handler(scanned_like()).represent(_ref(), Budget(max_chars=None))
    empty = await _handler(blank()).represent(_ref(), Budget(max_chars=None))
    assert scanned.text != empty.text


async def test_a_scan_without_a_vision_capability_says_so_and_does_not_ocr() -> None:
    """Negotiation working: no recogniser means no OCR and an honest report."""
    handler = _handler(scanned_like(), recognizer=None)
    rendered = await handler.represent(_ref(), Budget(max_chars=None))
    assert any(
        "no text could be extracted" in d.detail.lower() or "not attempted" in d.detail.lower()
        for d in rendered.degradations
    )


async def test_a_scan_with_a_recognizer_still_does_not_ocr_in_represent() -> None:
    """`represent()` never runs OCR, even when a recogniser is configured: OCR
    is `DEEP`, gated, and only reachable by name through `ocr_page`. A
    recogniser being present must not change what the flattened text or its
    degradation claims happened."""
    handler = _handler(scanned_like(), recognizer=VisionTextRecognizer(vision=FakeVision()))
    rendered = await handler.represent(_ref(), Budget(max_chars=None))

    assert "empty" not in rendered.text.lower()
    assert any("scan" in d.what.lower() or "image" in d.what.lower() for d in rendered.degradations)
    assert not any(
        "ocr" in d.detail.lower() and "not attempted" not in d.detail.lower()
        for d in rendered.degradations
    )


async def test_a_blank_page_is_called_blank_and_not_a_scan() -> None:
    """The other half of the distinction: a blank page must not be described as
    carrying image content nobody read."""
    rendered = await _handler(blank()).represent(_ref(), Budget(max_chars=None))
    assert "blank" in rendered.text.lower()
    assert all("scan" not in d.what.lower() for d in rendered.degradations)


async def test_a_page_that_extracts_nothing_still_owns_a_character() -> None:
    """`CharSpan` rejects a zero-width span, so a page contributing no text
    would break the map outright. It owns its separator instead."""
    rendered = await _handler(blank(3)).represent(_ref(), Budget(max_chars=None))
    assert len(rendered.locator_map.segments) == 3
    assert rendered.locator_map.resolve(len(rendered.text) - 1) == PageRef(3)


async def test_read_page_returns_that_page_located_at_that_page() -> None:
    handler = _handler(born_digital(["Alpha.", "Bravo.", "Charlie."]))
    rendition = await handler.invoke(_ref(), "read_page", ReadPageParams(page=2))
    assert isinstance(rendition.content, TextContent)
    assert "Bravo" in rendition.content.text
    assert rendition.locator == PageRef(2)


async def test_asking_for_a_page_past_the_end_degrades_rather_than_raising() -> None:
    """The handler never raises. An agent that guesses a page number gets a
    result it can read and correct, not an exception."""
    handler = _handler(born_digital(["Only page."]))
    rendition = await handler.invoke(_ref(), "read_page", ReadPageParams(page=99))
    assert rendition.degraded


async def test_page_region_bbox_uses_a_top_left_origin() -> None:
    """PDF points are bottom-left origin; BBox is top-left, as used by the
    image handler's crop. A missing flip yields upside-down citations that
    every test passes unless one asserts a known glyph's position.

    The fixture draws its text near the TOP of the page, so the top half must
    contain it and the bottom half must not.
    """
    handler = _handler(born_digital(["Alpha at the top."]))

    top = await handler.invoke(
        _ref(), "page_region", PageRegionParams(page=1, x=0.0, y=0.0, w=1.0, h=0.5)
    )
    bottom = await handler.invoke(
        _ref(), "page_region", PageRegionParams(page=1, x=0.0, y=0.5, w=1.0, h=0.5)
    )
    assert isinstance(top.content, TextContent)
    assert isinstance(bottom.content, TextContent)
    assert "Alpha" in top.content.text
    assert "Alpha" not in bottom.content.text


async def test_page_region_returns_a_bbox_carrying_its_page() -> None:
    """`BBox.page` has never been anything but None. This is its first real
    value."""
    handler = _handler(born_digital(["Alpha.", "Bravo."]))
    rendition = await handler.invoke(
        _ref(), "page_region", PageRegionParams(page=2, x=0.0, y=0.0, w=1.0, h=1.0)
    )
    assert isinstance(rendition.locator, BBox)
    assert rendition.locator.page == 2


async def test_page_image_returns_image_content_a_vision_tool_can_read() -> None:
    """A diagram on page 12 becomes describable through the existing tool pack
    without this handler knowing anything about vision."""
    handler = _handler(born_digital(["Alpha."]))
    rendition = await handler.invoke(_ref(), "page_image", PageImageParams(page=1, dpi=72))
    assert isinstance(rendition.content, ImageContent)
    assert rendition.content.mime == "image/png"
    assert rendition.content.data.startswith(b"\x89PNG")


async def test_ocr_output_is_marked_as_a_model_reading_not_extraction() -> None:
    """OCR is a model's reading of an image, not the document's own bytes. A
    consumer indexing it is entitled to know which it has — the same
    distinction drawn for synthesized text."""
    handler = _handler(scanned_like(), recognizer=VisionTextRecognizer(vision=FakeVision()))
    rendition = await handler.invoke(_ref(), "ocr_page", OcrPageParams(page=1))
    assert rendition.degraded


async def test_ocr_is_not_offered_without_a_vision_capability() -> None:
    """Negotiation, not a runtime apology: the affordance must not appear."""
    registry = MimeTypeRegistry(
        handlers=[_pdf_handler(recognizer=None)], capabilities=CapabilitySet.empty()
    )
    names = {a.name for a in registry.available_affordances(registry.handlers[0])}
    assert "ocr_page" not in names


async def test_represent_reports_started_and_finished_with_the_ref() -> None:
    recorder = RecordingObserver()
    ref = _ref()
    await _handler(COMPLIANCE_PDF, observer=recorder).represent(ref, Budget(max_chars=None))
    kinds = [type(e).__name__ for e in recorder.events]
    assert kinds == ["OperationStarted", "OperationFinished"]
    started = recorder.events[0]
    finished = recorder.events[1]
    assert isinstance(started, OperationStarted)
    assert isinstance(finished, OperationFinished)
    assert started.ref == ref
    assert finished.ref == ref
    assert finished.elapsed_s >= 0.0


async def test_attaching_an_observer_does_not_change_the_result() -> None:
    """Unobserved output and observed output are the same `Rendered`."""
    recorder = RecordingObserver()
    unobserved = await _handler(COMPLIANCE_PDF).represent(_ref(), Budget(max_chars=None))
    observed = await _handler(COMPLIANCE_PDF, observer=recorder).represent(
        _ref(), Budget(max_chars=None)
    )
    assert unobserved == observed
    assert recorder.events, "the observed arm must actually have been observed"


async def test_an_observer_that_raises_does_not_change_the_result() -> None:
    """A read must not fail — or differ — because progress reporting failed."""
    quiet = await _handler(COMPLIANCE_PDF).represent(_ref(), Budget(max_chars=None))
    noisy = await _handler(COMPLIANCE_PDF, observer=RaisingObserver()).represent(
        _ref(), Budget(max_chars=None)
    )
    assert quiet == noisy


class TestPdfHandlerCompliance(MediaHandlerCompliance):
    @pytest.fixture
    def handler(self) -> PdfHandler:
        return _handler(COMPLIANCE_PDF)

    @pytest.fixture
    def content(self) -> bytes:
        return COMPLIANCE_PDF
