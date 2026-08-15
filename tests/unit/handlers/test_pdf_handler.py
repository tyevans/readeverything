import pytest
from tests.fixtures_pdf import born_digital, many_pages

from readeverything.adapters.pdfium_probe import PdfiumProbe
from readeverything.domain.identity import ContentHash, MimeType, SourceRef
from readeverything.domain.locators import CharSpan, PageRef
from readeverything.domain.rendition import Budget
from readeverything.handlers.pdf import PdfHandler
from readeverything.testing.fakes import FakeSource
from readeverything.testing.handler_compliance import MediaHandlerCompliance

COMPLIANCE_PDF = born_digital(["Alpha.", "Bravo."])


def _ref(uri: str = "a.pdf", size_bytes: int = 1024) -> SourceRef:
    return SourceRef(
        uri=uri,
        mime=MimeType.parse("application/pdf"),
        content_hash=ContentHash("d" * 64),
        size_bytes=size_bytes,
    )


def _handler(data: bytes, *, recognizer: object | None = None) -> PdfHandler:
    return PdfHandler(
        source=FakeSource({"a.pdf": data, "somewhere/else": data, "compliance-subject": data}),
        probe=PdfiumProbe(),
        recognizer=recognizer,
    )


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
    assert card.facts["page_count"] == "12"
    assert "This is page" not in (card.excerpt or "")


async def test_the_card_is_a_pdf_by_its_mime_and_facts_not_by_a_new_media_kind() -> None:
    """`application/pdf` is `MediaKind.BINARY` — see `MediaKind`'s own docstring.

    Dispatch to this handler happens at the exact-mimetype step, long before
    the kind step, so adding a `DOCUMENT` member would change what `kind:`
    patterns mean without buying anything here.
    """
    card = await _handler(born_digital(["Alpha."])).describe(_ref())
    assert str(card.ref.mime) == "application/pdf"
    assert card.facts["page_count"] == "1"
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


class TestPdfHandlerCompliance(MediaHandlerCompliance):
    @pytest.fixture
    def handler(self) -> PdfHandler:
        return _handler(COMPLIANCE_PDF)

    @pytest.fixture
    def content(self) -> bytes:
        return COMPLIANCE_PDF
