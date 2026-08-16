"""The sheets handler: `CellRange`'s only producer, and the formula/value split."""

from __future__ import annotations

import pytest

from readeverything.adapters.ooxml import SHEETS_MIME
from readeverything.domain.identity import ContentHash, MimeType, SourceRef
from readeverything.domain.locators import ByteRange, CellRange
from readeverything.domain.rendition import Budget, TextContent
from readeverything.handlers.office_sheets import (
    ListSheetsParams,
    OfficeSheetsHandler,
    ReadCellsParams,
    ReadSheetParams,
    parse_a1,
    to_a1,
)
from readeverything.testing.fakes import FakeSource
from readeverything.testing.handler_compliance import MediaHandlerCompliance
from tests.fixtures_office import CACHED_FORMULA_VALUE, big_xlsx, ods_bytes, xlsx_bytes

URI = "book.xlsx"


def _handler(content: bytes) -> OfficeSheetsHandler:
    return OfficeSheetsHandler(source=FakeSource({URI: content, "somewhere/else": content}))


def _ref(content: bytes) -> SourceRef:
    return SourceRef(
        uri=URI,
        mime=MimeType.parse(SHEETS_MIME),
        content_hash=ContentHash("0" * 64),
        size_bytes=len(content),
    )


def _text(rendition: object) -> str:
    content = rendition.content  # type: ignore[attr-defined]
    assert isinstance(content, TextContent)
    return content.text


class TestSheetsCompliance(MediaHandlerCompliance):
    @pytest.fixture
    def content(self) -> bytes:
        return xlsx_bytes(formulas=True, cached=True)

    @pytest.fixture
    def handler(self, content: bytes) -> OfficeSheetsHandler:
        return OfficeSheetsHandler(
            source=FakeSource({"compliance-subject": content, "somewhere/else": content})
        )


def test_a1_round_trips_through_the_zero_indexed_domain() -> None:
    """A1 is 1-indexed and base-26; `CellRange` is 0-indexed. The conversion is
    where an off-by-one becomes a citation pointing at the wrong row.
    """
    assert parse_a1("A1", "Data") == CellRange(sheet="Data", row=0, col=0)
    assert parse_a1("B3", "Data") == CellRange(sheet="Data", row=2, col=1)
    assert parse_a1("A1:C3", "Data") == CellRange(sheet="Data", row=0, col=0, rows=3, cols=3)
    assert to_a1(0, 0) == "A1"
    assert to_a1(2, 1) == "B3"


def test_a1_handles_multi_letter_columns() -> None:
    """Column AA is 26, not 10. Base-26 with no zero digit is the trap."""
    assert parse_a1("AA1", "Data") == CellRange(sheet="Data", row=0, col=26)
    assert to_a1(0, 26) == "AA1"
    assert parse_a1("AB2", "Data") == CellRange(sheet="Data", row=1, col=27)
    assert to_a1(1, 27) == "AB2"


def test_a1_is_case_insensitive() -> None:
    assert parse_a1("b3", "Data") == parse_a1("B3", "Data")


def test_a_reversed_a1_range_is_normalised_rather_than_rejected() -> None:
    """`C3:A1` names the same block as `A1:C3`. A negative extent would raise
    from `CellRange`, so the corners are ordered before the block is built.
    """
    assert parse_a1("C3:A1", "Data") == CellRange(sheet="Data", row=0, col=0, rows=3, cols=3)


def test_an_unparseable_a1_range_yields_none_rather_than_a_wrong_cell() -> None:
    assert parse_a1("not a range", "Data") is None
    assert parse_a1("", "Data") is None
    assert parse_a1("1A", "Data") is None
    assert parse_a1("A0", "Data") is None


async def test_the_card_names_every_sheet_and_its_used_range() -> None:
    content = xlsx_bytes()
    card = await _handler(content).describe(_ref(content))
    assert [s.label for s in card.outline] == ["Data", "Notes"]
    assert card.facts["sheet_count"] == 2
    assert card.facts["sheet.Data.used_range"] == "A1:C4"
    assert card.facts["sheet.Data.rows"] == 4
    assert card.facts["sheet.Data.columns"] == 3


async def test_every_outline_segment_carries_a_cell_range() -> None:
    """`CellRange`'s reason for existing: a sheet is addressed as cells, not as
    a character offset into this handler's chosen delimiter.
    """
    content = xlsx_bytes()
    card = await _handler(content).describe(_ref(content))
    assert all(isinstance(s.locator, CellRange) for s in card.outline)
    assert card.outline[0].locator == CellRange(sheet="Data", row=0, col=0, rows=4, cols=3)


async def test_every_character_resolves_to_the_sheet_it_came_from() -> None:
    content = xlsx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    offset = rendered.text.index("Units are thousands.")
    locator = rendered.locator_map.resolve(offset)
    assert isinstance(locator, CellRange)
    assert locator.sheet == "Notes"


async def test_there_is_a_barrier_at_every_sheet_boundary() -> None:
    content = xlsx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert len(rendered.barriers) == 1
    barrier = rendered.barriers[0]
    assert rendered.locator_map.resolve(barrier) != rendered.locator_map.resolve(barrier - 1)


async def test_represent_shows_the_value_not_the_formula() -> None:
    """That is what the sheet MEANS. An auditor wants the formula and asks for
    it; a reader wants the number.
    """
    content = xlsx_bytes(formulas=True, cached=True)
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert str(CACHED_FORMULA_VALUE) in rendered.text
    assert "=B2*2" not in rendered.text


async def test_read_cells_shows_the_formula_when_asked() -> None:
    """Reporting only one of the two is how a spreadsheet lies to a reader."""
    content = xlsx_bytes(formulas=True, cached=True)
    rendition = await _handler(content).invoke(
        _ref(content), "read_cells", ReadCellsParams(name="Data", a1_range="C2", formulas=True)
    )
    assert "=B2*2" in _text(rendition)
    assert rendition.locator == CellRange(sheet="Data", row=1, col=2)


async def test_read_cells_shows_the_value_by_default() -> None:
    content = xlsx_bytes(formulas=True, cached=True)
    rendition = await _handler(content).invoke(
        _ref(content), "read_cells", ReadCellsParams(name="Data", a1_range="C2")
    )
    assert str(CACHED_FORMULA_VALUE) in _text(rendition)
    assert "=B2*2" not in _text(rendition)


async def test_a_formula_with_no_cached_value_is_reported_rather_than_shown_blank() -> None:
    """openpyxl computes nothing and many writers store no cached value. A
    blank cell and an uncomputed formula are different facts, and rendering
    both as blank is the spreadsheet's version of "this document is empty".
    """
    content = xlsx_bytes(formulas=True, cached=False)
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert any("formula" in d.what.lower() for d in rendered.degradations)
    assert any("cached" in d.detail.lower() for d in rendered.degradations)


async def test_the_uncomputed_formula_degradation_names_the_way_out() -> None:
    """A caller who reads it must learn to ask for formulas rather than
    concluding the sheet is empty.
    """
    content = xlsx_bytes(formulas=True, cached=False)
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    detail = next(d.detail for d in rendered.degradations if "formula" in d.what.lower())
    assert "read_cells" in detail
    assert "formulas" in detail


async def test_an_uncomputed_formula_shows_its_formula_rather_than_a_blank() -> None:
    content = xlsx_bytes(formulas=True, cached=False)
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert "=B2*2" in rendered.text


async def test_a_workbook_without_formulas_reports_no_formula_degradation() -> None:
    """A degradation that is always present tells a reader nothing."""
    content = xlsx_bytes(formulas=False)
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert not any("formula" in d.what.lower() for d in rendered.degradations)


async def test_read_sheet_pages_through_rows() -> None:
    content = big_xlsx(50)
    rendition = await _handler(content).invoke(
        _ref(content), "read_sheet", ReadSheetParams(name="Wide", offset=10, limit=5)
    )
    assert "region-9" in _text(rendition)
    assert "region-20" not in _text(rendition)
    assert isinstance(rendition.locator, CellRange)
    assert rendition.locator.row == 10
    assert rendition.locator.rows == 5


async def test_read_sheet_locates_what_it_returned_not_what_was_asked_for() -> None:
    """A locator claiming cells that were never read is a citation to nothing."""
    content = xlsx_bytes()
    rendition = await _handler(content).invoke(
        _ref(content), "read_sheet", ReadSheetParams(name="Data", offset=3, limit=100)
    )
    assert isinstance(rendition.locator, CellRange)
    assert rendition.locator.rows == 1


async def test_asking_for_a_sheet_that_is_not_there_degrades_rather_than_raising() -> None:
    content = xlsx_bytes()
    rendition = await _handler(content).invoke(
        _ref(content), "read_sheet", ReadSheetParams(name="Nope")
    )
    assert rendition.degraded
    assert isinstance(rendition.locator, ByteRange)


async def test_an_offset_past_the_end_degrades_rather_than_returning_nothing() -> None:
    content = xlsx_bytes()
    rendition = await _handler(content).invoke(
        _ref(content), "read_sheet", ReadSheetParams(name="Data", offset=9999)
    )
    assert rendition.degraded


async def test_an_unparseable_range_degrades_rather_than_citing_a_wrong_cell() -> None:
    content = xlsx_bytes()
    rendition = await _handler(content).invoke(
        _ref(content), "read_cells", ReadCellsParams(name="Data", a1_range="nonsense")
    )
    assert rendition.degraded


async def test_list_sheets_names_them_with_their_shapes() -> None:
    content = xlsx_bytes()
    rendition = await _handler(content).invoke(_ref(content), "list_sheets", ListSheetsParams())
    assert "Data" in _text(rendition)
    assert "Notes" in _text(rendition)


async def test_a_large_sheet_is_truncated_with_an_explicit_degradation() -> None:
    """A million-row sheet must be cut and must SAY it was cut."""
    content = big_xlsx(2000)
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=500))
    assert len(rendered.text) <= 500
    assert any("truncated" in d.what for d in rendered.degradations)


async def test_an_ods_reads_through_the_same_handler() -> None:
    content = ods_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert "north" in rendered.text
    card = await _handler(content).describe(_ref(content))
    assert [s.label for s in card.outline] == ["Data", "Notes"]


async def test_an_unreadable_workbook_degrades_rather_than_raising() -> None:
    content = b"not a workbook"
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert rendered.degradations
    assert rendered.text
    assert isinstance(rendered.locator_map.resolve(0), ByteRange)


async def test_an_unreadable_workbook_still_produces_a_card() -> None:
    content = b"not a workbook"
    card = await _handler(content).describe(_ref(content))
    assert card.facts["readable"] == "no"


async def test_every_sheet_owns_at_least_one_character() -> None:
    """`CharSpan` rejects a zero-width span, so an empty sheet between two full
    ones is what breaks the map.
    """
    content = xlsx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert rendered.locator_map.length == len(rendered.text)
    assert len(rendered.locator_map.segments) == 2


async def test_reading_a_workbook_needs_no_capability() -> None:
    assert _handler(xlsx_bytes()).requires() == frozenset()


@pytest.mark.parametrize(
    ("affordance", "params"),
    [
        ("read_sheet", ReadSheetParams()),
        ("read_cells", ReadCellsParams()),
        ("read_cells", ReadCellsParams(formulas=True)),
        ("list_sheets", ListSheetsParams()),
    ],
)
async def test_every_affordance_degrades_on_an_unreadable_workbook(
    affordance: str, params: object
) -> None:
    """No affordance raises about its input, and none of them reports an
    unopenable file as an empty one.
    """
    content = b"not a workbook"
    rendition = await _handler(content).invoke(_ref(content), affordance, params)  # type: ignore[arg-type]
    assert rendition.degraded


@pytest.mark.parametrize(
    ("affordance", "wrong"),
    [
        ("read_sheet", ListSheetsParams()),
        ("read_cells", ListSheetsParams()),
        ("list_sheets", ReadSheetParams()),
    ],
)
async def test_the_wrong_params_model_is_refused_rather_than_coerced(
    affordance: str, wrong: object
) -> None:
    content = xlsx_bytes()
    with pytest.raises(TypeError):
        await _handler(content).invoke(_ref(content), affordance, wrong)  # type: ignore[arg-type]


async def test_asking_for_a_missing_sheet_names_the_ones_that_exist() -> None:
    """A degradation that only says "no" makes an agent guess again. Naming the
    sheets turns one wrong call into one right one.
    """
    content = xlsx_bytes()
    rendition = await _handler(content).invoke(
        _ref(content), "read_cells", ReadCellsParams(name="Nope", a1_range="A1")
    )
    assert rendition.degraded
    assert "Data" in _text(rendition)
    assert "Notes" in _text(rendition)


async def test_reading_formulas_from_a_sheet_that_is_not_there_degrades() -> None:
    content = xlsx_bytes(formulas=True, cached=True)
    rendition = await _handler(content).invoke(
        _ref(content), "read_cells", ReadCellsParams(name="Nope", formulas=True)
    )
    assert rendition.degraded


async def test_an_empty_sheet_is_named_rather_than_rendered_as_nothing() -> None:
    """A sheet that exists and is empty is a different fact from a sheet that
    is missing, and the flattened text must be able to say which.
    """
    content = ods_bytes(sheets=(("Data", (("a",),)), ("Empty", ())))
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert "Empty" in rendered.text
    assert rendered.locator_map.length == len(rendered.text)
