from typing import get_args

import pytest

from readeverything.domain.locators import (
    BBox,
    ByteRange,
    CellRange,
    CharSpan,
    Locator,
    PageRef,
    TimeSpan,
)


def test_char_span_is_half_open_and_reports_length() -> None:
    assert CharSpan(0, 5).length == 5


def test_char_span_rejects_an_inverted_range() -> None:
    with pytest.raises(ValueError, match="start must be less than end"):
        CharSpan(5, 5)


def test_char_span_overlap_is_exclusive_at_the_boundary() -> None:
    assert CharSpan(0, 5).overlaps(CharSpan(4, 9))
    assert not CharSpan(0, 5).overlaps(CharSpan(5, 9))


def test_time_span_rejects_a_negative_start() -> None:
    with pytest.raises(ValueError, match="start_s must not be negative"):
        TimeSpan(-1.0, 2.0)


def test_page_ref_is_one_indexed() -> None:
    assert PageRef(1).page == 1
    with pytest.raises(ValueError, match="page must be at least 1"):
        PageRef(0)


def test_bbox_requires_normalised_coordinates() -> None:
    BBox(page=1, x=0.0, y=0.0, w=1.0, h=1.0)
    with pytest.raises(ValueError, match="must be within the unit square"):
        BBox(page=1, x=0.5, y=0.0, w=0.8, h=0.1)


def test_byte_range_rejects_an_inverted_range() -> None:
    with pytest.raises(ValueError, match="start must be less than end"):
        ByteRange(10, 2)


def test_a_cell_range_rejects_a_negative_origin() -> None:
    """0-indexed internally, so -1 is not "one before A1"; it is nothing."""
    with pytest.raises(ValueError, match="row"):
        CellRange(sheet="Data", row=-1, col=0)
    with pytest.raises(ValueError, match="col"):
        CellRange(sheet="Data", row=0, col=-1)


def test_a_cell_range_rejects_a_non_positive_extent() -> None:
    """A zero-row block addresses no cell, which is not a citation."""
    with pytest.raises(ValueError, match="rows"):
        CellRange(sheet="Data", row=0, col=0, rows=0)
    with pytest.raises(ValueError, match="cols"):
        CellRange(sheet="Data", row=0, col=0, cols=0)


def test_a_cell_range_rejects_a_blank_sheet_name() -> None:
    """A sheet name is how the citation is resolved back. Without one the
    locator points at a workbook rather than at a place in it.
    """
    with pytest.raises(ValueError, match="sheet"):
        CellRange(sheet="   ", row=0, col=0)


def test_a_single_cell_is_the_default_extent() -> None:
    cell = CellRange(sheet="Data", row=3, col=2)
    assert (cell.rows, cell.cols) == (1, 1)


def test_a_cell_range_is_hashable_and_compares_by_value() -> None:
    """Every other locator is, and handlers hold them in frozen dataclasses."""
    assert CellRange(sheet="Data", row=0, col=0) == CellRange(sheet="Data", row=0, col=0)
    assert len({CellRange(sheet="D", row=0, col=0), CellRange(sheet="D", row=0, col=0)}) == 1


def test_cell_range_is_in_the_locator_union() -> None:
    """A locator the union does not name cannot be returned by a handler
    without mypy --strict rejecting it at every call site.
    """
    assert CellRange in get_args(Locator.__value__)
