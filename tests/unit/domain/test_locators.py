import pytest

from readeverything.domain.locators import BBox, ByteRange, CharSpan, PageRef, TimeSpan


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
