import pytest
from hypothesis import given
from hypothesis import strategies as st

from readeverything.domain.locator_map import LocatorMap, LocatorSegment
from readeverything.domain.locators import CharSpan, TimeSpan


def _map() -> LocatorMap:
    return LocatorMap.build(
        (
            LocatorSegment(CharSpan(0, 10), TimeSpan(0.0, 1.0)),
            LocatorSegment(CharSpan(10, 25), TimeSpan(1.0, 2.5)),
            LocatorSegment(CharSpan(25, 30), TimeSpan(2.5, 3.0)),
        )
    )


def test_length_is_the_end_of_the_last_segment() -> None:
    assert _map().length == 30


def test_resolve_returns_the_containing_segments_locator() -> None:
    m = _map()
    assert m.resolve(0) == TimeSpan(0.0, 1.0)
    assert m.resolve(9) == TimeSpan(0.0, 1.0)
    assert m.resolve(10) == TimeSpan(1.0, 2.5)
    assert m.resolve(29) == TimeSpan(2.5, 3.0)


def test_resolve_rejects_an_offset_outside_the_map() -> None:
    with pytest.raises(ValueError, match="outside the map"):
        _map().resolve(30)
    with pytest.raises(ValueError, match="outside the map"):
        _map().resolve(-1)


def test_resolve_span_returns_every_overlapping_locator_in_order() -> None:
    assert _map().resolve_span(CharSpan(8, 26)) == (
        TimeSpan(0.0, 1.0),
        TimeSpan(1.0, 2.5),
        TimeSpan(2.5, 3.0),
    )


def test_resolve_span_of_a_single_segment_returns_one_locator() -> None:
    assert _map().resolve_span(CharSpan(11, 20)) == (TimeSpan(1.0, 2.5),)


def test_build_rejects_a_gap() -> None:
    with pytest.raises(ValueError, match="gapless"):
        LocatorMap.build(
            (
                LocatorSegment(CharSpan(0, 10), TimeSpan(0.0, 1.0)),
                LocatorSegment(CharSpan(12, 20), TimeSpan(1.0, 2.0)),
            )
        )


def test_build_rejects_segments_that_do_not_start_at_zero() -> None:
    with pytest.raises(ValueError, match="must start at 0"):
        LocatorMap.build((LocatorSegment(CharSpan(3, 10), TimeSpan(0.0, 1.0)),))


def test_build_rejects_unsorted_segments() -> None:
    with pytest.raises(ValueError, match="gapless"):
        LocatorMap.build(
            (
                LocatorSegment(CharSpan(10, 20), TimeSpan(1.0, 2.0)),
                LocatorSegment(CharSpan(0, 10), TimeSpan(0.0, 1.0)),
            )
        )


def test_empty_map_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one segment"):
        LocatorMap.build(())


@st.composite
def maps(draw: st.DrawFn) -> LocatorMap:
    lengths = draw(st.lists(st.integers(min_value=1, max_value=50), min_size=1, max_size=20))
    segments: list[LocatorSegment] = []
    cursor = 0
    for i, length in enumerate(lengths):
        segments.append(
            LocatorSegment(CharSpan(cursor, cursor + length), TimeSpan(float(i), float(i) + 1.0))
        )
        cursor += length
    return LocatorMap.build(tuple(segments))


@given(maps())
def test_resolution_is_total(m: LocatorMap) -> None:
    """Every offset in the map resolves. A hole here is an uncitable passage."""
    for offset in range(m.length):
        m.resolve(offset)


@given(maps())
def test_resolution_is_monotonic(m: LocatorMap) -> None:
    """Resolution never goes backwards as the offset advances."""
    seen: list[TimeSpan] = []
    for offset in range(m.length):
        locator = m.resolve(offset)
        assert isinstance(locator, TimeSpan)
        if not seen or seen[-1] != locator:
            seen.append(locator)
    assert seen == sorted(seen, key=lambda t: t.start_s)


@given(maps())
def test_resolve_span_agrees_with_pointwise_resolution(m: LocatorMap) -> None:
    """The span API is a compression of the pointwise one, not a second scheme."""
    span = CharSpan(0, m.length)
    pointwise: list[TimeSpan] = []
    for offset in range(m.length):
        locator = m.resolve(offset)
        assert isinstance(locator, TimeSpan)
        if not pointwise or pointwise[-1] != locator:
            pointwise.append(locator)
    assert m.resolve_span(span) == tuple(pointwise)
