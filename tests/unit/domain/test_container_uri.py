"""The `!` grammar, including the escape that would otherwise rot unnoticed."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from readeverything.domain.container_uri import container_of, join_uri, split_uri


def test_a_plain_path_is_one_segment() -> None:
    """Nothing changes for a uri with no `!`. This is the compatibility story."""
    assert split_uri("docs/report.pdf") == ("docs/report.pdf",)


def test_splits_on_the_separator() -> None:
    assert split_uri("docs.zip!nested.tar.gz!notes.txt") == (
        "docs.zip",
        "nested.tar.gz",
        "notes.txt",
    )


def test_a_doubled_separator_is_a_literal_one() -> None:
    assert split_uri("a.zip!od!!d.txt") == ("a.zip", "od!d.txt")


def test_join_escapes_a_literal_separator() -> None:
    assert join_uri(("a.zip", "od!d.txt")) == "a.zip!od!!d.txt"


def test_join_of_one_segment_is_that_segment() -> None:
    assert join_uri(("docs/report.pdf",)) == "docs/report.pdf"


def test_container_of_a_plain_path_is_none() -> None:
    assert container_of("docs/report.pdf") is None


def test_container_of_a_member_is_everything_to_its_left() -> None:
    assert container_of("docs.zip!nested.tar.gz!notes.txt") == "docs.zip!nested.tar.gz"


def test_container_of_preserves_the_escape() -> None:
    assert container_of("a.zip!od!!d.txt") == "a.zip"


def test_an_empty_segment_is_refused() -> None:
    with pytest.raises(ValueError, match="empty segment"):
        split_uri("a.zip!")


def test_an_empty_uri_is_refused() -> None:
    with pytest.raises(ValueError, match="empty segment"):
        split_uri("")


def test_joining_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one segment"):
        join_uri(())


def test_joining_an_empty_segment_is_refused() -> None:
    with pytest.raises(ValueError, match="empty segment"):
        join_uri(("a.zip", ""))


@given(st.lists(st.text(min_size=1), min_size=1, max_size=4))
def test_round_trips_through_join_and_split(segments: list[str]) -> None:
    """The escape is the whole reason this is a property test.

    Hypothesis generates `!` freely, which is the only way this catches a
    naive `str.split("!")` regression years from now.
    """
    assert split_uri(join_uri(segments)) == tuple(segments)
