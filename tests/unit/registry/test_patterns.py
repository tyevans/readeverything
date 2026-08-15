from readeverything.domain.identity import MimeType
from readeverything.registry.patterns import MatchRank, match_pattern


def test_an_exact_pattern_ranks_highest() -> None:
    assert match_pattern("video/mp4", MimeType.parse("video/mp4")) is MatchRank.EXACT


def test_a_suffix_pattern_matches_a_structured_subtype() -> None:
    assert match_pattern("+zip", MimeType.parse("application/epub+zip")) is MatchRank.SUFFIX
    assert match_pattern("+zip", MimeType.parse("application/pdf")) is None


def test_a_type_wildcard_matches_the_family() -> None:
    assert match_pattern("video/*", MimeType.parse("video/webm")) is MatchRank.TYPE
    assert match_pattern("video/*", MimeType.parse("audio/mp3")) is None


def test_a_kind_pattern_matches_the_media_kind() -> None:
    assert match_pattern("kind:text", MimeType.parse("text/markdown")) is MatchRank.KIND
    assert match_pattern("kind:binary", MimeType.parse("application/pdf")) is MatchRank.KIND


def test_the_star_pattern_always_matches_and_ranks_lowest() -> None:
    assert match_pattern("*", MimeType.parse("application/x-anything")) is MatchRank.FALLBACK


def test_a_non_matching_exact_pattern_returns_none() -> None:
    assert match_pattern("video/mp4", MimeType.parse("video/webm")) is None


def test_ranks_are_ordered_most_specific_first() -> None:
    assert MatchRank.EXACT < MatchRank.SUFFIX < MatchRank.TYPE < MatchRank.KIND < MatchRank.FALLBACK
