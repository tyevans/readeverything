"""Matching a mimetype against a handler's claims.

Five ranks, most specific first. The ranks are an `IntEnum` so that "more
specific" is expressible as `<`, which is what makes the registry's selection a
plain `min` rather than a chain of conditionals that has to be read to be
believed.
"""

from __future__ import annotations

from enum import IntEnum

from readeverything.domain.identity import MediaKind, MimeType


class MatchRank(IntEnum):
    """How specifically a pattern matched. Lower is more specific."""

    EXACT = 0
    SUFFIX = 1
    TYPE = 2
    KIND = 3
    FALLBACK = 4


def match_pattern(pattern: str, mime: MimeType) -> MatchRank | None:
    """The rank at which `pattern` matches `mime`, or None if it does not.

    Pattern forms:
      - `"video/mp4"`  exact mimetype
      - `"+zip"`       structured suffix
      - `"video/*"`    type wildcard
      - `"kind:text"`  media kind
      - `"*"`          always matches
    """
    if pattern == "*":
        return MatchRank.FALLBACK
    if pattern.startswith("kind:"):
        wanted = pattern.removeprefix("kind:")
        return MatchRank.KIND if MediaKind.for_mime(mime).value == wanted else None
    if pattern.startswith("+"):
        return MatchRank.SUFFIX if mime.suffix == pattern.removeprefix("+") else None
    if pattern.endswith("/*"):
        return MatchRank.TYPE if mime.type == pattern.removesuffix("/*") else None
    return MatchRank.EXACT if str(mime) == pattern else None
