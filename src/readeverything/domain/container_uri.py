"""Addressing something inside something else.

`ports/source.py` and `domain/identity.py` have both described this grammar in
prose since Spec 1 -- an archive member addressed as `"/a.zip!inner.txt"` --
without anything implementing it. This is that implementation, and it is
normative: Spec 9's fixtures reference these strings.

`!` was chosen because Java's `jar:` URLs have used it for two decades, it is
already written into two docstrings in this repository, and it is legal in
POSIX filenames but vanishingly rare. Rare is not impossible, which is why the
escape below exists rather than a claim that collision cannot happen.

Pure functions and no I/O, because this sits in `domain`: an adapter parses
these strings and so do test fixtures, and `domain` is the only layer both may
import.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Between a container and a member. A literal one inside a segment is doubled.
SEPARATOR = "!"


def split_uri(uri: str) -> tuple[str, ...]:
    """`"a!b!c"` -> `("a", "b", "c")`, honouring the `!!` escape.

    A SCAN, not `str.split(SEPARATOR)`. The difference only shows up on a
    member whose name contains a literal `!`, which is rare enough that a
    `str.split` regression here would be invisible for a year -- so the
    round-trip property test in `tests/unit/domain/test_container_uri.py` is
    the thing actually holding this correct, not review.
    """
    segments: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(uri):
        character = uri[index]
        if character == SEPARATOR:
            if uri[index + 1 : index + 2] == SEPARATOR:
                current.append(SEPARATOR)
                index += 2
                continue
            segments.append("".join(current))
            current = []
            index += 1
            continue
        current.append(character)
        index += 1
    segments.append("".join(current))
    if any(not segment for segment in segments):
        # An empty segment names nothing. Allowing it would let `"a.zip!"`
        # resolve to the archive itself by a second spelling, and two spellings
        # for one source means two provenance stories for one citation.
        raise ValueError(f"empty segment in container uri {uri!r}")
    return tuple(segments)


def join_uri(segments: Sequence[str]) -> str:
    """The inverse of `split_uri`, escaping any literal separator."""
    if not segments:
        raise ValueError("a container uri needs at least one segment")
    if any(not segment for segment in segments):
        raise ValueError(f"empty segment in {list(segments)!r}")
    return SEPARATOR.join(segment.replace(SEPARATOR, SEPARATOR * 2) for segment in segments)


def container_of(uri: str) -> str | None:
    """The uri of what holds `uri`, or None when nothing does.

    Re-joins rather than slicing the original string, so the escape survives:
    slicing at the last raw `!` would cut `"a.zip!od!!d.txt"` in the wrong
    place.
    """
    segments = split_uri(uri)
    if len(segments) == 1:
        return None
    return join_uri(segments[:-1])
