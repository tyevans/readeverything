r"""Addressing something inside something else.

`ports/source.py` and `domain/identity.py` have both described this grammar in
prose since Spec 1 -- an archive member addressed as `"/a.zip!inner.txt"` --
without anything implementing it. This is that implementation, and it is
normative: Spec 9's fixtures reference these strings.

`!` was chosen because Java's `jar:` URLs have used it for two decades, it is
already written into two docstrings in this repository, and it is legal in
POSIX filenames but vanishingly rare. Rare is not impossible, which is why the
escape below exists rather than a claim that collision cannot happen.

The escape is `\`, and it used to be a doubled `!`. Doubling cannot work, for a
reason that is easy to miss and that a property test found: `("0", "!0")` and
`("0!", "0")` both encode to `"0!!!0"`, so whatever a scan decides that string
means, it is wrong for one of them. A run of `!` in the output is
`2a + 1 + 2b` characters long and there is nothing in it saying where the
separator sits. CSV has the same shape of problem and solves it with quoting
rather than by doubling its comma; here an escape character distinct from the
separator is the smaller fix. `\` is legal in a POSIX filename too, so it
escapes itself -- and a zip written on Windows separates its member names with
`\`, so those names arrive here doubled. `join_uri` does that for you; the only
caller who notices is one hand-writing a uri.

Pure functions and no I/O, because this sits in `domain`: an adapter parses
these strings and so do test fixtures, and `domain` is the only layer both may
import.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Between a container and a member. A literal one inside a segment is escaped.
SEPARATOR = "!"

#: Before a literal separator, or before another escape.
ESCAPE = "\\"


def split_uri(uri: str) -> tuple[str, ...]:
    """`"a!b!c"` -> `("a", "b", "c")`, honouring the `\\!` escape.

    A SCAN, not `str.split(SEPARATOR)`. The difference only shows up on a
    member whose name contains a literal `!`, which is rare enough that a
    `str.split` regression here would be invisible for a year -- so the
    round-trip property test in `tests/unit/domain/test_container_uri.py` is
    the thing actually holding this correct, not review. It has already earned
    its keep once: it is what found the ambiguity described in the module
    docstring, years earlier than a bug report would have.
    """
    segments: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(uri):
        character = uri[index]
        if character == ESCAPE and index + 1 < len(uri):
            # Whatever follows is a literal, including another escape. A
            # trailing escape with nothing after it falls through and is kept
            # as itself, so no input is unparseable.
            current.append(uri[index + 1])
            index += 2
            continue
        if character == SEPARATOR:
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
    return SEPARATOR.join(
        # The escape first, or escaping the separator would escape its own
        # backslash a second time.
        segment.replace(ESCAPE, ESCAPE * 2).replace(SEPARATOR, ESCAPE + SEPARATOR)
        for segment in segments
    )


def container_of(uri: str) -> str | None:
    """The uri of what holds `uri`, or None when nothing does.

    Re-joins rather than slicing the original string, so the escape survives:
    slicing at the last raw `!` would cut `"a.zip!od\\!d.txt"` in the wrong
    place.
    """
    segments = split_uri(uri)
    if len(segments) == 1:
        return None
    return join_uri(segments[:-1])
