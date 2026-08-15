"""Character offsets to locators, and back again.

The structure on which citation correctness rests. A retrieval hit knows only
that it came from characters 4100-4380 of some flattened text; this is what
turns that into "00:42:15 to 00:42:31".

The map is required to be **gapless and starting at zero** rather than merely
sorted. A sparse map would let `resolve` fail for an offset that a chunker
happily produced, and it would fail at citation time — after the answer was
already computed, and far from whatever produced the hole. A handler that has
nothing to say about a region must say so with a segment, not with a gap.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Self

from readeverything.domain.locators import CharSpan, Locator


@dataclass(frozen=True, slots=True)
class LocatorSegment:
    """One contiguous run of text that shares a single locator."""

    span: CharSpan
    locator: Locator


@dataclass(frozen=True, slots=True)
class LocatorMap:
    """A total, monotonic mapping from character offset to locator.

    Construct with `build`, which validates and precomputes the bisection
    index. The constructor is not private, but a directly-constructed instance
    whose `starts` disagrees with its `segments` is rejected too.
    """

    segments: tuple[LocatorSegment, ...]
    starts: tuple[int, ...] = field(compare=False, repr=False)

    @classmethod
    def build(cls, segments: tuple[LocatorSegment, ...]) -> Self:
        return cls(segments=segments, starts=tuple(s.span.start for s in segments))

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("a locator map needs at least one segment")
        # Check that segments are sorted
        for i in range(len(self.segments) - 1):
            if self.segments[i].span.start >= self.segments[i + 1].span.start:
                raise ValueError(
                    f"segments must be sorted and gapless: segment {i + 1} starts at "
                    f"{self.segments[i + 1].span.start}, which is not after segment "
                    f"{i} at {self.segments[i].span.start}"
                )
        # Check that segments start at 0 and are gapless
        if self.segments[0].span.start != 0:
            raise ValueError(f"segments must start at 0, got {self.segments[0].span.start}")
        cursor = 0
        for segment in self.segments:
            if segment.span.start != cursor:
                raise ValueError(
                    f"segments must be sorted and gapless: expected a segment starting at "
                    f"{cursor}, got one starting at {segment.span.start}"
                )
            cursor = segment.span.end
        if self.starts != tuple(s.span.start for s in self.segments):
            raise ValueError("starts does not match segments; use LocatorMap.build")

    @property
    def length(self) -> int:
        """Total characters covered."""
        return self.segments[-1].span.end

    def resolve(self, offset: int) -> Locator:
        """The locator for a single character offset."""
        if not 0 <= offset < self.length:
            raise ValueError(f"offset {offset} is outside the map of length {self.length}")
        index = bisect_right(self.starts, offset) - 1
        return self.segments[index].locator

    def resolve_span(self, span: CharSpan) -> tuple[Locator, ...]:
        """Every locator overlapping `span`, in document order.

        Returns a tuple, not a single locator, and callers must not assume
        length 1: a chunk spanning 00:42:15-00:43:02 genuinely covers several
        transcript cues, and the honest citation is the union of them.
        """
        if span.end > self.length:
            raise ValueError(f"span {span} is outside the map of length {self.length}")
        first = bisect_right(self.starts, span.start) - 1
        return tuple(
            segment.locator for segment in self.segments[first:] if segment.span.overlaps(span)
        )
