"""Where in a source something is.

One vocabulary, shared by cards, affordance results, the locator map, chunk
barriers and citations. Every locator is pure data: speaker attribution is
*not* here, because a speaker is a property of an utterance rather than of a
position, and putting it on `TimeSpan` would mean every other locator carried a
field that is always `None`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CharSpan:
    """A half-open range of characters, `[start, end)`."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start must not be negative, got {self.start}")
        if self.start >= self.end:
            raise ValueError(f"start must be less than end, got {self.start} >= {self.end}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: CharSpan) -> bool:
        """True when the two ranges share at least one character.

        Exclusive at the boundary, because the ranges are half-open: `[0, 5)`
        and `[5, 9)` are adjacent, not overlapping. Getting this wrong would
        attach every chunk's provenance to its neighbour as well.
        """
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True, slots=True)
class ByteRange:
    """A half-open range of bytes, `[start, end)`."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start must not be negative, got {self.start}")
        if self.start >= self.end:
            raise ValueError(f"start must be less than end, got {self.start} >= {self.end}")


@dataclass(frozen=True, slots=True)
class TimeSpan:
    """A range of wall-clock time within a media stream, in seconds."""

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.start_s < 0:
            raise ValueError(f"start_s must not be negative, got {self.start_s}")
        if self.start_s >= self.end_s:
            raise ValueError(f"start_s must be less than end_s, got {self.start_s} >= {self.end_s}")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class PageRef:
    """A page of a paginated document, 1-indexed as a reader would count."""

    page: int

    def __post_init__(self) -> None:
        if self.page < 1:
            raise ValueError(f"page must be at least 1, got {self.page}")


@dataclass(frozen=True, slots=True)
class BBox:
    """A rectangle on a page, in normalised coordinates.

    Normalised rather than pixel coordinates so a locator survives the page
    being rendered at a different DPI — which it will be, since the card path
    and the VLM path render at different sizes.
    """

    page: int | None
    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"w and h must be positive, got {self.w}x{self.h}")
        if not (
            self.x >= 0.0 and self.y >= 0.0 and self.x + self.w <= 1.0 and self.y + self.h <= 1.0
        ):
            raise ValueError(
                f"must be within the unit square, got x={self.x} y={self.y} w={self.w} h={self.h}"
            )


type Locator = TimeSpan | PageRef | BBox | CharSpan | ByteRange
