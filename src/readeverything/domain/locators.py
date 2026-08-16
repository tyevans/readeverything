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


@dataclass(frozen=True, slots=True)
class CellRange:
    """A rectangular block of cells in a named sheet, 0-indexed internally.

    None of the five older locators addresses a cell. `CharSpan` into rendered
    text is not it: the rendering is an artifact of the sheets handler's own
    delimiter choice, so a citation into it stops meaning anything the moment
    that delimiter changes. A cell's address does not depend on how the cell
    was printed.

    0-indexed here and A1 outside. A1 notation is presentation — it is
    1-indexed, its columns are base-26 letters with no zero digit, and it
    belongs to the handler that talks to callers about spreadsheets. The domain
    counts from zero like everything else it addresses.
    """

    sheet: str
    row: int
    col: int
    rows: int = 1
    cols: int = 1

    def __post_init__(self) -> None:
        if not self.sheet.strip():
            raise ValueError("sheet must not be blank")
        if self.row < 0:
            raise ValueError(f"row must not be negative, got {self.row}")
        if self.col < 0:
            raise ValueError(f"col must not be negative, got {self.col}")
        if self.rows < 1:
            raise ValueError(f"rows must be at least 1, got {self.rows}")
        if self.cols < 1:
            raise ValueError(f"cols must be at least 1, got {self.cols}")


@dataclass(frozen=True, slots=True)
class PartSpan:
    """A character range inside one named part of a multi-part document.

    An EPUB is the case that needed this. Its text lives in a dozen separate
    XHTML files inside a zip, and none of the older locators can say where a
    sentence came from. `CharSpan` names offsets with no file, which in a book
    of twelve chapters is twelve possible answers. `ByteRange` into the epub
    itself addresses compressed bytes -- checkable only by a reader who
    reimplements DEFLATE, which is the opposite of what a citation is for.

    `part` is the member's name exactly as the container spells it, because
    that is the string that resolves it: `"OEBPS/ch3.xhtml"` unzips, and a
    prettified `"Chapter 3"` does not. `start` and `end` index that part's own
    decoded text, so a reader extracts one file and slices it.
    """

    part: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.part.strip():
            raise ValueError("part must not be blank")
        if self.start < 0:
            raise ValueError(f"start must not be negative, got {self.start}")
        if self.start >= self.end:
            raise ValueError(f"start must be less than end, got {self.start} >= {self.end}")

    @property
    def length(self) -> int:
        return self.end - self.start


type Locator = TimeSpan | PageRef | BBox | CharSpan | ByteRange | CellRange | PartSpan
