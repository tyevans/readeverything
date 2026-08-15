"""What is happening, said out loud, to whoever wants to hear it.

An operation that samples forty frames and calls a vision model on each of
them, sequentially, is silent by default: a caller waiting ninety seconds
cannot tell progress from a hang. These events are how a handler narrates
itself, without the library deciding where that narration goes.

Frozen `slots` dataclasses, matching every other domain type here. `Event` is
a PEP 695 union alias, the same shape as `Locator` and `RenditionContent`: one
`observe(event)` method with a typed union, not three methods with `**kwargs`
and a stringly-typed `op: str` a caller would have to guess at.
"""

from __future__ import annotations

from dataclasses import dataclass

from readeverything.domain.identity import SourceRef


@dataclass(frozen=True, slots=True)
class OperationStarted:
    """An operation began, on this source."""

    operation: str
    ref: SourceRef


@dataclass(frozen=True, slots=True)
class OperationProgressed:
    """`done` units complete, out of `total` if `total` is knowable.

    `total` is `int | None` because it is not always knowable. A video knows
    how many moments it will sample before it starts; a transcription does not
    know how many cues it will produce until it has produced them. Reporting a
    made-up total would be a number nothing measured — the field admits
    ignorance instead of inventing one.
    """

    operation: str
    ref: SourceRef
    done: int
    total: int | None

    def __post_init__(self) -> None:
        if self.done < 0:
            raise ValueError(f"done must not be negative, got {self.done}")
        if self.total is not None and self.done > self.total:
            raise ValueError(f"done must not exceed a known total: {self.done} of {self.total}")


@dataclass(frozen=True, slots=True)
class OperationFinished:
    """An operation ended, on this source, after `elapsed_s` seconds."""

    operation: str
    ref: SourceRef
    elapsed_s: float

    def __post_init__(self) -> None:
        if self.elapsed_s < 0:
            raise ValueError(f"elapsed_s must not be negative, got {self.elapsed_s}")


type Event = OperationStarted | OperationProgressed | OperationFinished
