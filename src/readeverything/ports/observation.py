"""The seam a caller uses to watch an operation without slowing it down.

`emit` is the containment point every caller goes through, and it is the whole
point of this module: a caller's observer must never break a read.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Protocol, runtime_checkable

from readeverything.domain.observation import Event


@runtime_checkable
class Observer(Protocol):
    def observe(self, event: Event) -> None: ...


def emit(observer: Observer | None, event: Event) -> None:
    """Deliver an event, or do nothing, and never let either fail a read.

    A caller's observer is arbitrary code the library invited into the middle
    of a read. It may raise; that is not the read's problem. Everything this
    library promises about never raising over bad input applies here too — a
    file that could have been read must not become unreadable because a
    progress callback threw.

    Deliberately broad within `Exception`, and deliberately no broader.
    Every subclass of `Exception` is contained — the library cannot know what
    a caller's observer does, so it cannot enumerate how that might fail, and
    a curated list would leave exactly the gaps this exists to close.

    `BaseException` is NOT contained, and widening this to `BaseException`
    would be a defect rather than more of a good thing. `KeyboardInterrupt`
    and `asyncio.CancelledError` are not failures of the observer; they are
    requests to stop, addressed to the whole process or the whole task.
    Swallowing one here would mean a Ctrl-C landing while a progress callback
    happened to be running is simply discarded, or a cancelled read carrying
    on because cancellation arrived at an unlucky instant — a read that
    ignores a stop signal is the hang this cycle exists to eliminate,
    arriving by another door. `SystemExit` and `MemoryError`
    (`MemoryError` being an `Exception`, and so still contained) mark the same
    boundary from the other side: the ones that must escape are the ones that
    are not about this observer at all.
    """
    if observer is None:
        return
    # `Exception`, never `BaseException`. See the docstring: containment is
    # the point, and KeyboardInterrupt/CancelledError are not what it contains.
    with suppress(Exception):
        observer.observe(event)
