"""The seam a caller uses to watch an operation without slowing it down.

`emit` is the containment point every caller goes through, and it is the whole
point of this module: a caller's observer must never break a read.
"""

from __future__ import annotations

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

    Deliberately broad: any exception at all, not a curated list. The library
    cannot know what a caller's observer does, so it cannot enumerate how that
    might fail, and guessing would leave exactly the gaps this exists to
    close.
    """
    if observer is None:
        return
    try:
        observer.observe(event)
    except Exception:  # noqa: BLE001 — see docstring; containment is the point
        pass
