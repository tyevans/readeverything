"""Bounding concurrency per capability with an `asyncio.Semaphore`.

An unconfigured capability is UNBOUNDED, never zero. A capability defaulting
to zero would deadlock every caller of it, and a deadlock looks exactly like
the hang this whole cycle exists to eliminate — the failure mode of a mistake
here must be visible load, not silence.
"""

from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from weakref import ReferenceType, ref

from readeverything.domain.capability import Capability

#: Conservative and reasoned, not measured. A vision endpoint's tolerance is
#: unknown to this library — it varies by provider, tier and account — so the
#: safe default is small: a caller raising the limit is an informed act, while
#: a caller discovering they flooded their endpoint is not. `ffmpeg` is
#: CPU-bound, so its default is bounded too, well under a typical core count,
#: for the same reason. Do not read these as tuned; they are a starting point
#: a caller is expected to override once they know their own environment.
DEFAULT_LIMITS: Mapping[Capability, int] = {
    Capability.VISION: 4,
    Capability.FFMPEG: 4,
}


class SemaphoreLimiter:
    """One `asyncio.Semaphore` per configured capability, per event loop.

    Semaphores are constructed lazily, on first use per capability, rather
    than eagerly in `__init__`. `asyncio.Semaphore()` binds no loop at
    construction time in Python 3.10+, but constructing lazily avoids relying
    on that detail and sidesteps a `SemaphoreLimiter` ever being built before
    an event loop exists.

    Lazy construction alone is not enough. A semaphore binds a loop the first
    time it is CONTENDED, and caching it on the capability alone would hand a
    second event loop — `asyncio.run` called twice, a test suite, a caller
    with a worker-per-loop design — a semaphore owned by the first, which
    raises `RuntimeError: ... is bound to a different event loop`. That error
    surfaces inside a handler's per-frame guard and is reported to the caller
    as a fact about their file. So the cache is keyed on the RUNNING LOOP as
    well as the capability: a new loop gets fresh semaphores, and the limits
    themselves — the caller's actual configuration — are shared across all of
    them.

    Entries for dead loops are dropped as they are noticed, so a process that
    creates many short-lived loops does not accumulate them. The keys are the
    loops' `id()`s paired with a weak reference, never the loops themselves:
    holding a loop object alive from a cache would be a leak of a different
    and worse kind.
    """

    def __init__(self, limits: Mapping[Capability, int] | None = None) -> None:
        self._limits: Mapping[Capability, int] = dict(DEFAULT_LIMITS if limits is None else limits)
        self._semaphores: dict[
            int, tuple[ReferenceType[AbstractEventLoop], dict[Capability, asyncio.Semaphore]]
        ] = {}

    def _semaphores_for_running_loop(self) -> dict[Capability, asyncio.Semaphore]:
        loop = asyncio.get_running_loop()
        key = id(loop)
        entry = self._semaphores.get(key)
        if entry is not None and entry[0]() is loop:
            return entry[1]
        # Either nothing cached, or `id()` was recycled by a loop that has
        # since been collected — the weak reference is what tells those apart
        # from a genuine hit. Drop whatever was there and start clean.
        self._prune()
        fresh: dict[Capability, asyncio.Semaphore] = {}
        self._semaphores[key] = (ref(loop), fresh)
        return fresh

    def _prune(self) -> None:
        for key in [key for key, (loop_ref, _) in self._semaphores.items() if loop_ref() is None]:
            del self._semaphores[key]

    def _semaphore_for(self, capability: Capability) -> asyncio.Semaphore | None:
        limit = self._limits.get(capability)
        if limit is None:
            return None
        semaphores = self._semaphores_for_running_loop()
        semaphore = semaphores.get(capability)
        if semaphore is None:
            semaphore = asyncio.Semaphore(limit)
            semaphores[capability] = semaphore
        return semaphore

    @asynccontextmanager
    async def limit(self, capability: Capability) -> AsyncIterator[None]:
        semaphore = self._semaphore_for(capability)
        if semaphore is None:
            # Unconfigured: unbounded, not zero. See module docstring.
            yield
            return
        async with semaphore:
            yield
