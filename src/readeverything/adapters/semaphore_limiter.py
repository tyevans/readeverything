"""Bounding concurrency per capability with an `asyncio.Semaphore`.

An unconfigured capability is UNBOUNDED, never zero. A capability defaulting
to zero would deadlock every caller of it, and a deadlock looks exactly like
the hang this whole cycle exists to eliminate — the failure mode of a mistake
here must be visible load, not silence.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

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
    """One `asyncio.Semaphore` per configured capability.

    Semaphores are constructed lazily, on first use per capability, rather
    than eagerly in `__init__`. `asyncio.Semaphore()` binds no loop at
    construction time in Python 3.10+, but constructing lazily avoids relying
    on that detail and sidesteps a `SemaphoreLimiter` ever being built before
    an event loop exists.
    """

    def __init__(self, limits: Mapping[Capability, int] | None = None) -> None:
        self._limits: Mapping[Capability, int] = dict(DEFAULT_LIMITS if limits is None else limits)
        self._semaphores: dict[Capability, asyncio.Semaphore] = {}

    def _semaphore_for(self, capability: Capability) -> asyncio.Semaphore | None:
        limit = self._limits.get(capability)
        if limit is None:
            return None
        semaphore = self._semaphores.get(capability)
        if semaphore is None:
            semaphore = asyncio.Semaphore(limit)
            self._semaphores[capability] = semaphore
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
