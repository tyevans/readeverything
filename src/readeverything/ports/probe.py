"""Asking the machine what it can actually do.

Capability negotiation was sound from Plan 1 and its inputs were not: a caller
hand-asserted that ffmpeg existed, and a wrong assertion registered affordances
that could not run — the exact failure negotiation exists to prevent. A probe
replaces an assertion with an observation.

A probe never raises and never guesses. `None` means unavailable, so under
uncertainty the library offers less rather than more.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from readeverything.domain.capability import Capability


@runtime_checkable
class CapabilityProbe(Protocol):
    async def revision(self, capability: Capability) -> str | None:
        """A revision string if this capability is genuinely available, else None."""
        ...
