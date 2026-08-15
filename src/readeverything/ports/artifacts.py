"""Storing what was expensive to derive.

Entries are immutable and content-addressed on the whole derivation, so there
is no invalidation protocol and no staleness: a different input, handler
version, parameter set or model revision is simply a different key.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ArtifactStore(Protocol):
    async def get(self, key: str) -> bytes | None:
        """The stored artifact, or None on a miss."""
        ...

    async def put(self, key: str, value: bytes) -> None:
        """Store an artifact. Storing an existing key is a no-op, not an error."""
        ...
