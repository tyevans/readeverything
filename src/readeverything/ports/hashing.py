"""Turning a source into a stable identity.

Split out after Plan 1, where `Perception` annotated the concrete
`ContentHasher` and so was the one collaborator in the core that could not be
substituted. import-linter permits it — `pipeline` sits above `adapters` — which
is exactly why it needed a human to notice.

The port exists for callers who already know the hash: a content-addressed
store that hands one over, a manifest, a build system that hashed the file
minutes ago. Re-reading a two-hour video to recompute what the caller already
has is the cost of not having this.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from readeverything.domain.identity import ContentHash


@runtime_checkable
class ContentHashing(Protocol):
    async def hash(self, uri: str) -> ContentHash:
        """The stable identity of the bytes at `uri`."""
        ...
