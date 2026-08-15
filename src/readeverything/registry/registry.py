"""Choosing a handler, and deciding what it may offer.

Two-stage capability filtering, and the order matters:

1. A handler whose `requires()` is unsatisfied is dropped *before* dispatch, so
   it cannot win a match it could not then serve.
2. A surviving handler's individual affordances are filtered by their own
   requirements.

The consequence is the design goal: with no ASR configured, video still works —
metadata, outline and frames are there, and `read_transcript` does not exist.
The agent never sees a tool it cannot use.
"""

from __future__ import annotations

from collections.abc import Sequence

from readeverything.domain.affordance import Affordance
from readeverything.domain.capability import CapabilitySet
from readeverything.domain.errors import DomainError
from readeverything.domain.identity import MimeType
from readeverything.ports.handler import MediaHandler
from readeverything.registry.patterns import MatchRank, match_pattern


class NoHandlerError(DomainError):
    """Nothing claimed this mimetype.

    Reachable only when no fallback handler is registered. A composition that
    includes `BinaryHandler` cannot produce this, which is why there is no
    "unsupported file" path in normal use.
    """

    def __init__(self, mime: MimeType) -> None:
        super().__init__(f"no handler for {mime}; register a fallback handler with pattern '*'")


class MimeTypeRegistry:
    """Dispatches a mimetype to the most specific handler that can serve it."""

    def __init__(
        self,
        *,
        handlers: Sequence[MediaHandler],
        capabilities: CapabilitySet,
    ) -> None:
        self._capabilities = capabilities
        self._handlers = tuple(h for h in handlers if capabilities.satisfies(h.requires()))

    @property
    def handlers(self) -> tuple[MediaHandler, ...]:
        """The handlers that survived capability filtering."""
        return self._handlers

    @property
    def capabilities(self) -> CapabilitySet:
        """What this deployment can do. Part of every artifact cache key."""
        return self._capabilities

    def resolve(self, mime: MimeType) -> MediaHandler:
        """The handler for `mime`: most specific rank, then highest priority."""
        best: tuple[MatchRank, int, int] | None = None
        chosen: MediaHandler | None = None
        for index, handler in enumerate(self._handlers):
            ranks = [
                rank
                for pattern in handler.mime_patterns
                if (rank := match_pattern(pattern, mime)) is not None
            ]
            if not ranks:
                continue
            # Negated priority so that a plain `<` comparison means "better":
            # lower rank wins, then higher priority, then earlier registration.
            candidate = (min(ranks), -handler.priority, index)
            if best is None or candidate < best:
                best = candidate
                chosen = handler
        if chosen is None:
            raise NoHandlerError(mime)
        return chosen

    def available_affordances(self, handler: MediaHandler) -> tuple[Affordance, ...]:
        """The affordances of `handler` this deployment can actually serve."""
        return tuple(a for a in handler.affordances() if a.is_available(self._capabilities))
