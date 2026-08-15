"""The exception taxonomy.

Two families under one root, following eventsource-py's split: a domain error
means the request did not make sense, an infrastructure error means the world
did not cooperate. The distinction is what lets a caller retry one and not the
other.

Note that the *tool pack* never raises any of these — it converts them to
structured results, because a traceback reaching a model is a wasted turn. See
`readeverything/agent/results.py`.
"""

from __future__ import annotations

from collections.abc import Iterable


class ReadEverythingError(Exception):
    """Root of every error this library raises."""


class DomainError(ReadEverythingError):
    """The request did not make sense."""


class InfrastructureError(ReadEverythingError):
    """The world did not cooperate."""


class UnknownAffordanceError(DomainError):
    """An affordance was invoked that the handler does not declare."""

    def __init__(self, name: str, available: Iterable[str]) -> None:
        offered = ", ".join(sorted(available)) or "none"
        super().__init__(f"unknown affordance {name!r}; this handler offers: {offered}")
        self.name = name


class CapabilityUnavailableError(DomainError):
    """Something required a capability this deployment does not have.

    Reaching this from the registry path is a bug: the registry filters
    unsatisfied handlers and affordances out before anything can invoke them.
    It exists for direct handler use, where no filtering happened.
    """

    def __init__(self, missing: frozenset[str] | set[str]) -> None:
        super().__init__(f"missing capabilities: {', '.join(sorted(missing))}")
        self.missing = frozenset(missing)


class SourceUnreadableError(InfrastructureError):
    """A source could not be read."""


class ProbeFailedError(InfrastructureError):
    """An external probe or tool failed."""
