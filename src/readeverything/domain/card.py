"""The cheap representation returned on first contact.

Producing a card must not invoke a model and must not process the whole file.
Its cost is bounded by a probe. Everything expensive is behind an affordance,
so the agent chooses what to spend rather than paying for a two-hour video
because it looked at a directory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from readeverything.domain.affordance import Affordance
from readeverything.domain.identity import MediaKind, SourceRef
from readeverything.domain.locators import Locator


@dataclass(frozen=True, slots=True)
class Segment:
    """One labelled region of a source: a scene, a chapter, a page, a cue group."""

    locator: Locator
    label: str


@dataclass(frozen=True, slots=True)
class Card:
    """What a source is, cheaply, and what can be done with it."""

    ref: SourceRef
    kind: MediaKind
    facts: Mapping[str, str | int | float]
    outline: tuple[Segment, ...]
    excerpt: str | None
    affordances: tuple[Affordance, ...]

    def affordance_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.affordances)
