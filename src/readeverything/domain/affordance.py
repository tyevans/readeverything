"""What a handler says it can do, before it does any of it.

An `Affordance` is a *declaration*, not a bound callable. That is the whole
design: the registry decides what to expose without executing anything, and the
tool pack materialises the survivors into tools. If an affordance were a
callable, deciding availability would mean holding a live handler, and
capability negotiation would have to happen at call time — which is exactly the
"tool exists but returns sorry" behaviour this avoids.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from readeverything.domain.capability import Capability, CapabilitySet


class DetailLevel(StrEnum):
    """How much work invoking this affordance is likely to be."""

    CARD = "card"
    SEGMENT = "segment"
    DEEP = "deep"


@dataclass(frozen=True, slots=True)
class Affordance:
    """One operation a handler offers over a source."""

    name: str
    description: str
    params: type[BaseModel]
    requires: frozenset[Capability]
    level: DetailLevel

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"name must be a valid identifier, got {self.name!r}")
        if not self.description.strip():
            raise ValueError("description must not be blank")

    def is_available(self, capabilities: CapabilitySet) -> bool:
        return capabilities.satisfies(self.requires)
