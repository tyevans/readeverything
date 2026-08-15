"""What every media handler must be able to say and do.

Handlers are stateless and receive every capability by constructor injection.
A handler never touches a filesystem, never shells out directly and never reads
the environment — it asks an injected port. That is what makes them unit
testable with fakes, and what keeps ffmpeg confined to one adapter module.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel

from readeverything.domain.affordance import Affordance
from readeverything.domain.capability import Capability
from readeverything.domain.card import Card
from readeverything.domain.identity import SourceRef
from readeverything.domain.rendition import Budget, Rendered, Rendition


@runtime_checkable
class MediaHandler(Protocol):
    #: Mimetype patterns this handler claims. See `registry.patterns`.
    mime_patterns: ClassVar[tuple[str, ...]]
    #: Higher wins a tie. Bundled handlers use 0; a caller shadows with 1+.
    priority: ClassVar[int]
    #: Stable identity, part of the artifact cache key.
    handler_id: ClassVar[str]
    #: Bumped when this handler's output changes for the same input.
    handler_version: ClassVar[int]

    def requires(self) -> frozenset[Capability]:
        """Capabilities without which this handler cannot function at all."""
        ...

    def affordances(self) -> tuple[Affordance, ...]:
        """Everything this handler can do, before capability filtering."""
        ...

    async def describe(self, ref: SourceRef) -> Card: ...

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition: ...

    async def represent(self, ref: SourceRef, budget: Budget) -> Rendered: ...
