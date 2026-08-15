"""Detect, dispatch, describe — the one object callers hold.

This is where a uri becomes a `SourceRef` and a `SourceRef` meets its handler.
It is deliberately thin: every decision it makes has been made somewhere
testable already, and it adds only the sequencing.

Params arrive as plain dicts from a tool call and are validated here against
the affordance's declared schema, so a handler's `invoke` may assume a
well-formed model. Validating in the handler instead would repeat the schema in
every handler and let a malformed call reach adapter code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from readeverything.domain.affordance import Affordance
from readeverything.domain.card import Card
from readeverything.domain.errors import UnknownAffordanceError
from readeverything.domain.identity import SourceRef
from readeverything.domain.rendition import Budget, Rendered, Rendition
from readeverything.pipeline.resolution import ResolutionMemo, stat_key
from readeverything.ports.artifacts import ArtifactStore
from readeverything.ports.detection import MimeDetector
from readeverything.ports.handler import MediaHandler
from readeverything.ports.hashing import ContentHashing
from readeverything.ports.source import FileSource
from readeverything.registry.registry import MimeTypeRegistry

_HEAD_BYTES = 4096


class Perception:
    """Everything an agent needs to see a filesystem."""

    def __init__(
        self,
        *,
        source: FileSource,
        detector: MimeDetector,
        hasher: ContentHashing,
        registry: MimeTypeRegistry,
        artifacts: ArtifactStore,
        memo: ResolutionMemo | None = None,
    ) -> None:
        self._source = source
        self._detector = detector
        self._hasher = hasher
        self._registry = registry
        self._artifacts = artifacts
        self._memo = memo

    async def _ref(self, uri: str) -> SourceRef:
        key = None if self._memo is None else await stat_key(self._source, uri)
        if self._memo is not None:
            cached = self._memo.get(uri, key)
            if cached is not None:
                return cached
        head = await self._source.read_range(uri, 0, _HEAD_BYTES)
        ref = SourceRef(
            uri=uri,
            mime=await self._detector.detect(uri, head),
            content_hash=await self._hasher.hash(uri),
            size_bytes=await self._source.size(uri),
        )
        if self._memo is not None:
            self._memo.put(uri, key, ref)
        return ref

    async def _resolve(self, uri: str) -> tuple[SourceRef, MediaHandler]:
        ref = await self._ref(uri)
        return ref, self._registry.resolve(ref.mime)

    def _affordance(self, handler: MediaHandler, name: str) -> Affordance:
        available = self._registry.available_affordances(handler)
        for affordance in available:
            if affordance.name == name:
                return affordance
        raise UnknownAffordanceError(name, (a.name for a in available))

    async def inspect(self, uri: str) -> Card:
        """The cheap card for `uri`, with affordances filtered to what works here."""
        ref, handler = await self._resolve(uri)
        card = await handler.describe(ref)
        return Card(
            ref=card.ref,
            kind=card.kind,
            facts=card.facts,
            outline=card.outline,
            excerpt=card.excerpt,
            affordances=self._registry.available_affordances(handler),
        )

    async def invoke(self, uri: str, name: str, params: Mapping[str, Any]) -> Rendition:
        """Invoke a named affordance. Raises if it is not available here."""
        ref, handler = await self._resolve(uri)
        affordance = self._affordance(handler, name)
        return await handler.invoke(ref, name, affordance.params.model_validate(dict(params)))

    async def represent(self, uri: str, budget: Budget) -> Rendered:
        """Flatten `uri` for indexing: text plus locator map plus barriers."""
        ref, handler = await self._resolve(uri)
        return await handler.represent(ref, budget)

    async def list(self, uri: str) -> Sequence[str]:
        """Every source under `uri`."""
        return await self._source.walk(uri)
