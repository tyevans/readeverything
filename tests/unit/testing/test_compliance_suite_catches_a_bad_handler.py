"""The compliance suite must fail an adapter that breaks a law.

A suite that passes everything is worse than no suite: it certifies nothing
while looking like it certifies the contract. So the suite is itself tested,
against a handler deliberately built to violate one law.
"""

import pytest
from pydantic import BaseModel

from readeverything.domain.affordance import Affordance, DetailLevel
from readeverything.domain.capability import Capability
from readeverything.domain.card import Card
from readeverything.domain.errors import UnknownAffordanceError
from readeverything.domain.identity import MediaKind, SourceRef
from readeverything.domain.locator_map import LocatorMap, LocatorSegment
from readeverything.domain.locators import CharSpan
from readeverything.domain.rendition import Budget, Degradation, Rendered, Rendition
from readeverything.testing.handler_compliance import MediaHandlerCompliance


class _Params(BaseModel):
    pass


class _LyingHandler:
    """Declares an affordance it does not implement — the drift law's target.

    Every other law is satisfied deliberately, so the suite's other tests must
    pass; only `test_declared_affordances_are_invocable` is meant to fail.
    """

    mime_patterns = ("text/plain",)
    priority = 0
    handler_id = "lying"
    handler_version = 1

    def requires(self) -> frozenset[Capability]:
        return frozenset()

    def affordances(self) -> tuple[Affordance, ...]:
        return (
            Affordance(
                name="declared_but_absent",
                description="Declared and never implemented.",
                params=_Params,
                requires=frozenset(),
                level=DetailLevel.DEEP,
            ),
        )

    async def describe(self, ref: SourceRef) -> Card:
        return Card(
            ref=ref, kind=MediaKind.TEXT, facts={}, outline=(), excerpt=None, affordances=()
        )

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition:
        declared = tuple(a.name for a in self.affordances())
        if name not in declared:
            raise UnknownAffordanceError(name, declared)
        raise NotImplementedError(name)

    async def represent(self, ref: SourceRef, budget: Budget) -> Rendered:
        text = "hello world"
        locator_map = LocatorMap.build(
            (LocatorSegment(span=CharSpan(0, len(text)), locator=CharSpan(0, len(text))),)
        )
        if budget.permits(len(text)):
            return Rendered(text=text, locator_map=locator_map, barriers=(), degradations=())
        truncated = text[: budget.max_chars] if budget.max_chars is not None else text
        truncated_map = LocatorMap.build(
            (LocatorSegment(span=CharSpan(0, len(truncated)), locator=CharSpan(0, len(truncated))),)
        )
        return Rendered(
            text=truncated,
            locator_map=truncated_map,
            barriers=(),
            degradations=(Degradation(what="text", detail="truncated to fit budget"),),
        )


class TestTheSuiteCatchesDrift(MediaHandlerCompliance):
    @pytest.fixture
    def handler(self) -> _LyingHandler:
        return _LyingHandler()

    @pytest.fixture
    def content(self) -> bytes:
        return b"hello"

    async def test_declared_affordances_are_invocable(self, handler, content, ref) -> None:  # type: ignore[no-untyped-def]
        """Override: assert the inherited law FAILS for this deliberately broken handler."""
        with pytest.raises(NotImplementedError):
            await super().test_declared_affordances_are_invocable(handler, content, ref)
