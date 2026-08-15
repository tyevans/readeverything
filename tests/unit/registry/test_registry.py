from typing import cast

import pytest
from pydantic import BaseModel

from readeverything.domain.affordance import Affordance, DetailLevel
from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.domain.card import Card
from readeverything.domain.identity import MimeType, SourceRef
from readeverything.domain.rendition import Budget, Rendered, Rendition
from readeverything.ports.handler import MediaHandler
from readeverything.registry.registry import MimeTypeRegistry, NoHandlerError


class _Params(BaseModel):
    pass


class _Stub:
    """A handler whose behaviour is declared entirely by construction.

    `mime_patterns`, `handler_id` and `priority` are declared `ClassVar` on
    `MediaHandler`, but this stub sets them per instance so each test can
    configure its own handlers without subclassing. `as_handler` below casts
    through that mismatch at the single point where a `_Stub` is handed to
    something typed `MediaHandler`, rather than weakening the protocol.
    """

    mime_patterns: tuple[str, ...] = ()
    priority: int = 0
    handler_id: str = "stub"
    handler_version: int = 1

    def __init__(
        self,
        *,
        patterns: tuple[str, ...],
        handler_id: str,
        requires: frozenset[Capability] = frozenset(),
        affordance_requires: frozenset[Capability] = frozenset(),
        priority: int = 0,
    ) -> None:
        self.mime_patterns = patterns
        self.handler_id = handler_id
        self.priority = priority
        self._requires = requires
        self._affordance_requires = affordance_requires

    def requires(self) -> frozenset[Capability]:
        return self._requires

    def affordances(self) -> tuple[Affordance, ...]:
        return (
            Affordance(
                name="free",
                description="Needs nothing.",
                params=_Params,
                requires=frozenset(),
                level=DetailLevel.CARD,
            ),
            Affordance(
                name="costly",
                description="Needs a capability.",
                params=_Params,
                requires=self._affordance_requires,
                level=DetailLevel.DEEP,
            ),
        )

    async def describe(self, ref: SourceRef) -> Card:
        raise NotImplementedError

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition:
        raise NotImplementedError

    async def represent(self, ref: SourceRef, budget: Budget) -> Rendered:
        raise NotImplementedError


def as_handler(stub: _Stub) -> MediaHandler:
    """Narrow a `_Stub` to `MediaHandler` for the one reason mypy can't: its
    class-var-typed attributes are set per instance for test convenience."""
    return cast(MediaHandler, stub)


def test_resolve_prefers_an_exact_match_over_a_wildcard() -> None:
    exact = _Stub(patterns=("video/mp4",), handler_id="exact")
    wild = _Stub(patterns=("video/*",), handler_id="wild")
    registry = MimeTypeRegistry(
        handlers=(as_handler(wild), as_handler(exact)), capabilities=CapabilitySet.empty()
    )
    assert registry.resolve(MimeType.parse("video/mp4")).handler_id == "exact"


def test_resolve_falls_back_through_the_ranks() -> None:
    wild = _Stub(patterns=("video/*",), handler_id="wild")
    star = _Stub(patterns=("*",), handler_id="star")
    registry = MimeTypeRegistry(
        handlers=(as_handler(wild), as_handler(star)), capabilities=CapabilitySet.empty()
    )
    assert registry.resolve(MimeType.parse("video/webm")).handler_id == "wild"
    assert registry.resolve(MimeType.parse("audio/mp3")).handler_id == "star"


def test_priority_breaks_a_tie_at_the_same_rank() -> None:
    """A caller shadows a bundled handler without forking it."""
    bundled = _Stub(patterns=("video/mp4",), handler_id="bundled", priority=0)
    custom = _Stub(patterns=("video/mp4",), handler_id="custom", priority=1)
    registry = MimeTypeRegistry(
        handlers=(as_handler(bundled), as_handler(custom)), capabilities=CapabilitySet.empty()
    )
    assert registry.resolve(MimeType.parse("video/mp4")).handler_id == "custom"


def test_a_handler_whose_capabilities_are_missing_is_dropped_entirely() -> None:
    needs_ffmpeg = _Stub(
        patterns=("video/mp4",), handler_id="video", requires=frozenset({Capability.FFMPEG})
    )
    star = _Stub(patterns=("*",), handler_id="star")
    registry = MimeTypeRegistry(
        handlers=(as_handler(needs_ffmpeg), as_handler(star)), capabilities=CapabilitySet.empty()
    )
    assert registry.resolve(MimeType.parse("video/mp4")).handler_id == "star"


def test_a_handler_is_kept_when_its_capabilities_are_present() -> None:
    needs_ffmpeg = _Stub(
        patterns=("video/mp4",), handler_id="video", requires=frozenset({Capability.FFMPEG})
    )
    registry = MimeTypeRegistry(
        handlers=(as_handler(needs_ffmpeg),),
        capabilities=CapabilitySet.of({Capability.FFMPEG: "7.1"}),
    )
    assert registry.resolve(MimeType.parse("video/mp4")).handler_id == "video"


def test_unsatisfied_affordances_are_filtered_from_a_surviving_handler() -> None:
    """Video still works with no ASR; read_transcript simply does not exist."""
    handler = _Stub(
        patterns=("video/mp4",),
        handler_id="video",
        affordance_requires=frozenset({Capability.ASR}),
    )
    registry = MimeTypeRegistry(handlers=(as_handler(handler),), capabilities=CapabilitySet.empty())
    names = tuple(a.name for a in registry.available_affordances(as_handler(handler)))
    assert names == ("free",)


def test_all_affordances_survive_when_capabilities_are_present() -> None:
    handler = _Stub(
        patterns=("video/mp4",),
        handler_id="video",
        affordance_requires=frozenset({Capability.ASR}),
    )
    registry = MimeTypeRegistry(
        handlers=(as_handler(handler),),
        capabilities=CapabilitySet.of({Capability.ASR: "whisper@1"}),
    )
    names = tuple(a.name for a in registry.available_affordances(as_handler(handler)))
    assert names == ("free", "costly")


def test_resolving_with_no_handler_at_all_raises() -> None:
    registry = MimeTypeRegistry(handlers=(), capabilities=CapabilitySet.empty())
    with pytest.raises(NoHandlerError, match="no handler"):
        registry.resolve(MimeType.parse("video/mp4"))
