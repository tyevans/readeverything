"""readeverything — give an agent eyes into a filesystem.

Imports are lazy (PEP 562). `import readeverything` loads no adapter and no
optional driver, so the base install stays light and a caller who wants only
the domain types never pays for langchain.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from readeverything.adapters.artifact_store import (
        FilesystemArtifactStore as FilesystemArtifactStore,
    )
    from readeverything.adapters.artifact_store import (
        InMemoryArtifactStore as InMemoryArtifactStore,
    )
    from readeverything.adapters.detection import PuremagicDetector as PuremagicDetector
    from readeverything.adapters.hashing import ContentHasher as ContentHasher
    from readeverything.adapters.hashing import StatMemo as StatMemo
    from readeverything.adapters.local_source import LocalFileSource as LocalFileSource
    from readeverything.domain.affordance import Affordance as Affordance
    from readeverything.domain.affordance import DetailLevel as DetailLevel
    from readeverything.domain.capability import Capability as Capability
    from readeverything.domain.capability import CapabilitySet as CapabilitySet
    from readeverything.domain.card import Card as Card
    from readeverything.domain.card import Segment as Segment
    from readeverything.domain.errors import (
        CapabilityUnavailableError as CapabilityUnavailableError,
    )
    from readeverything.domain.errors import DomainError as DomainError
    from readeverything.domain.errors import InfrastructureError as InfrastructureError
    from readeverything.domain.errors import ReadEverythingError as ReadEverythingError
    from readeverything.domain.errors import SourceUnreadableError as SourceUnreadableError
    from readeverything.domain.errors import UnknownAffordanceError as UnknownAffordanceError
    from readeverything.domain.identity import ContentHash as ContentHash
    from readeverything.domain.identity import MediaKind as MediaKind
    from readeverything.domain.identity import MimeType as MimeType
    from readeverything.domain.identity import SourceRef as SourceRef
    from readeverything.domain.locator_map import LocatorMap as LocatorMap
    from readeverything.domain.locator_map import LocatorSegment as LocatorSegment
    from readeverything.domain.locators import BBox as BBox
    from readeverything.domain.locators import ByteRange as ByteRange
    from readeverything.domain.locators import CharSpan as CharSpan
    from readeverything.domain.locators import PageRef as PageRef
    from readeverything.domain.locators import TimeSpan as TimeSpan
    from readeverything.domain.rendition import Budget as Budget
    from readeverything.domain.rendition import Degradation as Degradation
    from readeverything.domain.rendition import ImageContent as ImageContent
    from readeverything.domain.rendition import Rendered as Rendered
    from readeverything.domain.rendition import Rendition as Rendition
    from readeverything.domain.rendition import SpeakerId as SpeakerId
    from readeverything.domain.rendition import StructuredContent as StructuredContent
    from readeverything.domain.rendition import TextContent as TextContent
    from readeverything.domain.rendition import TranscriptCue as TranscriptCue
    from readeverything.handlers.binary import BinaryHandler as BinaryHandler
    from readeverything.handlers.text import TextHandler as TextHandler
    from readeverything.pipeline.perception import Perception as Perception
    from readeverything.registry.registry import MimeTypeRegistry as MimeTypeRegistry
    from readeverything.registry.registry import NoHandlerError as NoHandlerError

_LAZY: dict[str, str] = {
    "Affordance": "readeverything.domain.affordance",
    "BBox": "readeverything.domain.locators",
    "BinaryHandler": "readeverything.handlers.binary",
    "Budget": "readeverything.domain.rendition",
    "ByteRange": "readeverything.domain.locators",
    "Capability": "readeverything.domain.capability",
    "CapabilitySet": "readeverything.domain.capability",
    "CapabilityUnavailableError": "readeverything.domain.errors",
    "Card": "readeverything.domain.card",
    "CharSpan": "readeverything.domain.locators",
    "ContentHash": "readeverything.domain.identity",
    "ContentHasher": "readeverything.adapters.hashing",
    "Degradation": "readeverything.domain.rendition",
    "DetailLevel": "readeverything.domain.affordance",
    "DomainError": "readeverything.domain.errors",
    "FilesystemArtifactStore": "readeverything.adapters.artifact_store",
    "ImageContent": "readeverything.domain.rendition",
    "InMemoryArtifactStore": "readeverything.adapters.artifact_store",
    "InfrastructureError": "readeverything.domain.errors",
    "LocalFileSource": "readeverything.adapters.local_source",
    "LocatorMap": "readeverything.domain.locator_map",
    "LocatorSegment": "readeverything.domain.locator_map",
    "MediaKind": "readeverything.domain.identity",
    "MimeType": "readeverything.domain.identity",
    "MimeTypeRegistry": "readeverything.registry.registry",
    "NoHandlerError": "readeverything.registry.registry",
    "PageRef": "readeverything.domain.locators",
    "Perception": "readeverything.pipeline.perception",
    "PuremagicDetector": "readeverything.adapters.detection",
    "ReadEverythingError": "readeverything.domain.errors",
    "Rendered": "readeverything.domain.rendition",
    "Rendition": "readeverything.domain.rendition",
    "Segment": "readeverything.domain.card",
    "SourceRef": "readeverything.domain.identity",
    "SourceUnreadableError": "readeverything.domain.errors",
    "SpeakerId": "readeverything.domain.rendition",
    "StatMemo": "readeverything.adapters.hashing",
    "StructuredContent": "readeverything.domain.rendition",
    "TextContent": "readeverything.domain.rendition",
    "TextHandler": "readeverything.handlers.text",
    "TimeSpan": "readeverything.domain.locators",
    "TranscriptCue": "readeverything.domain.rendition",
    "UnknownAffordanceError": "readeverything.domain.errors",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_LAZY[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return __all__
