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
    from readeverything.adapters.binary_probe import BinaryProbe as BinaryProbe
    from readeverything.adapters.detection import PuremagicDetector as PuremagicDetector
    from readeverything.adapters.ffmpeg_audio import FfmpegAudio as FfmpegAudio
    from readeverything.adapters.hashing import ContentHasher as ContentHasher
    from readeverything.adapters.hashing import StatMemo as StatMemo
    from readeverything.adapters.local_source import LocalFileSource as LocalFileSource
    from readeverything.adapters.model_probe import ModelProbe as ModelProbe
    from readeverything.adapters.pdfium_probe import PdfiumProbe as PdfiumProbe
    from readeverything.adapters.probing import discover as discover
    from readeverything.adapters.semaphore_limiter import SemaphoreLimiter as SemaphoreLimiter
    from readeverything.adapters.vision_langchain import (
        LangChainVisionModel as LangChainVisionModel,
    )
    from readeverything.adapters.vision_langchain import (
        build_openai_vision_model as build_openai_vision_model,
    )
    from readeverything.adapters.vision_recognizer import (
        VisionTextRecognizer as VisionTextRecognizer,
    )
    from readeverything.adapters.whisper_transcriber import (
        WhisperTranscriber as WhisperTranscriber,
    )
    from readeverything.agent.results import ToolResult as ToolResult
    from readeverything.agent.tools import build_tools as build_tools
    from readeverything.composition import build_perception as build_perception
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
    from readeverything.domain.observation import Event as Event
    from readeverything.domain.observation import OperationFinished as OperationFinished
    from readeverything.domain.observation import OperationProgressed as OperationProgressed
    from readeverything.domain.observation import OperationStarted as OperationStarted
    from readeverything.domain.rendition import Budget as Budget
    from readeverything.domain.rendition import Degradation as Degradation
    from readeverything.domain.rendition import ImageContent as ImageContent
    from readeverything.domain.rendition import Rendered as Rendered
    from readeverything.domain.rendition import Rendition as Rendition
    from readeverything.domain.rendition import SpeakerId as SpeakerId
    from readeverything.domain.rendition import StructuredContent as StructuredContent
    from readeverything.domain.rendition import TextContent as TextContent
    from readeverything.domain.rendition import TranscriptCue as TranscriptCue
    from readeverything.handlers.audio import AudioHandler as AudioHandler
    from readeverything.handlers.binary import BinaryHandler as BinaryHandler
    from readeverything.handlers.image import ImageHandler as ImageHandler
    from readeverything.handlers.pdf import PdfHandler as PdfHandler
    from readeverything.handlers.text import TextHandler as TextHandler
    from readeverything.handlers.video import VideoHandler as VideoHandler
    from readeverything.pipeline.perception import Perception as Perception
    from readeverything.pipeline.resolution import ResolutionMemo as ResolutionMemo
    from readeverything.ports.artifacts import ArtifactStore as ArtifactStore
    from readeverything.ports.audio import AudioExtractor as AudioExtractor
    from readeverything.ports.detection import MimeDetector as MimeDetector
    from readeverything.ports.handler import MediaHandler as MediaHandler
    from readeverything.ports.hashing import ContentHashing as ContentHashing
    from readeverything.ports.limits import Limiter as Limiter
    from readeverything.ports.observation import Observer as Observer
    from readeverything.ports.probe import CapabilityProbe as CapabilityProbe
    from readeverything.ports.probe_media import DocumentFacts as DocumentFacts
    from readeverything.ports.probe_media import MediaProbe as MediaProbe
    from readeverything.ports.recognition import TextRecognizer as TextRecognizer
    from readeverything.ports.source import FileSource as FileSource
    from readeverything.ports.source import SourceReader as SourceReader
    from readeverything.ports.transcription import Transcriber as Transcriber
    from readeverything.ports.vision import VisionModel as VisionModel
    from readeverything.registry.registry import MimeTypeRegistry as MimeTypeRegistry
    from readeverything.registry.registry import NoHandlerError as NoHandlerError
    from readeverything.testing.artifact_compliance import (
        ArtifactStoreCompliance as ArtifactStoreCompliance,
    )
    from readeverything.testing.fakes import CountingLimiter as CountingLimiter
    from readeverything.testing.fakes import FakeDiarizer as FakeDiarizer
    from readeverything.testing.fakes import FakeSource as FakeSource
    from readeverything.testing.fakes import FakeTranscriber as FakeTranscriber
    from readeverything.testing.fakes import FakeVision as FakeVision
    from readeverything.testing.fakes import FakeVisionRefusing as FakeVisionRefusing
    from readeverything.testing.fakes import RaisingObserver as RaisingObserver
    from readeverything.testing.fakes import RecordingObserver as RecordingObserver
    from readeverything.testing.handler_compliance import (
        MediaHandlerCompliance as MediaHandlerCompliance,
    )

_LAZY: dict[str, str] = {
    "Affordance": "readeverything.domain.affordance",
    "ArtifactStore": "readeverything.ports.artifacts",
    "ArtifactStoreCompliance": "readeverything.testing.artifact_compliance",
    "AudioExtractor": "readeverything.ports.audio",
    "AudioHandler": "readeverything.handlers.audio",
    "BBox": "readeverything.domain.locators",
    "BinaryHandler": "readeverything.handlers.binary",
    "BinaryProbe": "readeverything.adapters.binary_probe",
    "Budget": "readeverything.domain.rendition",
    "ByteRange": "readeverything.domain.locators",
    "Capability": "readeverything.domain.capability",
    "CapabilityProbe": "readeverything.ports.probe",
    "CapabilitySet": "readeverything.domain.capability",
    "CapabilityUnavailableError": "readeverything.domain.errors",
    "Card": "readeverything.domain.card",
    "CharSpan": "readeverything.domain.locators",
    "ContentHash": "readeverything.domain.identity",
    "ContentHasher": "readeverything.adapters.hashing",
    "ContentHashing": "readeverything.ports.hashing",
    "CountingLimiter": "readeverything.testing.fakes",
    "Degradation": "readeverything.domain.rendition",
    "DetailLevel": "readeverything.domain.affordance",
    "DocumentFacts": "readeverything.ports.probe_media",
    "DomainError": "readeverything.domain.errors",
    "Event": "readeverything.domain.observation",
    "FakeDiarizer": "readeverything.testing.fakes",
    "FakeSource": "readeverything.testing.fakes",
    "FakeTranscriber": "readeverything.testing.fakes",
    "FakeVision": "readeverything.testing.fakes",
    "FakeVisionRefusing": "readeverything.testing.fakes",
    "FfmpegAudio": "readeverything.adapters.ffmpeg_audio",
    "FileSource": "readeverything.ports.source",
    "FilesystemArtifactStore": "readeverything.adapters.artifact_store",
    "ImageContent": "readeverything.domain.rendition",
    "ImageHandler": "readeverything.handlers.image",
    "InMemoryArtifactStore": "readeverything.adapters.artifact_store",
    "InfrastructureError": "readeverything.domain.errors",
    "LangChainVisionModel": "readeverything.adapters.vision_langchain",
    "Limiter": "readeverything.ports.limits",
    "LocalFileSource": "readeverything.adapters.local_source",
    "LocatorMap": "readeverything.domain.locator_map",
    "LocatorSegment": "readeverything.domain.locator_map",
    "MediaHandler": "readeverything.ports.handler",
    "MediaHandlerCompliance": "readeverything.testing.handler_compliance",
    "MediaKind": "readeverything.domain.identity",
    "MediaProbe": "readeverything.ports.probe_media",
    "MimeDetector": "readeverything.ports.detection",
    "MimeType": "readeverything.domain.identity",
    "MimeTypeRegistry": "readeverything.registry.registry",
    "ModelProbe": "readeverything.adapters.model_probe",
    "NoHandlerError": "readeverything.registry.registry",
    "Observer": "readeverything.ports.observation",
    "OperationFinished": "readeverything.domain.observation",
    "OperationProgressed": "readeverything.domain.observation",
    "OperationStarted": "readeverything.domain.observation",
    "PageRef": "readeverything.domain.locators",
    "PdfHandler": "readeverything.handlers.pdf",
    "PdfiumProbe": "readeverything.adapters.pdfium_probe",
    "Perception": "readeverything.pipeline.perception",
    "PuremagicDetector": "readeverything.adapters.detection",
    "RaisingObserver": "readeverything.testing.fakes",
    "ReadEverythingError": "readeverything.domain.errors",
    "RecordingObserver": "readeverything.testing.fakes",
    "Rendered": "readeverything.domain.rendition",
    "Rendition": "readeverything.domain.rendition",
    "ResolutionMemo": "readeverything.pipeline.resolution",
    "Segment": "readeverything.domain.card",
    "SemaphoreLimiter": "readeverything.adapters.semaphore_limiter",
    "SourceReader": "readeverything.ports.source",
    "SourceRef": "readeverything.domain.identity",
    "SourceUnreadableError": "readeverything.domain.errors",
    "SpeakerId": "readeverything.domain.rendition",
    "StatMemo": "readeverything.adapters.hashing",
    "StructuredContent": "readeverything.domain.rendition",
    "TextContent": "readeverything.domain.rendition",
    "TextHandler": "readeverything.handlers.text",
    "TextRecognizer": "readeverything.ports.recognition",
    "TimeSpan": "readeverything.domain.locators",
    "ToolResult": "readeverything.agent.results",
    "Transcriber": "readeverything.ports.transcription",
    "TranscriptCue": "readeverything.domain.rendition",
    "UnknownAffordanceError": "readeverything.domain.errors",
    "VideoHandler": "readeverything.handlers.video",
    "VisionModel": "readeverything.ports.vision",
    "VisionTextRecognizer": "readeverything.adapters.vision_recognizer",
    "WhisperTranscriber": "readeverything.adapters.whisper_transcriber",
    "build_openai_vision_model": "readeverything.adapters.vision_langchain",
    "build_perception": "readeverything.composition",
    "build_tools": "readeverything.agent.tools",
    "discover": "readeverything.adapters.probing",
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
