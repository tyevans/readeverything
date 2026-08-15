"""One function between a directory and a working `Perception`.

Everything here is a convenience over the public constructors and nothing more.
If this module can do something a caller assembling the pieces by hand cannot,
that is a bug in the constructors, not a feature of this file — which is why it
takes the same arguments they do and holds no state of its own.

It reads no environment variables. Every input is an argument, so two
differently-configured instances can run in one process.
"""

from __future__ import annotations

from pathlib import Path

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.binary_probe import BinaryProbe
from readeverything.adapters.detection import PuremagicDetector
from readeverything.adapters.hashing import ContentHasher, StatMemo
from readeverything.adapters.local_source import LocalFileSource
from readeverything.adapters.model_probe import ModelProbe
from readeverything.adapters.probing import discover
from readeverything.domain.capability import CapabilitySet
from readeverything.handlers.binary import BinaryHandler
from readeverything.handlers.text import TextHandler
from readeverything.pipeline.perception import Perception
from readeverything.pipeline.resolution import ResolutionMemo
from readeverything.ports.artifacts import ArtifactStore
from readeverything.ports.handler import MediaHandler
from readeverything.ports.probe import CapabilityProbe
from readeverything.ports.source import SourceReader
from readeverything.ports.vision import VisionModel
from readeverything.registry.registry import MimeTypeRegistry


def _optional_image_handler(source: SourceReader, vision: VisionModel | None) -> list[MediaHandler]:
    """`ImageHandler` when Pillow is importable, nothing when it is not.

    Pillow lives behind the `images` extra. A base install must yield a working
    `Perception` that handles text and binary — narrower, not broken — so the
    import failure is a registration decision here rather than an exception
    reaching the caller.
    """
    try:
        from readeverything.handlers.image import ImageHandler
    except ImportError:
        return []
    return [ImageHandler(source=source, vision=vision)]


async def build_perception(
    root: Path | str,
    *,
    vision: VisionModel | None = None,
    capabilities: CapabilitySet | None = None,
    artifacts: ArtifactStore | None = None,
    probe_binaries: bool = True,
) -> Perception:
    """A `Perception` over `root`, with everything else defaulted.

    `capabilities` given explicitly is used verbatim and nothing is probed —
    a test must be able to declare any capability set without depending on what
    happens to be installed on the machine running it.
    """
    source = LocalFileSource(root=root)
    if capabilities is None:
        probes: list[CapabilityProbe] = [ModelProbe(vision=vision)]
        if probe_binaries:
            probes.append(BinaryProbe())
        capabilities = await discover(probes=probes)
    handlers: list[MediaHandler] = [
        TextHandler(source=source),
        *_optional_image_handler(source, vision),
        # The fallback claims "*", so it must be last: the registry breaks a
        # rank tie by registration order, and a fallback registered first would
        # shadow nothing but would rank ahead of an equally-specific match.
        BinaryHandler(source=source),
    ]
    return Perception(
        source=source,
        detector=PuremagicDetector(),
        hasher=ContentHasher(source=source, memo=StatMemo()),
        registry=MimeTypeRegistry(handlers=handlers, capabilities=capabilities),
        artifacts=InMemoryArtifactStore() if artifacts is None else artifacts,
        memo=ResolutionMemo(),
    )
