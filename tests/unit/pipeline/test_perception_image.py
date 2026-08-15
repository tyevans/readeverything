"""Capability negotiation over an image, through the full stack, offline."""

import io
from pathlib import Path

import pytest
from PIL import Image

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.detection import PuremagicDetector
from readeverything.adapters.hashing import ContentHasher
from readeverything.adapters.local_source import LocalFileSource
from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.domain.errors import UnknownAffordanceError
from readeverything.domain.rendition import TextContent
from readeverything.handlers.binary import BinaryHandler
from readeverything.handlers.image import ImageHandler
from readeverything.pipeline.perception import Perception
from readeverything.registry.registry import MimeTypeRegistry
from readeverything.testing.fakes import FakeVision


def _perception(tmp_path: Path, *, seeing: bool) -> Perception:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 4), (0, 128, 0)).save(buffer, format="PNG")
    (tmp_path / "photo.png").write_bytes(buffer.getvalue())
    source = LocalFileSource(root=tmp_path)
    vision = FakeVision() if seeing else None
    capabilities = (
        CapabilitySet.of({Capability.VISION: FakeVision().model_id})
        if seeing
        else CapabilitySet.empty()
    )
    return Perception(
        source=source,
        detector=PuremagicDetector(),
        hasher=ContentHasher(source=source),
        registry=MimeTypeRegistry(
            handlers=(
                ImageHandler(source=source, vision=vision),
                BinaryHandler(source=source),
            ),
            capabilities=capabilities,
        ),
        artifacts=InMemoryArtifactStore(),
    )


async def test_a_png_dispatches_to_the_image_handler(tmp_path: Path) -> None:
    card = await _perception(tmp_path, seeing=False).inspect("photo.png")
    assert card.facts["width"] == 8


async def test_without_vision_the_agent_sees_only_crop(tmp_path: Path) -> None:
    card = await _perception(tmp_path, seeing=False).inspect("photo.png")
    assert card.affordance_names() == ("crop_region",)


async def test_with_vision_the_agent_sees_all_three(tmp_path: Path) -> None:
    card = await _perception(tmp_path, seeing=True).inspect("photo.png")
    assert set(card.affordance_names()) == {"crop_region", "describe_image", "ocr"}


async def test_without_vision_describe_is_not_invocable(tmp_path: Path) -> None:
    perception = _perception(tmp_path, seeing=False)
    with pytest.raises(UnknownAffordanceError):
        await perception.invoke("photo.png", "describe_image", {})


async def test_with_vision_describe_is_invocable(tmp_path: Path) -> None:
    rendition = await _perception(tmp_path, seeing=True).invoke("photo.png", "describe_image", {})
    assert isinstance(rendition.content, TextContent)
    assert rendition.content.text
