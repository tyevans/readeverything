"""The same negotiation, against the real model server.

Structure only — never assert on what the model says.
"""

import io
from pathlib import Path

import pytest
from PIL import Image

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.detection import PuremagicDetector
from readeverything.adapters.hashing import ContentHasher
from readeverything.adapters.local_source import LocalFileSource
from readeverything.adapters.vision_langchain import LangChainVisionModel
from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.domain.rendition import TextContent
from readeverything.handlers.binary import BinaryHandler
from readeverything.handlers.image import ImageHandler
from readeverything.pipeline.perception import Perception
from readeverything.registry.registry import MimeTypeRegistry

pytestmark = pytest.mark.live


def _perception(tmp_path: Path, vision: LangChainVisionModel) -> Perception:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (0, 128, 0)).save(buffer, format="PNG")
    (tmp_path / "photo.png").write_bytes(buffer.getvalue())
    source = LocalFileSource(root=tmp_path)
    return Perception(
        source=source,
        detector=PuremagicDetector(),
        hasher=ContentHasher(source=source),
        registry=MimeTypeRegistry(
            handlers=(
                ImageHandler(source=source, vision=vision),
                BinaryHandler(source=source),
            ),
            capabilities=CapabilitySet.of({Capability.VISION: vision.model_id}),
        ),
        artifacts=InMemoryArtifactStore(),
    )


async def test_a_real_model_describes_a_real_image(
    tmp_path: Path, live_vision: LangChainVisionModel
) -> None:
    perception = _perception(tmp_path, live_vision)
    card = await perception.inspect("photo.png")
    assert "describe_image" in card.affordance_names()
    rendition = await perception.invoke("photo.png", "describe_image", {})
    assert isinstance(rendition.content, TextContent)
    assert rendition.content.text.strip()


async def test_represent_against_a_real_model_reports_no_degradation(
    tmp_path: Path, live_vision: LangChainVisionModel
) -> None:
    """If the real model answers, nothing should be degraded at all.

    Asserting the exact set rather than filtering for one label: a degradation
    under any other name would otherwise pass silently, on the one test that
    is meant to prove a real model works.
    """
    from readeverything.domain.rendition import Budget

    perception = _perception(tmp_path, live_vision)
    rendered = await perception.represent("photo.png", Budget(max_chars=None))
    assert rendered.locator_map.length == len(rendered.text)
    assert rendered.degradations == (), (
        f"expected no degradations, got: {[d.what for d in rendered.degradations]}"
    )
