"""OCR of a scanned PDF page, against the real model server.

Marked `live` and deselected by default. Run with:
    uv run pytest tests/live -m live -v

These assert on STRUCTURE, never on the model's exact words: that text came
back, that it is not an echo of the prompt, and that the rendition is marked
as a model's reading rather than an extraction. Reading quality is a bench
concern, not a test concern — this project has held that line since its
first spec.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.cache_key import artifact_key
from readeverything.adapters.detection import PuremagicDetector
from readeverything.adapters.hashing import ContentHasher
from readeverything.adapters.local_source import LocalFileSource
from readeverything.adapters.pdfium_probe import PdfiumProbe
from readeverything.adapters.vision_langchain import LangChainVisionModel
from readeverything.adapters.vision_recognizer import _OCR_PROMPT, VisionTextRecognizer
from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.domain.identity import ContentHash
from readeverything.domain.rendition import TextContent
from readeverything.handlers.pdf import PdfHandler
from readeverything.pipeline.perception import Perception
from readeverything.registry.registry import MimeTypeRegistry
from tests.fixtures_pdf import scanned_like

pytestmark = pytest.mark.live


def _perception(tmp_path: Path, vision: LangChainVisionModel) -> Perception:
    (tmp_path / "scan.pdf").write_bytes(scanned_like())
    source = LocalFileSource(root=tmp_path)
    handler = PdfHandler(
        source=source, probe=PdfiumProbe(), recognizer=VisionTextRecognizer(vision=vision)
    )
    return Perception(
        source=source,
        detector=PuremagicDetector(),
        hasher=ContentHasher(source=source),
        registry=MimeTypeRegistry(
            handlers=(handler,),
            capabilities=CapabilitySet.of({Capability.VISION: vision.model_id}),
        ),
        artifacts=InMemoryArtifactStore(),
    )


async def test_a_real_model_reads_a_scanned_page(
    tmp_path: Path, live_vision: LangChainVisionModel
) -> None:
    """The whole scanned-PDF path against a real model rather than a fake.

    Asserts structure, never the model's exact words: that text came back,
    that it is not an echo of the OCR prompt, and that the rendition is marked
    as a model's reading (`degraded`) rather than an extraction.
    """
    perception = _perception(tmp_path, live_vision)
    rendition = await perception.invoke("scan.pdf", "ocr_page", {"page": 1})
    assert isinstance(rendition.content, TextContent)
    text = rendition.content.text.strip()
    assert text
    assert text != _OCR_PROMPT
    assert rendition.degraded


def test_ocr_artifacts_invalidate_when_the_model_changes(live_model_name: str) -> None:
    """Swapping the model must produce a different cache key.

    `capability_fingerprint` carries `VisionModel.model_id`, so two recognisers
    with different model ids must produce different `artifact_key` values for
    the same page — otherwise an index would silently mix OCR readings from
    two different models under one key. Computed directly, the same way
    `test_swapping_the_model_changes_every_cache_key` does for images in
    `tests/live/test_vision_endpoint.py`: this is a property of
    `capability_fingerprint`, not of the network call, so it needs no live
    server to demonstrate — it is here rather than in the unit tier because it
    is the OCR half of the same live-model contract as the rest of this file.
    """

    def key_for(model_id: str) -> str:
        return artifact_key(
            content_hash=ContentHash("a" * 64),
            handler_id=PdfHandler.handler_id,
            handler_version=PdfHandler.handler_version,
            affordance="ocr_page",
            params={"page": 1},
            capabilities=CapabilitySet.of({Capability.VISION: model_id}),
        )

    assert key_for(f"openai/{live_model_name}") != key_for(f"openai/{live_model_name}-swapped")
    assert key_for(f"openai/{live_model_name}") == key_for(f"openai/{live_model_name}")
