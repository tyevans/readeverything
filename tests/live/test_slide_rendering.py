"""`describe_slide` against a real model and a real LibreOffice.

The acceptance sentence's last clause: asking to describe slide 4 renders slide
4, hands it to the vision path, and the answer cites the slide.

Structure only, never the model's words — matching `test_office_vision.py`.
What is under test is that a converted page reaches an endpoint in a form it
accepts and that the answer is located at the slide asked about. Whether the
model is right about the slide is a bench concern.

Distinct from `test_office_vision.py`, which asks about a picture the author
EMBEDDED. This asks about the slide as an audience saw it, which is a question
no text extraction and no embedded image can answer.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.local_source import LocalFileSource
from readeverything.adapters.ooxml import SLIDES_MIME
from readeverything.adapters.soffice_renderer import SofficeRenderer
from readeverything.adapters.vision_langchain import LangChainVisionModel
from readeverything.domain.identity import ContentHash, MimeType, SourceRef
from readeverything.domain.locators import PageRef
from readeverything.domain.rendition import TextContent
from readeverything.handlers.office_slides import DescribeSlideParams, OfficeSlidesHandler
from tests.fixtures_office import pptx_bytes

pytestmark = pytest.mark.live

URI = "deck.pptx"
QUESTION = "Is there any text visible on this slide? Answer yes or no."


@pytest.fixture
def deck(tmp_path: Path) -> SourceRef:
    if shutil.which("soffice") is None:
        pytest.skip("soffice not available")
    content = pptx_bytes(titles=("Opening position", "The numbers", "What we decided"))
    (tmp_path / URI).write_bytes(content)
    return SourceRef(
        uri=URI,
        mime=MimeType.parse(SLIDES_MIME),
        content_hash=ContentHash("0" * 64),
        size_bytes=len(content),
    )


async def test_a_real_model_describes_a_converted_slide(
    live_vision: LangChainVisionModel, deck: SourceRef, tmp_path: Path
) -> None:
    handler = OfficeSlidesHandler(
        source=LocalFileSource(root=tmp_path),
        vision=live_vision,
        renderer=SofficeRenderer(
            artifacts=InMemoryArtifactStore(),
            profile_root=tmp_path / "profile",
            timeout_s=180.0,
        ),
    )

    rendition = await handler.invoke(
        deck, "describe_slide", DescribeSlideParams(page=3, question=QUESTION)
    )

    assert isinstance(rendition.content, TextContent)
    assert rendition.content.text.strip()
    # Not an echo of the prompt: a model that hands the question back is a
    # failure a "text came back" assertion alone would pass.
    assert rendition.content.text.strip() != QUESTION
    assert rendition.locator == PageRef(3), "the answer cites the slide asked about"
    assert not rendition.degraded


async def test_the_answer_carries_the_conversion_provenance(
    live_vision: LangChainVisionModel, deck: SourceRef, tmp_path: Path
) -> None:
    """The model looked at a rendering. An answer about how the type looks is
    an answer about the converter's font substitutions, and a reader weighing
    it needs to know that."""
    handler = OfficeSlidesHandler(
        source=LocalFileSource(root=tmp_path),
        vision=live_vision,
        renderer=SofficeRenderer(
            artifacts=InMemoryArtifactStore(),
            profile_root=tmp_path / "profile",
            timeout_s=180.0,
        ),
    )

    rendition = await handler.invoke(
        deck, "describe_slide", DescribeSlideParams(page=1, question=QUESTION)
    )

    assert any("font" in d.detail for d in rendition.degradations)
