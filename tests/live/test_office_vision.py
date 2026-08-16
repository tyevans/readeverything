"""`describe_slide_image` against a real model rather than a fake.

Structure only, never the model's words: that an answer came back, that it is
not an echo of the question, and that it is located at the slide the picture
was actually embedded in. Matching how the image and PDF specs treat vision.
"""

from __future__ import annotations

import pytest

from readeverything.adapters.ooxml import SLIDES_MIME
from readeverything.adapters.vision_langchain import LangChainVisionModel
from readeverything.domain.identity import ContentHash, MimeType, SourceRef
from readeverything.domain.locators import PageRef
from readeverything.domain.rendition import TextContent
from readeverything.handlers.office_slides import DescribeSlideImageParams, OfficeSlidesHandler
from readeverything.testing.fakes import FakeSource
from tests.fixtures_office import pptx_bytes

pytestmark = pytest.mark.live

URI = "deck.pptx"
QUESTION = "What is the dominant colour of this image?"


async def test_a_real_model_describes_a_picture_embedded_in_a_slide(
    live_vision: LangChainVisionModel,
) -> None:
    content = pptx_bytes(picture_on=(2,))
    handler = OfficeSlidesHandler(source=FakeSource({URI: content}), vision=live_vision)
    ref = SourceRef(
        uri=URI,
        mime=MimeType.parse(SLIDES_MIME),
        content_hash=ContentHash("0" * 64),
        size_bytes=len(content),
    )

    rendition = await handler.invoke(
        ref,
        "describe_slide_image",
        DescribeSlideImageParams(page=2, index=0, question=QUESTION),
    )

    assert isinstance(rendition.content, TextContent)
    assert rendition.content.text.strip()
    # Not an echo of the prompt: a model that returns the question back is a
    # failure that a "text came back" assertion alone would pass.
    assert rendition.content.text.strip() != QUESTION
    assert rendition.locator == PageRef(2)
    assert not rendition.degraded


async def test_a_real_model_is_never_asked_about_a_slide_with_no_picture(
    live_vision: LangChainVisionModel,
) -> None:
    """The degrade path must short-circuit BEFORE the endpoint is called.

    A handler that sent an empty payload and let the server reject it would
    spend a network round trip to learn something the deck already said.
    """
    content = pptx_bytes()
    handler = OfficeSlidesHandler(source=FakeSource({URI: content}), vision=live_vision)
    ref = SourceRef(
        uri=URI,
        mime=MimeType.parse(SLIDES_MIME),
        content_hash=ContentHash("0" * 64),
        size_bytes=len(content),
    )

    rendition = await handler.invoke(
        ref,
        "describe_slide_image",
        DescribeSlideImageParams(page=1, index=0, question=QUESTION),
    )

    assert rendition.degraded
