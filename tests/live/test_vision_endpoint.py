"""Does the real endpoint behave the way the design assumes?

Marked `live` and deselected by default. Run with:
    uv run pytest tests/live -m live -v

These assert on STRUCTURE, never on model text. What is being validated is that
the transport works, that the model accepts an inlined image, and that the
identity feeding the cache key is real — not that the model describes anything
well. Description quality is a bench concern, not a test.
"""

import io

import pytest
from PIL import Image

from readeverything.adapters.cache_key import artifact_key
from readeverything.adapters.vision_langchain import LangChainVisionModel
from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.domain.identity import ContentHash

pytestmark = pytest.mark.live


def _red_square_png() -> bytes:
    """A plain 64x64 red PNG.

    Deliberately not a 1x1 pixel: a real server's image loader rejects a
    degenerate image before the model ever sees it, which would make a
    payload failure look like an endpoint failure. Small and uniform is
    the property this fixture wants; single-pixel is not.
    """
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


RED_SQUARE_PNG = _red_square_png()


async def test_the_endpoint_answers_with_text(live_vision: LangChainVisionModel) -> None:
    """The transport works and the model accepts an inlined image."""
    answer = await live_vision.describe(
        RED_SQUARE_PNG, "image/png", "Describe this image in one short sentence."
    )
    assert answer.strip()


async def test_the_answer_is_not_an_echo_of_the_prompt(
    live_vision: LangChainVisionModel,
) -> None:
    """Guards the failure where a model returns the prompt back verbatim.

    That would pass a bare truthiness check while proving nothing about vision.
    """
    prompt = "Describe this image in one short sentence."
    answer = await live_vision.describe(RED_SQUARE_PNG, "image/png", prompt)
    assert answer.strip() != prompt


async def test_the_model_id_is_provider_qualified(
    live_vision: LangChainVisionModel, live_model_name: str
) -> None:
    """A bare family name makes 're-derive what the old model touched' unanswerable."""
    assert live_vision.model_id == f"openai/{live_model_name}"


def test_swapping_the_model_changes_every_cache_key(live_model_name: str) -> None:
    """The whole reason `model_id` exists.

    Without this, changing the model silently serves a mixture of descriptions
    produced by two different models under one key.
    """

    def key_for(model_id: str) -> str:
        return artifact_key(
            content_hash=ContentHash("a" * 64),
            handler_id="image",
            handler_version=1,
            affordance="describe_image",
            params={"prompt": "what is this"},
            capabilities=CapabilitySet.of({Capability.VISION: model_id}),
        )

    assert key_for(f"openai/{live_model_name}") != key_for("openai/some-other-model")
    assert key_for(f"openai/{live_model_name}") == key_for(f"openai/{live_model_name}")
