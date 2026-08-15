"""Does the real endpoint behave the way the design assumes?

Marked `live` and deselected by default. Run with:
    uv run pytest tests/live -m live -v

These assert on STRUCTURE, never on model text. What is being validated is that
the transport works, that the model accepts an inlined image, and that the
identity feeding the cache key is real — not that the model describes anything
well. Description quality is a bench concern, not a test.
"""

import pytest

from readeverything.adapters.cache_key import artifact_key
from readeverything.adapters.vision_langchain import LangChainVisionModel
from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.domain.identity import ContentHash

pytestmark = pytest.mark.live

#: A 1x1 red PNG. Small enough that a failure is the endpoint, not the payload.
RED_PIXEL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c636060f80f000101010018dd8db000"
    "00000049454e44ae426082"
)


async def test_the_endpoint_answers_with_text(live_vision: LangChainVisionModel) -> None:
    """The transport works and the model accepts an inlined image."""
    answer = await live_vision.describe(
        RED_PIXEL_PNG, "image/png", "Describe this image in one short sentence."
    )
    assert answer.strip()


async def test_the_answer_is_not_an_echo_of_the_prompt(
    live_vision: LangChainVisionModel,
) -> None:
    """Guards the failure where a model returns the prompt back verbatim.

    That would pass a bare truthiness check while proving nothing about vision.
    """
    prompt = "Describe this image in one short sentence."
    answer = await live_vision.describe(RED_PIXEL_PNG, "image/png", prompt)
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
