"""Turning pixels into words.

The whole multimodal strategy of this library rests on this one method: content
is described into text at the edge, and only the description enters an index or
a knowledge graph. A frame never becomes a claim — what a model asserts about
the frame does, with the frame's locator as provenance.

`model_id` is not used for dispatch. It feeds `CapabilitySet.fingerprint()`, so
that swapping the model changes every artifact cache key derived from it.
Without it the cache would serve a mixture of descriptions produced by two
different models, which is invisible until someone reads two answers side by
side and cannot explain the difference in voice.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VisionModel(Protocol):
    #: Provider-qualified and versioned, e.g. "openai/qwen3.8-27b-mtp@2026-08".
    #: A bare family name makes "re-derive everything the old model touched"
    #: unanswerable.
    model_id: str

    async def describe(self, data: bytes, mime: str, prompt: str) -> str:
        """Answer `prompt` about the image in `data`.

        Returns the model's text. Raises `InfrastructureError` if the model
        answered with nothing usable — see the adapter for why an empty
        completion is a real and common failure rather than a valid answer.
        """
        ...
