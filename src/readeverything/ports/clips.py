"""Watching a bounded range of a video, motion included.

Separate from `VisionModel` rather than a second method on it, because a
server that describes stills need not accept clips — ours did not until
2026-08-15, when the same request that had been failing with
"Failed to load image or audio file" began working — and a handler must be
able to offer frame description while truthfully reporting that it cannot
watch. A protocol with an optional half lies about half of its
implementations.

WHY A CAP IS THE CALLER'S JOB. Measured against llama.cpp b10438 serving
qwen3.8-27b-mtp on 2026-08-15, a clip costs about 2,100 prompt tokens per
second of DURATION — but only below roughly 40 seconds. Past that the line
stops: 60s, 300s and 1200s all cost exactly 88,033, a frame cap rather than a
rate, and 120s costs 218,435 for reasons that remain unexplained.

The rate cannot be reduced from the client either way: re-encoding the source
to a lower frame rate produces an identical token count, because the server
resamples by timestamp rather than reading the frames it was given.

What binds is the SERVER'S CONTEXT, not the clip's length. On a 65,536-token
window the largest clip that fits is about 30 seconds; on a larger one, the
88,033 plateau means even an hour would fit, which is how this model can
credibly claim hour-scale video. So the cap belongs to whoever knows the
window — the caller — and `video.py` refuses rather than truncates, because a
watch that silently covered part of a range would be a claim about time the
model never saw.

There is a second, earlier ceiling worth knowing: a 600-second clip encoded at
source resolution is ~336MB and the server rejects the request with HTTP 413
before context is ever consulted. Downscaling the clip fixes the transport
limit and does nothing for the context limit; they are independent.

`model_id` feeds `CapabilitySet.fingerprint()`, exactly as `VisionModel`'s
does, so swapping the model changes every artifact cache key derived from it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ClipModel(Protocol):
    #: Provider-qualified and versioned, e.g. "openai/qwen3.8-27b-mtp@2026-08".
    #: A bare family name makes "re-derive everything the old model touched"
    #: unanswerable.
    model_id: str

    async def watch(self, clip: bytes, mime: str, prompt: str) -> str:
        """Answer `prompt` about the video in `clip`.

        Returns the model's text. Raises `InfrastructureError` if the model
        answered with nothing usable — an empty completion is a real and
        common failure here rather than a valid answer, and more likely than
        it is for a still: a clip is many frames of prompt, so a reasoning
        model has more to think about before it starts writing.
        """
        ...
