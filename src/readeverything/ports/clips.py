"""Watching a bounded range of a video, motion included.

Separate from `VisionModel` rather than a second method on it, because a
server that describes stills need not accept clips — ours did not until
2026-08-15, when the same request that had been failing with
"Failed to load image or audio file" began working — and a handler must be
able to offer frame description while truthfully reporting that it cannot
watch. A protocol with an optional half lies about half of its
implementations.

WHY A CAP IS THE CALLER'S JOB. Measured against llama.cpp b10438 serving
qwen3.8-27b-mtp, a clip costs about 2,180 prompt tokens per second of
DURATION, and that figure cannot be reduced from the client: re-encoding the
source to a lower frame rate produced a byte-identical token count, because
the server resamples by timestamp rather than reading the frames it was
given. Cost is a function of duration alone. A 262k context therefore caps a
single clip near two minutes, and a caller who hands over ten gets a failure
a long way from the mistake — so callers bound the range before calling, and
`video.py` refuses rather than truncates.

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
