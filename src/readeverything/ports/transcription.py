"""Turning speech into words.

Symmetric with `ports/vision.py`: audio bytes are turned into text at the
edge, and only the text enters an index or a knowledge graph — the audio
itself never becomes a claim, only what a model transcribed from it does,
with the cue's `TimeSpan` as provenance.

Unlike `VisionModel.describe` and `AudioExtractor.extract`, `transcribe` MAY
raise `InfrastructureError`: a handler must never raise, but a transcriber is
allowed to, because the handler is what catches it and degrades.

`model_id` is not used for dispatch. It feeds `CapabilitySet.fingerprint()`,
so that swapping the model changes every artifact cache key derived from it —
the same reason `VisionModel` carries one.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from readeverything.domain.rendition import TranscriptCue


@runtime_checkable
class Transcriber(Protocol):
    #: Provider-qualified and versioned, e.g. "faster-whisper/base.en@int8".
    #: A bare family name makes "re-derive everything the old model touched"
    #: unanswerable.
    model_id: str

    async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]:
        """Transcribe `audio` into ordered, non-overlapping cues.

        May raise `InfrastructureError` if transcription failed outright — the
        handler catches it and degrades; this method is not required to.
        """
        ...
