"""`Transcriber` over `faster-whisper`, run entirely against local weights.

`local_files_only=True` is what makes "downloads nothing implicitly" enforced
rather than merely conventional — a local path alone does not prevent
`faster-whisper`/`huggingface_hub` from attempting a lookup; this flag is what
actually stops it. A missing or malformed `model_dir` therefore fails at
construction, loudly, which is the composition root's job to surface: the
HANDLER is what must never raise, and it is what catches `InfrastructureError`
from a transcriber it calls; construction happening once, at startup, is a
different concern from a call failing mid-request.

`WhisperModel.transcribe` accepts a `BinaryIO`, so the WAV bytes handed to
`transcribe()` are wrapped in `io.BytesIO` rather than written to a temp file
— nothing touches disk here that the caller didn't already put there.

`WhisperModel.transcribe` is synchronous and CPU-bound; it runs inside
`asyncio.to_thread` so it doesn't block the event loop.
"""

from __future__ import annotations

import asyncio
import io

from faster_whisper import WhisperModel

from readeverything.domain.errors import InfrastructureError
from readeverything.domain.locators import TimeSpan
from readeverything.domain.rendition import TranscriptCue


class WhisperTranscriber:
    """Local speech-to-text via `faster-whisper`, no network at any point."""

    def __init__(self, *, model_dir: str, compute_type: str = "int8", device: str = "cpu") -> None:
        try:
            self._model = WhisperModel(
                model_dir,
                device=device,
                compute_type=compute_type,
                local_files_only=True,
            )
        except Exception as exc:
            raise InfrastructureError(
                f"could not load a faster-whisper model from {model_dir!r}: {exc}"
            ) from exc
        # Identifies what actually ran — directory name plus compute type —
        # so it feeds CapabilitySet.fingerprint() meaningfully: swapping
        # either changes the cache key derived from it.
        self.model_id = f"faster-whisper/{model_dir.rstrip('/').rsplit('/', 1)[-1]}@{compute_type}"

    async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]:
        """Transcribe `audio` into ordered, non-overlapping cues.

        Runs the synchronous, CPU-bound `WhisperModel.transcribe` in a thread.
        `avg_logprob` is not treated as a confidence: it's an average
        token log-probability, whisper's own diagnostic, not a measured
        certainty about the transcript. `confidence` is left `None` on every
        cue — the field exists to admit ignorance, and presenting a derived
        number as a measurement is exactly the defect this project avoids.
        `None` needs no justification; it is the safer default.
        """
        buffer = io.BytesIO(audio)
        segments, _info = await asyncio.to_thread(
            self._model.transcribe, buffer, word_timestamps=False
        )
        cues = []
        for segment in segments:
            # Whisper occasionally emits a segment with start == end.
            # TimeSpan rejects a zero-width span, and widening it into one
            # would assert a duration nothing observed — so it's dropped.
            if segment.start >= segment.end:
                continue
            cues.append(
                TranscriptCue(
                    span=TimeSpan(segment.start, segment.end),
                    text=segment.text.strip(),
                    speaker=None,
                    confidence=None,
                )
            )
        return tuple(cues)
