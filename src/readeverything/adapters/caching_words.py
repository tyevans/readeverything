"""Caching the expensive half of reading a video's words.

TWO CACHES ALREADY EXIST AND NEITHER COVERS THIS. `Perception.invoke` caches a
*rendition* keyed on the whole derivation, which includes the affordance's
parameters — so `read_transcript(0, 120)` and `read_transcript(0, 240)` are
different entries, and each pays a full transcription to produce a different
window of the same transcript. And `represent()`'s cache is a third entry
again. The expensive step is identical in all three; only the slicing differs.

So these wrappers cache one layer down, at the port: the CUES, once per file
per producer, shared by every window and by `represent()`. Measured, that is
the difference between paying ~100 seconds of Whisper on every question about
a file and paying it once.

They are decorators rather than handler logic because a handler that knew
about a cache would be a handler that imports an adapter. Compose them at a
composition root and the handler cannot tell.

KEYING IS NOT THE SAME PROBLEM FOR BOTH, and the difference is worth stating:

- `Transcriber.transcribe` receives the audio BYTES, so its key is a hash of
  exactly what it was asked about. That is content-addressed in the same sense
  `artifact_key` is: a moved or renamed file hits, an edited one misses.
- `CaptionExtractor.extract` receives a PATH, and hashing a 195MB container to
  read its 40KB subtitle track would cost more than the extraction it is
  meant to save. Its key is `(resolved path, size, mtime_ns, track)` instead.
  That is a weaker guarantee — a file edited within the same mtime granularity
  and to the same size would hit a stale entry — and it is chosen knowingly:
  the failure requires a deliberately adversarial write, and the alternative
  makes the cache slower than the thing it caches.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from readeverything.domain.locators import TimeSpan
from readeverything.domain.rendition import CueSource, SpeakerId, TranscriptCue
from readeverything.ports.artifacts import ArtifactStore
from readeverything.ports.captions import CaptionExtractor
from readeverything.ports.transcription import Transcriber

#: Bumped when the encoding below changes shape. An old entry then misses
#: rather than decoding into something with the wrong fields.
_CODEC_VERSION = 1


def encode_cues(cues: tuple[TranscriptCue, ...]) -> bytes:
    """Cues as JSON.

    Hand-rolled rather than `TypeAdapter`, for the reason `rendition_codec`
    gives at length: a union resolved by shape is a silent wrong answer, and
    `CueSource` is exactly the field a shape-guess would drop. Here the risk is
    smaller — one concrete type, no unions — but the cost of being wrong is the
    same, so the source is written and read explicitly rather than defaulted.
    """
    return json.dumps(
        [
            {
                "start_s": cue.span.start_s,
                "end_s": cue.span.end_s,
                "text": cue.text,
                "speaker": cue.speaker,
                "confidence": cue.confidence,
                "source": cue.source.value,
                "version": _CODEC_VERSION,
            }
            for cue in cues
        ]
    ).encode("utf-8")


def decode_cues(data: bytes) -> tuple[TranscriptCue, ...]:
    """Cues from JSON. Raises `ValueError` on anything unrecognised.

    Callers treat a raise as a miss, never as a failure: a corrupt entry in a
    persistent store would otherwise poison this derivation forever, and
    recomputing costs time where trusting costs correctness.
    """
    rows = json.loads(data)
    if not isinstance(rows, list):
        raise ValueError("cached cues were not a list")
    cues: list[TranscriptCue] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("version") != _CODEC_VERSION:
            raise ValueError("cached cue was written by a different codec version")
        speaker = row.get("speaker")
        cues.append(
            TranscriptCue(
                span=TimeSpan(start_s=float(row["start_s"]), end_s=float(row["end_s"])),
                text=str(row["text"]),
                speaker=None if speaker is None else SpeakerId(str(speaker)),
                confidence=row.get("confidence"),
                source=CueSource(row["source"]),
            )
        )
    return tuple(cues)


async def _cached(
    store: ArtifactStore,
    key: str,
    produce: Callable[[], Awaitable[tuple[TranscriptCue, ...] | None]],
) -> tuple[TranscriptCue, ...] | None:
    """Shared get/decode/produce/put, so the two wrappers cannot drift."""
    stored = await store.get(key)
    if stored is not None:
        try:
            return decode_cues(stored)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass  # an unreadable entry is a miss, not a failure
    cues = await produce()
    if cues:
        # An empty result is NOT cached. A transcriber that heard nothing may
        # have failed transiently, and a persistent store has no eviction — so
        # caching the silence would make one bad run permanent.
        await store.put(key, encode_cues(cues))
    return cues


class CachingTranscriber:
    """A `Transcriber` that transcribes each distinct audio payload once.

    Keyed on a hash of the audio itself, so two videos sharing a soundtrack
    share the entry and a re-encoded one does not.
    """

    def __init__(self, *, inner: Transcriber, store: ArtifactStore) -> None:
        self._inner = inner
        self._store = store
        self.model_id = inner.model_id

    async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]:
        digest = hashlib.blake2b(audio, digest_size=32).hexdigest()
        key = f"cues/asr/{self.model_id}/{mime}/{digest}"
        cues = await _cached(self._store, key, lambda: self._inner.transcribe(audio, mime))
        return cues or ()


class CachingCaptionExtractor:
    """A `CaptionExtractor` that reads each track once.

    Cheap already — 0.16s for 848 cues on the reference file — so this exists
    less for the ffmpeg call than for uniformity: `read_transcript` and
    `represent()` should not each spawn a subprocess to produce identical cues,
    and a caller swapping captions for ASR should not also be swapping caching
    behaviour.
    """

    def __init__(self, *, inner: CaptionExtractor, store: ArtifactStore) -> None:
        self._inner = inner
        self._store = store

    async def extract(
        self, path: str, track: int | None = None
    ) -> tuple[TranscriptCue, ...] | None:
        try:
            stat = Path(path).stat()
        except OSError:
            # No file to describe means no key to cache under. Ask the inner
            # extractor, which answers `None` for exactly this case.
            return await self._inner.extract(path, track)
        key = f"cues/captions/{Path(path).resolve()}/{stat.st_size}/{stat.st_mtime_ns}/{track}"
        return await _cached(self._store, key, lambda: self._inner.extract(path, track))
