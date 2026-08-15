"""Transcribe media files into the artifact cache, ahead of any question.

Separates the slow deterministic step from the interesting one. Nothing here
calls a model endpoint: it is ffmpeg plus Whisper on CPU, writing cues into the
same `FilesystemArtifactStore` that `ask_agent.py` reads, so a later agent run
gets the words for free.

Captions are tried first, exactly as the handler's precedence does — a file
that carries its own words should never be transcribed.

    uv run python scripts/warm_transcripts.py media/*.mp4
"""

import asyncio
import sys
import time
from pathlib import Path

from readeverything import (
    CachingCaptionExtractor,
    CachingTranscriber,
    FfmpegCaptions,
    FilesystemArtifactStore,
    WhisperTranscriber,
)
from readeverything.adapters.ffmpeg_audio import FfmpegAudio
from readeverything.adapters.ffprobe_streams import FfprobeStreams

WHISPER_DIR = "models/faster-whisper-small"
CACHE_DIR = ".cache/readeverything"
EXTRACTED_MIME = "audio/wav"


async def warm(path: str) -> None:
    store = FilesystemArtifactStore(root=CACHE_DIR)
    captions = CachingCaptionExtractor(inner=FfmpegCaptions(), store=store)
    facts = await FfprobeStreams().probe(path)
    print(f"\n=== {Path(path).name} — {facts.duration_s / 60:.1f} min ===", flush=True)

    text_tracks = facts.text_subtitle_streams
    if text_tracks:
        track = next(i for i, s in enumerate(facts.subtitle_streams) if s.is_text)
        started = time.monotonic()
        cues = await captions.extract(path, track)
        if cues:
            print(
                f"  captions: {len(cues)} cues in {time.monotonic() - started:.1f}s (no model)",
                flush=True,
            )
            return
        print("  captions: declared a text track but yielded nothing; falling back", flush=True)

    if not Path(WHISPER_DIR).is_dir():
        print(f"  no captions and no weights at {WHISPER_DIR}; skipped", flush=True)
        return

    started = time.monotonic()
    audio = await FfmpegAudio().extract(path)
    if audio is None:
        print("  no audio track; nothing to transcribe", flush=True)
        return
    print(f"  audio: {len(audio) / 1e6:.1f} MB in {time.monotonic() - started:.1f}s", flush=True)

    transcriber = CachingTranscriber(inner=WhisperTranscriber(model_dir=WHISPER_DIR), store=store)
    started = time.monotonic()
    cues = await transcriber.transcribe(audio, EXTRACTED_MIME)
    elapsed = time.monotonic() - started
    words = sum(len(c.text.split()) for c in cues)
    print(
        f"  asr: {len(cues)} cues, {words} words in {elapsed:.1f}s "
        f"({elapsed / facts.duration_s:.0%} of realtime)",
        flush=True,
    )
    for cue in cues[:3]:
        print(f"    [{cue.span.start_s:7.1f}] {cue.text[:70]}", flush=True)


async def main() -> None:
    for path in sys.argv[1:]:
        await warm(path)


asyncio.run(main())
