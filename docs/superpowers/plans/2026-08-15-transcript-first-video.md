# Transcript-First Video Understanding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make an agent read a video's words before looking at its pictures, and give it a bounded way to look closely when the words are not enough.

**Architecture:** Captions become visible on the card and extractable through a new `CaptionExtractor` port; `TranscriptCue` gains a `CueSource` so authored text is never rendered as speech; `represent()` prefers captions to ASR. A separate `ClipModel` port carries `watch_segment` over llama.cpp's `input_video` content part with a load-bearing duration cap. A pure sampler then chooses frames from caption text rather than a fixed interval.

**Tech Stack:** Python 3.12+, pytest (asyncio_mode=auto), ffmpeg/ffprobe subprocesses, langchain-core/langchain-openai for model adapters, pydantic for affordance params.

**Spec:** `docs/superpowers/specs/2026-08-15-readeverything-transcript-first-video-design.md`

## Global Constraints

- **Nothing in `src/` reads the environment.** Configuration arrives as constructor arguments. `tests/unit/test_reads_no_environment.py` enforces this.
- **Handlers never raise about their input.** Every injected call is guarded; failures become `Degradation`, not exceptions. Adapters may raise `InfrastructureError`; handlers catch.
- **Ports use `None` as a normal answer**, never an error signal — matching `FrameExtractor.frame_at` and `AudioExtractor.extract`.
- **No adapter imports in handlers.** Everything arrives by injection. `tests/unit/test_dependencies_stay_confined.py` enforces layering.
- **Subprocess calls never build a shell string.** Use `asyncio.create_subprocess_exec` with the path as one argv element, wrapped in `asyncio.wait_for` with kill-and-reap.
- **`TimeSpan` forbids `start >= end`.** Any cue or span that would violate this is dropped or widened deliberately, never passed through.
- **Measured costs (live server, llama.cpp b10438, `qwen3.8-27b-mtp`):** one still ≈ 1,089 prompt tokens; video ≈ 2,180 prompt tokens per second of clip and NOT reducible client-side; `describe_frame` on 720x480 ≈ 65s.
- **Test media** lives in `media/` (gitignored). `media/mystery_subject.mp4` is the reference file: 2244.5s, 720x480 h264, aac, `mov_text` captions at subtitle index 1, `dvd_subtitle` bitmaps at subtitle index 0.

---

## Stage 1 — Captions

### Task 1: `StreamInfo` admits subtitle streams

**Files:**
- Modify: `src/readeverything/ports/streams.py:18-28`
- Modify: `src/readeverything/adapters/ffprobe_streams.py:56-68`
- Test: `tests/unit/ports/test_streams.py`, `tests/unit/adapters/test_ffprobe_streams.py`

**Interfaces:**
- Produces: `StreamInfo(kind: Literal["video","audio","subtitle"], codec: str, width, height, frame_rate, sample_rate, channels, language: str | None = None, is_text: bool = False)`
- Produces: `TEXT_SUBTITLE_CODECS: frozenset[str]` in `ports/streams.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/adapters/test_ffprobe_streams.py
from readeverything.adapters.ffprobe_streams import _parse_facts


def test_subtitle_streams_are_kept_and_classified():
    raw = {
        "format": {"duration": "10.0", "format_name": "mov,mp4"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "r_frame_rate": "30/1"},
            {"codec_type": "subtitle", "codec_name": "mov_text", "tags": {"language": "eng"}},
            {"codec_type": "subtitle", "codec_name": "dvd_subtitle"},
        ],
    }
    facts = _parse_facts(raw)
    subs = [s for s in facts.streams if s.kind == "subtitle"]
    assert [s.codec for s in subs] == ["mov_text", "dvd_subtitle"]
    assert subs[0].is_text is True
    assert subs[0].language == "eng"
    assert subs[1].is_text is False
    assert subs[1].language is None


def test_data_streams_are_still_dropped():
    raw = {
        "format": {"duration": "10.0", "format_name": "mov,mp4"},
        "streams": [{"codec_type": "data", "codec_name": "bin_data"}],
    }
    assert _parse_facts(raw).streams == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/adapters/test_ffprobe_streams.py -k subtitle -v`
Expected: FAIL — subtitle streams are dropped, so `subs` is empty.

- [ ] **Step 3: Write minimal implementation**

```python
# src/readeverything/ports/streams.py
#: Subtitle codecs that carry characters. Everything else a container calls a
#: subtitle carries pixels (`dvd_subtitle`, `hdmv_pgs_subtitle`) and needs OCR,
#: which costs about a hundred times what reading text costs. Presenting the
#: two as one kind of thing would send an agent down the expensive path when
#: the cheap one was sitting beside it.
TEXT_SUBTITLE_CODECS = frozenset(
    {"mov_text", "subrip", "srt", "ass", "ssa", "webvtt", "text"}
)


@dataclass(frozen=True, slots=True)
class StreamInfo:
    """One stream within a media container."""

    kind: Literal["video", "audio", "subtitle"]
    codec: str
    width: int | None
    height: int | None
    frame_rate: float | None
    sample_rate: int | None
    channels: int | None
    #: From `tags.language`; `None` when the container does not say. Which
    #: track to read is a real choice on a multi-track file.
    language: str | None = None
    #: Whether `codec` is in `TEXT_SUBTITLE_CODECS`. Always False for video
    #: and audio streams — the field is about subtitles.
    is_text: bool = False
```

```python
# src/readeverything/adapters/ffprobe_streams.py
from readeverything.ports.streams import TEXT_SUBTITLE_CODECS, MediaFacts, StreamInfo


def _parse_stream(raw: dict[str, Any]) -> StreamInfo | None:
    codec_type = raw.get("codec_type")
    if codec_type not in ("video", "audio", "subtitle"):
        return None
    codec = str(raw.get("codec_name", ""))
    tags = raw.get("tags")
    language = tags.get("language") if isinstance(tags, dict) else None
    return StreamInfo(
        kind=codec_type,
        codec=codec,
        width=raw.get("width"),
        height=raw.get("height"),
        frame_rate=_parse_frame_rate(raw.get("r_frame_rate")),
        sample_rate=int(raw["sample_rate"]) if raw.get("sample_rate") is not None else None,
        channels=raw.get("channels"),
        language=str(language) if language is not None else None,
        is_text=codec_type == "subtitle" and codec in TEXT_SUBTITLE_CODECS,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/adapters/test_ffprobe_streams.py tests/unit/ports -v`
Expected: PASS, including pre-existing tests (new fields have defaults, so existing `StreamInfo(...)` calls still work).

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/ports/streams.py src/readeverything/adapters/ffprobe_streams.py tests/unit/adapters/test_ffprobe_streams.py
git commit -m "Stop dropping subtitle streams on the floor"
```

---

### Task 2: The card says captions exist

**Files:**
- Modify: `src/readeverything/ports/streams.py` (MediaFacts properties)
- Modify: `src/readeverything/handlers/video.py` (card facts)
- Test: `tests/unit/handlers/test_video_handler.py`

**Interfaces:**
- Consumes: `StreamInfo.kind == "subtitle"`, `StreamInfo.is_text` from Task 1
- Produces: `MediaFacts.subtitle_streams`, `MediaFacts.text_subtitle_streams`
- Produces: card facts keys `subtitle_streams: int`, `text_captions: bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/handlers/test_video_handler.py
async def test_card_reports_text_captions(video_handler, ref_with_captions):
    """An agent that is not told captions exist will pay a vision model
    to learn what one ffmpeg call would have told it for free."""
    card = await video_handler.describe(ref_with_captions)
    assert card.facts["subtitle_streams"] == 2
    assert card.facts["text_captions"] is True
```

Add this fixture beside the existing video handler fixtures:

```python
@pytest.fixture
def ref_with_captions(tmp_path):
    """A ref whose probe reports one text and one bitmap subtitle track."""
    return _make_ref(tmp_path)  # reuse the module's existing ref helper
```

and give the fake probe used by `video_handler` these streams:

```python
StreamInfo(kind="subtitle", codec="mov_text", width=None, height=None,
           frame_rate=None, sample_rate=None, channels=None,
           language="eng", is_text=True),
StreamInfo(kind="subtitle", codec="dvd_subtitle", width=None, height=None,
           frame_rate=None, sample_rate=None, channels=None,
           language=None, is_text=False),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/handlers/test_video_handler.py -k captions -v`
Expected: FAIL with `KeyError: 'subtitle_streams'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/readeverything/ports/streams.py — on MediaFacts
    @property
    def subtitle_streams(self) -> tuple[StreamInfo, ...]:
        return tuple(s for s in self.streams if s.kind == "subtitle")

    @property
    def text_subtitle_streams(self) -> tuple[StreamInfo, ...]:
        """Subtitle streams that extract to characters rather than pixels."""
        return tuple(s for s in self.subtitle_streams if s.is_text)
```

In `video.py`'s card construction, alongside the existing `video_streams` / `audio_streams` entries:

```python
            "subtitle_streams": len(facts.subtitle_streams),
            "text_captions": bool(facts.text_subtitle_streams),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/handlers/test_video_handler.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/ports/streams.py src/readeverything/handlers/video.py tests/unit/handlers/test_video_handler.py
git commit -m "Tell the card's reader that captions are there"
```

---

### Task 3: `CueSource` — said is not written

**Files:**
- Modify: `src/readeverything/domain/rendition.py:54-61`
- Modify: `src/readeverything/handlers/video.py:96` and `:915-926`
- Test: `tests/unit/domain/test_rendition.py`, `tests/unit/handlers/test_video_handler.py`

**Interfaces:**
- Produces: `CueSource` enum with members `SAID` and `CAPTIONED` in `domain/rendition.py`
- Produces: `TranscriptCue.source: CueSource = CueSource.SAID`
- Produces: `CAPTION_MARKER = "(caption)"` in `handlers/video.py`, beside `SPEECH_MARKER`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/domain/test_rendition.py
from readeverything.domain.rendition import CueSource, TranscriptCue
from readeverything.domain.locators import TimeSpan


def test_cue_source_defaults_to_said():
    """Every producer that existed before captions was a transcriber. A
    default of CAPTIONED would mislabel all of them."""
    cue = TranscriptCue(span=TimeSpan(0.0, 1.0), text="hi", speaker=None, confidence=None)
    assert cue.source is CueSource.SAID
```

```python
# tests/unit/handlers/test_video_handler.py
async def test_captions_render_as_captions_not_speech(...):
    """A caption is authored text: condensed, sometimes non-verbal
    ("[music playing]"), sometimes translated. Rendering it as speech
    asserts someone said words they did not say."""
    cues = (
        TranscriptCue(span=TimeSpan(0.0, 2.0), text="[music playing]", speaker=None,
                      confidence=None, source=CueSource.CAPTIONED),
    )
    # drive represent() with a transcriber-substitute returning these cues
    rendered = await handler.represent(ref, Budget(max_chars=None))
    assert "(caption) [music playing]" in rendered.text
    assert "(speech) [music playing]" not in rendered.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/domain/test_rendition.py -k cue_source tests/unit/handlers/test_video_handler.py -k captions_render -v`
Expected: FAIL with `ImportError: cannot import name 'CueSource'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/readeverything/domain/rendition.py
from enum import Enum


class CueSource(Enum):
    """Where a cue's words came from.

    `video.py` already refuses to let a citation attribute speech to a
    picture. Captions need the same care for the same reason: they are
    authored text, frequently condensed for reading speed rather than
    verbatim, often describing non-speech sound, sometimes translated. A
    reader deciding how much to trust a quoted line needs to know which of
    these produced it.
    """

    SAID = "said"
    CAPTIONED = "captioned"


@dataclass(frozen=True, slots=True)
class TranscriptCue:
    """One utterance, with a speaker when diarization is available."""

    span: TimeSpan
    text: str
    speaker: SpeakerId | None
    confidence: float | None
    #: Defaults to SAID because every producer that predates this field is a
    #: transcriber. Only the caption adapter sets CAPTIONED.
    source: CueSource = CueSource.SAID
```

```python
# src/readeverything/handlers/video.py
#: What marks an authored caption in the merged timeline, as `SPEECH_MARKER`
#: marks heard speech. Two markers rather than one because the difference is
#: a difference in evidence, not in formatting.
CAPTION_MARKER = "(caption)"


def _spoken(cue: TranscriptCue) -> str:
    """One cue as a line of the merged timeline, marked by its source."""
    marker = SPEECH_MARKER if cue.source is CueSource.SAID else CAPTION_MARKER
    speaker = f"{cue.speaker} " if cue.speaker is not None else ""
    body = " ".join(cue.text.split())
    if not body:
        return f"{marker} (no text was produced for this cue)"
    return f"{marker} {speaker}{body}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit -v`
Expected: PASS. The whole unit suite runs because `TranscriptCue` is constructed in many tests; the default keeps them valid.

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/domain/rendition.py src/readeverything/handlers/video.py tests/unit
git commit -m "A caption is not a thing someone said"
```

---

### Task 4: `CaptionExtractor` port and its ffmpeg adapter

**Files:**
- Create: `src/readeverything/ports/captions.py`
- Create: `src/readeverything/adapters/ffmpeg_captions.py`
- Test: `tests/unit/adapters/test_ffmpeg_captions.py`

**Interfaces:**
- Consumes: `TranscriptCue`, `CueSource` (Task 3)
- Produces: `CaptionExtractor` protocol — `async def extract(self, path: str, track: int | None = None) -> tuple[TranscriptCue, ...] | None`
- Produces: `FfmpegCaptions(timeout_s: float = 30.0)` implementing it
- Produces: `parse_srt(text: str) -> tuple[TranscriptCue, ...]` — module-level, pure, testable without a subprocess

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/adapters/test_ffmpeg_captions.py
from readeverything.adapters.ffmpeg_captions import parse_srt
from readeverything.domain.rendition import CueSource

REAL_SAMPLE = """1
00:00:00,868 --> 00:00:03,202
<font size="24">[music playing]</font>

2
00:00:16,484 --> 00:00:18,617
<font size="24">When I say computer,
you probably</font>
"""


def test_parses_timing_and_strips_markup():
    """Every cue in the reference file is wrapped in <font size="24">.
    Indexed unstripped, that markup becomes content."""
    cues = parse_srt(REAL_SAMPLE)
    assert len(cues) == 2
    assert cues[0].span.start_s == pytest.approx(0.868)
    assert cues[0].span.end_s == pytest.approx(3.202)
    assert cues[0].text == "[music playing]"
    assert cues[1].text == "When I say computer, you probably"
    assert all(c.source is CueSource.CAPTIONED for c in cues)


def test_drops_degenerate_cues():
    """TimeSpan forbids start >= end. A malformed cue must not reach it."""
    cues = parse_srt("1\n00:00:05,000 --> 00:00:05,000\nzero width\n")
    assert cues == ()


def test_empty_input_is_no_cues_not_an_error():
    assert parse_srt("") == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/adapters/test_ffmpeg_captions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'readeverything.adapters.ffmpeg_captions'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/readeverything/ports/captions.py
"""Reading the words a container already carries.

A caption track is the cheapest thing in this library: ~8,700 tokens and one
second for a 37-minute lecture, against 12 model calls and five minutes to
learn the same thing from pixels. It exists in the port list so that a
handler can prefer it, and so that a caller who has no ffmpeg still gets a
handler that degrades rather than one that breaks.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from readeverything.domain.rendition import TranscriptCue


@runtime_checkable
class CaptionExtractor(Protocol):
    async def extract(
        self, path: str, track: int | None = None
    ) -> tuple[TranscriptCue, ...] | None:
        """`path`'s caption cues, or `None` if it has no text caption track.

        `None` means "nothing to read" — a normal answer, the same convention
        as `FrameExtractor.frame_at` and `AudioExtractor.extract`. A missing
        file, a bitmap-only subtitle track and a container with no subtitles
        all return `None` rather than raising.

        `track` indexes the container's SUBTITLE streams (0-based, as
        ffmpeg's `-map 0:s:N` does), not its streams overall. `None` takes
        the first text track.
        """
        ...
```

```python
# src/readeverything/adapters/ffmpeg_captions.py
"""`CaptionExtractor` over `ffmpeg`.

Converting to SRT rather than reading the native codec is deliberate: ffmpeg
normalises `mov_text`, `ass`, `webvtt` and the rest into one grammar, so this
module parses one format instead of six.

Real caption tracks carry markup. Every cue in the reference file is wrapped
in `<font size="24">`, which would be indexed as content if it were not
stripped — the kind of defect that is invisible until someone reads a
citation and finds a tag in it.
"""

from __future__ import annotations

import asyncio
import contextlib
import re

from readeverything.domain.locators import TimeSpan
from readeverything.domain.rendition import CueSource, TranscriptCue

_TIMING = re.compile(
    r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})\s*-->\s*(\d+):(\d\d):(\d\d)[,.](\d{1,3})"
)
_MARKUP = re.compile(r"<[^>]+>")


def _seconds(hours: str, minutes: str, secs: str, millis: str) -> float:
    return int(hours) * 3600 + int(minutes) * 60 + int(secs) + int(millis.ljust(3, "0")) / 1000.0


def parse_srt(text: str) -> tuple[TranscriptCue, ...]:
    """SRT text as cues. Degenerate and unparseable blocks are dropped.

    Dropped rather than raised: a caption track is a best-effort artifact of
    whoever authored the disc, and one malformed block must not cost the
    other 847.
    """
    cues: list[TranscriptCue] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        match = _TIMING.search(block)
        if match is None:
            continue
        start = _seconds(*match.group(1, 2, 3, 4))
        end = _seconds(*match.group(5, 6, 7, 8))
        if end <= start:
            continue  # TimeSpan forbids it, and a zero-width cue says nothing
        body = block[match.end() :]
        body = " ".join(_MARKUP.sub("", body).split())
        if not body:
            continue
        cues.append(
            TranscriptCue(
                span=TimeSpan(start_s=start, end_s=end),
                text=body,
                speaker=None,
                confidence=None,
                source=CueSource.CAPTIONED,
            )
        )
    return tuple(cues)


class FfmpegCaptions:
    """A container's caption track as cues, or `None` if it has none."""

    def __init__(self, *, timeout_s: float = 30.0) -> None:
        self._timeout_s = timeout_s

    async def extract(
        self, path: str, track: int | None = None
    ) -> tuple[TranscriptCue, ...] | None:
        index = 0 if track is None else track
        try:
            process = await asyncio.create_subprocess_exec(
                "ffmpeg", "-v", "error", "-i", path,
                "-map", f"0:s:{index}", "-c:s", "srt", "-f", "srt", "-",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError:
            return None
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=self._timeout_s)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            return None
        if process.returncode != 0:
            # No such track, a bitmap track ffmpeg will not convert to text,
            # an unreadable file: all "nothing to read", not errors.
            return None
        cues = parse_srt(stdout.decode("utf-8", errors="replace"))
        return cues or None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/adapters/test_ffmpeg_captions.py -v`
Expected: PASS

- [ ] **Step 5: Verify against the real file**

Run:
```bash
uv run python -c "
import asyncio
from readeverything.adapters.ffmpeg_captions import FfmpegCaptions
cues = asyncio.run(FfmpegCaptions().extract('media/mystery_subject.mp4', track=1))
print(len(cues), cues[0].text, cues[0].span)
"
```
Expected: 848 cues, first text `[music playing]`, span starting at 0.868. If this prints `None`, the track index is wrong — check `ffprobe` output before changing the adapter.

- [ ] **Step 6: Commit**

```bash
git add src/readeverything/ports/captions.py src/readeverything/adapters/ffmpeg_captions.py tests/unit/adapters/test_ffmpeg_captions.py
git commit -m "Read the caption track the container was carrying all along"
```

---

### Task 5: Precedence — captions beat ASR

**Files:**
- Modify: `src/readeverything/handlers/video.py:169-181` (constructor), `:556-640` (`_cues`)
- Test: `tests/unit/handlers/test_video_handler.py`

**Interfaces:**
- Consumes: `CaptionExtractor` (Task 4), `MediaFacts.text_subtitle_streams` (Task 2)
- Produces: `VideoHandler(..., captions: CaptionExtractor | None = None)`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/handlers/test_video_handler.py
async def test_captions_are_preferred_to_transcription(...):
    """Captions are free, already aligned, and written by someone who could
    hear the audio. Paying an ASR model when they exist is waste."""
    handler = VideoHandler(..., captions=FakeCaptions(cues=CAPTION_CUES),
                           transcriber=FakeTranscriber(cues=ASR_CUES), ...)
    rendered = await handler.represent(ref, Budget(max_chars=None))
    assert "(caption)" in rendered.text
    assert "(speech)" not in rendered.text


async def test_falls_back_to_asr_without_a_caption_track(...):
    handler = VideoHandler(..., captions=FakeCaptions(cues=None),
                           transcriber=FakeTranscriber(cues=ASR_CUES), ...)
    rendered = await handler.represent(ref, Budget(max_chars=None))
    assert "(speech)" in rendered.text


async def test_reports_captions_it_could_not_reach(...):
    """A file with captions read by ASR anyway is a worse answer produced at
    higher cost. Silence about that is how it stays that way."""
    handler = VideoHandler(..., captions=None,
                           transcriber=FakeTranscriber(cues=ASR_CUES), ...)  # probe says text_captions
    rendered = await handler.represent(ref_with_captions, Budget(max_chars=None))
    assert any(d.what == "captions not read" for d in rendered.degradations)
```

Add to `src/readeverything/testing/fakes.py`:

```python
class FakeCaptions:
    """A `CaptionExtractor` that returns what it was given."""

    def __init__(self, *, cues: tuple[TranscriptCue, ...] | None = ()) -> None:
        self._cues = cues
        self.calls: list[tuple[str, int | None]] = []

    async def extract(self, path: str, track: int | None = None):
        self.calls.append((path, track))
        return self._cues
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/handlers/test_video_handler.py -k "captions_are_preferred or falls_back or could_not_reach" -v`
Expected: FAIL with `TypeError: VideoHandler.__init__() got an unexpected keyword argument 'captions'`.

- [ ] **Step 3: Write minimal implementation**

Add `captions: CaptionExtractor | None = None` to `VideoHandler.__init__`, store as `self._captions`, and put this at the top of `_cues`, before the existing transcriber logic:

```python
        # PRECEDENCE. Captions first: they cost one ffmpeg call against a
        # model's whole run, they arrive already aligned to this timeline, and
        # a human who could hear the audio wrote them. ASR is the fallback for
        # files that carry no words of their own.
        text_tracks = facts.text_subtitle_streams
        if text_tracks and self._captions is not None:
            track = next(
                (i for i, s in enumerate(facts.subtitle_streams) if s.is_text), 0
            )
            try:
                captioned = await self._captions.extract(path, track)
            except Exception:
                captioned = None
            if captioned:
                cues, dropped = clamp_cues_to_duration(captioned, facts.duration_s)
                return cues, _dropped_degradations(dropped, facts.duration_s)
            # Fall through to ASR: a track that would not extract is exactly
            # the case the fallback exists for.
        if text_tracks and self._captions is None:
            degradation = Degradation(
                what="captions not read",
                detail=(
                    f"the container carries {len(text_tracks)} text caption track(s) but no "
                    "caption extractor is wired, so the words were read the expensive way "
                    "or not at all"
                ),
            )
            cues, asr_degradations = await self._transcribed_cues(path, facts)
            return cues, (degradation, *asr_degradations)
        return await self._transcribed_cues(path, facts)
```

Rename the existing transcriber body of `_cues` to `_transcribed_cues` with the same signature and return type, unchanged otherwise. Extract the existing "cues outside the file" block into `_dropped_degradations(dropped: int, duration_s: float) -> tuple[Degradation, ...]` so both paths report it identically.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/handlers/test_video_handler.py tests/integration -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/handlers/video.py src/readeverything/testing/fakes.py tests/unit/handlers/test_video_handler.py
git commit -m "Prefer the words the file already has"
```

---

### Task 6: Wire captions through composition

**Files:**
- Modify: `src/readeverything/composition.py:106-176` (`_video_handler`, `build_perception`)
- Modify: `src/readeverything/__init__.py` (exports)
- Test: `tests/unit/test_composition.py`, `tests/unit/test_public_surface.py`

**Interfaces:**
- Produces: `build_perception(..., captions: CaptionExtractor | None = None)`
- Produces: exports `CaptionExtractor`, `FfmpegCaptions`, `CueSource` from `readeverything`
- Note: when `captions is None` and ffmpeg is present, `build_perception` wires `FfmpegCaptions()` by default — captions cost nothing to offer, unlike a model.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_composition.py
async def test_captions_are_wired_by_default_when_ffmpeg_is_present(tmp_path):
    """Unlike a vision model, a caption extractor needs no configuration and
    costs nothing to offer. A caller should not have to know to ask."""
    perception = await build_perception(tmp_path)
    handler = _video_handler_from(perception)
    assert handler._captions is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_composition.py -k captions_are_wired -v`
Expected: FAIL — `AttributeError: 'VideoHandler' object has no attribute '_captions'` is already fixed by Task 5, so this fails on `assert None is not None`.

- [ ] **Step 3: Write minimal implementation**

In `_video_handler`, accept `captions: CaptionExtractor | None` and pass it to `VideoHandler`. In `build_perception`, add the parameter and default it exactly as the frame/audio extractors are defaulted when `Capability.FFMPEG` is available:

```python
    captions=FfmpegCaptions() if captions is None and has_ffmpeg else captions,
```

Add `CaptionExtractor`, `FfmpegCaptions` and `CueSource` to `__init__.py`'s exports and `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit tests/integration -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/composition.py src/readeverything/__init__.py tests
git commit -m "Wire captions by default; they cost nothing to offer"
```

---

### Task 7: End-to-end — the same question, answered from words

**Files:**
- Create: `tests/integration/test_captions.py`
- Test: real generated container with an embedded caption track

**Interfaces:**
- Consumes: everything from Tasks 1-6

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_captions.py
"""Captions through the whole stack, on a container ffmpeg actually made."""

import asyncio
import subprocess
from pathlib import Path

import pytest

from readeverything import Budget, build_perception

SRT = """1
00:00:00,500 --> 00:00:02,000
first thing said

2
00:00:03,000 --> 00:00:04,500
second thing said
"""


@pytest.fixture
def captioned_video(tmp_path: Path) -> Path:
    srt = tmp_path / "s.srt"
    srt.write_text(SRT)
    out = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc=size=64x48:rate=5:duration=5",
         "-i", str(srt), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:s", "mov_text", str(out)],
        check=True,
    )
    return out


async def test_timeline_reads_the_captions(captioned_video):
    perception = await build_perception(captioned_video.parent)
    rendered = await perception.represent(captioned_video.name, Budget(max_chars=None))
    assert "(caption) first thing said" in rendered.text
    assert "(caption) second thing said" in rendered.text
    # Every character maps to a span, or a hit cannot be cited.
    assert rendered.locator_map.covers(len(rendered.text))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_captions.py -v`
Expected: FAIL if any wiring is incomplete; PASS is the goal of this task.

- [ ] **Step 3: Fix whatever it surfaces**

No new code is planned here. This task exists to prove Tasks 1-6 compose. If it fails, the fix belongs in the task that owns the broken piece — go back and add a unit test there first.

- [ ] **Step 4: Measure the real win**

Run:
```bash
timeout 900 .venv/bin/python -u scratch/ask_agent.py "What is mystery_subject.mp4 about?" > run-captions.log 2>&1
```
(using the harness from `/tmp/.../scratchpad/ask_agent.py`, copied into the repo as `scripts/ask_agent.py`)

Record in the commit message: wall-clock time, number of vision calls, and whether the answer matches the known-correct one (a beginner Python course, Python 3.5.1, PyCharm, circa early 2016). The vision-only baseline was 4m54s and 12 vision calls.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_captions.py scripts/ask_agent.py
git commit -m "Prove the words path end to end"
```

---

## Stage 2 — `watch_segment`

### Task 8: `ClipModel` port

**Files:**
- Create: `src/readeverything/ports/clips.py`
- Test: `tests/unit/ports/test_clips.py`

**Interfaces:**
- Produces: `ClipModel` protocol — `model_id: str`, `async def watch(self, clip: bytes, mime: str, prompt: str) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/ports/test_clips.py
from readeverything.ports.clips import ClipModel


def test_a_minimal_implementation_satisfies_the_protocol():
    class Impl:
        model_id = "x/y"

        async def watch(self, clip: bytes, mime: str, prompt: str) -> str:
            return "ok"

    assert isinstance(Impl(), ClipModel)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/ports/test_clips.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/readeverything/ports/clips.py
"""Watching a bounded range of a video, motion included.

Separate from `VisionModel` rather than a method on it, because a server that
describes stills need not accept clips — ours did not until 2026-08-15 — and
a handler must be able to offer frame description while truthfully reporting
that it cannot watch. One protocol with an optional half is a protocol that
lies about half of its implementations.

`model_id` feeds `CapabilitySet.fingerprint()`, exactly as `VisionModel`'s
does, so swapping the model changes every artifact cache key derived from it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ClipModel(Protocol):
    #: Provider-qualified and versioned, e.g. "openai/qwen3.8-27b-mtp@2026-08".
    model_id: str

    async def watch(self, clip: bytes, mime: str, prompt: str) -> str:
        """Answer `prompt` about the video in `clip`.

        Raises `InfrastructureError` if the model answered with nothing
        usable. Callers bound the clip's DURATION before calling: cost is
        ~2,180 prompt tokens per second and cannot be reduced from the client.
        """
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/ports/test_clips.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/ports/clips.py tests/unit/ports/test_clips.py
git commit -m "A port for watching, distinct from a port for looking"
```

---

### Task 9: `LangChainClipModel` over `input_video`

**Files:**
- Create: `src/readeverything/adapters/clip_langchain.py`
- Test: `tests/unit/adapters/test_clip_langchain.py`

**Interfaces:**
- Consumes: `ClipModel` (Task 8)
- Produces: `LangChainClipModel(chat: BaseChatModel, model_id: str)`
- Produces: `build_openai_clip_model(base_url, model, api_key="not-needed", timeout_s=300.0, max_tokens=1500)`
- Reuses: `_flatten` from `adapters/vision_langchain.py` — import it rather than copying; the reasoning-channel and content-block cases are identical and a second copy would drift.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/adapters/test_clip_langchain.py
async def test_sends_an_input_video_content_part():
    """`input_video` is llama.cpp's extension, NOT OpenAI's `video_url`.
    Verified against the live server on 2026-08-15."""
    chat = RecordingChat(reply="a rainbow band scrolls")
    model = LangChainClipModel(chat=chat, model_id="openai/test")
    out = await model.watch(b"\x00\x01", "video/mp4", "what changes?")
    assert out == "a rainbow band scrolls"
    part = chat.last_message.content[1]
    assert part["type"] == "input_video"
    assert part["input_video"]["data"] == base64.b64encode(b"\x00\x01").decode()


async def test_empty_completion_is_a_failure_not_an_answer():
    """A reasoning model that spends its budget thinking returns empty
    content. Measured: a 2-frame call at max_tokens=300 returned 300 tokens
    of reasoning and no answer."""
    model = LangChainClipModel(chat=RecordingChat(reply=""), model_id="openai/test")
    with pytest.raises(InfrastructureError, match="empty completion"):
        await model.watch(b"\x00", "video/mp4", "?")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/adapters/test_clip_langchain.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/readeverything/adapters/clip_langchain.py
"""A `ClipModel` over llama.cpp's `input_video` content part.

`input_video` is a llama.cpp extension, not part of OpenAI's schema — the
OpenAI-shaped alternative (`video_url`) is what vLLM accepts, and the two
servers do not agree. Verified working against llama.cpp b10438 on
2026-08-15; before that build the same request returned
"Failed to load image or audio file" because ffmpeg was not reachable from
the server process.

The empty-completion and content-block handling is `vision_langchain`'s,
imported rather than copied: the failure modes are identical and two copies
would drift apart at the first fix.
"""

from __future__ import annotations

import base64

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from readeverything.adapters.vision_langchain import _flatten
from readeverything.domain.errors import InfrastructureError


class LangChainClipModel:
    """Describes a clip by sending it as an `input_video` content part."""

    def __init__(self, *, chat: BaseChatModel, model_id: str) -> None:
        self._chat = chat
        self.model_id = model_id

    async def watch(self, clip: bytes, mime: str, prompt: str) -> str:
        encoded = base64.b64encode(clip).decode("ascii")
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "input_video", "input_video": {"data": encoded}},
            ]
        )
        try:
            response = await self._chat.ainvoke([message])
        except Exception as exc:
            raise InfrastructureError(f"clip model call failed: {exc}") from exc
        flattened = _flatten(response.content)
        if flattened is None:
            raise InfrastructureError(
                f"clip model {self.model_id} returned an unrecognised content shape: "
                f"{type(response.content).__name__}"
            )
        text = flattened.strip()
        if not text:
            raise InfrastructureError(
                f"clip model {self.model_id} returned an empty completion; "
                f"a reasoning model may have spent its budget before answering"
            )
        return text


def build_openai_clip_model(
    *,
    base_url: str,
    model: str,
    api_key: str = "not-needed",
    timeout_s: float = 300.0,
    max_tokens: int = 1500,
) -> LangChainClipModel:
    """Build a clip model against an OpenAI-compatible endpoint.

    `max_tokens` defaults higher than `build_openai_vision_model`'s 1024:
    measured, a multi-frame call at 300 spent the whole budget on reasoning
    and returned nothing, and a clip is many frames.
    """
    from langchain_openai import ChatOpenAI

    chat = ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,  # type: ignore[arg-type]
        timeout=timeout_s,
        max_completion_tokens=max_tokens,
    )
    return LangChainClipModel(chat=chat, model_id=f"openai/{model}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/adapters/test_clip_langchain.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/adapters/clip_langchain.py tests/unit/adapters/test_clip_langchain.py
git commit -m "Speak llama.cpp's input_video, not OpenAI's video_url"
```

---

### Task 10: `ClipExtractor` — cutting a range out

**Files:**
- Create: `src/readeverything/ports/clip_source.py`
- Create: `src/readeverything/adapters/ffmpeg_clip.py`
- Test: `tests/unit/adapters/test_ffmpeg_clip.py`, `tests/integration/test_real_binaries.py`

**Interfaces:**
- Produces: `ClipExtractor` protocol — `async def clip(self, path: str, start_s: float, end_s: float) -> bytes | None`
- Produces: `FfmpegClip(timeout_s: float = 120.0)` implementing it
- Produces: `CLIP_MIME = "video/mp4"` in `adapters/ffmpeg_clip.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_real_binaries.py
async def test_clips_a_range_out_of_a_container(tmp_path):
    src = _make_test_video(tmp_path, duration=6)  # existing helper
    data = await FfmpegClip().clip(str(src), 2.0, 4.0)
    assert data is not None and data[4:8] == b"ftyp"  # a real mp4 box


async def test_a_range_past_the_end_is_none_not_an_error(tmp_path):
    src = _make_test_video(tmp_path, duration=2)
    assert await FfmpegClip().clip(str(src), 30.0, 32.0) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_real_binaries.py -k clip -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/readeverything/adapters/ffmpeg_clip.py
"""`ClipExtractor` over `ffmpeg`.

`-ss` before `-i` seeks the input, which is fast and accurate enough here:
the caller already knows the range it asked for, and `watch_segment` reports
the requested span as its locator rather than the one ffmpeg landed on.

`+faststart` is deliberately NOT used. It was measured on 2026-08-15: MOOV
atom position made no difference to whether llama.cpp could decode the clip,
and paying a second pass to move it would be cargo cult.
"""

from __future__ import annotations

import asyncio
import contextlib

CLIP_MIME = "video/mp4"


class FfmpegClip:
    """A bounded range of a container as mp4 bytes, or `None`."""

    def __init__(self, *, timeout_s: float = 120.0) -> None:
        self._timeout_s = timeout_s

    async def clip(self, path: str, start_s: float, end_s: float) -> bytes | None:
        if end_s <= start_s:
            return None
        try:
            process = await asyncio.create_subprocess_exec(
                "ffmpeg", "-v", "error",
                "-ss", f"{start_s:.3f}", "-i", path, "-t", f"{end_s - start_s:.3f}",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "pipe:1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError:
            return None
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=self._timeout_s)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            return None
        if process.returncode != 0 or not stdout:
            return None
        return stdout
```

The port file mirrors `ports/captions.py`'s docstring conventions and declares `clip` with the signature above.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/integration/test_real_binaries.py -k clip -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/ports/clip_source.py src/readeverything/adapters/ffmpeg_clip.py tests
git commit -m "Cut a bounded range out of a container"
```

---

### Task 11: `watch_segment`, and the cap that makes it safe

**Files:**
- Modify: `src/readeverything/handlers/video.py` (params, affordance, dispatch, handler)
- Modify: `src/readeverything/composition.py`
- Test: `tests/unit/handlers/test_video_handler.py`

**Interfaces:**
- Consumes: `ClipModel` (Task 8), `ClipExtractor` (Task 10)
- Produces: `VideoHandler(..., clips: ClipExtractor | None = None, watcher: ClipModel | None = None, max_clip_s: float = 30.0)`
- Produces: affordance `watch_segment` with params `start_s: float >= 0`, `end_s: float > 0`, `prompt: str`
- Produces: `MAX_CLIP_SECONDS = 30.0`, `TOKENS_PER_CLIP_SECOND = 2180` in `handlers/video.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/handlers/test_video_handler.py
async def test_watch_segment_describes_a_range_as_one_span(...):
    handler = VideoHandler(..., clips=FakeClips(data=b"mp4"), watcher=FakeWatcher(text="a demo"))
    result = await handler.invoke(ref, "watch_segment",
                                  {"start_s": 10.0, "end_s": 20.0, "prompt": "what happens?"})
    assert result.content.text == "a demo"
    assert result.locator == TimeSpan(start_s=10.0, end_s=20.0)


async def test_a_clip_over_the_cap_is_refused_with_its_cost(...):
    """Cost is ~2,180 tokens per second and cannot be reduced client-side, so
    a 10-minute request is 1.3M tokens. Truncating it and reporting success
    would be a claim about time the model never saw."""
    handler = VideoHandler(..., clips=..., watcher=..., max_clip_s=30.0)
    result = await handler.invoke(ref, "watch_segment",
                                  {"start_s": 0.0, "end_s": 600.0, "prompt": "?"})
    assert result.degraded is True
    assert "30" in result.content.text and "2180" in result.content.text.replace(",", "")


async def test_watch_segment_is_absent_without_a_watcher(...):
    handler = VideoHandler(..., clips=None, watcher=None)
    assert "watch_segment" not in [a.name for a in handler.affordances()]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/handlers/test_video_handler.py -k watch_segment -v`
Expected: FAIL — unexpected keyword argument `clips`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/readeverything/handlers/video.py
#: Measured against llama.cpp b10438 serving qwen3.8-27b-mtp on 2026-08-15:
#: a 2s clip cost 5,242 prompt tokens and a 10s clip 21,787. Re-encoding the
#: source to a lower fps changed NOTHING — the server resamples by timestamp,
#: so cost is a function of duration alone and a caller cannot turn it down.
TOKENS_PER_CLIP_SECOND = 2180

#: The default ceiling on one `watch_segment`, about 65k prompt tokens. A
#: request past it is refused rather than truncated: a watch that silently
#: covered the first 30 seconds of a 10-minute range and reported success
#: would be a claim about time the model never saw.
MAX_CLIP_SECONDS = 30.0


class WatchSegmentParams(BaseModel):
    start_s: float = Field(default=0.0, ge=0.0, description="Start of the range to watch.")
    end_s: float = Field(default=10.0, gt=0.0, description="End of the range to watch.")
    prompt: str = Field(
        default="Describe what happens in this segment.",
        description="What to ask the model about the segment.",
    )
```

Register the affordance when `self._clips is not None and self._watcher is not None`, dispatch `"watch_segment"` to `_watch_segment`, and implement:

```python
    async def _watch_segment(
        self, ref: SourceRef, start_s: float, end_s: float, prompt: str
    ) -> Rendition:
        if self._clips is None or self._watcher is None:
            raise UnknownAffordanceError("watch_segment", (a.name for a in self.affordances()))
        span_s = end_s - start_s
        if span_s <= 0:
            return self._refusal(
                start_s, end_s,
                f"a segment must end after it starts; got {start_s:g}s to {end_s:g}s",
            )
        if span_s > self._max_clip_s:
            return self._refusal(
                start_s, end_s,
                f"a {span_s:g}s segment costs about "
                f"{int(span_s * TOKENS_PER_CLIP_SECOND):,} prompt tokens at "
                f"{TOKENS_PER_CLIP_SECOND} tokens per second, and this handler's cap is "
                f"{self._max_clip_s:g}s; ask for a narrower range, or sample frames with "
                f"describe_frame to find the part worth watching",
            )
        path = ...  # same local-path resolution describe_frame uses
        try:
            data = await self._clips.clip(path, start_s, end_s)
        except Exception:
            data = None
        if data is None:
            return self._refusal(start_s, end_s, "no clip could be cut from that range")
        try:
            text = await self._watcher.watch(data, CLIP_MIME, prompt)
        except Exception as exc:
            return self._refusal(start_s, end_s, f"the model could not watch the segment ({exc})")
        return Rendition(
            locator=TimeSpan(start_s=start_s, end_s=end_s),
            content=TextContent(text=text),
        )
```

`_refusal` builds a `Rendition` with `degraded=True`, `TextContent` carrying the message, and the requested `TimeSpan` as its locator — the handler never raises about its input.

Wire `clips`, `watcher` and `max_clip_s` through `_video_handler` and `build_perception`; export `build_openai_clip_model` and `FfmpegClip`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit tests/integration -v`
Expected: PASS

- [ ] **Step 5: Live check (skipped without a server)**

Add to `tests/live/test_watch_segment.py`: a 10-second clip of `media/mystery_subject.mp4` through the real `build_openai_clip_model`, asserting non-empty text; and a 600-second request asserting `degraded is True` without any network call.

- [ ] **Step 6: Commit**

```bash
git add src/readeverything/handlers/video.py src/readeverything/composition.py tests
git commit -m "watch_segment: bounded, because the cost curve says so"
```

---

## Stage 3 — Where to look

> Gated on Task 7's measurement. If captions alone answer the question fast and correctly, re-ask whether this is worth building before starting.

### Task 12: A pure sampler

**Files:**
- Create: `src/readeverything/domain/sampling.py`
- Test: `tests/unit/domain/test_sampling.py`

**Interfaces:**
- Consumes: `TranscriptCue` (Task 3)
- Produces: `suggest_frames(cues, duration_s, budget) -> tuple[FrameSuggestion, ...]`
- Produces: `FrameSuggestion(seconds: float, reason: str)` frozen dataclass
- Produces: `DEICTIC_MARKERS: frozenset[str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/domain/test_sampling.py
def test_prefers_moments_where_the_speaker_pointed_at_something():
    """"As you can see here" is a moment the words do not describe. That is
    exactly where a frame earns its cost."""
    cues = (
        _cue(0, 5, "welcome to the course"),
        _cue(10, 15, "as you can see here, the syntax is simple"),
    )
    picks = suggest_frames(cues, duration_s=60.0, budget=1)
    assert picks[0].seconds == pytest.approx(10.0)
    assert "pointed" in picks[0].reason


def test_falls_back_to_gaps_when_nobody_points():
    """A stretch with no cues is silence or demonstration; on a lecture it is
    usually a demonstration."""
    cues = (_cue(0, 5, "hello"), _cue(50, 55, "goodbye"))
    picks = suggest_frames(cues, duration_s=60.0, budget=1)
    assert 5.0 < picks[0].seconds < 50.0


def test_no_cues_falls_back_to_even_spacing():
    picks = suggest_frames((), duration_s=100.0, budget=4)
    assert [p.seconds for p in picks] == [12.5, 37.5, 62.5, 87.5]


def test_never_exceeds_the_budget():
    cues = tuple(_cue(i, i + 1, "look here") for i in range(0, 100, 2))
    assert len(suggest_frames(cues, duration_s=100.0, budget=5)) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/domain/test_sampling.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Pure, no I/O, no model. Rank deictic cues first (midpoint of the cue), then gap midpoints longest-first, then even spacing to fill the budget; deduplicate timestamps within 1s; sort ascending before returning.

```python
DEICTIC_MARKERS = frozenset({
    "as you can see", "you can see", "look at", "over here", "right here",
    "this line", "watch what happens", "on the screen", "notice that",
})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/domain/test_sampling.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/domain/sampling.py tests/unit/domain/test_sampling.py
git commit -m "Choose frames by what the speaker pointed at"
```

---

### Task 13: `suggest_frames` affordance and the comparison

**Files:**
- Modify: `src/readeverything/handlers/video.py`
- Create: `tests/integration/test_sampling_strategy.py`

**Interfaces:**
- Consumes: `suggest_frames` (Task 12), caption precedence (Task 5)
- Produces: affordance `suggest_frames` with param `budget: int` (default 6), returning `StructuredContent` rows of `{"seconds": float, "reason": str}`

- [ ] **Step 1: Write the failing test**

```python
async def test_suggest_frames_uses_the_files_own_captions(...):
    result = await handler.invoke(ref, "suggest_frames", {"budget": 3})
    assert len(result.content.rows) == 3
    assert all("reason" in row for row in result.content.rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/handlers/test_video_handler.py -k suggest_frames -v`
Expected: FAIL — `UnknownAffordanceError`.

- [ ] **Step 3: Write minimal implementation**

Register unconditionally (it needs no model — only `{FFMPEG}`, already required), fetch cues through the same precedence path `_cues` uses, and return `StructuredContent`.

- [ ] **Step 4: Compare strategies on the real file**

Ask the same question three ways and record wall-clock, model calls and correctness:
1. captions only (no vision model wired)
2. captions + `suggest_frames`-guided `describe_frame`
3. the 2026-08-15 vision-only baseline (4m54s, 12 calls)

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/handlers/video.py tests/integration/test_sampling_strategy.py
git commit -m "Let the file say where looking is worth it"
```

---

## Self-Review

**Spec coverage:** §1 captions visible → Tasks 1-2. §2 CaptionExtractor → Task 4. §3 CueSource → Task 3. §4 precedence → Tasks 5-6. §5 watch_segment → Tasks 8-11. §6 sampling → Tasks 12-13. Testing section → Tasks 7, 11 (live), 13. Out-of-scope items (OCR, diarization, translation, sidecar `.srt`) appear in no task, as intended.

**Placeholder scan:** Task 7 Step 3 deliberately has no new code — it is a compose-check whose fix belongs in whichever earlier task broke. Task 10's port file and Task 12's ranking body are described rather than quoted in full; both are mechanical given the shown signatures and constants.

**Type consistency:** `TranscriptCue.source` (Task 3) is read by `_spoken` (Task 3), `FakeCaptions` (Task 5) and `suggest_frames` (Task 12). `CaptionExtractor.extract(path, track)` (Task 4) is called in Task 5 and wired in Task 6. `ClipModel.watch(clip, mime, prompt)` (Task 8) is implemented in Task 9 and called in Task 11 with `CLIP_MIME` from Task 10. `MediaFacts.text_subtitle_streams` (Task 2) is read in Tasks 5 and 6.
