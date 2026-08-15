# Transcript-first video understanding

**Status:** design
**Date:** 2026-08-15

## The problem

Asked "what is this video about?", an agent holding today's tools reaches
straight for the vision model. On a 37-minute lecture that cost 12 vision calls
and 4m54s to answer a question the file could have answered in one second.

The file carries 848 embedded caption cues — 6,412 words, roughly 8,700 tokens,
covering the whole timeline. `ffmpeg -map 0:s:1 -c:s srt` extracts them in about
a second with no model call at all. The agent never considered them because
nothing told it they existed: `adapters/ffprobe_streams.py` drops every stream
whose `codec_type` is not `video` or `audio`, so the card an agent inspects is
silent about captions.

Measured on the live server (`qwen3.8-27b-mtp`, llama.cpp b10438), the four ways
to learn what a video contains are not close:

| Path | Whole 37-min file | Model calls |
|---|---|---|
| Embedded captions | ~1 s, ~8.7k tokens | 0 |
| Whisper ASR | minutes of CPU | 0 (local weights) |
| Frame sampling | 4m54s at 12 frames | 12 |
| Native `input_video` | ~4.9M tokens — 19x over context | — |

The ordering is stable and it is not a close call. Text is three orders of
magnitude cheaper than pixels, and a lecture's meaning lives in what is said.

So the strategy is: **read the words first, look at pictures second, and look
only where the words are insufficient.** This document specifies the pieces that
are missing to make that possible, and the one piece — `watch_segment` — that
makes "look here specifically" cheap enough to be worth doing.

## What is missing

1. Captions are invisible to the card, so no agent can choose them.
2. Nothing extracts captions.
3. The domain cannot say a line of text was *written* rather than *heard*.
4. `represent()` has no precedence rule between captions and ASR.
5. `watch_segment` has been specced since perception-core and never built.
6. Nothing guides *where* to sample frames when the words run out.

## 1. Captions become visible

`StreamInfo.kind` widens from `Literal["video", "audio"]` to include
`"subtitle"`, and `ffprobe_streams.py` stops discarding those streams.
`StreamInfo` gains two fields:

- `language: str | None` — from the stream's `tags.language`, `None` when the
  container does not say. Which track to read is a real choice on a multi-track
  file and the agent cannot make it blind.
- `is_text: bool` — whether the codec carries characters (`mov_text`, `subrip`,
  `ass`, `ssa`, `webvtt`) or pixels (`dvd_subtitle`, `hdmv_pgs_subtitle`).

That last distinction is not cosmetic. The test file has **both**: a `mov_text`
track that extracts to characters in one second, and a `dvd_subtitle` track that
is a bitmap and would need OCR. Presenting them as one kind of thing would send
an agent down a path that costs a hundred times more than the one beside it.
Bitmap subtitle tracks are reported as present and not extractable. OCR over
them is out of scope here.

`MediaFacts` gains `subtitle_streams` and `text_subtitle_streams` properties,
mirroring `video_streams` and `audio_streams`.

## 2. `CaptionExtractor`

A new port, `ports/captions.py`:

```python
async def extract(self, path: str, track: int | None = None) -> tuple[TranscriptCue, ...] | None
```

`None` means "no text caption track" — the same normal-answer convention as
`FrameExtractor.frame_at` and `AudioExtractor.extract`. A handler must be able to
ask about anything and get an answer it can act on, never an exception.

`track` selects among several; `None` takes the first text track. The adapter
(`adapters/ffmpeg_captions.py`) shells out to ffmpeg, converts to SRT, parses it
into cues, and strips the markup real files carry — the test file's every cue is
wrapped in `<font size="24">`, which would otherwise be indexed as content.

Captions are not resampled or re-tiled here. They arrive with their own spans
and go through `domain.timeline.tile` exactly as ASR cues do, so a caption and a
transcript cue are interchangeable downstream.

## 3. `CueSource`: said versus captioned

`video.py` already refuses to let a citation attribute speech to a picture — its
`SPEECH_MARKER` exists so an agent can tell what was *seen* from what was
*said*. Captions are a third kind of evidence and need the same care.

A caption is authored text. It is frequently condensed for reading speed, so it
is not verbatim; it describes non-speech sound (`[music playing]` is the first
cue in the test file, and nobody said it); and it can be a translation. Marking
it as speech would assert something false in exactly the register this project
is built to avoid.

So `TranscriptCue` gains:

```python
source: CueSource = CueSource.SAID   # SAID | CAPTIONED
```

Same type, same tiling, same locator map — a citation resolves identically
either way. Only the claim about provenance differs. The default is `SAID`
because every producer that exists today is a transcriber, and only the caption
adapter sets `CAPTIONED`; a default of `CAPTIONED` would mislabel every existing
cue, which is the failure this field exists to prevent.

Rendering picks the marker from the source: `(speech)` as today, `(caption)` for
authored text. Both `audio.py` and `video.py` use the same rule.

## 4. Precedence in `represent()`

When a video has both a text caption track and a transcriber available, captions
win. They are free, they are already aligned to the timeline, and they were
written by a human who could hear the audio.

The full order, per file:

1. A text caption track, if one exists and a `CaptionExtractor` is wired.
2. ASR over the extracted audio, if a `Transcriber` is wired.
3. Neither — the timeline is frame descriptions only, exactly as today.

Whichever ran is reported as a `Degradation` when it was not the best available
— "captions were present but no extractor was wired", "no caption track; used
ASR" — because the difference is invisible in the output and changes how much a
reader should trust the words. A file with no words at all is not a degradation;
it is a fact about the file.

`audio.py` follows the same rule for containers that carry both.

## 5. `watch_segment`

The affordance perception-core specified and never built. It answers "what
happens between 12:00 and 12:30", which frame sampling answers badly: sampled
stills miss motion, and the thing an agent wants to zoom into is usually motion.

A new port, `ports/clips.py`:

```python
class ClipModel(Protocol):
    model_id: str
    async def watch(self, clip: bytes, mime: str, prompt: str) -> str: ...
```

Separate from `VisionModel` rather than an overload of it, because a server that
describes stills need not accept clips — ours did not until it was fixed on
2026-08-15 — and a handler must be able to offer frame description while
truthfully reporting that it cannot watch.

`adapters/clip_langchain.py` implements it over the `input_video` content part:

```json
{"type": "input_video", "input_video": {"data": "<base64>"}}
```

This is a llama.cpp extension, not OpenAI's `video_url`. It is verified working
against the live server. `adapters/ffmpeg_clip.py` cuts `[start, end)` into mp4
bytes; `+faststart` is not required — measured, not assumed.

**The duration cap is the whole design.** Measured cost is ~2,180 prompt tokens
per second of clip, and it cannot be reduced from the client: re-encoding the
source to a lower fps produced a byte-identical token count, because the server
resamples by timestamp. Cost is a function of duration alone. Against a 262k
context that caps a single clip near two minutes, and a caller who asks for ten
gets a failure a long way from the mistake.

So `watch_segment` takes `max_clip_s` (default 30, about 65k tokens) at
construction. A request longer than the cap is refused with a message naming the
cap and the measured rate, not silently truncated — a truncated watch that
reports success is a claim about time the model never saw.

The affordance returns one `Rendition` whose `LocatorSegment` spans the whole
requested `TimeSpan`. It is a joint description of a range; splitting it
per-second would invent precision the model never had.

## 6. Where to look, when words are not enough

Captions answer "what is this about". They do not answer "what is on screen at
14:02", and lectures are full of moments where the words point at something they
do not describe: *"as you can see here"*, *"this line"*, *"watch what happens"*.

`FrameExtractor.scene_cuts` already exists and is unused by any strategy. Two
signals, combined, choose frames far better than a fixed interval:

- **Deictic cues.** Caption text containing pointing words is a moment where the
  speaker referred to something visual. That is precisely where a frame earns
  its cost.
- **Caption gaps.** A stretch with no cues is either silence or non-verbal
  demonstration. On a lecture it is usually a demonstration.

This lands as `domain/sampling.py` — a pure function from cues, duration and a
frame budget to a tuple of timestamps, with no I/O and no model. Pure because it
is the piece most worth testing exhaustively and least worth mocking.

It is used by a new `suggest_frames` affordance (returning the timestamps and
why each was chosen, so an agent can decide) rather than being imposed on
`represent()`. `represent()` keeps its fixed interval: it must stay predictable
and total, and a heuristic that samples unevenly would make the indexed timeline
depend on caption quality.

## Testing

The existing seams carry this. `FakeTranscriber` gains a caption sibling; the
pure sampler needs no fakes at all.

- **Unit:** SRT parsing including the `<font>` markup and multi-line cues; cue
  source rendering; precedence selection under each combination of
  captions/transcriber/neither; the duration cap's refusal; the sampler's
  choices given synthetic cues.
- **Integration:** a generated container with a real embedded caption track,
  through `represent()`, asserting the timeline interleaves `(caption)` with
  frame descriptions and that every character maps to a span.
- **Live (skipped by default):** `watch_segment` against the real server,
  asserting a clip under the cap returns text and one over it is refused.

The end-to-end check is the one that matters: ask the agent the same question
about `mystery_subject.mp4` and compare against the known-correct vision-only
answer — same conclusion, and it should arrive in seconds rather than five
minutes, with citations to spoken lines rather than to frames.

## Order of work

1. Captions visible + extracted + `CueSource` + precedence. This is the whole
   speed win; everything else is refinement.
2. `watch_segment` + `ClipModel` + clip extraction + the cap.
3. `domain/sampling.py` + `suggest_frames`, and the end-to-end comparison.

Staged so that step 1 can be measured before step 2 is written. If step 1 makes
the agent fast and correct on its own, steps 2 and 3 are answering a question we
should re-ask rather than assume.

## Out of scope

OCR over bitmap subtitle tracks. Diarization (speaker turns remain a
perception-core concern). Translation of foreign-language caption tracks.
Downloading sidecar `.srt` files that sit beside a video — a real case, but a
source-resolution concern, not a video one.
