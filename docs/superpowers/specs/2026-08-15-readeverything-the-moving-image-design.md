# readeverything: The Moving Image

**Date:** 2026-08-15
**Status:** Approved for planning
**Predecessors:** Spec 1 (perception core), Spec 3 (integration), Spec 4 (documents)
**Landed:** Plan 4 merged `d4f564b` — 381 tests, 94.16% coverage

---

## 1. Why this, and what it closes

Spec 4 gave `PageRef` and `Rendered.barriers` their first producers. One locator
remains unproduced:

| Locator | Producer |
| --- | --- |
| `CharSpan` | text, image, binary handlers |
| `ByteRange` | binary handler |
| `BBox` | image handler; PDF, with a real page |
| `PageRef` | PDF handler (Spec 4) |
| `TimeSpan` | **none** |

`TimeSpan` has existed since Plan 1. Nothing has ever emitted one.

This cycle also makes ffmpeg the **first real consumer of capability
discovery**. Every capability the library has negotiated so far has been a
model. `Capability.FFMPEG` has been an enum member with nothing behind it, and
`BinaryProbe` — which exists to observe OS binaries rather than assume them —
has never been run against the binaries it names.

### 1.1 It was broken, and nobody knew

Run against this machine's real executables for the first time:

```
ffmpeg       on-disk=installed  probe='ffmpeg version 6.1.1-3ubuntu5 …'
exiftool     on-disk=installed  probe=None
libreoffice  on-disk=installed  probe='Warning: -version is deprecated.  Use --version instead.'
tesseract    on-disk=ABSENT     probe=None
```

Two defects, and they are the same defect this project keeps finding.

**exiftool is installed and reported absent.** `BinaryProbe` uses one
`version_flag="-version"` for every executable. exiftool's is `-ver`. A real
capability is invisible, and "absent" is indistinguishable from "not installed".

**libreoffice's recorded revision is a deprecation warning.** The probe takes
the first line of stdout and calls it a version. That string enters
`CapabilitySet.fingerprint()` and therefore every artifact cache key — so a
warning is currently part of this library's cache identity, and if libreoffice
stops emitting it every cached artifact silently invalidates.

Nothing established that the captured line was a version. The probe written to
replace assertion with observation asserts.

Its unit tests used `echo` as a stand-in, which is the right way to test the
*contract* — but nobody ever pointed it at the real thing. **A probe is the one
component whose correctness cannot be established against a stand-in.**

### 1.2 Acceptance

> Point the library at a directory containing a video. `inspect` reports its
> duration, resolution and codecs without decoding a frame. `represent` returns
> a timeline of frame descriptions whose every character resolves to the moment
> it describes, with barriers at scene cuts. Asking for the frame at 12.4s
> returns that frame as an image an agent can already route into
> `describe_image`. On a machine without ffmpeg, none of those affordances are
> offered and nothing breaks.

---

## 2. Scope

**In scope**

1. Fixing `BinaryProbe` (§1.1), with a test against real executables.
2. A `MediaProbe` for audio/video via `ffprobe`: duration, streams, codecs,
   dimensions — without decoding.
3. A `FrameExtractor` port and its ffmpeg adapter.
4. `VideoHandler`: card, affordances, and `represent()` producing a frame
   timeline with `TimeSpan` locators and scene-cut barriers.
5. Generated media fixtures — owed since Spec 1 §14b.

**Out of scope**

- **Transcription and audio.** Cycle 6, "the spoken word": `AudioExtractor`,
  `Transcriber`, `AudioHandler`, and video gaining transcript interleaving.
  Split out because the model server's support for timestamped transcription is
  **unverified**, and designing an honest degradation for that deserves its own
  attention rather than being appended here.
- **Diarization.** Later still. Research recommends sherpa-onnx (Apache-2.0,
  ONNX, CPU, no gated weights) over pyannote (gated Hugging Face weights, pulls
  torch) — which flips Spec 1 §7's default. That is a real decision and it is
  not this cycle's.
- **Scene detection beyond a simple threshold.** §6.
- **Concurrency.** Still Cycle 7. Frame extraction is per-frame subprocess work
  and is the strongest argument yet for it, which is why it stays deliberate
  rather than accidental.

---

## 3. Fixing the probe first

**The version flag belongs to the executable, not to the probe.** `BinaryProbe`
gets a per-capability mapping of executable *and* flag:

| Capability | Executable | Flag |
| --- | --- | --- |
| `FFMPEG` | `ffmpeg` | `-version` |
| `EXIFTOOL` | `exiftool` | `-ver` |
| `LIBREOFFICE` | `libreoffice` | `--version` |
| `TESSERACT` | `tesseract` | `--version` |

**And a captured line must look like an identity before it is called one.** At
minimum: non-empty, and not a warning or error. Under uncertainty the probe
returns `None`, which its own contract already requires — "under uncertainty the
library offers less, never more".

**A test must run against the real executables.** It cannot assert *which*
binaries exist, since that varies by machine — but it can assert the property
that failed here: **for every capability whose executable is on `PATH`, the
probe returns a revision; for every one that is not, it returns `None`.** That
is machine-independent and it is exactly the bug. Marked `integration`, since it
touches the real system.

---

## 4. `ffprobe` for the card

One invocation, headers only, no decode:

```
ffprobe -v error -show_format -show_streams -of json <path>
```

Returns `{"streams": [...], "format": {...}}`. `format` carries `duration`,
`format_name`, `size`, `bit_rate`. Each stream carries `codec_type`
(`video`/`audio`), `codec_name`, and per-type keys — `width`, `height`,
`r_frame_rate` for video; `sample_rate`, `channels` for audio.

This is an index read, not a decode, so the card stays cheap — the same
discipline `MediaProbe` already enforces for PDF, where the type carries no text
and has no way to return any.

`DocumentFacts` does not fit: it is pages. Audio/video get `MediaFacts` —
`duration_s`, and a tuple of stream descriptions. Two record types behind one
`MediaProbe` protocol is the wrong shape, so the protocol takes the mimetype and
returns a union, or there are two protocols. **Decision: two protocols.**
`MediaProbe` stays the paginated-document probe Spec 4 built; audio/video gets
`StreamProbe`. A protocol whose return type depends on its input is a protocol
that has two jobs.

---

## 5. Frames, and a failure mode that does not fail

```
ffmpeg -ss <T> -i <path> -frames:v 1 -f image2 -vcodec png -loglevel error -y -
```

`-ss` **before** `-i` seeks at the demuxer and is fast; after `-i` it decodes
from the start and is frame-accurate but slow on long files. Cheap-first is this
library's whole design, so `-ss` goes first, and the card's `r_frame_rate` tells
a caller what precision they are getting.

**The trap, measured:** seeking past the end produces **exit code 0, zero bytes
on stdout, and no stderr**. ffmpeg does not error.

A handler trusting the return code would hand back an empty PNG as "the frame at
t=999" — asserting an observation nothing made, this project's recurring defect,
handed to a model as an image. **Frame extraction is validated by output length,
never by exit status.**

And the asymmetry matters: a *missing audio stream* does exit non-zero (234).
Two ffmpeg failure modes, two different checks. Anyone unifying them
reintroduces the bug, so both adapters say so in a comment.

---

## 6. `represent()`: a timeline that is total

`represent()` samples frames at a fixed interval, describes each through the
injected `VisionModel`, and concatenates the descriptions in timeline order.
Every character resolves to the `TimeSpan` of the moment it describes.

Three constraints shape this, and all three are structural:

**`TimeSpan` forbids `start >= end`.** A frame is a point in time and a point is
inexpressible. So a frame's span is `[t, t + frame_duration)`, taken from
`r_frame_rate` — the honest width of one frame, not an arbitrary epsilon.

**`LocatorMap` must be total and gapless.** So the timeline cannot skip the
stretches between sampled frames. Each sample's segment runs from its own
timestamp to the next sample's, and the description sits inside it. The map
covers the whole duration because the duration is what the map is *of*.

**Barriers go at scene cuts.** For PDF the barrier was the page break; here it
is the cut, detected with ffmpeg's `select='gt(scene,<threshold>)'`. A chunker
must not casually merge across a cut, because the frames either side depict
different things. Where scene detection is unavailable or the threshold finds
nothing, there are no barriers — an empty tuple is honest, and `barriers=()` is
what every handler produced before Spec 4 anyway.

**Without a vision capability** there are no descriptions. `represent()` then
produces the timeline's *structure* — the moments, their spans, and a statement
that nothing described them — rather than an empty string. A video is not empty
because nothing looked at it, which is the scanned-PDF lesson at a new site.

---

## 7. Affordances

| Name | Level | Locator | Requires |
| --- | --- | --- | --- |
| `frame_at(seconds)` | SEGMENT | `TimeSpan` | `FFMPEG` |
| `describe_frame(seconds, prompt)` | DEEP | `TimeSpan` | `FFMPEG`, `VISION` |

`frame_at` returns `ImageContent`, so an agent can already route it into the
existing `describe_image` affordance through the tool pack — the same
composition PDF's `page_image` gained for free.

`describe_frame` exists anyway because the round trip through the agent costs a
turn and this is the operation an agent actually wants.

Both require `FFMPEG`, so on a machine without it the handler registers and
offers nothing but its card. That is negotiation working: the file is still
identified, its duration still reported from... nothing, because `ffprobe` is
also absent. **So the handler itself requires `FFMPEG`** and does not register at
all without it. Video files then fall to the binary fallback, which is honest.

---

## 8. Safety

These commands take file paths and this library hands results to an agent, so a
malicious file is adversarial input.

- `create_subprocess_exec` with an argv vector. Never a shell string. The path
  is one element; timestamps are separate elements. This is already
  `BinaryProbe`'s pattern.
- Every invocation gets a wall-clock timeout with kill-and-reap, as
  `BinaryProbe.revision` already does.
- `-analyzeduration` and `-probesize` bounded, so a malformed header cannot make
  ffprobe work indefinitely.
- `-frames:v 1` caps decode for frame extraction; a duration cap bounds anything
  that streams.
- Output goes to stdout (`-`), never a path derived from the input, so there is
  no filesystem write to sanitise.
- A stream descriptor claiming absurd dimensions is rejected from the card
  before any `ffmpeg` call, because the card is supposed to be cheap and a
  crafted descriptor is the cheapest possible attack.

---

## 9. Acceptance

1. §1.2's sentence is true, demonstrated by an integration test.
2. `TimeSpan` has a producer. Every locator type in the domain now has one.
3. `BinaryProbe` returns a revision for every capability whose executable is on
   `PATH`, and `None` for every one that is not — tested against the real
   system, not a stand-in.
4. No captured revision is a warning or an error message.
5. Seeking past a video's end degrades, and the degradation says the time was
   past the end. No empty image is ever presented as a frame.
6. `represent()` covers the full duration with no gaps, and a video with no
   vision capability still reports its timeline rather than nothing.
7. Barriers land at scene cuts where detection finds them; an empty tuple where
   it does not.
8. A machine without ffmpeg registers no video handler, and video files fall to
   the binary fallback.
9. All gates green, coverage floor holds at 92.

---

## 10. Risks

| Risk | Mitigation |
| --- | --- |
| Fixing `BinaryProbe` changes discovered capabilities and so changes cache keys | It does, and that is correct — the old keys embedded a warning string. Over-invalidation costs recomputation, never correctness. Recorded, not hidden. |
| Frame sampling on a long video is slow and serial | The sampling interval is a parameter with a conservative default, and `represent()` is not the cheap path — `inspect` is. Concurrency lands in Cycle 7 and this is its motivating case. |
| Scene detection quality varies wildly by content | Which is why barriers are advisory, and why "no barriers found" is a valid, honest result rather than a failure. |
| ffmpeg's silent-success-with-no-output shows up somewhere new | §5 makes output-length validation the rule rather than a special case, and the adapters say why in comments so it is not "simplified" away. |
| `ffprobe` on an adversarial file is itself the attack | Bounded `-analyzeduration`/`-probesize`, a hard timeout, and dimension sanity-checking before any decode. |
