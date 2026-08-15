# readeverything: The Spoken Word

**Date:** 2026-08-15
**Status:** Approved for planning
**Predecessors:** Spec 1 (perception core), Spec 3 (integration), Spec 4 (documents), Spec 5 (the moving image)
**Landed:** Plan 5 merged `da4d0e6` — 453 tests, 92.57% coverage

---

## 1. Why this

Audio is the last common media family with no handler, and a recorded meeting or
interview is the file people most want to ask a question of. Text, images, PDFs
and video are read; a `.wav` falls to the binary fallback and reports its size.

`TranscriptCue(span: TimeSpan, text: str, speaker: SpeakerId | None,
confidence: float | None)` has existed in the domain since Spec 1 and, like
`TimeSpan` before Cycle 5, has never had a producer. This gives it one.

### 1.1 The server cannot do this, and we know because we asked

Spec 5 deferred transcription because the model server's support for timestamped
transcription was unverified. It is now verified, by request:

```
POST http://192.168.1.14:8080/v1/audio/transcriptions
{"error":{"code":501,"message":"The current model does not support audio input.",
          "type":"not_supported_error"}}
```

Identical with and without `response_format=verbose_json` and
`timestamp_granularities[]`. The route exists; no loaded model implements it.
`GET /v1/models` lists `gemma-4-26b-qat`, `muse-glimmer-30b`, `nomic-embed-text`,
`qwen3.8-27b-mtp` — no audio model.

This mattered enough to split a cycle over. The OpenAI specification documents
`timestamp_granularities` thoroughly, and a design built from that documentation
would have targeted an endpoint that returns 501. **Transcription is local.**

### 1.2 Acceptance

> Point the library at a directory containing an audio file. `inspect` reports
> its duration, codec, sample rate and channels without decoding it. With a
> transcriber configured, `represent` returns the spoken text, every character
> resolving to the moment it was said. Without one, `represent` says the audio
> is present and untranscribed rather than returning nothing, and the file stays
> discoverable by its facts. Asking for a time range returns what was said then.

---

## 2. Scope

**In scope**

1. An `AudioExtractor` port and its ffmpeg adapter — the audio track of any
   container as 16 kHz mono WAV.
2. A `Transcriber` port and a `faster-whisper` adapter, behind an extra.
3. `AudioHandler`: card from `StreamProbe`, `represent()` producing a cue
   timeline located by `TimeSpan`, and a `read_span` affordance.
4. `VideoHandler` gains transcript interleaving when a transcriber is present.
5. Generated audio fixtures.

**Out of scope**

- **Diarization.** Research recommends sherpa-onnx (Apache-2.0, ONNX, CPU, no
  gated weights) over pyannote (gated Hugging Face weights, pulls torch), which
  flips Spec 1 §7's default. That is a real decision and it deserves its own
  cycle, not a footnote in this one. Until then `TranscriptCue.speaker` stays
  `None` — the field exists and admits ignorance rather than guessing.
- **Silence detection.** See §5.
- **Observability and concurrency.** Cycle 7. Transcription is the strongest
  argument yet for both, which is exactly why it stays deliberate.

---

## 3. `Transcriber`, and a model this library will not download

```python
@runtime_checkable
class Transcriber(Protocol):
    model_id: str
    async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]: ...
```

`model_id` feeds the capability fingerprint, so changing the ASR model
invalidates transcripts and leaves everything else alone — the same property
`VisionModel.model_id` already gives OCR and frame descriptions.

**The adapter takes a required local model directory, with no default.**
`faster-whisper` downloads weights from Hugging Face on first construction
unless given a path. This library reads no environment and downloads nothing
implicitly, so a missing configuration must fail at the composition root rather
than quietly pulling several hundred megabytes on a caller's first `represent()`.

A path that does not exist raises. The port catches it and degrades, because a
handler never raises — but the composition root is where a caller learns they
have not configured a transcriber, and it is loud there.

**On this machine there are no weights**, so the default path degrades. That is
not a shortcoming of the cycle: the degradation is a real, tested behaviour that
a user with no ASR model will actually experience, and the moment a model
directory is supplied the same code transcribes. Unit tests use a
`FakeTranscriber` in the shape of the existing `FakeVision`; a live test runs
against real weights when they are present and skips when they are not.

---

## 4. `AudioExtractor`

```
ffmpeg -i <path> -vn -map 0:a:0 -ac 1 -ar 16000 -f wav -loglevel error -y -
```

16 kHz mono WAV on stdout, which is what ASR wants.

**Verified:** with several audio streams, ffmpeg's default selector picks exactly
one and does not mix — output is byte-identical with and without `-map 0:a:0`.
The map is included anyway: it says which stream was chosen, survives a change
in ffmpeg's default selection, and is the only way to pick among language tracks
later.

**Also verified, and it differs from frame extraction:** a file with no audio
stream exits **234** with `Output file does not contain any stream` on stderr.
Frames fail silently with exit 0; streams fail loudly. Spec 5 §5 established that
asymmetry and this adapter is the other half of it — **check the exit status
here, and output length there.** Both adapters say so, each naming the other.

---

## 5. `represent()`: cues, and what fills the silence

The transcriber returns cues with gaps between them — people stop talking.
`LocatorMap` must be total, gapless and zero-start, so something must occupy
those gaps.

**Each cue's span extends to the next cue's start; the last extends to the file
duration.** Exactly `VideoHandler._bounds`, one medium over: the stretches
between cues belong to the cue that most recently spoke. A citation landing in a
pause resolves to the utterance before it, which is the reading a person
scrubbing a transcript would make anyway.

The alternative — emitting explicit `(silence)` segments — requires a
silence-detection pass with thresholds and false positives. That is a second
inference-shaped decision on top of transcription, unverified, and it would put
text in the index describing something nothing measured. Rejected.

**Zero cues** — silent or unintelligible audio — falls back to a single segment
over the whole duration saying so, like video's no-video-stream case. Silence is
a result; an empty `Rendered` would be a claim that there was nothing to hear,
which is different and which nothing established.

**Barriers: none.** Video's scene cuts answer "did the visual content change",
which ffmpeg genuinely measures. The honest audio analogue is a speaker turn,
which needs diarization, which is out of scope. `barriers=()` is what four of
five handlers already produce and it is honest here. Inventing a barrier at
every cue boundary would make every pause a hard chunk boundary, which is a
claim about structure that transcription does not support.

---

## 6. Where the PDF precedent stops applying

Spec 4 ruled OCR out of `represent()`: it is `DEEP`, gated, and asked for by
name. **This spec rules transcription IN, and the difference is not
inconsistency.**

A scanned PDF whose text is not OCR'd still contributes page count, page
structure and an outline to an index. A silent video still contributes its
visual timeline. **An audio file has no cheaper layer.** Without transcription it
contributes a duration and a codec — nothing anyone can ask a question of. The
choice is not "expensive now versus cheap now"; it is "expensive now versus
never".

So transcription happens in `represent()`, gated on an injected `Transcriber |
None`, exactly as video's frame descriptions are gated on `VisionModel | None`.
`describe()` stays probe-only and cheap: duration, codec, sample rate, channels,
all from `StreamProbe`, no decode. The expensive tier is `represent()`, which is
already where video samples frames and calls a vision model.

`Budget` truncates the flattened text afterwards. It never transcribes less
audio — a small budget must not silently drop the tail, which would be a
transcript that stops without saying it stopped.

---

## 7. Video gains a transcript

`VideoHandler` already interleaves frame descriptions on a timeline. With a
transcriber it also interleaves cues, merged in timestamp order — the design
Spec 1 §7 called for and nothing has been able to build until now.

The interleaving keeps `LocatorMap` total: the merged sequence is still ordered
moments, each owning the span to the next. A frame description and a cue that
overlap in time are two entries at their own timestamps, not a conflict.

Without a transcriber, video behaves exactly as it does today.

---

## 8. Affordances

| Name | Level | Locator | Requires |
| --- | --- | --- | --- |
| `read_span(start_s, end_s)` | SEGMENT | `TimeSpan` | `FFMPEG`, `ASR` |

One affordance. `read_span` returns what was said in a time range — the
operation an agent actually wants after reading a card that says "42 minutes of
audio".

`Capability.ASR` exists in the enum and has never had a consumer. It gets one
here, discovered from the injected transcriber the way `ModelProbe` derives
`VISION` from the injected vision model — so the capability cannot disagree with
the model actually present.

`AudioHandler.requires()` is `{FFMPEG}` — without ffmpeg there is no way to
extract the audio track at all, so the registry drops the handler and audio
files fall to the binary fallback. Transcription is gated at the affordance
level, not the handler level, because a card is still worth having without ASR.

---

## 9. Acceptance

1. §1.2's sentence is true, demonstrated by an integration test.
2. `TranscriptCue` has a producer.
3. An audio file's card reports duration, codec, sample rate and channels with
   no decode.
4. With a transcriber, `represent()` covers the whole duration with no gaps and
   every character resolves to the moment it was said.
5. Without a transcriber, `represent()` reports that audio is present and
   untranscribed — never an empty string, never a claim that there was silence.
6. Zero cues from a real transcription attempt is distinguishable from no
   transcriber configured. Both are honest and they are not the same fact.
7. A file with no audio stream degrades on exit status, and the message says
   there is no audio track.
8. A missing or invalid model directory fails at the composition root, and does
   not trigger a download.
9. Video with a transcriber interleaves cues and frame descriptions in timestamp
   order, with the map still total.
10. All gates green, coverage floor holds at 92.

---

## 10. Risks

| Risk | Mitigation |
| --- | --- |
| `faster-whisper`'s API differs from the researched shape | It was not verified by running — no weights were downloaded. The plan's first task confirms attribute names against the installed package before any adapter is written, exactly as Cycle 4 did for the deepagents protocol and Cycle 5 for ffprobe's JSON. |
| CPU transcription is too slow to be usable | Small/base with `int8` is the default and the model size is the caller's argument. Speed is a bench question, not a correctness one; the tests use a fake. |
| No weights on this machine means the real path ships untested | The degraded path is what this machine exercises and it is fully tested. The transcribing path is covered by a `FakeTranscriber` in unit tests and a live test that skips without weights — the same discipline the vision live tests already follow. |
| Transcription in `represent()` surprises a caller with cost | §6. It is gated on an injected port: a caller who supplies no transcriber pays nothing. Supplying one is the act of asking for it. |
| Interleaving breaks video's existing timeline | §7 keeps the merged sequence ordered and total. Video without a transcriber must be byte-identical to today, and that is a test. |
