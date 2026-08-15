# The Spoken Word Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read audio, give `TranscriptCue` its first producer, and let video carry a transcript.

**Architecture:** An `AudioExtractor` port pulls a container's audio track as 16 kHz mono WAV via ffmpeg, checking exit status because missing streams fail loudly. A `Transcriber` port turns those bytes into cues; its `faster-whisper` adapter takes a required local model directory and refuses to reach the network. `AudioHandler` builds a gapless cue timeline located by `TimeSpan`. `VideoHandler` merges cues into its existing frame timeline when a transcriber is present.

**Tech Stack:** Python 3.13, ffmpeg 6.1, faster-whisper 1.2.1 (MIT), pydantic v2, pytest, mypy --strict, ruff, import-linter, bandit, coverage.

**Spec:** `docs/superpowers/specs/2026-08-15-readeverything-the-spoken-word-design.md`

## Global Constraints

- **The library reads NO environment variables under `src/`**, and **downloads nothing implicitly**.
- **Python 3.13, PEP 695 inline type parameters.** A module-level `TypeVar` is a defect.
- **`mypy --strict`, `warn_unused_ignores = true`**, over `src` and `tests`. No new `# type: ignore` without a comment naming why.
- **Import-linter layers**, outermost first: `composition, testing, agent, pipeline, registry, handlers, adapters, ports, domain`. **Handlers must not import from `adapters/`.**
- **No handler ever raises** from `describe`, `invoke`, or `represent`, except `UnknownAffordanceError` for a name it does not offer.
- **Never assert on model text.** Assert structure and locators.
- **Every subprocess uses `create_subprocess_exec` with an argv vector**, a timeout, and kill-and-reap.
- **Coverage floor 92.** Run `make check`.
- **A law must be able to fail.**

---

## Measured facts (verified by the plan author, by running)

**The model server cannot transcribe.** `POST /v1/audio/transcriptions` →
`501 {"message":"The current model does not support audio input."}`, with and
without `verbose_json`/`timestamp_granularities`. `/v1/models` lists no audio
model. Transcription is local; do not add a server-backed adapter.

**`faster-whisper` 1.2.1, MIT.** Signatures confirmed against the installed
package:

```python
WhisperModel(model_size_or_path: str, device="auto", device_index=0,
             compute_type="default", cpu_threads=0, num_workers=1,
             download_root=None, local_files_only=False, files=None,
             revision=None, use_auth_token=None, **model_kwargs)

model.transcribe(audio: str | BinaryIO | np.ndarray, ..., word_timestamps=False,
                 vad_filter=False, ...) -> tuple[Iterable[Segment], TranscriptionInfo]
```

**Two facts the research missed and both matter:**

1. **`local_files_only=True` exists.** It guarantees no network access. Pass it
   — that makes "downloads nothing implicitly" *enforced* rather than merely
   conventional. A local path alone does not stop a lookup; this does.
2. **`transcribe` accepts `BinaryIO`.** The WAV bytes from ffmpeg go in as
   `io.BytesIO` with no temp file, so nothing is written to disk.

Field shapes, confirmed:

```
Segment: id, seek, start, end, text, tokens, avg_logprob, compression_ratio,
         no_speech_prob, words, temperature
Word:    start, end, word, probability
TranscriptionInfo: language, language_probability, duration, duration_after_vad, …
```

**`TimeSpan`'s fields are `start_s` and `end_s`**, and it rejects
`start_s >= end_s`.

**Audio extraction:**
```
ffmpeg -i <path> -vn -map 0:a:0 -ac 1 -ar 16000 -f wav -loglevel error -y -
```
With several audio streams ffmpeg's default selector picks exactly one and does
not mix — output is byte-identical with and without `-map 0:a:0`. The map is
kept for explicitness.

**A file with no audio stream exits 234** with `Output file does not contain any
stream` on stderr. **This is the opposite convention from frame extraction**,
which exits 0 with empty output when the seek is past the end. Streams are
checked by **exit status**; frames by **output length**. Both adapters must say
so, each naming the other.

**`avg_logprob` is not a confidence.** It is an average token log-probability.
`exp(avg_logprob)` is a defensible derivation, but it is an interpretation, not
a number whisper reported. If `TranscriptCue.confidence` is populated from it,
the code says so in a comment — this project does not present derived numbers as
measured ones.

---

## File Structure

**Created:**

| File | Responsibility |
| --- | --- |
| `src/readeverything/ports/audio.py` | `AudioExtractor` protocol. |
| `src/readeverything/ports/transcription.py` | `Transcriber` protocol. |
| `src/readeverything/adapters/ffmpeg_audio.py` | `AudioExtractor` over ffmpeg. |
| `src/readeverything/adapters/whisper_transcriber.py` | `Transcriber` over faster-whisper. |
| `src/readeverything/handlers/audio.py` | `AudioHandler`. |

**Modified:**

| File | Change |
| --- | --- |
| `src/readeverything/handlers/video.py` | Interleave cues when a transcriber is present. |
| `src/readeverything/adapters/model_probe.py` | Derive `Capability.ASR` from the injected transcriber. |
| `src/readeverything/composition.py` | Register `AudioHandler`; thread the transcriber. |
| `src/readeverything/__init__.py` | Export the new ports, adapters, handler. |
| `tests/fixtures_media.py` | Audio fixtures. |
| `pyproject.toml` | `transcription` extra. |

---

## Task 1: `AudioExtractor` and its ffmpeg adapter

**Files:**
- Create: `src/readeverything/ports/audio.py`, `src/readeverything/adapters/ffmpeg_audio.py`
- Test: `tests/unit/adapters/test_ffmpeg_audio.py`
- Modify: `tests/fixtures_media.py` (an `audio_only()` generator), `tests/unit/test_dependencies_stay_confined.py`

**Interfaces:**
- Produces: `class AudioExtractor(Protocol)` with `async def extract(self, path: str) -> bytes | None`. **`None` means the file has no audio track** — a normal answer, not an error.
- `class FfmpegAudio` implementing it.

- [ ] **Step 1: Write the failing tests**

```python
async def test_a_video_with_sound_yields_wav_bytes(sample_video: str) -> None:
    wav = await FfmpegAudio().extract(sample_video)
    assert wav is not None
    assert wav.startswith(b"RIFF")


async def test_a_file_with_no_audio_stream_returns_none(video_only_path: str) -> None:
    """The opposite convention from frame extraction, and both are measured.

    ffmpeg exits 234 with "Output file does not contain any stream" here, but
    exits 0 with empty output when a frame seek is past the end. Streams are
    checked by exit status; frames by output length. An adapter that checked the
    wrong one for either would be silently wrong in one direction and noisily
    wrong in the other.
    """
    assert await FfmpegAudio().extract(video_only_path) is None


async def test_a_file_that_is_not_media_returns_none(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not media at all")
    assert await FfmpegAudio().extract(str(junk)) is None


async def test_the_extractor_never_raises(tmp_path: Path) -> None:
    """A handler must be able to ask about anything."""
    assert await FfmpegAudio().extract(str(tmp_path / "absent.mp4")) is None
```

- [ ] **Step 2: Run to verify failure, then implement**

Use the verified command. Check `process.returncode != 0` → `None`. Also treat
empty output as `None` — belt and braces, and cheap. Comment the asymmetry with
`ffmpeg_frames.py`, naming it, as `ffmpeg_frames.py` names this one.

argv vector; timeout with kill-and-reap; output to stdout.

- [ ] **Step 3: Run and commit**

```bash
uv run --all-extras pytest -q && uv run --all-extras mypy && uv run --all-extras bandit -c pyproject.toml -r src -q
uv run --all-extras ruff format src tests
git add src/readeverything/ports/audio.py src/readeverything/adapters/ffmpeg_audio.py tests/
git commit -m "feat: extract a container's audio track, and know when there is none"
```

---

## Task 2: `Transcriber` and the faster-whisper adapter

**Files:**
- Create: `src/readeverything/ports/transcription.py`, `src/readeverything/adapters/whisper_transcriber.py`
- Test: `tests/unit/adapters/test_whisper_transcriber.py`
- Modify: `pyproject.toml`, `tests/unit/test_dependencies_stay_confined.py`, `src/readeverything/testing/fakes.py`

**Interfaces:**
- Produces:
  - `class Transcriber(Protocol)`: `model_id: str`; `async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]`.
  - `class WhisperTranscriber` — `__init__(self, *, model_dir: str, compute_type: str = "int8", device: str = "cpu")`. **`model_dir` is required and has no default.**
  - `FakeTranscriber` in `readeverything.testing.fakes`, shaped like the existing `FakeVision`: deterministic cues derived mechanically from input length, never real speech.

- [ ] **Step 1: Add the extra**

```toml
transcription = ["faster-whisper>=1.2,<2"]
```

Register `faster_whisper` in the confinement test for `whisper_transcriber.py` only.

- [ ] **Step 2: Write the failing tests**

```python
def test_the_model_directory_is_required() -> None:
    """No default, so a caller who has not configured ASR finds out at the
    composition root rather than when a first `represent()` quietly pulls
    several hundred megabytes from Hugging Face."""
    with pytest.raises(TypeError):
        WhisperTranscriber()  # type: ignore[call-arg]  # the point of the test


def test_construction_never_reaches_the_network(tmp_path: Path) -> None:
    """`local_files_only=True` is what makes "downloads nothing implicitly"
    enforced rather than merely conventional. A local path alone does not
    prevent a lookup; this flag does.
    """
    source = Path("src/readeverything/adapters/whisper_transcriber.py").read_text()
    assert "local_files_only=True" in source


def test_a_missing_model_directory_fails_loudly(tmp_path: Path) -> None:
    """Loud here, because this is the composition root's job to surface. The
    HANDLER is what must never raise, and it catches this."""
    with pytest.raises(InfrastructureError):
        WhisperTranscriber(model_dir=str(tmp_path / "nope"))


def test_the_fake_produces_cues_that_tile_without_gaps() -> None:
    """The fake stands in for a real transcriber in every unit test, so its
    shape has to be the shape the handler must cope with — cues in order, with
    silence between them."""
    cues = await FakeTranscriber().transcribe(b"x" * 1000, "audio/wav")
    assert cues
    assert all(c.span.start_s < c.span.end_s for c in cues)
    assert all(a.span.end_s <= b.span.start_s for a, b in zip(cues, cues[1:], strict=False))
```

- [ ] **Step 3: Implement**

`WhisperTranscriber.__init__` constructs `WhisperModel(model_dir, device=device,
compute_type=compute_type, local_files_only=True)` and translates any failure
into `InfrastructureError`. `model_id` is derived from the directory name plus
the compute type, so it identifies what actually ran and feeds the capability
fingerprint.

`transcribe` wraps the bytes in `io.BytesIO` — **`transcribe` accepts
`BinaryIO`, so nothing is written to disk** — calls
`model.transcribe(buffer, word_timestamps=False)`, and maps each `Segment` to:

```python
TranscriptCue(
    span=TimeSpan(segment.start, segment.end),
    text=segment.text.strip(),
    speaker=None,   # diarization is out of scope; None admits ignorance
    confidence=...,  # see below
)
```

**On `confidence`:** whisper returns `avg_logprob`, an average token
log-probability, not a confidence. `math.exp(avg_logprob)` is a defensible
derivation and it is an *interpretation*. If you populate `confidence`, say so
in a comment. **Passing `None` is also correct and is the safer default** — the
field admits ignorance, and this project does not present derived numbers as
measured ones. Choose, and justify the choice in the docstring.

`WhisperModel.transcribe` is synchronous and CPU-bound: run it in
`asyncio.to_thread`.

**A zero-length segment is possible** (whisper occasionally emits `start ==
end`). `TimeSpan` rejects it. Drop such segments and say so in a comment —
do not widen them into a span nothing observed.

- [ ] **Step 4: Run and commit**

---

## Task 3: `AudioHandler`

**Files:**
- Create: `src/readeverything/handlers/audio.py`
- Test: `tests/unit/handlers/test_audio_handler.py`

**Interfaces:**
- Produces: `AudioHandler(*, source, probe: StreamProbe, audio: AudioExtractor, transcriber: Transcriber | None = None)`; ClassVars `mime_patterns = ("kind:audio",)`, `priority = 0`, `handler_id = "audio"`, `handler_version = 1`; `requires()` → `frozenset({Capability.FFMPEG})`.
- Affordance `read_span(start_s, end_s)`, SEGMENT, requires `{FFMPEG, ASR}`.

- [ ] **Step 1: Write the failing tests**

```python
async def test_the_card_reports_codec_and_sample_rate_without_decoding(
    audio_path: str,
) -> None:
    card = await _handler(audio_path).describe(_ref())
    assert card.facts["audio_codec"]
    assert card.facts["sample_rate"] == 44100


async def test_every_character_resolves_to_the_moment_it_was_said(
    audio_path: str,
) -> None:
    """`TranscriptCue`'s first producer, and the property the cycle exists for."""
    rendered = await _handler(audio_path, transcriber=FakeTranscriber()).represent(
        _ref(), Budget(max_chars=None)
    )
    assert isinstance(rendered.locator_map.resolve(0), TimeSpan)


async def test_the_timeline_covers_the_whole_duration_with_no_gaps(
    audio_path: str,
) -> None:
    """Cues have silence between them, and `LocatorMap` must be total. Each
    cue's span extends to the next cue's start; the last extends to the file
    duration — exactly `VideoHandler._bounds`, one medium over.
    """
    rendered = await _handler(audio_path, transcriber=FakeTranscriber()).represent(
        _ref(), Budget(max_chars=None)
    )
    spans = [s.locator for s in rendered.locator_map.segments]
    assert spans[0].start_s == 0.0
    for earlier, later in zip(spans, spans[1:], strict=False):
        assert earlier.end_s == pytest.approx(later.start_s)


async def test_without_a_transcriber_it_says_so_and_is_not_empty(
    audio_path: str,
) -> None:
    """An audio file has no cheaper layer — without transcription it has only a
    duration and a codec. An empty string would claim there was nothing to hear."""
    rendered = await _handler(audio_path, transcriber=None).represent(
        _ref(), Budget(max_chars=None)
    )
    assert rendered.text.strip()
    assert any("transcri" in d.what.lower() for d in rendered.degradations)


async def test_silence_is_distinguishable_from_no_transcriber(audio_path: str) -> None:
    """Two different facts. A transcriber that ran and heard nothing is not the
    same as no transcriber, and reporting them identically would lose the
    difference between "this file is silent" and "nobody listened"."""
    silent = await _handler(audio_path, transcriber=_EmptyTranscriber()).represent(
        _ref(), Budget(max_chars=None)
    )
    absent = await _handler(audio_path, transcriber=None).represent(
        _ref(), Budget(max_chars=None)
    )
    assert silent.text != absent.text


async def test_a_file_with_no_audio_track_degrades_and_says_which(
    video_only_path: str,
) -> None:
    rendered = await _handler(video_only_path).represent(_ref(), Budget(max_chars=None))
    assert rendered.degradations
    assert "audio" in rendered.degradations[0].detail.lower()
```

- [ ] **Step 2: Implement**

Follow `handlers/video.py` closely — it solved the same shape. `describe()` uses
`StreamProbe` only, no extraction. `represent()` extracts, transcribes, and
tiles cue spans to the next cue's start with the last running to duration.

`barriers=()`. Video's scene cuts measure something real; the audio analogue is
a speaker turn, which needs diarization. An empty tuple is honest.

Truncation drops nothing but text and reports `kept {len(text)} of {len(full)}`.

- [ ] **Step 3: Run and commit**

---

## Task 4: `Capability.ASR`, composition, and the front door

**Files:**
- Modify: `src/readeverything/adapters/model_probe.py`, `src/readeverything/composition.py`, `src/readeverything/__init__.py`
- Test: `tests/unit/adapters/test_model_probe.py`, `tests/unit/test_composition.py`, `tests/integration/test_audio.py`

**Interfaces:**
- `ModelProbe(*, vision=None, transcriber=None)` — derives `Capability.ASR` from the injected transcriber's `model_id`.
- `build_perception(..., transcriber: Transcriber | None = None)`.

- [ ] **Step 1: Write the failing tests**

```python
async def test_asr_is_derived_from_the_injected_transcriber() -> None:
    """The same seam `ModelProbe` closes for VISION. A capability cannot
    disagree with the model actually present, because it is read from it."""
    probe = ModelProbe(transcriber=FakeTranscriber())
    assert await probe.revision(Capability.ASR) == FakeTranscriber().model_id


async def test_no_transcriber_means_no_asr_capability() -> None:
    assert await ModelProbe(vision=FakeVision()).revision(Capability.ASR) is None


async def test_read_span_is_not_offered_without_a_transcriber(media_root) -> None:
    """Negotiation, not a runtime apology."""
    perception = await build_perception(media_root, probe_binaries=False)
    card = await perception.inspect("clip.wav")
    assert "read_span" not in {a.name for a in card.affordances}
```

- [ ] **Step 2: Implement, run `make check`, commit**

`_audio_handler` in the composition root follows `_video_handler`: constructed
unconditionally, dropped by the registry when `FFMPEG` is absent. Register before
`BinaryHandler`, which stays last.

---

## Task 5: Video carries a transcript

**Files:**
- Modify: `src/readeverything/handlers/video.py`, `src/readeverything/composition.py`
- Test: `tests/unit/handlers/test_video_handler.py`

**Interfaces:** `VideoHandler` gains `audio: AudioExtractor | None = None` and `transcriber: Transcriber | None = None`.

- [ ] **Step 1: Write the failing tests**

```python
async def test_video_without_a_transcriber_is_unchanged(sample_video: str) -> None:
    """The regression guard. Existing behaviour must be byte-identical, because
    every video test in the suite depends on it."""
    before = await _handler(sample_video, vision=FakeVision()).represent(
        _ref(), Budget(max_chars=None)
    )
    after = await _handler(
        sample_video, vision=FakeVision(), transcriber=None
    ).represent(_ref(), Budget(max_chars=None))
    assert before.text == after.text


async def test_cues_and_frames_interleave_in_timestamp_order(sample_video: str) -> None:
    rendered = await _handler(
        sample_video, vision=FakeVision(), audio=FfmpegAudio(), transcriber=FakeTranscriber()
    ).represent(_ref(), Budget(max_chars=None))
    spans = [s.locator for s in rendered.locator_map.segments]
    assert spans == sorted(spans, key=lambda s: s.start_s)
    assert spans[0].start_s == 0.0
    for earlier, later in zip(spans, spans[1:], strict=False):
        assert earlier.end_s == pytest.approx(later.start_s)
```

- [ ] **Step 2: Implement**

Merge the sampled moments and the cues into one timestamp-ordered sequence, then
apply the same tiling: each entry's span runs to the next entry's start, the
last to the duration. A frame description and a cue at the same instant are two
entries, not a conflict.

- [ ] **Step 3: Run `make check` and commit**

---

## Task 6: Live validation

**Files:** `tests/live/test_transcription.py`

Marked `live`, deselected by default, and **skipped when no model directory is
configured** — which on this machine is always, since no weights are present.
The skip is the honest state, not a gap: the degraded path is what this machine
exercises and Task 3 tests it fully.

Assert structure only: cues came back, spans are ordered and non-degenerate, and
the rendition is located in time. **Never assert on transcribed text.**

**Do not run these without telling the human partner** — a live transcription
run uses their machine's CPU heavily.

---

## Plan Self-Review

**Spec coverage.** §3 `Transcriber` → Task 2. §4 `AudioExtractor` → Task 1. §5
timeline and silence → Task 3. §6 transcription in `represent()` → Task 3. §7
video interleaving → Task 5. §8 affordances and `Capability.ASR` → Tasks 3, 4.
§9 acceptance 1–10 → Tasks 1, 3, 4, 5.

**Ordering.** Extraction before transcription because the transcriber consumes
its bytes. The handler before composition. Video interleaving last among the
code tasks, because it modifies a handler every existing test depends on and its
regression guard is the first test written for it.

**Known risks a reviewer should hold me to.**
- Task 2's `confidence` decision is deliberately left to the implementer with a
  stated default (`None`) and a requirement to justify. If it populates the
  field from `avg_logprob` without a comment, that is exactly the defect this
  project keeps finding.
- Task 5 modifies `VideoHandler`, and the regression guard compares text
  equality with and without a transcriber. If the merge changes formatting even
  when no cues exist, every video test moves at once.
- Task 3's silence-versus-no-transcriber test needs an `_EmptyTranscriber` that
  returns zero cues. That is not the same as `FakeTranscriber`, and writing it as
  a variant that returns `()` is the point.
