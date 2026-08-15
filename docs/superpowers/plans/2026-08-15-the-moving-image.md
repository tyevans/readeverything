# The Moving Image Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read video, give `TimeSpan` its first producer, and fix the capability probe that was broken without anyone knowing.

**Architecture:** `BinaryProbe` gains a per-executable version flag and refuses to call a warning a version. A `StreamProbe` port answers duration/streams/codecs from one `ffprobe` header read. A `FrameExtractor` port pulls a single frame at a timestamp, validated by output length because ffmpeg fails silently. `VideoHandler` builds a gapless timeline of frame descriptions located by `TimeSpan`, with barriers at scene cuts.

**Tech Stack:** Python 3.13, ffmpeg/ffprobe 6.1 via `asyncio.create_subprocess_exec`, pydantic v2, pytest, mypy --strict, ruff, import-linter, bandit, coverage.

**Spec:** `docs/superpowers/specs/2026-08-15-readeverything-the-moving-image-design.md`

## Global Constraints

- **The library reads NO environment variables under `src/`.** Enforced by `tests/unit/test_reads_no_environment.py`.
- **Python 3.13, PEP 695 inline type parameters.** A module-level `TypeVar` is a defect.
- **`mypy --strict`, `warn_unused_ignores = true`**, over `src` and `tests`. No new `# type: ignore` without a comment naming why.
- **Import-linter layered contract**, outermost first: `composition, testing, agent, pipeline, registry, handlers, adapters, ports, domain`. **Handlers must not import from `adapters/`** — see the PDF precedent.
- **`readeverything.testing` may import only `ports` and `domain`.**
- **Third-party and stdlib-subprocess imports are pinned** by `tests/unit/test_dependencies_stay_confined.py`.
- **No handler ever raises** from `describe`, `invoke`, or `represent`. They degrade.
- **Never assert on model text.** Assert structure and locators.
- **Every subprocess uses `create_subprocess_exec` with an argv vector.** Never a shell string, never `shell=True`. Paths and timestamps are separate argv elements.
- **Every subprocess has a wall-clock timeout with kill-and-reap.**
- **Coverage floor 92.** Run the full gate set with `make check`.
- **A law must be able to fail.**

---

## Measured facts (verified by the plan author against ffmpeg/ffprobe 6.1.1)

Use these. Do not re-derive them.

**Probe, one header read, no decode:**
```
ffprobe -v error -show_format -show_streams -of json <path>
```
```
format keys: bit_rate, duration, filename, format_long_name, format_name,
             nb_programs, nb_streams, probe_score
format.duration = "5.000000"   (a STRING)
format.format_name = "mov,mp4,m4a,3gp,3g2,mj2"   (a comma-joined LIST)
stream 0 video h264 320 240 r_frame_rate="10/1"  sample_rate=None
stream 1 audio aac  None None r_frame_rate="0/0" sample_rate="44100"
```
Note three traps in that output: `duration` is a string; `format_name` is a
comma-joined list of candidate formats, not one name; and `r_frame_rate` is a
**rational string** (`"10/1"`), which for a non-video stream is `"0/0"` — parse
it by splitting on `/` and guard the zero denominator.

**Frame extraction:**
```
ffmpeg -ss <T> -i <path> -frames:v 1 -f image2 -vcodec png -loglevel error -y -
```

**THE TRAP, measured on a real 5-second file:**

| Case | exit | stdout bytes | stderr bytes |
| --- | --- | --- | --- |
| `-ss 2.5` (in range) | 0 | 16942 | 0 |
| `-ss 999` (past end) | **0** | **0** | **0** |

Identical exit status. No stderr either way. **Only the byte count distinguishes
success from silent failure.** A handler checking `returncode` would hand an
empty PNG to a model as "the frame at t=999".

**And the asymmetry, also measured** — extracting audio from a file with no
audio stream:

| Case | exit | stdout bytes | stderr |
| --- | --- | --- | --- |
| no audio stream | **234** | 0 | `Output file does not contain any stream` |

So ffmpeg has two different failure conventions. Frames fail silently with exit
0; missing streams fail loudly with a non-zero exit. **Check output length for
frames, exit status for streams, and never unify the two.**

`-ss` **before** `-i` seeks at the demuxer (fast, keyframe-adjacent). After
`-i` it decodes from the start (accurate, slow). This plan uses before-`-i`;
`r_frame_rate` tells a caller the precision they got.

**Fixture generation** (no committed binaries):
```
ffmpeg -y -loglevel error -f lavfi -i testsrc=duration=5:size=320x240:rate=10 \
  -f lavfi -i sine=frequency=440:duration=5 -c:v libx264 -c:a aac -shortest out.mp4
```

---

## File Structure

**Created:**

| File | Responsibility |
| --- | --- |
| `src/readeverything/ports/streams.py` | `StreamProbe` protocol, `MediaFacts`, `StreamInfo`. |
| `src/readeverything/ports/frames.py` | `FrameExtractor` protocol. |
| `src/readeverything/adapters/ffprobe_streams.py` | `StreamProbe` over `ffprobe`. |
| `src/readeverything/adapters/ffmpeg_frames.py` | `FrameExtractor` over `ffmpeg`. |
| `src/readeverything/handlers/video.py` | `VideoHandler`. |
| `tests/fixtures_media.py` | Generated media, never committed. |

**Modified:**

| File | Change |
| --- | --- |
| `src/readeverything/adapters/binary_probe.py` | Per-executable flags; refuse a warning as a version. |
| `src/readeverything/composition.py` | Register `VideoHandler` when ffmpeg is discovered. |
| `src/readeverything/__init__.py` | Export the new ports, adapters, handler. |
| `pyproject.toml` / confinement test | No new third-party dependency — ffmpeg is an OS binary. |

**Note:** this cycle adds **no Python dependency**. ffmpeg is discovered, not imported, which is why `Capability.FFMPEG` exists.

---

## Task 1: Fix `BinaryProbe`

**Files:**
- Modify: `src/readeverything/adapters/binary_probe.py`
- Test: `tests/unit/adapters/test_binary_probe.py`, and a new `tests/integration/test_real_binaries.py`

**Interfaces:**
- Produces: `DEFAULT_EXECUTABLES: Mapping[Capability, tuple[str, str]]` — capability to `(executable, version_flag)`. This replaces the old `Mapping[Capability, str]` plus a single global `version_flag`.
- `BinaryProbe(*, executables=None, timeout_s=5.0)` — the `version_flag` constructor parameter is **removed**; it was the bug.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/adapters/test_binary_probe.py
async def test_each_capability_carries_its_own_version_flag() -> None:
    """One global flag was the bug.

    `exiftool -version` is not exiftool's version invocation — it wants `-ver`.
    With a single flag for every executable, exiftool was installed on the
    machine and reported absent, and "absent" is indistinguishable from "not
    installed".
    """
    assert DEFAULT_EXECUTABLES[Capability.EXIFTOOL] == ("exiftool", "-ver")
    assert DEFAULT_EXECUTABLES[Capability.FFMPEG] == ("ffmpeg", "-version")
    assert DEFAULT_EXECUTABLES[Capability.LIBREOFFICE] == ("libreoffice", "--version")


async def test_a_warning_is_not_a_version(tmp_path: Path) -> None:
    """`libreoffice -version` printed a deprecation warning, and the probe
    recorded it as the revision. That string then entered the capability
    fingerprint and therefore every artifact cache key — a warning became part
    of this library's cache identity.

    Nothing established that the captured line was a version. Under uncertainty
    the probe returns None, which its own contract already requires.
    """
    script = tmp_path / "warner"
    script.write_text("#!/bin/sh\necho 'Warning: -version is deprecated.  Use --version instead.'\n")
    script.chmod(0o755)
    probe = BinaryProbe(executables={Capability.FFMPEG: (str(script), "-version")})
    assert await probe.revision(Capability.FFMPEG) is None


async def test_an_error_line_is_not_a_version(tmp_path: Path) -> None:
    script = tmp_path / "errorer"
    script.write_text("#!/bin/sh\necho 'Error: no such option'\n")
    script.chmod(0o755)
    probe = BinaryProbe(executables={Capability.FFMPEG: (str(script), "-version")})
    assert await probe.revision(Capability.FFMPEG) is None


async def test_a_real_version_line_is_accepted(tmp_path: Path) -> None:
    """The rejection must not be so eager that it rejects genuine versions."""
    script = tmp_path / "versioner"
    script.write_text("#!/bin/sh\necho 'ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023'\n")
    script.chmod(0o755)
    probe = BinaryProbe(executables={Capability.FFMPEG: (str(script), "-version")})
    revision = await probe.revision(Capability.FFMPEG)
    assert revision is not None and "6.1.1" in revision
```

```python
# tests/integration/test_real_binaries.py
"""The probe, against the machine's real executables.

Its unit tests use shell-script stand-ins, which is the right way to test the
contract — but a probe is the one component whose correctness cannot be
established against a stand-in. These ran clean while exiftool was installed and
reported absent, and while libreoffice's recorded revision was a deprecation
warning.

This test cannot assert WHICH binaries exist, since that varies by machine. It
asserts the property that actually failed: presence on PATH and a discovered
revision must agree.
"""

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("capability", list(DEFAULT_EXECUTABLES))
async def test_presence_on_path_agrees_with_discovery(capability: Capability) -> None:
    executable, _flag = DEFAULT_EXECUTABLES[capability]
    on_path = shutil.which(executable) is not None
    revision = await BinaryProbe().revision(capability)

    if on_path:
        assert revision is not None, (
            f"{executable} is on PATH but the probe reports it unavailable"
        )
        assert not revision.lower().startswith(("warning", "error")), (
            f"{executable}'s revision is a diagnostic, not a version: {revision!r}"
        )
    else:
        assert revision is None
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run --all-extras pytest tests/unit/adapters/test_binary_probe.py tests/integration/test_real_binaries.py -v
```
Expected: FAIL — `DEFAULT_EXECUTABLES` maps to a bare string, and the warning is accepted as a revision. **On this machine the integration test fails on exiftool specifically.**

- [ ] **Step 3: Implement**

```python
#: The executable each capability is provided by, and the flag that makes THAT
#: executable print a version and exit. The flag belongs to the executable, not
#: to the probe: a single global `-version` reported exiftool absent while it
#: was installed, because exiftool's flag is `-ver`.
DEFAULT_EXECUTABLES: Mapping[Capability, tuple[str, str]] = {
    Capability.FFMPEG: ("ffmpeg", "-version"),
    Capability.EXIFTOOL: ("exiftool", "-ver"),
    Capability.LIBREOFFICE: ("libreoffice", "--version"),
    Capability.TESSERACT: ("tesseract", "--version"),
}

#: Prefixes that mean the tool talked to us rather than identified itself.
_NOT_A_VERSION = ("warning", "error", "usage")


def _as_revision(stdout: bytes) -> str | None:
    """The first line, if it plausibly identifies the tool.

    `libreoffice -version` prints "Warning: -version is deprecated…", and the
    probe recorded that as the revision. It then entered the capability
    fingerprint and therefore every artifact cache key. Nothing had established
    that the captured line was a version — this is the probe's own contract
    turned on itself, so it now checks.
    """
    lines = stdout.decode("utf-8", errors="replace").strip().splitlines()
    if not lines:
        return None
    first = lines[0].strip()
    if not first or first.lower().startswith(_NOT_A_VERSION):
        return None
    return first
```

`revision()` unpacks `(executable, flag)` from the mapping and passes both to
`create_subprocess_exec`. The `version_flag` constructor parameter is removed.

- [ ] **Step 4: Run everything**

```bash
uv run --all-extras pytest -q && uv run --all-extras mypy && uv run --all-extras bandit -c pyproject.toml -r src -q
```

**Report the integration test's result explicitly** — which capabilities this
machine discovered, and whether exiftool now reports a revision. That is the
bug, and its fix must be observed rather than assumed.

- [ ] **Step 5: Commit**

```bash
uv run --all-extras ruff format src tests
git add src/readeverything/adapters/binary_probe.py tests/
git commit -m "fix(adapters): a version flag belongs to its executable, and a warning is not a version"
```

---

## Task 2: `StreamProbe` and the ffprobe adapter

**Files:**
- Create: `src/readeverything/ports/streams.py`, `src/readeverything/adapters/ffprobe_streams.py`
- Test: `tests/unit/ports/test_streams.py`, `tests/unit/adapters/test_ffprobe_streams.py`
- Modify: `tests/unit/test_dependencies_stay_confined.py`

**Interfaces:**
- Produces:
  - `StreamInfo` — frozen: `kind: Literal["video","audio"]`, `codec: str`, `width: int | None`, `height: int | None`, `frame_rate: float | None`, `sample_rate: int | None`, `channels: int | None`.
  - `MediaFacts` — frozen: `duration_s: float`, `container: str`, `streams: tuple[StreamInfo, ...]`. Helpers `video_streams` / `audio_streams`.
  - `class StreamProbe(Protocol)`: `async def probe(self, path: str) -> MediaFacts`.
  - `class FfprobeStreams` implementing it.

**Note on the port split.** `MediaProbe` (Spec 4) answers paginated documents and returns `DocumentFacts`. This is a different question with a different answer, so it is a different protocol. A protocol whose return type depends on its input has two jobs.

- [ ] **Step 1: Write the failing tests**

```python
async def test_a_probe_reports_duration_and_both_streams(sample_video: str) -> None:
    facts = await FfprobeStreams().probe(sample_video)
    assert facts.duration_s == pytest.approx(5.0, abs=0.2)
    assert len(facts.video_streams) == 1
    assert len(facts.audio_streams) == 1


async def test_duration_is_a_float_though_ffprobe_returns_a_string(sample_video: str) -> None:
    """`format.duration` comes back as the string "5.000000". A caller doing
    arithmetic on a string gets a confusing failure a long way from here."""
    facts = await FfprobeStreams().probe(sample_video)
    assert isinstance(facts.duration_s, float)


async def test_the_frame_rate_rational_is_parsed(sample_video: str) -> None:
    """`r_frame_rate` is "10/1", not 10. And for an audio stream it is "0/0",
    which is a division by zero waiting for whoever forgets to guard it."""
    facts = await FfprobeStreams().probe(sample_video)
    assert facts.video_streams[0].frame_rate == pytest.approx(10.0)
    assert facts.audio_streams[0].frame_rate is None


async def test_a_file_that_is_not_media_raises_infrastructure_error(tmp_path: Path) -> None:
    """The probe may raise; the HANDLER must not. Keeping it loud here lets the
    handler decide how to degrade rather than receive fabricated facts."""
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    with pytest.raises(InfrastructureError):
        await FfprobeStreams().probe(str(junk))


async def test_probing_does_not_decode(sample_video: str) -> None:
    """The card must stay cheap. Asserted as a bound on wall time rather than
    left as a claim — a decode of even this tiny file is orders of magnitude
    slower than a header read."""
    ...  # see Step 3 note
```

- [ ] **Step 2: Run to verify failure, then implement**

Requirements, all load-bearing:

- `format.duration` is a **string** — parse to float, and treat a missing or
  unparseable duration as `InfrastructureError` rather than defaulting to `0.0`.
  A fabricated zero duration would make every `TimeSpan` in `represent()` a lie.
- `format.format_name` is a **comma-joined list of candidate formats**
  (`"mov,mp4,m4a,3gp,3g2,mj2"`). Keep the whole string; do not pick one and
  present it as *the* container, which would assert an identification ffprobe
  declined to make.
- `r_frame_rate` is a **rational string**; `"0/0"` for non-video. Parse by
  splitting on `/`, and return `None` when the denominator is zero rather than
  raising.
- Bound the work: `-analyzeduration` and `-probesize` at fixed small values, and
  an `asyncio.wait_for` timeout with kill-and-reap.
- argv vector, never a shell string.

For the cheapness test in Step 1, assert the probe completes well within a
generous bound (e.g. 5 s) — enough to catch an accidental full decode without
being a flaky timing assertion.

- [ ] **Step 3: Commit**

```bash
uv run --all-extras ruff format src tests
git add src/readeverything/ports/streams.py src/readeverything/adapters/ffprobe_streams.py tests/
git commit -m "feat: read a media file's shape without decoding a frame"
```

---

## Task 3: Media fixtures, generated

**Files:**
- Create: `tests/fixtures_media.py`
- Modify: `tests/integration/conftest.py`

**Interfaces:**
- Produces: `video_with_audio(seconds=5, size="320x240", rate=10) -> bytes`, `video_only(seconds=2) -> bytes`, `scene_cuts(...) -> bytes` (concatenated distinct sources so a cut detector has something to find).

- [ ] **Step 1: Write the generators**

Use the verified command from Measured Facts. `ffmpeg` writes to a path, so
generate into a `tmp_path` and read the bytes back.

**Skip cleanly when ffmpeg is absent.** These fixtures cannot exist without it,
so the module exposes `ffmpeg_available() -> bool` and the tests that need media
are skipped rather than failed on a machine without it. The library's own
behaviour without ffmpeg is tested separately, by Task 6, and that test must not
be skipped.

- [ ] **Step 2: A test guarding the fixtures**

```python
def test_the_video_only_fixture_really_has_no_audio_stream() -> None:
    """Task 5's "no audio stream" path is tested against this fixture. If it
    ever gains an audio track, that test silently becomes a different test."""
    facts = _ffprobe(video_only())
    assert not facts.audio_streams


def test_the_scene_cut_fixture_really_contains_a_cut() -> None:
    """Barrier tests depend on a detectable cut existing. A fixture of uniform
    content would make "no barriers found" look correct."""
    ...
```

- [ ] **Step 3: Commit**

---

## Task 4: `FrameExtractor`, and the silent failure

**Files:**
- Create: `src/readeverything/ports/frames.py`, `src/readeverything/adapters/ffmpeg_frames.py`
- Test: `tests/unit/adapters/test_ffmpeg_frames.py`

**Interfaces:**
- Produces: `class FrameExtractor(Protocol)` with `async def frame_at(self, path: str, seconds: float) -> bytes | None` — **`None` means no frame at that time**, not an error. And `class FfmpegFrames` implementing it.

- [ ] **Step 1: Write the failing tests**

```python
async def test_a_frame_in_range_comes_back_as_a_png(sample_video: str) -> None:
    png = await FfmpegFrames().frame_at(sample_video, 2.5)
    assert png is not None
    assert png.startswith(b"\x89PNG")


async def test_seeking_past_the_end_returns_none_rather_than_empty_bytes(
    sample_video: str,
) -> None:
    """The measured trap. ffmpeg exits 0 with zero bytes and no stderr when the
    seek is past the end — it does not error:

        -ss 2.5  -> exit 0, 16942 bytes, no stderr
        -ss 999  -> exit 0,     0 bytes, no stderr

    An adapter checking `returncode` would return b"" and a handler would hand
    an empty PNG to a model as "the frame at t=999". Output length is the only
    thing that distinguishes success here.
    """
    assert await FfmpegFrames().frame_at(sample_video, 999.0) is None


async def test_a_negative_time_returns_none(sample_video: str) -> None:
    assert await FfmpegFrames().frame_at(sample_video, -1.0) is None


async def test_a_file_that_is_not_media_returns_none(tmp_path: Path) -> None:
    """The extractor answers "no frame", never raises. A handler must be able to
    call it about anything."""
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    assert await FfmpegFrames().frame_at(str(junk), 0.0) is None
```

- [ ] **Step 2: Implement**

```python
async def frame_at(self, path: str, seconds: float) -> bytes | None:
    """One frame as PNG bytes, or None if there is no frame at that time.

    ffmpeg has two failure conventions and they are not interchangeable.
    Seeking past the end exits ZERO with empty stdout and no stderr — measured,
    not assumed. Extracting a missing stream exits 234 with a message. So a
    frame is validated by OUTPUT LENGTH and a stream by exit status, and
    unifying the two checks reintroduces the silent one.

    Returning None rather than b"" matters: an empty bytes object is a value a
    caller can pass on, and an empty PNG presented as a frame is an observation
    nothing made.
    """
```

argv vector; `-ss` before `-i`; `-frames:v 1`; output to `-`; timeout with
kill-and-reap; `-loglevel error`.

- [ ] **Step 3: Run and commit**

---

## Task 5: `VideoHandler` — card and the `TimeSpan` timeline

**Files:**
- Create: `src/readeverything/handlers/video.py`
- Test: `tests/unit/handlers/test_video_handler.py`

**Interfaces:**
- Produces: `VideoHandler(*, source, probe: StreamProbe, frames: FrameExtractor, vision: VisionModel | None = None)`; ClassVars `mime_patterns = ("kind:video",)`, `priority = 0`, `handler_id = "video"`, `handler_version = 1`.
- `requires()` returns `frozenset({Capability.FFMPEG})` — without ffmpeg the handler does not register at all, and video files fall to the binary fallback.

- [ ] **Step 1: Write the failing tests**

```python
async def test_the_card_reports_duration_and_resolution_without_decoding(
    sample_video: str,
) -> None:
    card = await _handler(sample_video).describe(_ref())
    assert card.facts["duration_s"] == pytest.approx(5.0, abs=0.2)
    assert card.facts["width"] == 320
    assert card.facts["height"] == 240


async def test_every_character_resolves_to_the_moment_it_describes(
    sample_video: str,
) -> None:
    """`TimeSpan`'s first producer. The property the cycle exists for."""
    rendered = await _handler(sample_video, vision=FakeVision()).represent(
        _ref(), Budget(max_chars=None)
    )
    first = rendered.locator_map.resolve(0)
    last = rendered.locator_map.resolve(len(rendered.text) - 1)
    assert isinstance(first, TimeSpan) and isinstance(last, TimeSpan)
    assert first.start == 0.0
    assert last.end == pytest.approx(5.0, abs=0.5)


async def test_the_timeline_is_gapless_and_covers_the_whole_duration(
    sample_video: str,
) -> None:
    """`LocatorMap` requires total gapless coverage, so the stretches between
    sampled frames belong to the sample that starts them. A timeline with holes
    is a timeline that cannot answer "what was on screen at 3.1 seconds"."""
    rendered = await _handler(sample_video, vision=FakeVision()).represent(
        _ref(), Budget(max_chars=None)
    )
    spans = [s.locator for s in rendered.locator_map.segments]
    assert spans[0].start == 0.0
    for earlier, later in zip(spans, spans[1:], strict=False):
        assert earlier.end == pytest.approx(later.start)


async def test_a_frame_span_is_never_zero_width(sample_video: str) -> None:
    """`TimeSpan.__post_init__` rejects start >= end, so a frame — a point in
    time — is inexpressible as a point. Its span is one frame's duration, taken
    from r_frame_rate: the honest width, not an arbitrary epsilon."""
    rendered = await _handler(sample_video, vision=FakeVision()).represent(
        _ref(), Budget(max_chars=None)
    )
    for segment in rendered.locator_map.segments:
        assert segment.locator.end > segment.locator.start


async def test_without_vision_the_timeline_still_reports_its_structure(
    sample_video: str,
) -> None:
    """A video is not empty because nothing looked at it. The scanned-PDF lesson
    at a new site: report what is there and say what was not done."""
    rendered = await _handler(sample_video, vision=None).represent(
        _ref(), Budget(max_chars=None)
    )
    assert rendered.text.strip()
    assert rendered.locator_map.length == len(rendered.text)
    assert any("vision" in d.what.lower() or "describe" in d.what.lower()
               for d in rendered.degradations)


async def test_an_unreadable_video_degrades_rather_than_raising(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    rendered = await _handler(str(junk)).represent(_ref(), Budget(max_chars=None))
    assert rendered.degradations
```

- [ ] **Step 2: Implement**

Requirements:

- The card comes from `StreamProbe` only. No frame is extracted during
  `describe` — the cheapness discipline `MediaProbe` already enforces for PDF.
- `represent()` samples at a fixed interval (default 5 s, a constructor
  argument), describes each sampled frame through `vision` when present, and
  builds one `LocatorSegment` per sample running from that sample's timestamp to
  the next sample's — the last running to the duration.
- Truncation under `Budget` drops barriers beyond the kept text, exactly as the
  three existing handlers do, and reports `kept {len(text)} of {len(full)}`.
- The handler never raises.

- [ ] **Step 3: Run and commit**

---

## Task 6: Affordances, composition, and the no-ffmpeg path

**Files:**
- Modify: `src/readeverything/handlers/video.py`, `src/readeverything/composition.py`, `src/readeverything/__init__.py`
- Test: `tests/unit/handlers/test_video_handler.py`, `tests/unit/test_composition.py`, `tests/integration/test_video.py`

**Interfaces:**
- Produces: `FrameAtParams(seconds: float)`, `DescribeFrameParams(seconds: float, prompt: str = ...)`; affordances `frame_at` (SEGMENT, requires `FFMPEG`) and `describe_frame` (DEEP, requires `FFMPEG` + `VISION`).

- [ ] **Step 1: Write the failing tests**

```python
async def test_frame_at_returns_an_image_located_in_time(sample_video: str) -> None:
    rendition = await _handler(sample_video).invoke(
        _ref(), "frame_at", FrameAtParams(seconds=2.5)
    )
    assert isinstance(rendition.content, ImageContent)
    assert rendition.content.data.startswith(b"\x89PNG")
    assert isinstance(rendition.locator, TimeSpan)


async def test_a_frame_past_the_end_degrades_and_says_why(sample_video: str) -> None:
    """Never an empty image presented as a frame — the measured trap."""
    rendition = await _handler(sample_video).invoke(
        _ref(), "frame_at", FrameAtParams(seconds=999.0)
    )
    assert rendition.degraded
    assert not isinstance(rendition.content, ImageContent)


async def test_no_ffmpeg_means_no_video_handler(tmp_path, monkeypatch) -> None:
    """Negotiation against a real OS dependency, for the first time.

    The handler REQUIRES ffmpeg, so without it the registry drops the handler
    entirely and video files fall to the binary fallback. Narrower, not broken.
    """
    perception = await build_perception(
        tmp_path, capabilities=CapabilitySet.empty(), probe_binaries=False
    )
    assert "VideoHandler" not in {type(h).__name__ for h in perception.registry.handlers}
```

- [ ] **Step 2: Implement**

`_optional_video_handler` follows `_optional_pdf_handler`'s shape, but the gate
is a **capability**, not an import — there is no Python package to import. The
registry already drops handlers whose `requires()` is unsatisfied, so the
composition root constructs it and the registry filters it. Register before
`BinaryHandler`, which stays last.

Export from the front door, `_LAZY` and `TYPE_CHECKING` in sync and sorted.

- [ ] **Step 3: Run `make check` and commit**

---

## Task 7: Barriers at scene cuts

**Files:**
- Modify: `src/readeverything/adapters/ffmpeg_frames.py` (or a sibling), `src/readeverything/handlers/video.py`
- Test: `tests/unit/adapters/`, `tests/unit/handlers/test_video_handler.py`

**Interfaces:**
- Produces: `async def scene_cuts(self, path: str, threshold: float = 0.4) -> tuple[float, ...]` on the frame adapter — timestamps where the content changes sharply.

- [ ] **Step 1: Write the failing tests**

```python
async def test_a_video_with_a_cut_reports_a_cut(scene_cut_video: str) -> None:
    cuts = await FfmpegFrames().scene_cuts(scene_cut_video)
    assert cuts


async def test_uniform_content_reports_no_cuts(sample_video: str) -> None:
    """"No cuts found" must be a real answer, not a stand-in for "detection
    failed" — otherwise a caller cannot tell an unedited video from a broken
    detector."""
    assert await FfmpegFrames().scene_cuts(sample_video) == ()


async def test_barriers_land_at_cuts(scene_cut_video: str) -> None:
    rendered = await _handler(scene_cut_video, vision=FakeVision()).represent(
        _ref(), Budget(max_chars=None)
    )
    assert rendered.barriers
    for barrier in rendered.barriers:
        assert 0 < barrier < len(rendered.text)
```

- [ ] **Step 2: Implement**

`ffmpeg -i <path> -filter:v "select='gt(scene,<threshold>)',showinfo" -f null -`
writes `pts_time:` values to stderr. Parse them. **Detection failing and
detection finding nothing must be distinguishable** — the first returns an empty
tuple *and* records a degradation; the second returns an empty tuple silently.

Map each cut timestamp to the character offset where its sample's description
begins, and pass those as `barriers`.

- [ ] **Step 3: Run `make check` and commit**

---

## Plan Self-Review

**Spec coverage.** §3 probe fix → Task 1. §4 ffprobe card → Task 2. §5 frames and
the silent failure → Task 4. §6 timeline → Task 5. §7 affordances → Task 6. §8
safety → Tasks 1, 2, 4 (argv, timeouts, bounded probing). §9 acceptance 1–9 →
Tasks 1, 5, 6, 7.

**Ordering.** The probe fix comes first because everything downstream negotiates
on it. Fixtures precede the adapters that consume them. The card precedes the
timeline that uses its frame rate. Affordances precede barriers, which are the
most optional piece and the easiest to drop if scene detection disappoints.

**Known risks a reviewer should hold me to.**
- Task 5's gapless-timeline requirement is stated and its arithmetic is not
  shown. The zip-based test is what catches an off-by-one, and it must run on a
  video whose duration is not an exact multiple of the sample interval.
- Task 7's scene detection is the piece most likely to be flaky across ffmpeg
  builds. Its "no cuts found" case is deliberately a valid result so that a
  disappointing detector degrades to Spec 4's status quo rather than a failure.
- The `sample_video` fixture is shared by most tests here; if it changes shape
  (duration, frame rate, stream count) several assertions move together. That is
  a coupling I accept for one generated fixture, but it should be a named
  constant, not a literal repeated across files.
