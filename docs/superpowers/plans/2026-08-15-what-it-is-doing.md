# What It Is Doing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the library say what it is doing while it does it, and let its most expensive work run bounded-concurrently.

**Architecture:** An `Observer` port taking a typed event union, injected and defaulting to `None`. A `Limiter` port bounding work per `Capability`, injected and defaulting to unbounded. `VideoHandler.represent()` fetches its sampled moments concurrently under those bounds, then assembles them in the existing sequential order.

**Tech Stack:** Python 3.13 stdlib only — `asyncio.Semaphore`, `asyncio.gather`, `contextlib.asynccontextmanager`. No new dependency. pytest, mypy --strict, ruff, import-linter, bandit, coverage.

**Spec:** `docs/superpowers/specs/2026-08-15-readeverything-what-it-is-doing-design.md`

## Global Constraints

- **The library reads NO environment variables under `src/`**, configures no logging, installs no handler, claims no logger name, and prints nothing.
- **Python 3.13, PEP 695 inline type parameters** (`type Event = ...`). A module-level `TypeVar` is a defect.
- **`mypy --strict`, `warn_unused_ignores = true`**, over `src` and `tests`. No new `# type: ignore` without a comment naming why.
- **Import-linter layers**, outermost first: `composition, testing, agent, pipeline, registry, handlers, adapters, ports, domain`. **Handlers must not import from `adapters/`** or from each other.
- **No handler ever raises** from `describe`, `invoke`, or `represent`, except `UnknownAffordanceError` for a name it does not offer. **An injected observer is now part of that surface.**
- **Never assert on model text.** Assert structure, locators, and counts.
- **No new third-party dependency.** This cycle adds none; if you reach for one, stop and report.
- **Coverage floor 92.** Run `make check`.

---

## Measured facts (verified by the plan author)

**There is no observability and no concurrency.** `grep -rn "gather\|Semaphore\|TaskGroup" src/readeverything/` returns nothing. Logging/tracing/metrics matches are prose in docstrings only.

**The loop to parallelise is `handlers/video.py:442-470`**, inside `_timeline`:

```python
for index, ((sampled_at, cue), (start, end)) in enumerate(zip(entries, bounds, strict=True)):
    if cue is None:
        body, state = await self._moment(path, sampled_at)     # <- the sequential await
        ...
    else:
        body = _spoken(cue)
    chunk = f"[{_timestamp(start)}] {body}{MOMENT_SEPARATOR}"
    if index:
        entry_barriers.append(cursor)
    entry_offsets.append(cursor)
    segments.append(LocatorSegment(CharSpan(cursor, cursor + len(chunk)), TimeSpan(start, end)))
    cursor += len(chunk)
    chunks.append(chunk)
```

**Critically: this loop does two different jobs.** It fetches each moment (the slow part, independent per entry) *and* it accumulates `cursor`, `entry_offsets`, `entry_barriers` and `segments` in strict order (fast, and order-dependent by construction).

**So the change is a split, not a rewrite.** Fetch every moment concurrently first, into a list indexed by entry; then run the existing loop unchanged, reading from that list instead of awaiting. Ordering is then preserved *by construction* rather than by trusting `gather`'s return order — and a reviewer can see the assembly is untouched.

`_moment(path, sampled_at) -> tuple[str, str]` returns `(body, state)` where state is `"ok"`, `"missing"` or `"failed"`. Only entries with `cue is None` call it.

**`_moment` internally calls** `self._frames.frame_at(...)` (ffmpeg subprocess) and, when vision is configured, `self._vision.describe(...)` (model call). Those are the two things needing different limits.

---

## File Structure

**Created:**

| File | Responsibility |
| --- | --- |
| `src/readeverything/domain/observation.py` | The event union: `OperationStarted`, `OperationProgressed`, `OperationFinished`, `type Event`. |
| `src/readeverything/ports/observation.py` | `Observer` protocol. |
| `src/readeverything/ports/limits.py` | `Limiter` protocol. |
| `src/readeverything/adapters/semaphore_limiter.py` | `SemaphoreLimiter`, per-`Capability` bounds. |
| `src/readeverything/testing/fakes.py` (modify) | `RecordingObserver`, `RaisingObserver`, `CountingLimiter`. |

**Modified:**

| File | Change |
| --- | --- |
| `src/readeverything/handlers/video.py` | Concurrent moment fetch; observer calls. |
| `src/readeverything/handlers/audio.py` | Observer calls (no concurrency — one transcription call). |
| `src/readeverything/composition.py` | Accept and thread `observer` and `limiter`. |
| `src/readeverything/__init__.py` | Export the new names. |

**Layering note.** The events are domain types (they name a `SourceRef` and describe work, not infrastructure). The protocols are ports. The semaphore adapter is an adapter. Handlers receive both by injection and import neither concretely.

---

## Task 1: The event union and the `Observer` port

**Files:**
- Create: `src/readeverything/domain/observation.py`, `src/readeverything/ports/observation.py`
- Test: `tests/unit/domain/test_observation.py`

**Interfaces:**
- Produces:
  - `OperationStarted(operation: str, ref: SourceRef)`
  - `OperationProgressed(operation: str, ref: SourceRef, done: int, total: int | None)`
  - `OperationFinished(operation: str, ref: SourceRef, elapsed_s: float)`
  - `type Event = OperationStarted | OperationProgressed | OperationFinished`
  - `class Observer(Protocol)` with `def observe(self, event: Event) -> None: ...`
  - `def emit(observer: Observer | None, event: Event) -> None` — the containment helper every caller uses.

- [ ] **Step 1: Write the failing tests**

```python
def test_progress_reports_what_is_done_and_what_is_known(...) -> None:
    """`total` is `int | None` because it is not always knowable.

    A video knows how many moments it will sample before it starts. A
    transcription does not know how many cues it will produce until it has
    produced them. Reporting a made-up total would be a number nothing
    measured — the field admits ignorance instead.
    """
    known = OperationProgressed(operation="represent", ref=_ref(), done=3, total=40)
    unknown = OperationProgressed(operation="represent", ref=_ref(), done=3, total=None)
    assert known.total == 40
    assert unknown.total is None


def test_a_progressed_event_rejects_a_negative_count() -> None:
    with pytest.raises(ValueError):
        OperationProgressed(operation="represent", ref=_ref(), done=-1, total=None)


def test_done_may_not_exceed_a_known_total() -> None:
    """"7 of 5 complete" is a claim about work nobody scheduled."""
    with pytest.raises(ValueError):
        OperationProgressed(operation="represent", ref=_ref(), done=7, total=5)


def test_elapsed_may_not_be_negative() -> None:
    with pytest.raises(ValueError):
        OperationFinished(operation="represent", ref=_ref(), elapsed_s=-0.1)


def test_emit_contains_an_observer_that_raises() -> None:
    """THE RULE THAT MATTERS MOST. A read must not fail because the progress
    reporting failed. The library has promised since its first spec that a
    handler never raises about its input; an injected callback is now part of
    that surface.
    """
    emit(_RaisingObserver(), OperationStarted(operation="represent", ref=_ref()))
    # reaching here is the assertion


def test_emit_with_no_observer_does_nothing() -> None:
    emit(None, OperationStarted(operation="represent", ref=_ref()))
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run --all-extras pytest tests/unit/domain/test_observation.py -v
```
Expected: FAIL — the module does not exist.

- [ ] **Step 3: Implement**

Frozen `slots` dataclasses with `__post_init__` validation, matching every other domain type here. `type Event = ...` as a PEP 695 alias.

`emit` is the containment point:

```python
def emit(observer: Observer | None, event: Event) -> None:
    """Deliver an event, or do nothing, and never let either fail a read.

    A caller's observer is arbitrary code the library invited into the middle
    of a read. It may raise; that is not the read's problem. Everything this
    library promises about never raising over bad input applies here too — a
    file that could have been read must not become unreadable because a
    progress callback threw.

    Deliberately broad: any exception at all, not a curated list. The library
    cannot know what a caller's observer does, so it cannot enumerate how that
    might fail, and guessing would leave exactly the gaps this exists to close.
    """
    if observer is None:
        return
    try:
        observer.observe(event)
    except Exception:  # noqa: BLE001 — see docstring; containment is the point
        pass
```

- [ ] **Step 4: Run and commit**

```bash
uv run --all-extras pytest tests/unit -q && uv run --all-extras mypy
uv run --all-extras ruff format src tests
git add src/readeverything/domain/observation.py src/readeverything/ports/observation.py tests/
git commit -m "feat: say what is happening, and never fail a read for saying it"
```

---

## Task 2: The `Limiter` port and its semaphore adapter

**Files:**
- Create: `src/readeverything/ports/limits.py`, `src/readeverything/adapters/semaphore_limiter.py`
- Test: `tests/unit/adapters/test_semaphore_limiter.py`

**Interfaces:**
- Produces:
  - `class Limiter(Protocol)` with `def limit(self, capability: Capability) -> AbstractAsyncContextManager[None]: ...`
  - `class SemaphoreLimiter` — `__init__(self, limits: Mapping[Capability, int] | None = None)`.
  - `DEFAULT_LIMITS: Mapping[Capability, int]`.

- [ ] **Step 1: Write the failing tests**

```python
async def test_concurrency_is_bounded_by_the_configured_limit() -> None:
    """Asserted by observing peak in-flight count, not by timing.

    A timing assertion would be flaky and would not actually establish the
    bound — a slow machine passes it for the wrong reason.
    """
    limiter = SemaphoreLimiter({Capability.VISION: 2})
    peak = 0
    in_flight = 0

    async def worker() -> None:
        nonlocal peak, in_flight
        async with limiter.limit(Capability.VISION):
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0)
            in_flight -= 1

    await asyncio.gather(*(worker() for _ in range(10)))
    assert peak == 2


async def test_an_unconfigured_capability_is_unbounded_not_zero() -> None:
    """The failure mode of a mistake here must be visible load, never silence.

    A capability defaulting to zero would deadlock, and a deadlock looks
    exactly like the hang this whole cycle exists to eliminate.
    """
    limiter = SemaphoreLimiter({Capability.VISION: 1})
    ran = False
    async with limiter.limit(Capability.FFMPEG):
        ran = True
    assert ran


async def test_different_capabilities_do_not_share_a_bound() -> None:
    """A vision endpoint tolerating four in flight and ffmpeg bounded by cores
    are different constraints; one global number starves one or floods the
    other."""
    ...


async def test_the_limit_is_released_when_the_body_raises() -> None:
    """Otherwise one failure permanently narrows the pipeline."""
    limiter = SemaphoreLimiter({Capability.VISION: 1})
    with contextlib.suppress(RuntimeError):
        async with limiter.limit(Capability.VISION):
            raise RuntimeError("boom")
    async with limiter.limit(Capability.VISION):
        pass  # reaching here is the assertion
```

- [ ] **Step 2: Run to verify failure, then implement**

`contextlib.asynccontextmanager` over an `asyncio.Semaphore` per configured capability. An unconfigured capability yields immediately.

**Note on `asyncio.Semaphore` and event loops:** a semaphore created outside a running loop is fine in Python 3.10+, but construct lazily on first use per capability if you hit any binding issue, and say why in a comment.

`DEFAULT_LIMITS` is conservative and its reasoning is in a comment: a vision endpoint's tolerance is unknown to this library, ffmpeg is CPU-bound, and the safe default is small because a caller raising a limit is an informed act while a caller discovering they flooded their endpoint is not.

- [ ] **Step 3: Run and commit**

---

## Task 3: Video fetches its moments concurrently

**The centrepiece.** Read `src/readeverything/handlers/video.py:419-480` in full before touching it.

**Files:**
- Modify: `src/readeverything/handlers/video.py`
- Test: `tests/unit/handlers/test_video_handler.py`

**Interfaces:**
- `VideoHandler.__init__` gains `observer: Observer | None = None` and `limiter: Limiter | None = None`.

- [ ] **Step 1: Write the failing tests — the ordering one first**

```python
async def test_the_timeline_is_identical_when_moments_complete_out_of_order(
    sample_video: str,
) -> None:
    """THE TEST THIS TASK EXISTS FOR.

    A naive `gather` that appended results in completion order would scramble
    the timeline, and every locator with it. Fetch order must not reach the
    output at all.

    The fake extractor here returns each frame after a delay inversely
    proportional to its timestamp, so completion order is exactly reversed.
    """
    ordered = await _handler(sample_video, vision=FakeVision()).represent(
        _ref(), Budget(max_chars=None)
    )
    reversed_completion = await _handler(
        sample_video, vision=FakeVision(), frames=_ReverseOrderFrames()
    ).represent(_ref(), Budget(max_chars=None))
    assert ordered.text == reversed_completion.text
    assert [s.locator for s in ordered.locator_map.segments] == [
        s.locator for s in reversed_completion.locator_map.segments
    ]


async def test_frame_work_is_bounded_by_the_vision_limit(sample_video: str) -> None:
    """Asserted by peak in-flight count against a counting limiter, not timing."""
    limiter = CountingLimiter({Capability.VISION: 2})
    await _handler(sample_video, vision=FakeVision(), limiter=limiter).represent(
        _ref(), Budget(max_chars=None)
    )
    assert limiter.peak(Capability.VISION) <= 2


async def test_without_a_limiter_behaviour_is_unchanged(sample_video: str) -> None:
    """Every existing video test rests on this."""
    before = await _handler(sample_video, vision=FakeVision()).represent(
        _ref(), Budget(max_chars=None)
    )
    after = await _handler(sample_video, vision=FakeVision(), limiter=None).represent(
        _ref(), Budget(max_chars=None)
    )
    assert before.text == after.text


async def test_an_observer_that_raises_does_not_change_the_result(
    sample_video: str,
) -> None:
    """A read must not fail — or differ — because progress reporting failed."""
    quiet = await _handler(sample_video, vision=FakeVision()).represent(
        _ref(), Budget(max_chars=None)
    )
    noisy = await _handler(
        sample_video, vision=FakeVision(), observer=RaisingObserver()
    ).represent(_ref(), Budget(max_chars=None))
    assert quiet.text == noisy.text
    assert quiet.locator_map.length == noisy.locator_map.length


async def test_progress_reaches_the_observer_in_order(sample_video: str) -> None:
    recorder = RecordingObserver()
    await _handler(sample_video, vision=FakeVision(), observer=recorder).represent(
        _ref(), Budget(max_chars=None)
    )
    kinds = [type(e).__name__ for e in recorder.events]
    assert kinds[0] == "OperationStarted"
    assert kinds[-1] == "OperationFinished"
    dones = [e.done for e in recorder.events if isinstance(e, OperationProgressed)]
    assert dones == sorted(dones)
```

- [ ] **Step 2: Implement — as a SPLIT, not a rewrite**

The existing loop does two jobs: it fetches each moment (slow, independent) and it accumulates `cursor`, `entry_offsets`, `entry_barriers` and `segments` in strict order (fast, order-dependent).

**Fetch first, then assemble.** Build a list of results indexed by entry — `None` for cue entries, the `(body, state)` tuple for sampled ones — using `asyncio.gather` over only the `cue is None` entries. Then run the existing assembly loop **unchanged**, reading from that list instead of awaiting.

Ordering is then preserved *by construction*: the assembly never sees completion order. A reviewer can diff the assembly and find it untouched.

Wrap each fetch in the `VISION` limit when a vision model is configured and the `FFMPEG` limit around the extraction, using `emit` to report progress as each completes.

**`asyncio.gather` propagates the first exception.** `_moment` already returns a state rather than raising, but if anything else can raise inside the gathered coroutine, use `return_exceptions=True` and map failures to the existing `"failed"` state — the handler must not start raising.

- [ ] **Step 3: Run and commit**

```bash
uv run --all-extras pytest tests/unit -q && uv run --all-extras mypy && make check
```

---

## Task 4: Audio reports progress

**Files:**
- Modify: `src/readeverything/handlers/audio.py`
- Test: `tests/unit/handlers/test_audio_handler.py`

Audio transcribes in one call, so there is no concurrency to add. It still
reports `OperationStarted` and `OperationFinished`, and `OperationProgressed`
per cue as the timeline is assembled — with `total=None` until the cue count is
known, because a transcription does not know how many cues it will produce
until it has produced them, and a made-up total is a number nothing measured.

Same tests as Task 3 for the observer-raises and no-observer-unchanged cases.

- [ ] Write, run, commit.

---

## Task 5: Compose and export

**Files:**
- Modify: `src/readeverything/composition.py`, `src/readeverything/__init__.py`
- Test: `tests/unit/test_composition.py`, `tests/integration/test_observation.py`

**Interfaces:** `build_perception(..., observer: Observer | None = None, limiter: Limiter | None = None)`.

- [ ] **Step 1: Write the failing tests**

```python
async def test_an_observer_sees_a_whole_directory_being_read(media_root) -> None:
    """§1.1's sentence: which file took the time, answered.

    The caller writes the loop over `list()`, so the observer sees each file's
    start and finish and can attribute elapsed time per uri without threading
    any state of its own.
    """
    recorder = RecordingObserver()
    perception = await build_perception(
        media_root, vision=FakeVision(), observer=recorder, probe_binaries=False
    )
    for uri in await perception.list("."):
        await perception.represent(uri, Budget(max_chars=None))

    finished = [e for e in recorder.events if isinstance(e, OperationFinished)]
    assert {e.ref.uri for e in finished} == set(await perception.list("."))
    assert all(e.elapsed_s >= 0.0 for e in finished)


async def test_defaults_change_nothing(media_root) -> None:
    """No observer, no limiter — today's behaviour exactly."""
    ...
```

- [ ] **Step 2: Implement**

Thread both through to the handlers that take them. **Defaults stay `None`**, so a caller who wants nothing pays nothing and gets today's code paths.

Export from the front door, `_LAZY` and `TYPE_CHECKING` sorted and in sync:
`Event`, `Limiter`, `Observer`, `OperationFinished`, `OperationProgressed`,
`OperationStarted`, `SemaphoreLimiter`, and the fakes.

- [ ] **Step 3: `make check`, commit.**

---

## Task 6: The README earns its claim

**Files:** `README.md`, `tests/integration/test_readme_example.py`

The README says what the library does. It should now show, in a few lines, how a
caller learns what it is doing — an observer that prints, and a limiter that
bounds a vision endpoint to four. Add it to the example the existing test
already executes, so it cannot rot.

- [ ] Write, run, commit.

---

## Plan Self-Review

**Spec coverage.** §3 `Observer` → Task 1. §4 no dependency → the Global
Constraints forbid one. §5 `Limiter` → Task 2. §6 concurrency → Task 3. §7
answerable questions → Task 5's directory test. §8 acceptance 1–9 → Tasks 1, 2,
3, 5.

**Ordering.** Ports before the handlers that consume them; video before audio
because video is where concurrency lands and audio only reports; composition
after both; README last, since it documents what exists.

**Known risks a reviewer should hold me to.**
- Task 3 is the whole cycle's risk. The split-not-rewrite instruction is what
  keeps the assembly loop reviewable; if an implementer rewrites the assembly,
  the ordering guarantee becomes a thing to re-verify rather than a thing to
  read.
- `test_the_timeline_is_identical_when_moments_complete_out_of_order` needs a
  fake that genuinely reverses completion order. If it returns in call order,
  the test passes vacuously and proves nothing — that failure mode has already
  happened twice in this project.
- `DEFAULT_LIMITS`' values are reasoned, not measured. They are defaults a
  caller overrides, and the spec's risk table says so; do not present them as
  tuned.
