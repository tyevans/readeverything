# readeverything: What It Is Doing

**Date:** 2026-08-15
**Status:** Approved for planning
**Predecessors:** Specs 1, 3, 4, 5, 6
**Landed:** Plan 6 merged `2e86b6c` — 537 tests, 92.62% coverage

---

## 1. Why this, and why now

Grep the source tree for logging, tracing, or metrics. The only matches are
prose in docstrings. Grep it for `gather`, `Semaphore`, or `TaskGroup`. There
are none.

So: the library performs its two most expensive operations — sampling a video's
frames and calling a vision model on each, and transcribing an hour of audio —
**sequentially, and in complete silence.** A caller waiting ninety seconds
cannot distinguish progress from a hang, and afterwards cannot say which of
forty files consumed the time.

Spec 1 §12 specified per-capability semaphores and deferred them on explicit
grounds: nothing yet did concurrent expensive work. Cycles 5 and 6 ended that.
The deferral's own condition has expired.

### 1.1 Acceptance

> Point the library at a directory of videos with a vision model configured.
> With an observer injected, a caller sees each file start, each frame complete,
> and each file finish with its elapsed time — enough to answer "which file took
> the ninety seconds" and "is it still working". Frames within one video are
> fetched and described concurrently, bounded by a limit the caller set, so a
> vision endpoint that tolerates four in flight is given four and not forty.
> With no observer and no limits injected, behaviour is exactly what it is
> today.

---

## 2. Scope

**In scope**

1. An `Observer` port and the events it receives.
2. A `Limiter` port with per-`Capability` bounds, and its default adapter.
3. `VideoHandler.represent()` sampling frames concurrently under those bounds.
4. Observer calls from both expensive `represent()` paths — video and audio.
5. Wiring both through `build_perception`, defaulting to today's behaviour.

**Out of scope**

- **Concurrency inside `Perception.invoke`.** Concurrent `invoke` on one
  `(uri, affordance)` races the artifact store, and nothing has analysed that.
  `represent()` is the measured cost; `invoke` is not.
- **Batch concurrency across files inside `Perception`.** `Perception.list()`
  returns uris and the caller writes the loop. Adding fan-out here would bound
  it twice — once by us, once by the caller — and the caller already knows how
  many files they want in flight.
- **OpenTelemetry, or any tracing dependency.** §4.
- **Audio transcription concurrency.** `faster-whisper` transcribes a file in
  one call; chunking it is a different feature with its own quality questions.
  Audio still reports progress.

---

## 3. `Observer`: process-shaped, not result-shaped

The library already has a channel for "something did not go as asked":
`Degradation`, attached to a `Rendered` after the work is done. It is tempting
to route progress through it.

**It is the wrong channel, and the difference is lifecycle.** A `Degradation` is
a *fact about a result* — it exists because the result exists, and a consumer
reads it to decide whether to trust what it got. Progress is a *fact about work
in flight* — it exists before any result does, and its entire value is being
visible while the caller is still waiting. Folding them together would mean a
caller learns a file was slow only once it is no longer slow.

So a separate, injected port:

```python
@runtime_checkable
class Observer(Protocol):
    def observe(self, event: Event) -> None: ...
```

**One method taking a typed event union**, not three methods with `**kwargs`.
The domain already expresses variants this way — `Locator` and
`RenditionContent` are both PEP 695 unions — and a stringly-typed `op: str` with
loose `**meta` would be the one place in this library where a caller has to
guess at a contract.

```python
type Event = OperationStarted | OperationProgressed | OperationFinished
```

Each a frozen dataclass carrying the `SourceRef` it concerns, so an observer can
attribute an event to a file without threading state.

### 3.1 Rules, and the one that matters most

- **A caller's observer must never break a read.** Every call into it is wrapped;
  an observer that raises is contained and the operation continues. The library
  has said since Spec 1 that a handler never raises about its input, and an
  injected callback is now part of that surface. A read that fails because the
  progress reporting failed would be absurd.
- **`observe` is synchronous and must not block.** The library cannot enforce
  that — a caller can always do something slow — but it can and does contain the
  damage, and the port says so plainly rather than leaving a caller to discover
  it under load.
- **No observer means no cost.** The default is `None`, and the code paths are
  the ones that exist today.
- **The library never decides where events go.** No logger is configured, no
  handler installed, no name claimed in the logging hierarchy, nothing printed.
  A caller who wants OpenTelemetry writes ten lines mapping events to spans.

---

## 4. No tracing dependency

`opentelemetry-api` is genuinely stable, SemVer-versioned, explicitly designed
to be embedded in shared libraries, and inert without an SDK. It is still the
wrong choice here.

This library's base install is deliberately light and everything heavy sits
behind an extra. More importantly, taking the dependency would mean adopting
OTel's vocabulary — spans, attributes, a particular notion of a trace — and
prescribing it to every caller. **That vocabulary is not this library's to
choose.** A typed event port costs nothing, composes with a ten-line OTel
adapter for callers who want one, and equally with a progress bar, a print, or
nothing at all.

---

## 5. `Limiter`: bounded per capability, because the costs differ

Spec 1 §12 already reasoned this out and was right: *"ASR, VLM and ffmpeg have
very different cost profiles… Limits are constructor arguments."*

A vision model behind an HTTP endpoint is bounded by what that server tolerates
— often four or eight. ffmpeg on the local CPU is bounded by cores. One global
number would either starve the CPU or flood the endpoint.

```python
@runtime_checkable
class Limiter(Protocol):
    def limit(self, capability: Capability) -> AbstractAsyncContextManager[None]: ...
```

Used as `async with limiter.limit(Capability.VISION):` around each concurrent
unit.

### 5.1 Rules

- **Limits are constructor arguments**, set at the composition root. Not read
  from the environment — that constraint has held since Spec 1 §3 and holds
  here.
- **A capability with no configured limit is unbounded**, not zero. An
  unconfigured limiter must not deadlock; the failure mode of a mistake here is
  "too much concurrency", which is visible, rather than "no progress ever",
  which looks exactly like the hang this cycle exists to eliminate.
- **The default limiter is injected, not constructed inside a handler.** A
  handler that made its own semaphore would bound itself independently of every
  other handler sharing the same endpoint.

---

## 6. What actually runs concurrently

**Frames within one video's `represent()`.** That is the win and it is safe:
`VideoHandler` is stateless, holding only injected ports, and each sample's
extraction and description touch nothing shared. Sampled moments are gathered
under the `VISION` limit for description and the `FFMPEG` limit for extraction.

**Nothing else, this cycle.** Not files — the caller owns that loop. Not
affordance invocations — concurrent `invoke` on one key races the artifact store
and nobody has analysed it. Both are recorded rather than quietly skipped.

**Order must not change.** The timeline is sorted by timestamp before tiling, so
completion order is irrelevant to the result — but that is a property worth a
test, because it is exactly what a naive `gather` would break by appending in
completion order.

---

## 7. What this makes answerable

Today, a caller waiting on a directory of recordings can answer none of these:

- Is it still working, or has it hung?
- Which file consumed the ninety seconds?
- Was the time in ffmpeg or in the model?
- How many model calls did indexing this directory actually make?

Afterwards, an observer of a dozen lines answers all four, and the caller
chooses where the answers go.

---

## 8. Acceptance

1. §1.1's sentence is true, demonstrated by an integration test.
2. An observer that raises on every event does not fail a read, and the
   `Rendered` produced is identical to one produced with no observer at all.
3. With no observer and no limiter injected, behaviour and output are unchanged
   from today — asserted by comparison, not by inspection.
4. Frames within one video are described concurrently, and the concurrency is
   bounded by the configured limit — asserted by observing peak in-flight count
   against a fake, not by timing.
5. A video's timeline is identical whether frames complete in order or out of
   it.
6. A capability with no configured limit runs unbounded rather than deadlocking.
7. Events carry the `SourceRef` they concern, so a caller can attribute time to
   a file without threading state.
8. Nothing under `src/` configures logging, installs a handler, prints, or reads
   an environment variable.
9. All gates green, coverage floor holds at 92.

---

## 9. Risks

| Risk | Mitigation |
| --- | --- |
| Concurrency changes a video's output | §6 — the timeline is sorted before tiling, and criterion 5 asserts identical output under out-of-order completion. A naive `gather` appending in completion order is exactly the defect this guards. |
| A slow observer stalls the pipeline | The port says `observe` must not block, and the library contains exceptions but cannot contain slowness. Recorded honestly rather than claimed away: a caller who does a network POST per frame will feel it. |
| Unbounded default deadlocks something | §5.1 makes "no limit configured" mean unbounded, never zero. The mistake's failure mode is visible load, not silence. |
| Observer events become a de facto API nobody can change | They are a typed union in `domain/`, versioned with the library like every other domain type. Adding a variant is additive; a consumer matching exhaustively will fail loudly at type-check time rather than silently at runtime. |
| Parallel ffmpeg saturates the machine | The `FFMPEG` limit defaults to something conservative rather than core count, and the default is a constructor argument a caller can raise. Reasoned, not measured — flagged for tuning against a real workload. |
