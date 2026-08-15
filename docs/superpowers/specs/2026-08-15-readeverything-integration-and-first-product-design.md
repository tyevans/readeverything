# readeverything: Integration, Hardening, and the First Usable Product

**Date:** 2026-08-15
**Status:** Approved for planning
**Predecessor:** `2026-08-14-readeverything-perception-core-design.md` (Spec 1)
**Plans landed against Spec 1:** Plan 1 (perception core, 150 tests),
Plan 2 (vision and the image family, 230 tests)

---

## 1. The problem this spec exists to solve

Two plans have landed. Every gate is green: 230 tests in 1.16 seconds,
92.28% coverage against a 90% floor, `mypy --strict` clean, both
import-linter contracts kept, the wheel builds and imports on base
dependencies alone.

None of that is in question here, and none of it is what is wrong.

What is wrong is that **the components are excellent and the seams are
empty**. Every defect found below sits between two well-built pieces
that have never been run together, or at the boundary where the library
meets a person trying to use it. The evidence:

- A caller must hand-construct seven to ten objects in dependency order
  before reading one file. Every real assembly of this library that
  exists anywhere lives inline in a test fixture.
- `Perception` accepts an `ArtifactStore`, assigns `self._artifacts`,
  and never reads it. Both store adapters, the key derivation, and their
  unit tests are complete, correct, and unreachable.
- `tests/` contains `unit/` and `live/`. The `integration` marker is
  declared in `pyproject.toml` and matches zero tests.
- `ArtifactStoreCompliance` ships in the wheel and is subclassed by
  nobody. This project's stated principle is that a law must be able to
  fail; this one has never been given the chance.
- `README.md` is zero bytes, and `pyproject.toml` ships it as the
  wheel's long description.
- The distribution is named `deepagents-read-everything` and the string
  `deepagents` appears nowhere in `src/` or `tests/`.

The through-line is that Plans 1 and 2 built *downward* — domain, ports,
adapters, handlers — with great discipline, and never built *outward*.
This spec builds outward.

### 1.1 What "fully usable and useful" means here

Concretely, and as the acceptance test for this whole spec:

> A developer who has never seen this repository can `pip install
> deepagents-read-everything`, read the README, write **under ten lines**,
> and have an agent that can look at a directory of mixed files —
> including images — and answer questions about them with locators. On a
> machine with no ffmpeg, no tesseract, and no model server, the same ten
> lines work and simply offer fewer affordances.

Everything in scope below is chosen because that sentence is currently
false. Everything out of scope is excluded because that sentence stays
true without it.

---

## 2. Scope

**In scope**

1. Wiring the artifact cache into `Perception`, and removing the
   duplicated work between `inspect` and `invoke`.
2. Capability *discovery* — the missing half of capability negotiation.
3. A composition root, plus the front-door and extras fixes that make
   the public surface honest.
4. An integration test layer, and running the shipped `ArtifactStore`
   law against both real stores.
5. A deepagents integration that earns the distribution's name.
6. A README that makes §1.1 achievable.
7. The residual defects in §8.

**Out of scope** — deliberately, with reasons

- **New media families** (audio, video, PDF, office, archives) and the
  seven unbuilt ports they need. This spec makes the existing three
  handlers into a product; widening the handler set is Spec 4's job and
  is strictly easier once the composition root and integration layer
  exist to receive it.
- **The query interface** (`ask(path, question)`), retrieval, chunking
  into redstring `SourceDocument`s, and citation rendering. Still Spec 2.
- **The redstring RFC.** Unchanged, still owed, still independent.
- **Per-capability concurrency semaphores.** Nothing yet does concurrent
  expensive work. Spec 1's reasoning holds.
- **A `synthesized` marker on `LocatorSegment`.** See §8.1 — the cheap
  honest fix lands now; the type change lands when citations make it
  load-bearing.

---

## 3. Wiring the cache, and paying for work once

### 3.1 What is wrong

`Perception._ref` performs four operations per call — `read_range` for
the head bytes, `detector.detect`, `hasher.hash` over the whole file, and
`source.size`. `_resolve` calls `_ref` and then `registry.resolve`. All
of `inspect`, `invoke`, and `represent` call `_resolve`.

So for one `inspect` followed by one `invoke` on the same path, the
library hashes the entire file twice, detects the mimetype twice, stats
twice, resolves the handler twice, and computes `available_affordances`
twice. Through the tool pack this repeats on every model turn, with no
state carried between turns. Content hashing is the expensive one and it
scales with file size.

### 3.2 The design

Two distinct caches, because they answer two different questions, and
conflating them is how cache bugs happen.

**A resolution memo** answers "what is this path?" — the `SourceRef` and
its handler. Keyed on `(uri, size, mtime_ns)`, which is exactly the
`StatMemo` key `ContentHasher` already uses; reuse that type rather than
inventing a second staleness rule. Bounded, in-memory, per-`Perception`.
This is what removes the duplicated hashing.

**The artifact store** answers "what did we derive from this content?" —
the existing `ArtifactStore`, keyed by `artifact_key` over the full
derivation including `capability_fingerprint` and `handler_version`.
Content-addressed, so it is safe across processes and across time.

The resolution memo is a performance optimisation over a mutable
filesystem and must be invalidated by stat. The artifact store is
content-addressed and never needs invalidating. Keeping them separate
means the second one keeps that property.

### 3.3 Rules

- A stat change (size or mtime) invalidates the memo entry for that uri.
  A file rewritten between two calls must produce a fresh `SourceRef`.
- `Perception` gains no environment reads and no I/O it did not already
  do. The memo is populated from work already performed.
- Cache participation is a handler's decision, not the pipeline's. A
  handler declares a `handler_version`; `Perception` supplies the key
  material. A handler that declares no version does not get cached.
- **A cache hit and a cache miss must be indistinguishable in their
  result.** This is the property the integration tests assert, not the
  speedup.

---

## 4. Capability discovery — the missing half

### 4.1 What is wrong

`CapabilitySet` takes `Mapping[Capability, str]` — capability to
revision string. `Capability` has eight members: `ASR`, `DIARIZATION`,
`EXIFTOOL`, `FFMPEG`, `LIBREOFFICE`, `TESSERACT`, `TEXT_LLM`, `VISION`.

Nothing in the library discovers any of them. The caller hand-asserts
that ffmpeg exists and at what revision. If they assert wrongly,
handlers register affordances that cannot run — which is precisely and
exactly the failure that capability negotiation was designed to prevent.
The negotiation logic is sound. Its inputs are unverified assertions.

This is the same defect shape both prior review waves kept finding, at a
higher altitude: **something asserted as observed that nothing
observed.** There it was a degradation message; here it is the entire
capability layer.

### 4.2 The design

A `CapabilityProbe` port with one method: given a `Capability`, return a
revision string if it is genuinely available here, or `None`.

Two adapters:

- **`BinaryProbe`** — for `FFMPEG`, `EXIFTOOL`, `LIBREOFFICE`,
  `TESSERACT`. Locates the executable and executes its version flag.
  Availability means *it ran and reported a version*, not that a file
  exists at a path. A binary that is present and broken is not a
  capability.
- **`ModelProbe`** — for `VISION`, `TEXT_LLM`, `ASR`, `DIARIZATION`.
  Derives the revision from the injected model object's own identity
  (`VisionModel.model_id`), not from configuration.

`ModelProbe` closes the seam Plan 2's final review flagged: the VISION
revision and `VisionModel.model_id` stop being independent inputs that
happen to agree, because one is *derived from* the other. It becomes
impossible to build a `CapabilitySet` that disagrees with the model
actually injected.

### 4.3 Rules

- Probing is explicit and eager, performed once at composition time. It
  is never lazy and never inside a handler — a handler asking "do I have
  ffmpeg?" mid-request is the design this replaces.
- A probe never raises. A probe that cannot determine availability
  returns `None`, and `None` means unavailable. Under uncertainty the
  library offers less, never more.
- Probe execution has a hard timeout. A hung `--version` call must not
  hang composition.
- **Hand-assertion remains fully supported.** `CapabilitySet` built from
  an explicit mapping keeps working exactly as it does today. Discovery
  is a convenience for the default path, not a new requirement, and
  tests must be able to construct any capability set they like without
  touching the machine.
- The library still reads no environment variables. A probe inspects the
  system's executables; it does not read config from the environment.

---

## 5. The composition root and an honest public surface

### 5.1 What is wrong

Four separate problems, all on the boundary between this library and a
person:

1. **No composition root.** Seven to ten objects, in order, with the
   caller expected to know which handler classes exist and that each
   needs the same `source`.
2. **The only real vision adapter is unreachable from the front door.**
   `LangChainVisionModel` and `build_openai_vision_model` are absent
   from `_LAZY`. Callers must import `readeverything.adapters.
   vision_langchain` directly — the private path a lazy front door
   exists to make unnecessary.
3. **`ImageHandler` is exported but unusable on a base install.** The
   name resolves lazily; instantiating it raises `ModuleNotFoundError:
   PIL` unless the `images` extra is present. The front door advertises
   a capability the install cannot honour, and the failure names Pillow
   rather than the extra.
4. **`README.md` is zero bytes**, shipped as the wheel's long
   description.

### 5.2 The design

One function is the whole product surface:

> `build_perception(root, *, vision=None, capabilities=None,
> artifacts=None)` returns a working `Perception`.

Its contract:

- `root` is the only required argument. Everything else has a working
  default.
- It assembles the source, detector, hasher, handler set, registry, and
  artifact store.
- It **registers only handlers whose dependencies are actually
  importable.** `ImageHandler` is included when Pillow imports and
  omitted when it does not. A base install yields a `Perception` that
  handles text and binary and says so; no import error reaches the
  caller.
- With `capabilities=None` it probes (§4). With an explicit
  `CapabilitySet` it uses it verbatim and probes nothing.
- With `vision=None` the image affordances requiring VISION do not
  register. This is negotiation working, not degradation.

Plus the surface fixes: export the vision adapter builders from the
front door; make the Pillow-missing path raise an error naming the
`images` extra rather than `PIL`; write the README.

### 5.3 Rules

- `build_perception` reads no environment variables. Every input is an
  argument. This is the constraint that has held since Spec 1 §3 and it
  holds here.
- It is a convenience, never a requirement. Direct construction of
  `Perception` stays fully supported and stays the thing the composition
  root itself uses. If `build_perception` can do something the public
  constructors cannot, that is a bug in the constructors.
- Adding a handler to the library must mean adding it in exactly one
  place. A handler registered in the composition root but absent from
  the front door — or the reverse — is the defect this section exists to
  prevent, and it gets a test.

### 5.4 The README

The README's job is §1.1 and nothing else. It must contain a copyable
example that works on a base install, the same example extended with
vision, a statement of what the library does and does not read
(filesystem yes, environment no), and an honest table of which media
types are supported today. It must not describe unbuilt handlers as
though they exist.

---

## 6. The integration layer

### 6.1 What is wrong

`tests/` has `unit/` and `live/`. The `integration` marker is declared
and matches zero tests. Every component is verified against fakes; the
two tests that happen to wire real components together
(`test_tools.py`, `test_perception_image.py`) are filed as unit tests and
neither reaches vision through the agent surface.

Specifically unverified today:

- The agent tool pack has never met `ImageHandler` with any
  `VisionModel`, real or fake. The image capability has never reached
  the agent-facing surface in a test.
- No test asserts caching works, because there is nothing to test.
- `ArtifactStoreCompliance` has never been run against `InMemory-` or
  `FilesystemArtifactStore`.

### 6.2 The design

A third test tier, `tests/integration/`, under the existing declared
marker. Its distinguishing rule:

> An integration test constructs real components through
> `build_perception` and asserts on behaviour crossing at least two
> module boundaries. It may use a fake **model**, because model output
> is nondeterministic and this project has never asserted on model text.
> It may not use a fake **source, detector, hasher, store, or
> registry** — those are the seams under test.

Required coverage:

- Both artifact stores subclass `ArtifactStoreCompliance`. The law gets
  the chance to fail.
- The full agent path: `build_perception` → `build_tools` → invoke every
  tool by name, on a real directory of real fixture files, including an
  image, with a fake vision model.
- Cache behaviour: repeated `inspect`/`invoke` on one path produces
  identical results, and the second call does not re-hash. Assert
  identity of result; assert the absence of repeated work by counting
  calls through a counting decorator over a real adapter, not by timing.
- Capability negotiation end to end: the same directory under a
  capability set with VISION and without, asserting the affordance list
  differs and that nothing unavailable is ever offered.
- Composition-root fallback: with Pillow importable and with it
  unavailable, both produce a usable `Perception`.
- A file rewritten between two calls produces a fresh `SourceRef`.

### 6.3 Rules

- Integration tests run in the default suite. A tier that is deselected
  by default is a tier that rots — that is what happened to the
  `integration` marker already.
- They use real temporary directories and real files.
- They never require a network or a model server. Those are what `live`
  is for, and `live` stays deselected.
- The existing five gates keep passing unchanged. The coverage floor
  rises to 92% — it currently sits at 92.28% with under a point of
  headroom, and this spec adds substantial tested surface.

---

## 7. Earning the name: deepagents

### 7.1 What is wrong

The distribution is `deepagents-read-everything`. `grep -rn "deepagents"
src/ tests/` returns nothing. Spec 1 §10 specified a `MediaAwareBackend`
and it was never built.

### 7.2 The design

`build_tools(perception) -> list[BaseTool]` already returns plain
LangChain tools, which is the correct framework-agnostic core and stays
exactly as it is. What is missing is the last mile.

Ship `readeverything.agent.deepagents` behind a `deepagents` extra,
containing one helper that goes from a directory to a ready deep agent.
The framework-agnostic path stays first-class and stays the one the
library itself uses; the deepagents helper is a thin convenience over
it, importing `deepagents` only inside the extra-guarded module.

**This section's design is provisional pending verification of the
current `deepagents` public API.** The plan's first task in this area
must confirm, against the installed package, the exact agent-construction
signature and whether a third-party filesystem/store backend protocol
exists. If a backend protocol exists and is implementable within this
library's read-only, no-environment constraints, implementing it is
strictly better than a construction helper — it would give a deep
agent's *built-in* file tools media understanding transparently, which
is what Spec 1 §10 originally envisioned. If it does not exist or
demands mutability this library will not offer, the construction helper
is the honest answer and the spec's §10 ambition is formally retired
with a note saying why.

Choosing between these two on evidence is a task in the plan, not a
guess in the spec.

### 7.3 Rules

- `deepagents` is imported in exactly one module, guarded by its extra,
  and the layered import contract is extended to keep it there.
- Nothing in `domain`, `ports`, `adapters`, `handlers`, `registry`, or
  `pipeline` learns that deepagents exists.
- If the extra is absent, every other part of the library works
  unchanged.

---

## 8. Residual defects

An adversarial shape-sweep across the whole source tree confirmed the
prior waves' fixes are holding: all three "kept N of M" degradations
report `len(text)`; `_reject_non_primitives` validates both mapping keys
and values; the empty-versus-truncated guard in `TextHandler` is
correct. The sweep's own clean verdicts are recorded with the evidence
that produced them.

The following remain.

### 8.1 Synthesized text carries a fabricated locator — MAJOR (design)

`BinaryHandler.represent` builds `ByteRange(0, max(1, ref.size_bytes))`.
The `max(1, ...)` exists because `ByteRange` rejects `start >= end`. For
a zero-byte file, the locator asserts a byte that does not exist.
`TextHandler.represent` does the same in the other direction, mapping a
24-character `[empty text file: ...]` placeholder over a `CharSpan` into
a file containing zero characters.

Both are honest in their comments about why. Both still hand an indexer a
span that is not there.

The real gap is that the domain cannot express the difference between
text **extracted from** a file and text **synthesized about** one. Every
`Rendered` must carry a `LocatorMap` whose length matches its text, so
synthesized description is structurally compelled to fabricate a span.
This is the §1 shape once more — asserting as observed what nothing
observed — and it is the last instance, at the type level.

**Ruling.** The correct long-term answer is a `synthesized` marker on
`LocatorSegment`, and it lands when citations render it load-bearing,
which is Spec 2. It is not built here.

What lands here: every rendition whose text is synthesized rather than
extracted announces itself through the `Degradation` channel that
already exists for "this is not what you asked for" — `Degradation(what=
"synthesized description", detail="no content was extracted; this text
describes the file")`. Cheap, uses existing vocabulary, and makes the
fiction visible to any consumer instead of silent. A consumer that
respects degradations can already act on it.

### 8.2 The tool pack directs the model to a tool that does not exist — MAJOR

`agent/tools.py` `_render_rendition` renders image content as
`"[{mime} image, {len} bytes — pass to a vision tool to read it]"`. The
pack exposes `inspect_path`, `list_paths`, and `invoke_affordance`. There
is no vision tool. The string instructs a model toward an affordance
that is not there, and the text reaches the model directly.

This is the same shape as §8.1 and the prior waves' findings, surfacing
at the product boundary after two review passes cleaned it out of the
handlers — which is the strongest available argument for hunting this
shape by shape rather than by site.

**Fix:** name the actual route. When the file's card offers
`describe_image` or `ocr`, say so by name. When it does not — because no
vision capability is registered — say that the image cannot be read
here, which is true and actionable, rather than pointing at a tool that
does not exist.

### 8.3 `TextHandler.invoke` silently discards `end` — MINOR

`start = min(params.start, len(text) - 1)`. Whenever `start >=
len(text) - 1`, the caller's `end` is ignored and exactly one character
comes back. Untested at `len(text) == 1, start == 1`. Clamp both ends
symmetrically, and test the boundary.

### 8.4 Untested error paths — MINOR

`adapters/detection.py:39` (`except Exception` around puremagic);
`adapters/local_source.py` OSError branches in `size`, `stream`, and
`walk`, and `stream`'s `finally: close()` under error;
`adapters/hashing.py` `StatMemo._key`'s OSError on a file that vanishes
between memo lookup and stat.

Each is a branch that has never run. Each gets a test that forces it.

### 8.5 Spec 1 §14b is stale — documentation

§14b lists two items as owed that were closed in Plan 2: the
`Perception.hasher` port annotation (`perception.py` already types the
port) and `artifact_key`'s `default=str` collision (`cache_key.py` now
raises). Correct §14b to reflect what the code does. A deferral list
that misreports its own status is worse than none, because it is
believed.

---

## 9. What this spec does not fix, and why that is acceptable

- **Seven capabilities still have no adapter.** `ASR`, `DIARIZATION`,
  `TEXT_LLM`, `EXIFTOOL`, `FFMPEG`, `LIBREOFFICE`, `TESSERACT` remain
  enum members with nothing behind them. After §4 the library will
  *correctly report them as unavailable*, which is honest and is a
  strict improvement over asserting them. They gain adapters alongside
  the handlers that need them.
- **Three media families.** Text, image, binary fallback. §1.1 is
  achievable with three, and the composition root is what makes the
  fourth cheap.
- **No performance benchmarks.** §3 asserts correctness of caching, not
  a speedup number. Timing assertions are flaky and this project has
  been right to avoid them.

---

## 10. Acceptance

This spec is satisfied when:

1. The §1.1 sentence is true, demonstrated by an integration test that
   performs exactly what it describes.
2. `tests/integration/` runs in the default suite and both artifact
   stores pass the shipped compliance law.
3. `Perception` consults its artifact store, and one `inspect` followed
   by one `invoke` on the same path hashes the file once.
4. A `CapabilitySet` built by discovery cannot disagree with the vision
   model actually injected.
5. `build_perception(root)` on a base install returns a working
   `Perception` with no import error.
6. Every defect in §8 is closed, each with a test that fails without its
   fix.
7. All five existing gates stay green, with the coverage floor at 92%.

---

## 11. Risks

| Risk | Mitigation |
| --- | --- |
| The deepagents API is not what §7 assumes | §7.2 makes verification a plan task with two pre-authorised outcomes, so neither is a stall. |
| Cache wiring introduces a stale-result bug — the worst class of bug this library could ship | Two caches with separate invalidation rules (§3.2); the content-addressed one never invalidates. Integration tests assert hit and miss are indistinguishable, and that a rewritten file produces a fresh ref. |
| Capability probing executes subprocesses, a new risk surface | Probes run fixed version flags on located executables, never caller-supplied strings; hard timeout; never raise. Bandit stays in the gate set. |
| The composition root becomes a god-function that grows with every handler | Its contract (§5.3) is that it may do nothing the public constructors cannot. It is tested as a convenience over them, not as a separate path. |
| Coverage floor at 92% leaves too little headroom | The floor rises only after the integration tier lands, which adds more tested surface than it adds code. If the margin is under a point at the end, the floor stays at 90 and that is recorded rather than forced. |
