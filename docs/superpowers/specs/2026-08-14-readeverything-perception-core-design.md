# readeverything — Perception Core Design

**Date:** 2026-08-14
**Status:** Implemented and amended. The perception core is built and merged
(`main`, 150 tests, five quality gates green). This spec was amended afterwards
against what execution actually taught — the architecture held, but five
statements here were wrong or incomplete and are corrected in place:

| § | Amendment |
| --- | --- |
| §4.7 | Withdrew the "barriers are a subset of segment boundaries" law — over-specified, and it would forbid a correct video representation |
| §5 | Added the missing `ContentHashing` port |
| §6 | The no-unsupported-file guarantee is a property to defend, not assume |
| §10 | Recorded the three-tools decision; named where the never-raises guarantee actually begins |
| §13 | Added "a law must be able to fail", after two shipped laws turned out to constrain nothing |

Eight further defects were found in the *implementation plan* during execution
and corrected there. That the spec needed five corrections and the plan eight,
while the architecture itself needed none, is the useful signal.
**Scope:** Spec 1 of 2. This spec covers the perception core: domain model, ports,
mimetype registry, handler families, tool pack, artifact cache, and the deepagents
backend adapter. Spec 2 will cover the query interface (chunking into redstring,
retrieval, and the sourced-answer loop). The `represent()` contract that Spec 2
depends on is fixed here.

---

## 1. Purpose

Give an agent eyes into a filesystem.

Today an agent handed `/data/lecture.mp4` gets nothing useful. `readeverything`
turns any file — by detected mimetype, not extension — into a cheap, meaningful
representation plus a declared set of deeper operations the agent can choose to
pay for. Everything it produces carries a locator, so any claim derived from a
file can be traced back to a timestamp, a page, a bounding box, or a character
span.

The library owns everything that turns **bytes into claims-with-locators**.
`redstring` owns everything that turns **claims into a queryable graph**. That
boundary is load-bearing and is defended in §9.

### Non-goals

- Not a knowledge graph. That is redstring's job.
- Not an agent framework. The core is framework-agnostic; deepagents is one
  optional adapter.
- Not a media editing or transcoding toolkit. Transcoding exists only in service
  of representation.
- Not event-sourced. See §2.

---

## 2. Architectural decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Persistence model | Plain DDD/hexagonal, **no event sourcing** | Media derivation is deterministic derivation, not decision-making. A content-addressed cache gives replay and provenance without a log. |
| Representation | Progressive: cheap card + declared affordances | An agent should choose what to spend on a two-hour video, not have it chosen for it. |
| Agent surface | Framework-agnostic tool pack; deepagents backend decorator optional | Core must be usable from plain LangChain/LangGraph. |
| Retrieval | Index-then-retrieve via redstring (**core dependency**) | redstring already owns chunking, embedding, vector and graph retrieval, with shipped compliance suites. |
| Capability gaps | Structural: unsatisfied handlers/affordances never register | No dead tools, no wasted agent turns, no silent degradation. |
| Provenance | Opaque payload upstreamed into redstring | A citation must be self-describing. See §9. |
| Import name | `readeverything` (dist: `deepagents-read-everything`) | The core does not depend on deepagents; the name must not claim otherwise. |

---

## 3. Layering

Enforced by `import-linter` with `exhaustive = true` (a new top-level package
fails the gate until it is placed), plus a companion AST test pinning each
third-party client to a single directory — import-linter cannot see third-party
imports, and that test is what actually stops a leak.

```
composition/            the only layer that may name redstring + handlers + tools together
    |
agent/                  LangChain tool pack; deepagents backend decorator (extra)
    |
pipeline/               detect -> dispatch -> card; represent-for-index
    |
handlers/  registry/    MediaHandler implementations; MimeTypeRegistry dispatch
    |
adapters/               ffmpeg, exiftool, pdf, local FS, VLM/ASR/diarizer, caches, redstring
    |
ports/                  Protocols only
    |
domain/                 pure; stdlib + pydantic only
```

`testing/` sits above all rings with a `forbidden` contract restricting it to
`ports` + `domain`, and ships in the wheel.

### Third-party confinement (asserted by test)

| Dependency | Permitted location |
| --- | --- |
| `subprocess` / ffmpeg / ffprobe | `adapters/ffmpeg.py` |
| `exiftool` | `adapters/exiftool.py` |
| `pypdfium2` | `adapters/pdf.py` |
| `PIL`, `pillow_heif` | `adapters/images.py` |
| `faster_whisper` | `adapters/asr_local.py` |
| `pyannote` | `adapters/diarize_pyannote.py` |
| `trafilatura` | `adapters/html.py` |
| `tesseract` | `adapters/ocr_tesseract.py` |
| `pyarrow` | `adapters/tabular.py` |
| `langchain*` | `adapters/langchain_*.py`, `agent/` |
| `redstring` | `adapters/redstring_sink.py` |
| `deepagents` | `agent/deepagents_backend.py` |

### The library reads no environment

Enforced by test, as in redstring. All configuration is constructor arguments;
`composition/` receives them explicitly from the caller. This includes secrets
such as the Hugging Face token required by the pyannote adapter.

---

## 4. Domain model

Pure, `stdlib` + `pydantic` only. Value objects are
`@dataclass(frozen=True, slots=True)`; user-facing models are pydantic.

### 4.1 Locators

One vocabulary shared by cards, affordance results, the locator map, chunk
barriers, and citations.

```python
type Locator = TimeSpan | PageRef | BBox | CharSpan | ByteRange
```

- `TimeSpan(start_s: float, end_s: float)`
- `PageRef(page: int)` — 1-indexed
- `BBox(page: int | None, x: float, y: float, w: float, h: float)` — normalized 0..1
- `CharSpan(start: int, end: int)` — half-open
- `ByteRange(start: int, end: int)` — half-open

Locators are pure data. Speaker attribution is **not** part of a locator; it
lives on the transcript cue (§4.5).

### 4.2 Source identity

```python
@dataclass(frozen=True, slots=True)
class SourceRef:
    uri: str                  # opaque to the domain; "path!member" for archive members
    mime: MimeType
    content_hash: ContentHash # blake2b of content
    size_bytes: int
```

Handlers receive a `SourceRef` and never touch a filesystem directly. Bytes are
obtained through the injected `FileSource` port.

### 4.3 Affordance

A **declaration**, not a bound callable. The registry can decide what to expose
without executing anything.

```python
@dataclass(frozen=True, slots=True)
class Affordance:
    name: str                        # "get_frame_at"
    description: str                 # used verbatim as the tool docstring
    params: type[BaseModel]          # becomes the tool args schema
    requires: frozenset[Capability]
    level: DetailLevel               # CARD | SEGMENT | DEEP
```

### 4.4 Card

The cheap representation returned on first contact.

```python
@dataclass(frozen=True, slots=True)
class Card:
    ref: SourceRef
    kind: MediaKind
    facts: Mapping[str, str | int | float]   # probe metadata, flat and renderable
    outline: tuple[Segment, ...]             # scenes, chapters, pages, cue groups
    excerpt: str | None                      # only when cheap to obtain
    affordances: tuple[Affordance, ...]      # already capability-filtered
```

`Segment` is `(locator: Locator, label: str)`.

Producing a card must not invoke a model and must not process the whole file.
Cost is bounded by a probe plus, for video, a scene-change pass.

### 4.5 Transcripts and speakers

```python
@dataclass(frozen=True, slots=True)
class TranscriptCue:
    span: TimeSpan
    text: str
    speaker: SpeakerId | None
    confidence: float | None
```

Speaker attribution is modelled from the start. When `Capability.DIARIZATION` is
unsatisfied every cue carries `speaker=None` and the entire pipeline is
unchanged. This is deliberate: retrofitting a speaker field later would touch
cues, `represent()`, the locator map, chunk barriers, and the redstring
provenance payload simultaneously.

Rationale: redstring builds a graph of claims, and the highest-value edge over
conversational media is `Person --asserted--> Claim`. Without diarization every
statement in a multi-speaker recording attributes to the file rather than to a
person.

### 4.6 Renditions and rendered text

```python
@dataclass(frozen=True, slots=True)
class Rendition:
    locator: Locator
    content: TextContent | ImageContent | StructuredContent
    degraded: bool = False

@dataclass(frozen=True, slots=True)
class Rendered:
    text: str
    locator_map: LocatorMap
    barriers: tuple[int, ...]           # hard chunk boundaries, char offsets
    degradations: tuple[Degradation, ...]
```

Every `Rendition` carries a locator. A frame returns image content tagged with a
`TimeSpan`; a transcript range returns text tagged with a `TimeSpan`; OCR returns
text tagged with a `BBox`.

### 4.7 LocatorMap

The structure on which citation correctness rests. A sorted, non-overlapping,
gapless sequence of `(CharSpan, Locator)` segments over `[0, len(text))`, with a
pure `resolve(offset) -> Locator` and `resolve_span(CharSpan) -> tuple[Locator, ...]`.

Laws (property-tested):
- **Total**: every offset in `[0, len(text))` resolves.
- **Monotonic**: resolution never goes backwards.
- **Non-overlapping**: segments partition the range.

An earlier draft required **barriers to be a subset of segment boundaries**.
That is withdrawn: it is over-specified and would forbid a correct video
representation. A scene cut can fall mid-utterance — someone keeps talking
across the cut — so a barrier at the cut lands inside a transcript cue's
segment. Chunking there is fine: both halves still resolve to that cue's
`TimeSpan`, so the citation stays correct. What `Rendered` actually enforces
is that barriers lie within the text and are sorted and unique, which is
sufficient.

### 4.8 Capability

```python
class Capability(StrEnum):
    VISION = "vision"
    ASR = "asr"
    DIARIZATION = "diarization"
    TEXT_LLM = "text_llm"
    FFMPEG = "ffmpeg"
    EXIFTOOL = "exiftool"
    LIBREOFFICE = "libreoffice"
    TESSERACT = "tesseract"
```

Model capabilities and OS binaries are the same kind of thing. A missing ffmpeg
degrades exactly like a missing VLM. One mechanism, no special cases.

---

## 5. Ports

All `Protocol` + `@runtime_checkable`, decomposed into narrow capability slices;
collaborators annotate the slimmest slice they use.

| Port | Responsibility |
| --- | --- |
| `SourceStat` / `SourceReader` / `SourceLister` (`FileSource`) | Existence and size; streamed bytes and ranged reads; directory walking. |
| `MimeDetector` | Content-sniffed mimetype; filename is a tiebreak only. |
| `ContentHashing` | `async hash(uri) -> ContentHash`. Added after Plan 1: `Perception` initially depended on the concrete `ContentHasher`, the one non-hexagonal seam in the core. import-linter cannot catch it, because `pipeline` legitimately sits above `adapters`. Without this port a caller cannot supply a precomputed or remote hash without subclassing. |
| `ArtifactStore` | Content-addressed immutable put/get of derived artifacts. |
| `MediaProbe` | Container/stream metadata without decoding. |
| `FrameExtractor` | Frame at time, frames over a range, scene-change detection. |
| `AudioExtractor` | Extract/normalize an audio track or clip. |
| `Transcriber` | Audio to `tuple[TranscriptCue, ...]` with word-level timing. |
| `Diarizer` | Audio to speaker turns; may be satisfied by a diarizing `Transcriber`. |
| `VisionModel` | Image content to text (description, OCR, chart reading). |
| `TextRecognizer` | Pure OCR: image to text plus `BBox` locators. Satisfied by `VisionModel` when `Capability.VISION` is present, or by a tesseract adapter when it is not. |
| `TextModel` | Text completion, for summarization within handlers. |
| `BinaryProbe` | Resolve OS binaries on PATH into capabilities. |

Ports raise typed exceptions rooted at `ReadEverythingError`, split into domain
and infrastructure families following eventsource-py's convention.

---

## 6. Registry and dispatch

```python
class MediaHandler(Protocol):
    mime_patterns: ClassVar[tuple[MimePattern, ...]]
    priority: ClassVar[int]
    handler_id: ClassVar[str]
    handler_version: ClassVar[int]

    def requires(self) -> frozenset[Capability]: ...
    def affordances(self) -> tuple[Affordance, ...]: ...
    async def describe(self, ref: SourceRef) -> Card: ...
    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition: ...
    async def represent(self, ref: SourceRef, budget: Budget) -> Rendered: ...
```

Handlers are stateless and receive every capability by constructor injection.

### Resolution order (deterministic)

1. Exact mimetype match
2. Structured suffix (`+zip`, `+xml`, `+json`)
3. `type/*`
4. `MediaKind` fallback
5. Binary fallback handler

Ties are broken by explicit `priority`, so a caller can shadow a bundled handler
without forking. **There is no "unsupported file" error path** — the worst
outcome is a thin facts-only card.

That last property is not free, and Plan 1 found it briefly untrue. It holds
only if nothing between the caller and the fallback can raise, and detection
sits in that path: an unguarded `puremagic.magic_string` would have propagated
instead of falling through to `application/octet-stream`, and the dependency
floor is unbounded so a future version could start raising. **Every call on the
path to the fallback must be unable to escape it** — this is a property to be
defended in code, not an assumption to be stated here.

### Capability filtering (two stages)

1. A handler whose `requires()` is unsatisfied is dropped from the registry.
2. A surviving handler's individual affordances are filtered by their own
   `requires`.

With no ASR configured, video still works: metadata, outline, and frames are
available, and `read_transcript` simply does not exist.

---

## 7. Handler families

| Family | Card from | Deep affordances | Adapter |
| --- | --- | --- | --- |
| `video/*` | ffprobe JSON + scene-change outline | `get_frame_at`, `describe_frame`, `read_transcript`, `search_transcript`, `list_speakers`, `watch_segment` | ffmpeg/ffprobe |
| `audio/*` | ffprobe metadata | `read_transcript`, `search_transcript`, `list_speakers`, `get_clip` | ASR port (+ diarizer) |
| `image/*` | Pillow dimensions + exiftool | `describe_image`, `ocr`, `crop_region` | Pillow, VLM |
| `application/pdf` | page count, per-page text density | `read_pages`, `render_page`, `describe_page` | pypdfium2 |
| Office documents | via PDF conversion | inherits PDF's | `libreoffice --convert-to pdf`, delegates |
| `text/html` | title/meta + prose extraction | `read_range` | trafilatura |
| `text/*`, source code | encoding, line count, heading/symbol outline | `read_range` | charset-normalizer, stdlib |
| Tabular (`csv`, `parquet`) | inferred schema + head | `query_columns`, `sample_rows` | stdlib csv, pyarrow |
| Archives (`zip`, `tar`) | entry listing | members addressed as `path!member` | stdlib |
| Fallback | exiftool facts + hexdump excerpt | — | always succeeds |

### Defended choices

**pypdfium2 over PyMuPDF.** PyMuPDF is AGPL, which is unacceptable in a library
that research-team embeds. pypdfium2 is permissively licensed and provides both
text extraction and in-process page rendering. Rendering matters: the scanned-PDF
path is "page has near-zero extractable text, therefore render and hand to the
VLM," which removes the need for tesseract on the critical path.

**faster-whisper for the local ASR adapter**, chosen for word-level timestamps —
these are what make `TimeSpan` locators precise enough to cite honestly. CPU-only
inference is slow, so a server ASR endpoint is the default and local is the
fallback.

**pyannote.audio as the primary diarizer**, in a `[diarization]` extra. Best
accuracy; costs a torch dependency and gated weights. The HF token is a
constructor argument. The `Diarizer` port keeps sherpa-onnx and server-side
diarizing ASR as drop-in alternatives.

**Video representation is interleaved, not concatenated.** `represent()` emits a
single timeline-ordered stream mixing transcript cues and periodic frame
descriptions, each segment carrying its `TimeSpan` into the locator map, with
hard barriers at scene boundaries **and speaker turns**. A chunk spanning two
speakers produces a mis-attributed claim, so speaker barriers are a correctness
requirement, not a nicety. Qwen3.8's native video support is used by
`watch_segment` for bounded ranges; the indexing path stays frame-sampled so cost
remains predictable.

---

## 8. Caching

The cache key is the whole derivation, not just the file:

```
key = H(content_hash, handler_id, handler_version, affordance, params, capability_fingerprint)
```

- `content_hash` — a moved or renamed file is a hit; an edited one is a miss.
  **There is no staleness protocol because there is no mutable key.**
- `handler_version` — a fixed extraction bug invalidates exactly what it should.
- `capability_fingerprint` — model id plus revision per capability. Swapping the
  VLM must not silently serve a mixture of descriptions from two models.

Entries are immutable; eviction is size/LRU and never correctness-relevant.

Content hashing uses streamed `hashlib.blake2b` (stdlib, no dependency), fronted
by a `(device, inode, size, mtime_ns) -> hash` memo. The memo is an optimization
only: a miss costs a rehash, never a wrong answer.

---

## 9. redstring integration and the provenance RFC

### Boundary rule

Upstream a change to redstring **only if it can be justified with a purely-text
example**. If the change requires redstring to know what a pixel or a second is,
it stays here.

### Prerequisite RFC

Specified in full at `docs/rfcs/0001-source-spans-carry-caller-provenance.md`.

Investigation of redstring's actual code collapsed the three changes originally
scoped here into **one new concept**: a `SourceDocument` may carry `spans` —
sorted, non-overlapping character ranges over its own text, each with an opaque
payload and an optional `barrier` flag. From that single addition all three
follow.

Three findings drove the collapse:

- `StoredChunk.metadata` already exists and round-trips through the postgres
  `jsonb` column, but `extraction/corpus.py::build_stored_chunks` never
  populates it from `Chunk.metadata`. The channel is plumbed end to end and
  structurally unreachable.
- Retrieval already returns `StoredChunk`s via `ScoredChunk`, so provenance at
  retrieval needs a test rather than an implementation.
- The real blocker is that a caller cannot annotate chunks it does not create.
  redstring chunks internally from `SourceDocument.text`, and
  `SourceDocument.metadata` is document-scoped — the granularity that makes a
  citation useless.

One consequence propagates into this library's design: because `chunk_id` is
content-addressed on text, identical passages collapse to one chunk, so
provenance is **multi-valued**. Payloads accumulate in document order, mirroring
how `entity_ids` already accumulates. `LocatorMap.resolve_span()` therefore
returns a tuple of locators, and citation rendering must handle a passage with
several origins — the repeated-transcript-cue and repeated-PDF-footer cases.

No new dependencies, no new ports, and redstring remains pure and text-only.

### Explicitly not upstreamed

- **Image/video embeddings, multi-vector retrieval.** Instead, content is
  described into text at this library's edge by the VLM and redstring embeds the
  description. This is the honest model: redstring's graph is a graph of claims,
  and a claim is textual. A frame does not enter the graph; what the VLM asserts
  about the frame does, with the frame's locator as provenance.
- **Multimodal extraction** (handing `LlmProvider.extract` image blocks). Changes
  redstring's core port to serve one caller. Revisit only if text descriptions
  demonstrably lose too much on diagrams and charts.
- **Any mimetype, ffmpeg, ASR, or codec awareness.**

### Sequencing

The RFC is a hard prerequisite. Land the three redstring changes, cut a redstring
release, pin it, then build here. No development against an unreleased sibling.

### Ingestion identity

`source_id = uuid5(namespace, content_hash)` — derived, not stored, mirroring how
redstring derives its stream ids. Re-ingesting the same bytes is a no-op
regardless of path; ingesting an edited file produces a new document rather than
corrupting the old one's provenance.

---

## 10. Agent surface

### Tool pack (framework-agnostic)

**Three tools, not one per affordance.** Affordances are per-mimetype and
therefore per-file, so a tool-per-affordance would need a tool list that changes
with whatever the agent last looked at — which no agent framework supports and
no model handles well. Instead `inspect_path` returns the card *including each
affordance's JSON schema*, and `invoke_affordance` runs one by name;
`list_paths` walks a tree. **The card is the discovery mechanism**, which is
progressive disclosure made concrete. The consequence is that the model learns
this file's tools at runtime rather than from its system prompt, and any other
agent surface — including the deepagents adapter below — must stay consistent
with that.

**The tool pack never raises.** Every tool returns a structured result; the
raise-to-return conversion is a single decorator in `agent/`. A traceback
reaching a model is a wasted and unrecoverable turn. This mirrors deepagents'
`BackendProtocol`, whose methods return structured results rather than raising.

The guarantee has a boundary worth naming, because Plan 1 found it leaking
there. The decorator wraps the tool *body*, but LangChain's `StructuredTool`
validates arguments against `args_schema` **before** the body runs, so a
malformed call raised `ValidationError` out of `ainvoke` untouched — on exactly
the input the guarantee exists for, since tool arguments are model-authored and
therefore untrusted. Framework-level validation must be routed into the same
structured shape (`handle_validation_error`, or validating inside the body).
**The guarantee begins at the framework boundary, not at our function.**

### deepagents backend decorator (optional, `[deepagents]` extra)

`MediaAwareBackend` wraps any `BackendProtocol`:

- `read` on a non-text mimetype returns the card instead of garbage or an error.
- `ls`, `glob`, `grep`, `write`, `edit` delegate unchanged to the wrapped backend.

This follows research-team's `EventSourcedBackend` pattern: override the minimum
surface, inherit the rest verbatim. It composes with `CompositeBackend`, so media
awareness can be scoped to specific path prefixes.

---

## 11. Composition root

`composition/` is a thin layer of at most two modules, following redstring's
discipline that a candidate for this layer must **name the forbidden pair it
joins**:

- `perceive.py` — joins registry + handlers + adapters + capability negotiation
  into a ready `Perception` object.
- `ingest.py` — joins `Perception` with the redstring sink.

This is what makes the library adoptable as research-team's knowledge substrate:
research-team's `KnowledgePort` adapter points here instead of at redstring
directly, and its wiring is deleted rather than duplicated.

---

## 12. Error handling and budgets

**Raise vs. return splits by audience.** Ports and handlers raise typed domain
exceptions rooted at `ReadEverythingError`. The tool pack returns structured
results.

**Budget is passed into `represent()`, not enforced around it.** A handler that
cannot fit its content degrades on its own terms — video reduces frame-sampling
density before dropping transcript; a PDF drops page renders before page text —
and reports what it dropped in `Rendered.degradations`. Silent truncation is the
failure mode most deliberately designed out, because it is invisible in exactly
the case where the answer is wrong.

**Concurrency** is async end to end, with a bounded semaphore **per capability**
rather than one global limit: ASR, VLM, and ffmpeg have very different cost
profiles and saturating one must not stall the others. Limits are constructor
arguments.

---

## 13. Testing strategy

### Shipped conformance suites

`readeverything/testing/`, in the wheel, `forbidden` from importing above
`ports` + `domain`: `MediaHandlerCompliance`, `ArtifactStoreCompliance`,
`FileSourceCompliance`, `MimeDetectorCompliance`, `TranscriptionCompliance`,
`DiarizerCompliance`, `VisionCompliance`.

### A law must be able to fail

These suites ship in the wheel, so whatever they actually check is what
third-party handler authors inherit. **A law that cannot fail is worse than no
law: it certifies nothing while looking rigorous.** Plan 1 shipped two such
laws before they were caught.

- One asserted `budget.permits(len(text)) or degradations` — satisfiable by a
  handler that truncated silently (it fits) *and* by one that reported a
  degradation without ever truncating (it cried wolf). It constrained neither
  direction. The fix compares against an unbounded render: shorter than
  unbounded *requires* a degradation, equal to unbounded *forbids* one.
- Another asserted that a `Rendered`'s map covers its text — which
  `Rendered.__post_init__` already enforces, so any handler returning a
  `Rendered` at all passed.

Two practices follow, and both are requirements rather than suggestions:
**every law is exercised against a deliberately-broken handler** that violates
exactly that law and no other; and when writing a law, state what a broken
implementation would do differently, and check the assertion distinguishes it.

### Handler laws (encoded in the suites)

- `describe()` depends only on content: same bytes, different path, identical card.
- Every affordance in a card is invocable; every invocable name is declared.
  Drift between the two would make capability negotiation a lie.
- Every `Rendition` locator lies within the source's bounds.
- `LocatorMap` is total, monotonic, and non-overlapping (Hypothesis).
- Truncation is announced and announcements are truthful, checked in both
  directions against an unbounded render.

### Fixtures are generated, not committed

A session fixture synthesizes tiny deterministic media with ffmpeg — color bars
with burnt-in timecode, a sine tone at known intervals, a two-page PDF. This
gives exact ground truth ("the frame at 1.5s is green", "the tone starts at
0.8s") without binaries in git, and skips cleanly when ffmpeg is absent.

### Model nondeterminism

Unit tests never assert on model text. They use `FakeVision`, `FakeTranscriber`,
and `FakeDiarizer` from `testing/` and assert on structure, locators, and
degradations. Model quality is **measured, not asserted**: a `bench/` harness
with committed results and drift runs, following redstring's pattern.

### Markers

`unit` (default), `integration` (needs binaries), `live` (needs the model
server), `accuracy` (bench), `slow`.

### Enforcement tests (ported from redstring)

Dependencies-confined AST test; reads-no-environment; public-surface-is-
self-contained; quality-gates-agree (pre-commit entry equals CI string);
coverage ratchet with committed baseline; stale-exemption detection.

---

## 14. Packaging

- hatchling + uv, `py.typed`, `requires-python >= 3.13`
- mypy strict, PEP 695 inline type parameters, ruff pinned, bandit, pip-audit
- import-linter contracts with `exhaustive = true`
- mkdocs Diátaxis plus numbered ADRs
- Lazy PEP-562 front door (`_LAZY` name-to-module map, eventsource-py's pattern):
  `import readeverything` loads no driver.

Base install is pure-Python and light: `pydantic`, `redstring`, mime detection,
`charset-normalizer`.

Extras: `[pdf]`, `[images]`, `[html]`, `[asr]`, `[diarization]`, `[tabular]`,
`[langchain]`, `[deepagents]`, `[all]`.

OS binaries (ffmpeg, ffprobe, exiftool, libreoffice, tesseract) are discovered by
`BinaryProbe` at composition time with install hints in the error path, never at
import.

### Reference model deployment

OpenAI-compatible endpoint at `http://192.168.1.14:8080/v1/`, model
`qwen3.8-27b-mtp` (multimodal: images and video, documents, STEM diagrams,
long-form video).

This is configuration, not environment: the base URL, model id and API key are
constructor arguments to the adapter, and `test_reads_no_environment` enforces
that nothing below the composition root reads them from the process
environment. The value is recorded here so the `capability_fingerprint` has a
known referent — swapping this model must change the artifact cache key.

Live tests against it carry the `live` marker and are deselected by default.

### Environment status on the reference machine

Present: ffmpeg, ffprobe, exiftool, pdftotext, pdftoppm, libreoffice.
Absent: tesseract, pandoc, mediainfo. No NVIDIA GPU — diarization and local ASR
are CPU-bound, which is why server-side ASR is the default.

---

## 14b. Carried out of Plan 1

Recorded here rather than lost, because each is a decision someone will
otherwise rediscover.

**Owed early in the next plan, and getting more expensive:**
- Annotate `Perception.hasher` against the new `ContentHashing` port (§5). Every
  handler added first makes this a wider change.
- Fix `artifact_key`'s `json.dumps(..., default=str)`: `{"path": Path("a")}` and
  `{"path": "a"}` collide on one key. Free now — no caller passes non-primitives
  and the cache is not wired — and a silent wrong-answer bug once it is.

**Owed with cache wiring:** `Perception._resolve` re-reads, re-detects and
re-hashes on every `inspect`/`invoke`/`represent`, so a large-file `invoke`
re-hashes per affordance call. Three call sites change together.

**Ports specified but deliberately unbuilt**, each landing with the handler that
implements it: `MediaProbe`, `FrameExtractor`, `AudioExtractor`, `Transcriber`,
`Diarizer`, `VisionModel`, `TextModel`, `TextRecognizer`, `BinaryProbe`. A
Protocol with no implementer is untested surface.

**Also unbuilt:** per-capability concurrency semaphores (nothing yet does
concurrent expensive work); generated media fixtures (they arrive with the
handlers needing them).

**Small and carried:** ~~no test for `BinaryHandler` at `max_chars=0` (correct by
trace — it returns one character because `CharSpan(0, 0)` raises, and the
overrun is announced)~~ — **wrong, and closed in Plan 2.** The trace was right
that a character comes back and wrong that the announcement was true: the
degradation reported the budget, so `max_chars=0` claimed "kept 0" of one
character. Reading the trace confirmed the behaviour and never checked the
claim the behaviour makes about itself. All three handlers now report
`len(text)`; `resolve_span` scans to the end of the map rather than
stopping at the first non-overlapping segment; `LocatorMap.__post_init__` makes
two O(n) passes; coverage passes at 91% against a 90% floor, which is under a
point of headroom.

**The largest unvalidated assumption:** nothing has touched a real model server.
Capability negotiation, the `VisionModel` port shape and the
`capability_fingerprint` are all proven against fakes only. The next plan should
close that before widening handler coverage.

## 15. Deferred to Spec 2

- Chunking `Rendered` into redstring `SourceDocument`s
- Retrieval and the sourced-answer loop (`ask(path, question)`)
- Citation rendering from provenance through `LocatorMap.resolve()`
- Incremental re-indexing of changed trees

---

## 16. Open risks

| Risk | Mitigation |
| --- | --- |
| redstring RFC lands slower than expected, blocking everything | The RFC is three small changes with text-only justification; perception core §4–§8 and §10 have no dependency on it and can proceed in parallel. |
| CPU-only diarization is too slow to be usable | `Diarizer` is a port with server-side and sherpa-onnx alternatives; diarization is capability-gated, so degradation is already a supported path. |
| VLM frame descriptions are too lossy for diagrams | Measured by the `bench/` harness rather than assumed. If confirmed, the escape hatch is upstreaming multimodal extraction, explicitly deferred in §9. |
| Interleaved video representation produces poor chunks | Barriers at scene and speaker boundaries; chunk quality is a bench metric with committed baselines. |
| Cache key omits something material | `capability_fingerprint` covers model swaps; `handler_version` covers logic changes. A wrong key is a stale artifact, detectable by bumping `handler_version` and comparing. |
