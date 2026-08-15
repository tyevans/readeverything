# readeverything — autonomous development roadmap

**Standing instruction (2026-08-15):** run research → spec → plan →
subagent-driven development → merge, on a loop, without stopping, until the
user says stop. Each cycle identifies the next most valuable step and lands it
on `main`.

**The destination**, in the user's words: *"a full suite of tools that let me
ask the filesystem or any file a question and get real answers"* — robust,
observable, with strong DDD and SOLID vision, built from real user stories.

## Landed

| Spec | Plan | Merge | Result |
| --- | --- | --- | --- |
| 1. Perception core | Plan 1 | — | 150 tests |
| 1 (vision half) | Plan 2 | `618d1a2` | 230 tests, image family, live model |
| 3. Integration & first product | Plan 3 | `60ed781` | 335 tests, 94.69%, composition root, cache wired, capability discovery, integration tier, README |
| 4. The document family | Plan 4 | `d4f564b` | 381 tests, 94.16%. PDF via pypdfium2. First honest producers of `PageRef`, `BBox`-on-a-page, and `Rendered.barriers` |
| 5. The moving image | Plan 5 | `da4d0e6` | 453 tests, 92.57%. Video via ffmpeg. `TimeSpan` gets its producer — **every locator type is now real**. `BinaryProbe` fixed |
| 6. The spoken word | Plan 6 | `2e86b6c` | 537 tests, 92.62%. Audio via faster-whisper; `TranscriptCue` gets its producer; video interleaves cues with frames; `domain/timeline.py` shared by both |

## The two tracks, and why this order

The destination needs both:

- **"any file"** — media handlers. Today: text, images, PDF, video, audio,
  binary fallback. Still missing: office documents, archives.
- **"ask a question, get real answers"** — the query layer: chunk `Rendered`,
  index it, retrieve, and answer with citations that resolve to exact source
  locations.

They are largely independent — the query layer's design does not change with
handler count; more handlers just give it more to chew on. But building the
query layer against three handlers and re-validating it against ten is wasted
work, and the file types where "ask it a question" is most valuable (PDF,
recorded audio, video) are exactly the ones missing.

**So: media first, query second.** Each cycle re-evaluates.

## Planned cycles

**Cycle 4 — the document family (PDF first).** ✅ **Done.** `MediaProbe`,
`TextRecognizer`, `PdfHandler` with `read_page`/`page_region`/`page_image`/
`ocr_page`, page-mapped `represent()` with page-break barriers, and the
scanned-versus-blank distinction (they are identical through the text layer;
`page.get_objects()` is what tells them apart).

**Cycle 5 — the moving image.** ✅ **Done.** `StreamProbe`, `FrameExtractor`,
`VideoHandler`, barriers at scene cuts, and the `BinaryProbe` fix. Split from
its original scope: transcription moved to Cycle 6 because the model server's
support for timestamped transcription is unverified and designing an honest
degradation for that deserves its own cycle.

**Cycle 6 — the spoken word.** ✅ **Done.** `AudioExtractor`, `Transcriber`,
`AudioHandler`, video transcript interleaving, and `domain/timeline.py` — one
tiling rule for three callers. Transcription is local (the server returns 501,
settled by request rather than by reading a spec). Diarization stayed out:
flipping Spec 1 §7's pyannote default to sherpa-onnx (Apache-2.0, ONNX, CPU, no
gated weights) is a real decision, not a footnote. `TranscriptCue.speaker`
remains `None` — the field admits ignorance rather than guessing.

**Cycle 7 — observability and concurrency. NEXT.**
Carried since Plan 3: the library has NO logging, tracing, or metrics anywhere
in `src/`, and nothing runs concurrent work. The case is now overwhelming:
`represent()` on a video samples frames and calls a vision model per sample, and
on audio runs a whole transcription — the two most expensive operations the
library performs, both sequential, both invisible while they run. A caller
waiting ninety seconds on a long recording has no way to tell progress from a
hang.

Per-capability semaphores were deferred from Spec 1 §14b because nothing did
concurrent expensive work. Cycles 5 and 6 ended that.

This cycle is also the natural home for the OCR-in-`represent()` budget owed
since Cycle 4, since both need the same thing: a way to bound and observe
expensive model work.

**Cycle 8 — the query layer.**
`ask(path, question) -> sourced answer`. Chunking `Rendered` into retrievable
units that keep provenance, retrieval, and citations that resolve through
`LocatorMap` back to exact file locations. Depends on the redstring decision
(see Open questions).

**Cycle 9+ — re-evaluate.** Candidates: office documents, archives, incremental
re-indexing of changed trees, a whole-tree `ask`, richer agent tooling.

## Owed, discovered during Cycle 4

**Per-affordance cache keys.** `CapabilitySet.fingerprint()` digests the whole
capability set, not the capabilities a given affordance requires. Measured:
swapping a vision model changes the key for `read_page`, `hexdump` and
`read_range` — none of which depend on any model — so one model swap discards
every cached artifact for every file. Safe (over-invalidation recomputes and
gets the same answer) but wasteful. The fix keys each artifact on only the
capabilities its own affordance requires, which changes `artifact_key`'s
signature and every call site. Inherent to the key derivation wired in Spec 3;
PDF only made it visible by being the first handler with both model-dependent
and model-independent affordances.

**OCR during `represent()`, with a budget.** `represent()` deliberately does not
OCR: it is the indexing feed, and OCR is one model call per page — four hundred
sequential calls for a four-hundred-page scan. The cost is real and recorded:
a scanned document contributes no readable text to an index built only from
`represent()`. It stays readable through `ocr_page`. Closing this needs
model-call budget semantics ("OCR up to N pages") and the concurrency to make it
bearable, so it lands with or after Cycle 7.

**Two open sites for pdfium.** `adapters/pdfium_probe.open_document` raises
`InfrastructureError`; `PdfHandler._open` returns `None` and degrades. The
difference is intentional — a probe may fail loudly, a handler must never raise
— and both docstrings now say so. Worth revisiting only if a third opener
appears.

## Owed, discovered during Cycle 7

**Observer emission from `invoke()`, with cache hits distinguishable from real
model calls.** Spec §7 asks "How many model calls did indexing this directory
actually make?" and **nothing in this library can answer it.** Cycle 7 wired
`Observer` into all six handlers' `represent()`, and that is genuinely useful,
but it is a different question: a `represent()` start/finish pair says a file
was represented, not that a model was consulted. Every affordance invocation —
`describe_image`, `ocr`, `ocr_page`, `describe_frame` — is silent, and those are
where the model calls actually happen.

This was deliberately not fixed alongside the rest of Cycle 7's observability,
because it is design work rather than instrumentation. Affordance invocation
goes through a content-addressed artifact cache, so an emission wrapped naively
around `invoke()` would count a cache hit as a model call and give a caller a
number that is wrong in the direction that matters most — the whole point of the
cache is that the second read costs nothing, and a counter that cannot see that
is worse than no counter. The event vocabulary needs a way to distinguish "a
model was called" from "an artifact was served", and that belongs in its own
cycle with its own spec.

Until then: **no document should claim §7 is answerable.**

## Open questions the research must settle

- **redstring**: is it the retrieval substrate, does it need RFC 0001 landed
  first, or is it the wrong tool? RFC 0001 has never landed and is owed.
- **PDF licensing**: pymupdf is AGPL; this library is MIT. If pymupdf is the
  best extractor, that is a licensing conflict, not a preference.
- ~~**Transcription**: local versus the model server~~ — **SETTLED by testing.**
  The server returns `501 {"message":"The current model does not support audio
  input."}` on `/v1/audio/transcriptions`, and `/v1/models` lists no audio model.
  Transcription must be local (`faster-whisper`, MIT, explicit local model
  directory since this library downloads nothing implicitly).

**Discovered while testing that:** `/v1/models` also lists **`nomic-embed-text`**
— an embedding model, on hardware already running. Cycle 8's query layer needs
embeddings for retrieval, and redstring's `EmbeddingProvider` is an injected
Protocol with a LangChain adapter for any OpenAI-compatible server. The
retrieval substrate's embedding half may already be available. Recorded here so
Cycle 8 does not re-derive it.

## Standing constraints (every cycle)

- Model server for live tests: `http://192.168.1.14:8080/v1/`, model
  `qwen3.8-27b-mtp`. **Tell the user before a live run** so they can stop other
  inference.
- The library reads NO environment variables under `src/`.
- Python 3.13, PEP 695 inline type parameters.
- `mypy --strict`; import-linter layered contract, `exhaustive = true`.
- Base install stays light; heavy dependencies behind extras.
- Coverage floor 92.
- A law must be able to fail. A test that cannot fail is not a test.

## The recurring defect shape — check every cycle

**Text asserting something nothing established.** Found in every review wave so
far, at every altitude: a degradation naming a cause nothing checked; a cache
key claiming two derivations are identical; a message pointing at a tool that
does not exist; a locator describing a span that is not there; a test docstring
promising a guarantee its assertions do not provide; a spec claiming a test
exists when it does not; a capability probe recording a deprecation warning as a
version number; a test name promising an explanation its assertions never check.

Hunt it by shape across the whole tree, never by site — briefs that name known
sites produce reviewers who stay inside them.

**Five cycles in, the pattern is clear enough to state as a rule.** It appears
most often in the components written to prevent it: the probe built to replace
assumption with observation was asserting; the spec written to hunt the shape
contradicted itself two sections apart; the tests written to prove a base
install works proved nothing. This is not carelessness in any one place. It is
the natural decay of text written beside code and never re-checked against it.

So: **a claim about what code does belongs in a spec, a docstring, a comment or
a test name only after running the code.** Every ruling this loop makes is
measured before it is written down, and every cycle's most valuable finding so
far has come from running something nobody had run — a probe against real
binaries, a seek past the end of a real video, a serializer against a real
union.

**Cycle 7's instance, and it is the sharpest yet.** The shape appeared in a test
written *to enforce the shape's absence*. Spec §1.1 promised a caller "sees each
file start... and each file finish", so the integration test asserted every file
in a directory emits `OperationFinished`. Nothing in the design ever made that
true: `grep -c "emit(" src/readeverything/handlers/*.py` returned text 0,
binary 0, image 0, pdf 0, audio 3, video 3. Two of six handlers spoke. The spec
had contradicted itself again — §2.4 scoped emission to "both expensive paths,
video and audio" while §7 promised the caller could count *model calls*, which
`ImageHandler` and `PdfHandler`'s OCR path make and neither reported.

**And the correction did not close §7.** All six handlers now emit from
`represent()`, but a `represent()` event is not a model call and must not be
read as one: it says a file's representation started or finished, nothing about
whether a model was consulted along the way. **§7's question — "How many model
calls did indexing this directory actually make?" — remains unanswerable**, and
any ruling of this loop that said otherwise was wrong. No `invoke()` path emits
at all: `describe_image`, `ocr`, `ocr_page` and `describe_frame` are silent. See
"Owed, discovered during Cycle 7" below. This paragraph exists because the same
loop that named prose-decay as its own recurring defect committed it again in
the ruling that named it.

The specific lesson, distinct from prior cycles: **an acceptance sentence is a
claim about code too.** §1.1 was written before the handlers existed, was never
re-checked against them, and its authority made a false test look correct — the
test was graded against the prose rather than the program. Prose in a spec decays
exactly like a docstring; being labelled "acceptance" grants it no immunity, and
a test derived from unverified prose inherits the error instead of catching it.
Both counts above came from running `grep`, not from reading either document.
