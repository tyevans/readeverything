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
| 4. The document family | Plan 4 | pending | 381 tests, 94.16%. PDF via pypdfium2. First honest producers of `PageRef`, `BBox`-on-a-page, and `Rendered.barriers` |

## The two tracks, and why this order

The destination needs both:

- **"any file"** — media handlers. Today: text, images, binary fallback. No
  audio, video, PDF, office, or archives.
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

**Cycle 5 — the time-based family (audio, video).**
Ports: `AudioExtractor`, `FrameExtractor`, `Transcriber`, `Diarizer`. `TimeSpan`
already exists as a locator and has never had a producer. ffmpeg becomes the
first real `BinaryProbe` consumer, which finally exercises capability
negotiation against a genuine OS dependency.

**Cycle 6 — observability and concurrency.**
Carried from Plan 3's findings: the library has NO logging, tracing, or metrics
anywhere in `src/`, and nothing runs concurrent work. A whole-tree index and
ffmpeg-driven extraction both need concurrency; debugging either needs
observability. Per-capability semaphores were deferred from Spec 1 §14b on the
grounds that nothing did concurrent expensive work — Cycle 5 ends that.

**Cycle 7 — the query layer.**
`ask(path, question) -> sourced answer`. Chunking `Rendered` into retrievable
units that keep provenance, retrieval, and citations that resolve through
`LocatorMap` back to exact file locations. Depends on the redstring decision
(see Open questions).

**Cycle 8+ — re-evaluate.** Candidates: office documents, archives, incremental
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
bearable, so it lands with or after Cycle 6.

**Two open sites for pdfium.** `adapters/pdfium_probe.open_document` raises
`InfrastructureError`; `PdfHandler._open` returns `None` and degrades. The
difference is intentional — a probe may fail loudly, a handler must never raise
— and both docstrings now say so. Worth revisiting only if a third opener
appears.

## Open questions the research must settle

- **redstring**: is it the retrieval substrate, does it need RFC 0001 landed
  first, or is it the wrong tool? RFC 0001 has never landed and is owed.
- **PDF licensing**: pymupdf is AGPL; this library is MIT. If pymupdf is the
  best extractor, that is a licensing conflict, not a preference.
- **Transcription**: local (faster-whisper) versus the existing
  OpenAI-compatible model server at `http://192.168.1.14:8080/v1/`.

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
exists when it does not.

Hunt it by shape across the whole tree, never by site — briefs that name known
sites produce reviewers who stay inside them.
