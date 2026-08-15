# readeverything: The Document Family

**Date:** 2026-08-15
**Status:** Approved for planning
**Predecessors:** Spec 1 (perception core), Spec 3 (integration and first product)
**Landed:** Plan 1 (150 tests), Plan 2 (230), Plan 3 (335, merged `60ed781`)

---

## 1. Why this, and why now

The library reads text, images, and an honest binary fallback. It cannot read
a PDF — the single format most likely to hold the answer to a question someone
actually has.

But the better argument is structural. `readeverything` has five locator types:

| Locator | Producer today |
| --- | --- |
| `CharSpan` | `TextHandler`, `ImageHandler`, `BinaryHandler` |
| `ByteRange` | `BinaryHandler` |
| `BBox` | `ImageHandler` — with `page=None` only |
| `PageRef` | **none** |
| `TimeSpan` | **none** |

And `Rendered` carries a field it has never once populated:

```python
barriers: tuple[int, ...]
```

Declared, validated in `__post_init__` for sortedness, uniqueness and range,
documented as "hard chunk barriers" — and set to `()` by all three handlers
(`text.py:149`, `binary.py:147`, `image.py:294`). Zero producers in `src/`.

This is the defect shape this project keeps finding, at the type level: a
field whose validation asserts a capability nothing provides. `PageRef` and
`BBox(page=...)` are the same — a vocabulary designed for documents the library
cannot read.

**A PDF handler is the first thing that can honestly produce a `PageRef`, the
first that can produce a `BBox` on a real page, and the first with a natural
barrier: the page break.** It does not merely add a file type. It gives three
declared domain concepts their first producer, and turns RFC 0001's motivating
example from hypothetical into real.

### 1.1 Acceptance

> Point the library at a directory containing a born-digital PDF and a scanned
> one. `inspect` tells you the page count without extracting anything. `represent`
> returns text whose every character resolves to the page it came from, with
> barriers at page breaks. Asking for page 7 returns page 7. The scanned PDF
> says plainly that its text is in images, and — where a vision capability
> exists — reads it anyway and says that it did.

---

## 2. Scope

**In scope**

1. A `MediaProbe` port and a pypdfium2 adapter: page count and page dimensions,
   cheaply, without extracting text.
2. `PdfHandler`: card, affordances (`read_page`, `page_region`, `page_image`),
   and `represent()` producing page-mapped text with page-break barriers.
3. The scanned-PDF problem: detecting that a page has no extractable text, and
   never reporting that as "empty".
4. A `TextRecognizer` port, with the existing `VisionModel` as its first adapter
   — OCR by rendering the page and asking the model, reusing what exists rather
   than adding tesseract.
5. Real PDF fixtures, generated not committed, and live tests.

**Out of scope**

- **Audio and video.** Cycle 5. `TimeSpan` waits its turn; the ports
  (`AudioExtractor`, `FrameExtractor`, `Transcriber`, `Diarizer`) land with the
  handlers that implement them.
- **Office documents and archives.** Cycle 6+. Mechanically similar, different
  libraries, and the archive security work deserves its own attention.
- **The query layer.** Cycle 7. This spec produces `Rendered` with real
  barriers; consuming them is that cycle's job.
- **RFC 0001.** Still owed upstream, still independent. This spec makes its
  motivating example concrete, which strengthens it, but does not depend on it.
- **Observability and concurrency.** Cycle 6, and this spec deliberately does
  not add per-page concurrency — see §9.

---

## 3. The library choice, and a licensing constraint

**pypdfium2.** Apache-2.0/BSD, wrapping Google's PDFium — the engine in Chrome.
Ships a bundled binary in the wheel, so it needs no OS dependency and no
`Capability` member. Gives per-character and per-textrun bounding boxes with
page numbers, which is the hard requirement here.

**pymupdf is rejected on licensing, not on merit.** It is AGPL-3.0. This library
is MIT and is meant to be redistributed. Depending on it would force AGPL
disclosure on every downstream user or a commercial licence purchase. That is a
conflict, not a preference, and it is recorded here so nobody re-opens it as a
performance question later.

**pdfplumber** (MIT) is the runner-up — comparable bbox data, better table
ergonomics, measured 8–12× slower. If pypdfium2's bundled binary ever becomes a
packaging problem, this is the swap, and the `MediaProbe`/handler split is what
keeps that swap cheap.

Dependencies land behind an extra, `documents`, exactly as Pillow sits behind
`images`. The composition root already omits handlers whose imports fail
(`composition.py::_optional_image_handler`); `PdfHandler` follows that pattern
unchanged.

---

## 4. `MediaProbe` — cheap facts before expensive ones

Spec 1 specified this port and never built it. Its job is the card: what can be
said about a document without extracting its content?

```python
@runtime_checkable
class MediaProbe(Protocol):
    async def probe(self, data: bytes, mime: MimeType) -> DocumentFacts: ...
```

`DocumentFacts` carries `page_count`, per-page `(width, height)` in points, and
whatever metadata the format offers (title, author, producer). It carries **no
text**.

The separation is the point. `inspect` must stay cheap — Spec 1's whole
progressive-disclosure design rests on a card costing no real work — and page
count is exactly the fact that makes an agent's next decision. A card that
extracted text to count pages would defeat the design.

---

## 5. `represent()`, page mapping, and the first real barriers

### 5.1 What it produces

For a born-digital PDF, `represent` extracts each page's text in reading order,
concatenates it, and builds a `LocatorMap` in which every character resolves to
the `PageRef` of the page it came from.

`LocatorMap` requires total, gapless, zero-start coverage. One `LocatorSegment`
per page satisfies that naturally: pages partition the text, in order, with no
gaps. This is the first `LocatorMap` in the project whose segments are not
either "the whole thing" or a synthetic single span.

**Barriers go at page boundaries** — the character offset where each page after
the first begins. That is what a barrier is for: a point a chunker must not
split across, because text either side of it belongs to different pages and a
chunk spanning it would carry two `PageRef`s and cite both. `resolve_span`
already returns a tuple precisely to express that, so a chunk *may* cross a
barrier — the barrier says it should not do so casually.

### 5.2 Page granularity, and why not finer

`represent` maps to `PageRef`, not `BBox`. pypdfium2 can give character-level
boxes, and the temptation is to map every character to its rectangle.

Rejected, for three reasons. `LocatorMap` must be total and gapless, so
per-character boxes force a rectangle for whitespace and line breaks, which have
no honest rectangle. `BBox.__post_init__` rejects `w <= 0` or `h <= 0`, so
degenerate glyphs would have to be fabricated into non-degenerate boxes — the
defect shape again. And a per-character map over a 400-page document is a large
object built on every `represent`, to serve a granularity nothing has asked for.

Page granularity is honest, cheap, and total. **`BBox` belongs on the `invoke`
path**, where a caller asks about a specific region and pays for the precision
they requested. That is exactly the `DetailLevel` distinction the domain already
draws: `CARD` is free, `SEGMENT` is cheap, `DEEP` costs.

---

## 6. The scanned PDF, and the honest-failure requirement

A PDF whose pages are images carries no extractable text. `pdfium` returns the
empty string, and the naive handler reports an empty document.

**That is this project's recurring defect, at a new site.** "This document is
empty" is a claim about the document. What was actually observed is "the text
layer is empty" — which for a scan is not the same thing at all, and is the
difference between an agent concluding a contract says nothing and an agent
knowing to look harder.

### 6.1 Rules

- A page with no extractable text is **never** reported as empty text. It is
  reported as a page whose text could not be extracted, with a `Degradation`
  naming the reason.
- A handler must distinguish three states and say which: text extracted; no text
  layer but the page has image content (a scan); genuinely blank page.
  The third is rare and the first two are what matter, but reporting a scan as a
  blank page is the failure this section exists to prevent.
- When a page's text is recovered by OCR rather than extracted from the text
  layer, the rendition says so through a `Degradation`. OCR output is a model's
  reading, not the document's own bytes, and a consumer indexing it is entitled
  to know which it has — the same distinction §8.1 of Spec 3 drew for
  synthesized text.

### 6.2 `TextRecognizer`

```python
@runtime_checkable
class TextRecognizer(Protocol):
    model_id: str
    async def recognize(self, image: bytes, mime: str) -> str: ...
```

The first adapter wraps the existing `VisionModel`: render the page to a bitmap
with pypdfium2, ask the model to transcribe it. This reuses a capability that
already exists, already negotiates, and already contributes to the cache
fingerprint. A tesseract-backed adapter is a later addition behind
`Capability.TESSERACT`, which already exists in the enum and has never had a
consumer.

OCR is gated on `Capability.VISION`, so a deployment without a model gets the
honest degradation from §6.1 and no OCR — negotiation working exactly as
designed, and the same shape as `describe_image` today.

---

## 7. Affordances

| Name | Level | Locator returned | Requires |
| --- | --- | --- | --- |
| `read_page(page)` | SEGMENT | `PageRef` | — |
| `page_region(page, x, y, w, h)` | SEGMENT | `BBox` | — |
| `page_image(page, dpi)` | SEGMENT | `PageRef` | — |
| `ocr_page(page)` | DEEP | `PageRef` | `VISION` |

`page_region` is where `BBox` earns its place: the caller names a rectangle in
normalised coordinates and gets back the text inside it. Normalised rather than
points because `BBox`'s docstring already commits to surviving a DPI change, and
PDF pages vary in size — the handler normalises by the page dimensions
`MediaProbe` reported.

`page_image` returns `ImageContent`, which means an agent can already route it
into `describe_image` through the existing tool pack. A diagram on page 12
becomes describable without this spec building anything new for it.

---

## 8. Cache participation

`PdfHandler` declares a `handler_version`, so its renditions cache. This matters
more here than for text: extracting a 400-page PDF is real work, and OCR is a
model call per page.

The existing key already covers what changes the answer — `content_hash`,
`handler_id`, `handler_version`, affordance, params, and the capability
fingerprint.

**Correction, measured during execution.** This section originally claimed that
"swapping the model invalidates OCR artifacts and leaves extracted text alone,
because extraction does not depend on the model", and called it "the design
working". That is false, and it was written without checking:

```
OCR       model-a vs model-b differ: True
read_page model-a vs model-b differ: True     # ← should not
```

`CapabilitySet.fingerprint()` digests the **whole** capability set, not the
capabilities a given affordance requires. So swapping a vision model invalidates
every cached `read_page`, `hexdump` and `read_range` artifact for every file —
none of which depend on any model.

The behaviour is **safe**: over-invalidation means recomputing and getting the
same answer, never serving a wrong one. It is wasteful, not incorrect. And it is
not a PDF defect — it is inherent to the key derivation as wired in Spec 3, and
affects every handler. PDF merely made it visible by being the first handler
with both model-dependent and model-independent affordances.

**Deliberately not fixed here.** The real fix keys each artifact on only the
capabilities its own affordance requires, which changes `artifact_key`'s
signature and every call site. That deserves its own cycle and its own review
rather than being rushed into this branch. Recorded as owed in the roadmap.

The lesson is the one this project keeps paying for: a claim about what code
does belongs in a spec only after running it. "That falls out of the existing
design" was reasoning, presented as observation.

---

## 9. What this spec deliberately does not do

- **No per-page concurrency.** Extracting 400 pages in parallel is the obvious
  optimisation and it is Cycle 6's, with the semaphores and observability that
  make concurrent work debuggable. Adding a bare `gather` here would be the
  first concurrent work in the library with no instrumentation to see it and no
  bound on it.
- **No table extraction.** pdfplumber is better at it and this is not the cycle
  to change libraries over. A table currently reads as its text in reading
  order, which is honest and often enough.
- **No form fields, annotations, or embedded attachments.** Each is a real
  feature and none is on the path to answering a question about a document's
  text.

---

## 10. Acceptance

1. The §1.1 sentence is true, demonstrated by an integration test that performs
   exactly what it describes.
2. `PageRef` has a producer. `BBox` is produced with a real page number.
3. `Rendered.barriers` is non-empty for a multi-page PDF, with one barrier per
   page break, and a test asserts the barrier offsets match the page starts in
   the `LocatorMap`.
4. Every character of a multi-page PDF's `represent()` output resolves to the
   page it came from — asserted across page boundaries, not just in the middle
   of page one.
5. A scanned PDF is never reported as empty, with or without a vision
   capability. Both paths are tested.
6. OCR'd text is distinguishable from extracted text by a consumer.
7. A base install without the `documents` extra still builds a working
   `Perception`; PDFs fall through to the binary fallback.
8. All gates green, coverage floor holds at 92.

---

## 11. Risks

| Risk | Mitigation |
| --- | --- |
| pypdfium2's bundled binary complicates packaging on some platform | The `MediaProbe` port and handler split keep pdfplumber a contained swap. Wheels cover the platforms this targets; a source build is the fallback. |
| Page text extraction order differs from reading order for multi-column layouts | Real, and not fully solvable at this layer. `represent` commits to pdfium's order and says so; `page_region` gives a caller who cares a way to ask precisely. Do not silently claim reading order the extractor does not provide. |
| A malformed or encrypted PDF crashes the extractor | The handler never raises; it degrades, like every other handler. Encrypted-and-unopenable is a `Degradation`, not an exception. Fuzz-adjacent fixtures included. |
| OCR cost surprises a caller | It is `DEEP`, gated on `VISION`, cached, and never runs during `inspect` or `represent`. A caller must ask for it by name. |
| The 400-page `LocatorMap` is large | One segment per page, not per character (§5.2) — a 400-page map is 400 segments. Measured in a test, not assumed. |
