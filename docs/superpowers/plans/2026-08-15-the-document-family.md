# The Document Family Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read PDFs, and in doing so give `PageRef`, `BBox`-on-a-page, and `Rendered.barriers` their first honest producers.

**Architecture:** A `MediaProbe` port answering cheap document facts without extracting text; a `PdfHandler` whose `represent()` maps every character to the page it came from and drops a barrier at every page break; a `TextRecognizer` port whose first adapter reuses the existing `VisionModel` to read scanned pages. pypdfium2 behind a `documents` extra, registered by the composition root only when importable.

**Tech Stack:** Python 3.13, pypdfium2 5.13 (Apache-2.0/BSD), pydantic v2, reportlab (test fixtures only), pytest, mypy --strict, ruff, import-linter, bandit, coverage.

**Spec:** `docs/superpowers/specs/2026-08-15-readeverything-the-document-family-design.md`

## Global Constraints

- **The library reads NO environment variables under `src/`.** Enforced by `tests/unit/test_reads_no_environment.py`.
- **Python 3.13, PEP 695 inline type parameters** (`class Foo[T]`, `type X = ...`). A module-level `TypeVar` is a defect.
- **`mypy --strict`, `warn_unused_ignores = true`**, over `src` and `tests`. No new `# type: ignore` without a comment naming why.
- **Import-linter layered contract, `exhaustive = true`.** Outermost first: `composition, testing, agent, pipeline, registry, handlers, adapters, ports, domain`.
- **`readeverything.testing` may import only `ports` and `domain`.**
- **Third-party imports are pinned by an AST test** (`tests/unit/test_dependencies_stay_confined.py`). Any new third-party import must be registered there with a comment naming why.
- **Handlers never import from each other.** Each owns its constants.
- **No handler ever raises from `describe`, `invoke`, or `represent`.** They degrade.
- **Never assert on model text.** Assert structure and locators.
- **Coverage floor 92.** Run the full gate set with `make check`.
- **A law must be able to fail.**

---

## Measured facts (verified by the plan author against pypdfium2 5.13.0)

Use these values; do not re-derive them.

```python
doc = pdfium.PdfDocument(data)          # data: bytes, also accepts a path
len(doc)                                 # page count
doc[i]                                   # PdfPage, 0-indexed
doc.get_metadata_dict()                  # {'Title','Author','Subject','Keywords','Creator','Producer','CreationDate','ModDate'}
page.get_size()                          # (width, height) in POINTS, e.g. (612.0, 792.0)
tp = page.get_textpage()
tp.count_chars()                         # int
tp.get_text_range(index=0, count=-1)     # str — the whole page by default
tp.get_charbox(index, loose=False)       # (left, bottom, right, top) in POINTS
tp.get_text_bounded(left, bottom, right, top)   # str inside a rectangle, POINTS
page.get_objects()                       # iterator of page objects
page.render(scale=1)                     # PdfBitmap; .to_pil() gives a PIL Image
```

**The scanned-versus-blank discriminator, measured on real files:**

| Page | `count_chars()` | `get_text_range()` | `len(list(get_objects()))` |
| --- | --- | --- | --- |
| Scanned (image, no text layer) | 0 | `''` | 1 |
| Genuinely blank | 0 | `''` | 0 |

Through the text layer alone the two are identical. `get_objects()` is what tells them apart, and that difference is the whole of Spec §6.

**Coordinate systems disagree and you must convert.** PDF points have a
**bottom-left** origin. `BBox` is normalised 0–1 and is used elsewhere in this
codebase with a **top-left** origin (`ImageHandler`'s crop maps directly onto
PIL, which is top-left). So:

```python
width, height = page.get_size()
left, bottom, right, top = tp.get_charbox(i)
bbox = BBox(
    page=page_number,            # 1-indexed
    x=left / width,
    y=1.0 - (top / height),      # flip: PDF top is a large y, BBox top is 0
    w=(right - left) / width,
    h=(top - bottom) / height,
)
```

Getting this wrong produces upside-down citations that no test catches unless it
asserts on a known glyph's position — Task 4 has that test.

---

## File Structure

**Created:**

| File | Responsibility |
| --- | --- |
| `src/readeverything/ports/probe_media.py` | `MediaProbe` protocol and the `DocumentFacts` it returns. |
| `src/readeverything/ports/recognition.py` | `TextRecognizer` protocol. |
| `src/readeverything/adapters/pdfium_probe.py` | `MediaProbe` over pypdfium2: page count, sizes, metadata. No text. |
| `src/readeverything/adapters/vision_recognizer.py` | `TextRecognizer` wrapping the existing `VisionModel`. |
| `src/readeverything/handlers/pdf.py` | `PdfHandler`: card, four affordances, `represent`. |
| `tests/integration/conftest.py` (modify) | Generated PDF fixtures — born-digital, scanned, blank, many-page. |

**Modified:**

| File | Change |
| --- | --- |
| `src/readeverything/composition.py` | Register `PdfHandler` when pypdfium2 imports. |
| `src/readeverything/__init__.py` | Export the new ports, adapters, and handler. |
| `pyproject.toml` | `documents` extra; reportlab in `dev`; confinement entries. |
| `tests/unit/test_dependencies_stay_confined.py` | `pypdfium2`, `reportlab`. |

**Layering note.** `handlers/pdf.py` imports `ports/` only, never `adapters/` — the probe and recogniser arrive by constructor injection, exactly as `ImageHandler` receives its `VisionModel`.

---

## Task 1: `MediaProbe` and `DocumentFacts`

**Files:**
- Create: `src/readeverything/ports/probe_media.py`
- Test: `tests/unit/ports/test_probe_media.py`

**Interfaces:**
- Produces:
  - `DocumentFacts` — frozen dataclass: `page_count: int`, `page_sizes: tuple[tuple[float, float], ...]`, `metadata: Mapping[str, str]`.
  - `class MediaProbe(Protocol)` with `async def probe(self, data: bytes) -> DocumentFacts`.

- [ ] **Step 1: Write the failing test**

```python
def test_document_facts_rejects_a_page_count_that_disagrees_with_its_sizes() -> None:
    """The two fields describe the same document and must not contradict.

    A probe that reported 10 pages and 3 sizes would produce a card claiming a
    page count nothing measured — and `read_page(7)` would then fail on a
    document the card said had page 7.
    """
    with pytest.raises(ValueError, match="page_count"):
        DocumentFacts(page_count=10, page_sizes=((612.0, 792.0),), metadata={})


def test_document_facts_rejects_a_non_positive_page_size() -> None:
    with pytest.raises(ValueError):
        DocumentFacts(page_count=1, page_sizes=((0.0, 792.0),), metadata={})


def test_document_facts_carries_no_text() -> None:
    """The card path must stay cheap. A probe that extracted text to answer
    'how many pages' would defeat the progressive-disclosure design this
    library is built on."""
    assert not any(
        "text" in f.name for f in dataclasses.fields(DocumentFacts)
    )
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run --all-extras pytest tests/unit/ports/test_probe_media.py -v
```
Expected: FAIL — the module does not exist.

- [ ] **Step 3: Implement**

```python
"""What can be said about a document without reading its content.

`inspect` must stay cheap: Spec 1's progressive-disclosure design rests on a
card costing no real work, and page count is exactly the fact that shapes an
agent's next move. A probe that extracted text in order to count pages would
defeat that, so this type carries no text and the protocol has no way to
return any.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class DocumentFacts:
    """Cheap facts about a paginated document."""

    page_count: int
    page_sizes: tuple[tuple[float, float], ...]
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.page_count < 0:
            raise ValueError(f"page_count must not be negative, got {self.page_count}")
        if len(self.page_sizes) != self.page_count:
            # These two fields describe one document. Disagreement means a card
            # claiming a page count nothing measured.
            raise ValueError(
                f"page_count {self.page_count} disagrees with {len(self.page_sizes)} page sizes"
            )
        for width, height in self.page_sizes:
            if width <= 0 or height <= 0:
                raise ValueError(f"page size must be positive, got {width}x{height}")


@runtime_checkable
class MediaProbe(Protocol):
    """Cheap structural facts about a document, without extracting content."""

    async def probe(self, data: bytes) -> DocumentFacts: ...
```

- [ ] **Step 4: Run the tests**

```bash
uv run --all-extras pytest tests/unit/ports -q && uv run --all-extras mypy
```
Expected: PASS, clean.

- [ ] **Step 5: Commit**

```bash
uv run --all-extras ruff format src tests
git add src/readeverything/ports/probe_media.py tests/unit/ports/test_probe_media.py
git commit -m "feat(ports): cheap document facts, with no way to return text"
```

---

## Task 2: The pypdfium2 probe adapter

**Files:**
- Create: `src/readeverything/adapters/pdfium_probe.py`
- Test: `tests/unit/adapters/test_pdfium_probe.py`
- Modify: `pyproject.toml`, `tests/unit/test_dependencies_stay_confined.py`

**Interfaces:**
- Consumes: `DocumentFacts`, `MediaProbe` from Task 1.
- Produces: `class PdfiumProbe` implementing `MediaProbe`; module-level helper `open_document(data: bytes) -> pdfium.PdfDocument` used by Task 3's handler so the pdfium import lives in one adapter module.

- [ ] **Step 1: Add the extra and register the import**

In `pyproject.toml`:
```toml
documents = ["pypdfium2>=5.13,<6"]
```
Add `reportlab>=4.2` to the `dev` extra — test fixtures only, never imported by `src/`.

Register `pypdfium2` (for `adapters/pdfium_probe.py`) and `reportlab` (for the test fixture module) in `tests/unit/test_dependencies_stay_confined.py`, each with a comment naming why.

- [ ] **Step 2: Write the failing tests**

```python
async def test_a_three_page_document_reports_three_pages(three_page_pdf: bytes) -> None:
    facts = await PdfiumProbe().probe(three_page_pdf)
    assert facts.page_count == 3
    assert len(facts.page_sizes) == 3


async def test_page_sizes_come_back_in_points(three_page_pdf: bytes) -> None:
    """US Letter is 612x792 points. The handler normalises BBoxes by these
    numbers, so a wrong unit here produces citations off by a factor of 72."""
    facts = await PdfiumProbe().probe(three_page_pdf)
    assert facts.page_sizes[0] == pytest.approx((612.0, 792.0))


async def test_metadata_is_string_to_string(three_page_pdf: bytes) -> None:
    facts = await PdfiumProbe().probe(three_page_pdf)
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in facts.metadata.items())


async def test_a_probe_of_bytes_that_are_not_a_pdf_raises_infrastructure_error() -> None:
    """The probe may raise; the HANDLER is what must never raise. Keeping the
    failure loud here means the handler decides how to degrade, rather than
    receiving a fabricated zero-page document."""
    with pytest.raises(InfrastructureError):
        await PdfiumProbe().probe(b"this is not a pdf at all")


async def test_probing_does_not_extract_text(three_page_pdf: bytes) -> None:
    """The cheapness claim, asserted rather than trusted."""
    facts = await PdfiumProbe().probe(three_page_pdf)
    assert "alpha" not in repr(facts)
```

- [ ] **Step 3: Run to verify they fail**

```bash
uv run --all-extras pytest tests/unit/adapters/test_pdfium_probe.py -v
```
Expected: FAIL — module does not exist.

- [ ] **Step 4: Implement**

```python
"""`MediaProbe` over pypdfium2.

pypdfium2 wraps Google's PDFium under Apache-2.0/BSD and ships a bundled
binary, so this needs no OS dependency and no `Capability` member — it is a
Python import, gated like Pillow.

pymupdf is faster at some of this and is not used: it is AGPL-3.0 and this
library is MIT. That is a licensing conflict, not a performance trade-off, and
it is written here so it is not reopened as one.

pdfium is synchronous and CPU-bound, so every call runs in a thread.
"""

from __future__ import annotations

import asyncio

import pypdfium2 as pdfium

from readeverything.domain.errors import InfrastructureError
from readeverything.ports.probe_media import DocumentFacts


def open_document(data: bytes) -> pdfium.PdfDocument:
    """Open bytes as a PDF, or raise `InfrastructureError`.

    One place opens documents so the pdfium import stays in one adapter module
    and every caller gets the same translated failure.
    """
    try:
        return pdfium.PdfDocument(data)
    except Exception as exc:
        raise InfrastructureError(f"could not open as a PDF: {exc}") from exc


def _probe_sync(data: bytes) -> DocumentFacts:
    document = open_document(data)
    try:
        sizes = tuple(document[i].get_size() for i in range(len(document)))
        raw = document.get_metadata_dict()
        metadata = {str(k): str(v) for k, v in raw.items() if v}
        return DocumentFacts(page_count=len(document), page_sizes=sizes, metadata=metadata)
    finally:
        document.close()


class PdfiumProbe:
    """Page count, page sizes and metadata. No text."""

    async def probe(self, data: bytes) -> DocumentFacts:
        return await asyncio.to_thread(_probe_sync, data)
```

- [ ] **Step 5: Run the tests**

```bash
uv run --all-extras pytest tests/unit/adapters -q && uv run --all-extras mypy && uv run --all-extras lint-imports
```

- [ ] **Step 6: Commit**

```bash
uv run --all-extras ruff format src tests
git add src/readeverything/adapters/pdfium_probe.py tests/unit/adapters/test_pdfium_probe.py pyproject.toml tests/unit/test_dependencies_stay_confined.py
git commit -m "feat(adapters): probe a PDF's shape without reading it"
```

---

## Task 3: PDF fixtures, generated not committed

**Files:**
- Create: `tests/fixtures_pdf.py`
- Modify: `tests/integration/conftest.py`

**Interfaces:**
- Produces, importable from `tests/fixtures_pdf.py`:
  - `born_digital(pages: Sequence[str]) -> bytes`
  - `scanned_like(pages: int = 1) -> bytes` — image content, no text layer
  - `blank(pages: int = 1) -> bytes` — no objects at all
  - `many_pages(count: int) -> bytes`

- [ ] **Step 1: Write the fixture module**

```python
"""PDFs generated at test time rather than committed as binaries.

Committed binary fixtures rot: nobody can read a diff of them, nobody can tell
what a change to one did, and a corrupted byte looks identical to an
intentional edit. Generating them keeps the input to every PDF test readable in
the same file as the test.

reportlab is a dev dependency only. Nothing under `src/` imports it.
"""

from __future__ import annotations

import io
from collections.abc import Sequence

from PIL import Image
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def born_digital(pages: Sequence[str]) -> bytes:
    """A PDF with a real text layer, one string per page."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    for text in pages:
        pdf.drawString(72, 720, text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def scanned_like(pages: int = 1) -> bytes:
    """Image content and NO text layer — what a scan looks like to an extractor.

    Indistinguishable from `blank()` through the text layer: both report zero
    characters and empty text. `page.get_objects()` is what tells them apart,
    and that difference is the whole point of the scanned-PDF handling.
    """
    image = Image.new("RGB", (400, 200), (30, 30, 30))
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    for _ in range(pages):
        pdf.drawImage(ImageReader(image), 72, 500, width=300, height=150)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def blank(pages: int = 1) -> bytes:
    """Genuinely empty pages: no text, no objects."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    for _ in range(pages):
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def many_pages(count: int) -> bytes:
    """For asserting the locator map's size and the barrier count."""
    return born_digital([f"This is page {i + 1}." for i in range(count)])
```

- [ ] **Step 2: Assert the fixtures are actually distinguishable**

This test guards the fixtures themselves — if `scanned_like` ever stops
differing from `blank`, every scanned-PDF test silently becomes a blank-page
test.

```python
def test_the_scanned_and_blank_fixtures_differ_only_in_page_objects() -> None:
    """Both report zero characters. That is the point: the text layer cannot
    tell them apart, so the handler must use something else."""
    scan = pdfium.PdfDocument(scanned_like())
    empty = pdfium.PdfDocument(blank())

    assert scan[0].get_textpage().count_chars() == 0
    assert empty[0].get_textpage().count_chars() == 0

    assert len(list(scan[0].get_objects())) > 0
    assert len(list(empty[0].get_objects())) == 0
```

- [ ] **Step 3: Add integration fixtures**

In `tests/integration/conftest.py`, add a `documents_root` fixture writing
`report.pdf` (born-digital, 3 pages), `scan.pdf`, and `blank.pdf` into a
`tmp_path`.

- [ ] **Step 4: Run and commit**

```bash
uv run --all-extras pytest tests/unit -q -k fixture
uv run --all-extras ruff format tests
git add tests/fixtures_pdf.py tests/integration/conftest.py tests/unit/test_pdf_fixtures.py
git commit -m "test: generate PDF fixtures rather than committing binaries"
```

---

## Task 4: `PdfHandler` — the card, and `represent()` with real barriers

The centrepiece. This is where `PageRef` and `barriers` get their first producers.

**Files:**
- Create: `src/readeverything/handlers/pdf.py`
- Test: `tests/unit/handlers/test_pdf_handler.py`

**Interfaces:**
- Consumes: `MediaProbe`, `DocumentFacts`, `SourceReader`, `open_document` from Task 2.
- Produces: `PdfHandler(*, source: SourceReader, probe: MediaProbe, recognizer: TextRecognizer | None = None)`; ClassVars `mime_patterns = ("application/pdf",)`, `priority = 0`, `handler_id = "pdf"`, `handler_version = 1`.

- [ ] **Step 1: Write the failing tests — start with the barrier and page-mapping ones**

```python
async def test_every_character_resolves_to_the_page_it_came_from() -> None:
    """The property the whole handler exists to provide.

    Asserted across a page boundary, not just inside page one — an off-by-one
    in the segment starts passes any test that only samples the middle.
    """
    handler = _handler(born_digital(["Alpha.", "Bravo.", "Charlie."]))
    rendered = await handler.represent(_ref(), Budget(max_chars=None))

    for page_number, word in ((1, "Alpha"), (2, "Bravo"), (3, "Charlie")):
        offset = rendered.text.index(word)
        assert rendered.locator_map.resolve(offset) == PageRef(page_number)
        # and the LAST character of that word, which is where an off-by-one shows
        assert rendered.locator_map.resolve(offset + len(word) - 1) == PageRef(page_number)


async def test_barriers_sit_at_page_breaks() -> None:
    """`Rendered.barriers` has never had a producer until now.

    A barrier marks a point a chunker must not casually split across, because
    text either side belongs to different pages. So there is one barrier per
    page break — page count minus one — and each sits exactly where a new
    page's first character begins.
    """
    handler = _handler(born_digital(["Alpha.", "Bravo.", "Charlie."]))
    rendered = await handler.represent(_ref(), Budget(max_chars=None))

    assert len(rendered.barriers) == 2
    for barrier in rendered.barriers:
        # the character at a barrier is the first of its page, so the character
        # before it belongs to the previous page
        assert rendered.locator_map.resolve(barrier) != rendered.locator_map.resolve(barrier - 1)


async def test_a_chunk_spanning_a_page_break_cites_both_pages() -> None:
    """Why `resolve_span` returns a tuple, demonstrated on a real document."""
    handler = _handler(born_digital(["Alpha.", "Bravo."]))
    rendered = await handler.represent(_ref(), Budget(max_chars=None))
    barrier = rendered.barriers[0]

    pages = rendered.locator_map.resolve_span(CharSpan(barrier - 2, barrier + 2))
    assert pages == (PageRef(1), PageRef(2))


async def test_the_card_reports_the_page_count_without_extracting_text() -> None:
    handler = _handler(many_pages(12))
    card = await handler.describe(_ref())
    assert card.facts["page_count"] == "12"
    assert "This is page" not in (card.excerpt or "")


async def test_a_four_hundred_page_map_has_four_hundred_segments() -> None:
    """Page granularity, not character granularity — spec §5.2.

    A per-character map over a long document would be enormous and would have
    to fabricate a rectangle for every space and newline.
    """
    handler = _handler(many_pages(400))
    rendered = await handler.represent(_ref(), Budget(max_chars=None))
    assert len(rendered.locator_map.segments) == 400
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run --all-extras pytest tests/unit/handlers/test_pdf_handler.py -v
```
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the card and `represent`**

Key requirements, all load-bearing:

- One `LocatorSegment` per page, `CharSpan(page_start, page_end)` → `PageRef(i + 1)`. `PageRef` is 1-indexed; pdfium pages are 0-indexed. Off-by-one here is the single most likely defect in this task.
- `LocatorMap` requires total, gapless, zero-start coverage. Pages that extract to empty text would produce a zero-width `CharSpan`, which raises. **A page contributing no characters must still contribute at least one** — join pages with a separator (`"\n"`) so every page owns at least its separator character, and say so in a comment. Otherwise an empty page in the middle of a document breaks the map.
- Barriers are the character offsets where pages 2..N begin.
- `represent` applies `Budget.max_chars` by truncating, and reports `kept {len(text)} of {len(full)} characters` — matching all three existing handlers. Truncation must also drop barriers beyond the kept text, or `Rendered.__post_init__` rejects out-of-range barriers.
- The handler never raises. An unopenable PDF returns a `Rendered` describing the failure with a `Degradation`, exactly as `ImageHandler` does for an undecodable image.

- [ ] **Step 4: Run the tests**

```bash
uv run --all-extras pytest tests/unit/handlers -q
```

- [ ] **Step 5: Commit**

```bash
uv run --all-extras ruff format src tests
git add src/readeverything/handlers/pdf.py tests/unit/handlers/test_pdf_handler.py
git commit -m "feat(handlers): read a PDF, and give PageRef and barriers their first producer"
```

---

## Task 5: The scanned PDF, told apart from an empty one

**Files:**
- Modify: `src/readeverything/handlers/pdf.py`
- Test: `tests/unit/handlers/test_pdf_handler.py`

**Interfaces:** no signature change.

- [ ] **Step 1: Write the failing tests**

```python
async def test_a_scanned_page_is_never_reported_as_empty() -> None:
    """The project's recurring defect at a new site.

    "This document is empty" is a claim about the document. What was observed
    is "the text layer is empty" — for a scan those are different, and the
    difference is whether an agent concludes a contract says nothing or knows
    to look harder.
    """
    handler = _handler(scanned_like())
    rendered = await handler.represent(_ref(), Budget(max_chars=None))

    assert "empty" not in rendered.text.lower()
    assert any("scan" in d.what.lower() or "image" in d.what.lower()
               for d in rendered.degradations)


async def test_a_blank_page_and_a_scanned_page_do_not_produce_the_same_text() -> None:
    """Both have zero characters in the text layer. If the handler cannot tell
    them apart, it is guessing about one of them."""
    scanned = await _handler(scanned_like()).represent(_ref(), Budget(max_chars=None))
    empty = await _handler(blank()).represent(_ref(), Budget(max_chars=None))
    assert scanned.text != empty.text


async def test_a_scan_without_a_vision_capability_says_so_and_does_not_ocr() -> None:
    """Negotiation working: no recogniser means no OCR and an honest report."""
    handler = _handler(scanned_like(), recognizer=None)
    rendered = await handler.represent(_ref(), Budget(max_chars=None))
    assert any("no text could be extracted" in d.detail.lower()
               or "not attempted" in d.detail.lower() for d in rendered.degradations)
```

- [ ] **Step 2: Run to verify they fail**

Expected: FAIL — the handler currently treats a scan as an empty page.

- [ ] **Step 3: Implement the three-state distinction**

```python
def _page_state(page: pdfium.PdfPage, text: str) -> _PageState:
    """Extracted, scanned, or genuinely blank.

    Measured on real files: a scanned page and a blank page are IDENTICAL
    through the text layer — both report zero characters and empty text. The
    page's object list is what distinguishes them, a scan carrying at least one
    image object where a blank page carries none.

    Without this, a scanned contract is reported as an empty document, which is
    a false claim about the file rather than an honest report about the text
    layer.
    """
    if text.strip():
        return _PageState.EXTRACTED
    if any(True for _ in page.get_objects()):
        return _PageState.SCANNED
    return _PageState.BLANK
```

Each state produces different placeholder text and a different `Degradation`.
A scanned page says its text is in images; a blank page says it is blank.

- [ ] **Step 4: Run and commit**

```bash
uv run --all-extras pytest tests/unit/handlers -q
uv run --all-extras ruff format src tests
git add src/readeverything/handlers/pdf.py tests/unit/handlers/test_pdf_handler.py
git commit -m "fix(handlers): a scanned page is not an empty one"
```

---

## Task 6: Affordances — `read_page`, `page_region`, `page_image`

**Files:**
- Modify: `src/readeverything/handlers/pdf.py`
- Test: `tests/unit/handlers/test_pdf_handler.py`

**Interfaces:**
- Produces params models: `ReadPageParams(page: int)`, `PageRegionParams(page: int, x: float, y: float, w: float, h: float)`, `PageImageParams(page: int, dpi: int = 150)`.

- [ ] **Step 1: Write the failing tests**

```python
async def test_read_page_returns_that_page_located_at_that_page() -> None:
    handler = _handler(born_digital(["Alpha.", "Bravo.", "Charlie."]))
    rendition = await handler.invoke(_ref(), "read_page", ReadPageParams(page=2))
    assert "Bravo" in rendition.content.text
    assert rendition.locator == PageRef(2)


async def test_asking_for_a_page_past_the_end_degrades_rather_than_raising() -> None:
    """The handler never raises. An agent that guesses a page number gets a
    result it can read and correct, not an exception."""
    handler = _handler(born_digital(["Only page."]))
    rendition = await handler.invoke(_ref(), "read_page", ReadPageParams(page=99))
    assert rendition.degraded


async def test_page_region_bbox_uses_a_top_left_origin() -> None:
    """PDF points are bottom-left origin; BBox is top-left, as used by the
    image handler's crop. A missing flip yields upside-down citations that
    every test passes unless one asserts a known glyph's position.

    The fixture draws its text near the TOP of the page, so the top half must
    contain it and the bottom half must not.
    """
    handler = _handler(born_digital(["Alpha at the top."]))

    top = await handler.invoke(
        _ref(), "page_region", PageRegionParams(page=1, x=0.0, y=0.0, w=1.0, h=0.5)
    )
    bottom = await handler.invoke(
        _ref(), "page_region", PageRegionParams(page=1, x=0.0, y=0.5, w=1.0, h=0.5)
    )
    assert "Alpha" in top.content.text
    assert "Alpha" not in bottom.content.text


async def test_page_region_returns_a_bbox_carrying_its_page() -> None:
    """`BBox.page` has never been anything but None. This is its first real
    value."""
    handler = _handler(born_digital(["Alpha.", "Bravo."]))
    rendition = await handler.invoke(
        _ref(), "page_region", PageRegionParams(page=2, x=0.0, y=0.0, w=1.0, h=1.0)
    )
    assert isinstance(rendition.locator, BBox)
    assert rendition.locator.page == 2


async def test_page_image_returns_image_content_a_vision_tool_can_read() -> None:
    """A diagram on page 12 becomes describable through the existing tool pack
    without this handler knowing anything about vision."""
    handler = _handler(born_digital(["Alpha."]))
    rendition = await handler.invoke(_ref(), "page_image", PageImageParams(page=1, dpi=72))
    assert isinstance(rendition.content, ImageContent)
    assert rendition.content.mime == "image/png"
    assert rendition.content.data.startswith(b"\x89PNG")
```

- [ ] **Step 2: Run to verify they fail, then implement**

The Y-flip formula is in the Measured Facts section above. Use it verbatim.

`page_image` renders with `page.render(scale=dpi / 72)` and saves the resulting
bitmap as PNG through Pillow. Pillow is already a dependency of the `images`
extra — if it is absent, `page_image` degrades rather than raising, and the
affordance still registers because the rest of the handler works.

- [ ] **Step 3: Run and commit**

```bash
uv run --all-extras pytest tests/unit/handlers -q
uv run --all-extras ruff format src tests
git add src/readeverything/handlers/pdf.py tests/unit/handlers/test_pdf_handler.py
git commit -m "feat(handlers): read a page, a region of a page, or a page as an image"
```

---

## Task 7: `TextRecognizer` and OCR over a rendered page

**Files:**
- Create: `src/readeverything/ports/recognition.py`, `src/readeverything/adapters/vision_recognizer.py`
- Modify: `src/readeverything/handlers/pdf.py`
- Test: `tests/unit/adapters/test_vision_recognizer.py`, `tests/unit/handlers/test_pdf_handler.py`

**Interfaces:**
- Produces:
  - `class TextRecognizer(Protocol)`: `model_id: str`; `async def recognize(self, image: bytes, mime: str) -> str`.
  - `class VisionTextRecognizer` wrapping a `VisionModel`.
  - `ocr_page` affordance, `DetailLevel.DEEP`, requiring `Capability.VISION`.

- [ ] **Step 1: Write the failing tests**

```python
async def test_the_recognizer_carries_the_model_id_it_recognises_with() -> None:
    """It feeds the capability fingerprint, so OCR artifacts invalidate when
    the model changes while extracted text — which does not depend on a model
    — stays cached."""
    recognizer = VisionTextRecognizer(vision=FakeVision())
    assert recognizer.model_id == FakeVision().model_id


async def test_ocr_output_is_marked_as_a_model_reading_not_extraction() -> None:
    """OCR is a model's reading of an image, not the document's own bytes. A
    consumer indexing it is entitled to know which it has — the same
    distinction drawn for synthesized text."""
    handler = _handler(scanned_like(), recognizer=VisionTextRecognizer(vision=FakeVision()))
    rendition = await handler.invoke(_ref(), "ocr_page", OcrPageParams(page=1))
    assert rendition.degraded


async def test_ocr_is_not_offered_without_a_vision_capability() -> None:
    """Negotiation, not a runtime apology: the affordance must not appear."""
    registry = MimeTypeRegistry(
        handlers=[_pdf_handler(recognizer=None)], capabilities=CapabilitySet.empty()
    )
    names = {a.name for a in registry.available_affordances(registry.handlers[0])}
    assert "ocr_page" not in names
```

- [ ] **Step 2: Implement, run, commit**

`VisionTextRecognizer.recognize` renders nothing itself — the handler renders
the page and hands it bytes. The prompt asks for a verbatim transcription and
says to reply with a known marker if there is no text, matching how
`ImageHandler`'s OCR prompt already works.

```bash
uv run --all-extras ruff format src tests
git add src/readeverything/ports/recognition.py src/readeverything/adapters/vision_recognizer.py src/readeverything/handlers/pdf.py tests/unit/
git commit -m "feat: read a scanned page by asking a model to look at it"
```

---

## Task 8: Wire into the composition root and the front door

**Files:**
- Modify: `src/readeverything/composition.py`, `src/readeverything/__init__.py`
- Test: `tests/unit/test_composition.py`, `tests/integration/test_documents.py`

- [ ] **Step 1: Write the failing tests**

```python
async def test_a_pdf_is_handled_when_the_documents_extra_is_present(documents_root) -> None:
    perception = await build_perception(documents_root, probe_binaries=False)
    card = await perception.inspect("report.pdf")
    assert card.kind == "document"
    assert card.facts["page_count"] == "3"


async def test_a_base_install_without_pypdfium2_falls_back_to_binary(
    documents_root, monkeypatch
) -> None:
    """Narrower, not broken — the same contract Pillow already has."""
    monkeypatch.setitem(sys.modules, "pypdfium2", None)
    perception = await build_perception(documents_root, probe_binaries=False)
    card = await perception.inspect("report.pdf")
    assert card.kind == "binary"


async def test_the_agent_can_ask_a_pdf_for_a_page(documents_root) -> None:
    """End to end through the tools an agent actually holds."""
    perception = await build_perception(documents_root, probe_binaries=False)
    tools = {t.name: t for t in build_tools(perception)}
    result = await tools["invoke_affordance"].ainvoke(
        {"uri": "report.pdf", "affordance": "read_page", "params": {"page": 2}}
    )
    assert "PageRef(page=2)" in result
```

- [ ] **Step 2: Implement**

Add `_optional_pdf_handler(source, probe, recognizer)` beside
`_optional_image_handler`, following it exactly: `try: import` / `except
ImportError: return []`. Register **before** `BinaryHandler`, which must stay
last.

Build the recogniser from `vision` when one was supplied, so OCR negotiates on
the capability the caller already provided.

Export from the front door, keeping `_LAZY` and the `TYPE_CHECKING` block sorted
and in sync: `DocumentFacts`, `MediaProbe`, `PdfHandler`, `PdfiumProbe`,
`TextRecognizer`, `VisionTextRecognizer`.

- [ ] **Step 3: Run the full gates and commit**

```bash
make check
uv run --all-extras ruff format src tests
git add src/readeverything/composition.py src/readeverything/__init__.py tests/
git commit -m "feat(composition): register the PDF handler when pypdfium2 is importable"
```

---

## Task 9: Live validation against a real model

**Files:**
- Create: `tests/live/test_pdf_ocr.py`

**Interfaces:** consumes the live-test config in `tests/live/conftest.py` — `DEFAULT_BASE_URL = "http://192.168.1.14:8080/v1/"`, `DEFAULT_MODEL = "qwen3.8-27b-mtp"`.

**Note for the executor:** these are `live`-marked and deselected by default. Do not run them without telling the human partner first — they need to stop other inference on that server.

- [ ] **Step 1: Write the tests**

```python
@pytest.mark.live
async def test_a_real_model_reads_a_scanned_page(live_vision) -> None:
    """The whole scanned-PDF path against a real model rather than a fake.

    Asserts structure, never the model's exact words: that text came back, that
    it is not an echo of the prompt, and that the rendition is marked as a
    model reading rather than an extraction.
    """
    ...


@pytest.mark.live
async def test_ocr_artifacts_invalidate_when_the_model_changes(live_vision) -> None:
    """Swapping the model must produce a different cache key, or an index
    silently mixes two models' readings of the same page."""
    ...
```

- [ ] **Step 2: Ask the human partner before running, then run and commit**

---

## Plan Self-Review

**Spec coverage.** §3 library choice → Tasks 2, 3. §4 `MediaProbe` → Tasks 1, 2. §5 page mapping and barriers → Task 4. §6 scanned PDFs and `TextRecognizer` → Tasks 5, 7. §7 affordances → Tasks 6, 7. §8 caching → falls out of `handler_version` set in Task 4. §10 acceptance 1–8 → Tasks 4, 5, 6, 8, 9.

**Ordering.** The port comes first so the adapter has something to satisfy; fixtures come before the handler that needs them; the card and `represent` come before affordances because the affordances reuse their page-opening code; OCR comes after the scanned-page detection that decides when to call it; composition last.

**Known risks a reviewer should hold me to.**
- Task 4's 1-indexed `PageRef` against 0-indexed pdfium pages is the likeliest defect in the plan. The test asserts the last character of each page's word, which is where an off-by-one shows.
- Task 4's separator requirement (every page owning at least one character) is stated but not shown as code. If an implementer skips it, a document with an empty middle page raises from `CharSpan`.
- Task 6's Y-flip is given as a formula; the test that catches a missing flip asserts on a fixture whose text is near the top of the page, so the fixture and the test are coupled — if the fixture's drawing position changes, that test stops meaning anything.
