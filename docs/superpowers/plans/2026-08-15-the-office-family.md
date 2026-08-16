# The Office Family Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read `.docx`, `.pptx`, `.xlsx` and their ODF equivalents as documents rather than as hex dumps of a zip file.

**Architecture:** Three handlers (`office_word`, `office_slides`, `office_sheets`) over two thin, I/O-free adapters (`ooxml.py` for zip-part access and container sniffing, `odf.py` for lxml text extraction). Detection gains a content-based refinement that classifies a zip by the part names visible in its first 4096 bytes. `CellRange` joins the locator vocabulary because no existing locator addresses a cell. Everything new sits behind an `office` extra and is guarded exactly as `handlers/pdf.py` guards pypdfium2.

**Tech Stack:** Python 3.13, python-docx 1.2, python-pptx 1.0, openpyxl 3.1, lxml 5 (all `office` extra), pydantic v2, pytest, mypy --strict, ruff, import-linter, coverage.

**Spec:** `docs/superpowers/specs/2026-08-15-readeverything-the-office-family-design.md`

## Global Constraints

- **The library reads NO environment variables under `src/`.** Enforced by `tests/unit/test_reads_no_environment.py`. Every input is an explicit argument.
- **Python 3.13, PEP 695 inline type parameters.** A module-level `TypeVar` is a defect.
- **`mypy --strict`, `warn_unused_ignores = true`**, over `src` and `tests`. No new `# type: ignore` without a comment naming why.
- **Import-linter layered contract, `exhaustive = true`.** Outermost first: `composition, testing, agent, pipeline, registry, handlers, adapters, ports, domain`. `handlers` may import `adapters`; `adapters` may not import `handlers`. Modules *within* one layer may import each other (`clip_langchain` already imports `vision_langchain`).
- **Third-party imports are pinned by an AST test** (`tests/unit/test_dependencies_stay_confined.py`). Any new third-party import must be registered there with a comment naming why.
- **New dependencies go in the `office` extra ONLY**, never in core `dependencies`. Every import guarded so the library works without the extra.
- **No handler ever raises from `describe`, `invoke`, or `represent`.** They degrade.
- **Never assert on model text.** Assert structure and locators.
- **Coverage floor 92.**
- **Do NOT run the full suite or `make check`** — that is CI's job. Run only the tests named in each task, plus `mypy`/`ruff` on touched files.
- **Stay in lane.** Edits to `composition.py` and `adapters/detection.py` must be minimal and purely additive: Spec 8 is being built concurrently in another worktree and touches both.
- **Do not build anything archive-aware.** No `adapters/nested_source.py`, no `handlers/archive.py`, no archive descent. These handlers read bytes through `SourceReader` and cannot tell disk from tarball.

---

## Measured facts (verified by the plan author against the real libraries)

Use these values; do not re-derive them. Every one was run.

### The detection facts, which contradict the spec's §3 sketch

Spec §3 says to "peek at the container's `[Content_Types].xml`". **Two measurements say that cannot work as written**, and this plan therefore refines §3 rather than following it literally:

1. **`openpyxl` writes `[Content_Types].xml` LAST.** Measured entry order of an openpyxl workbook:
   `docProps/app.xml, docProps/core.xml, xl/theme/theme1.xml, xl/worksheets/sheet1.xml, …, xl/workbook.xml, xl/_rels/workbook.xml.rels, [Content_Types].xml`.
   Detection only ever sees `head = read_range(uri, 0, 4096)` (`pipeline/perception.py:34,68`). A zip's central directory is at the *end* of the file, so `zipfile` cannot be used on a head at all, and `[Content_Types].xml` is simply unreachable for any openpyxl-written `.xlsx`. python-docx and python-pptx *do* put it first, so a `[Content_Types].xml`-only rule would detect two families of three and silently fail the third.

2. **The `[Content_Types].xml` override is not the document mimetype anyway.** Measured overrides:
   `/word/document.xml → …wordprocessingml.document.main+xml`, `/ppt/presentation.xml → …presentationml.presentation.main+xml`, `/xl/workbook.xml → …spreadsheetml.sheet.main+xml`. The `.main+xml` suffix would have to be stripped.

3. **puremagic's answer for OOXML is worthless and is already discarded.** For a `.docx`, a `.pptx` *and* an `.xlsx`, `magic_string(head)` returns the identical list — `wordprocessingml.document`, `presentationml.presentation`, `spreadsheetml.sheet`, `ms-excel.sheet.binary` — every one at **confidence 0.40**. That is below the existing `_SIGNATURE_FLOOR = 0.5`, so today every one is `continue`d and detection falls through to the filename. Taking puremagic's top match would label every `.xlsx` a Word document.

**Therefore the refinement classifies by zip PART NAMES, walked from the local file headers inside the head.** Measured reachable-within-4096 part names:

| File | First part names visible in head | Discriminator hit |
| --- | --- | --- |
| `.docx` (python-docx) | `[Content_Types].xml`, … | `word/…` |
| `.pptx` (python-pptx) | `[Content_Types].xml`, … | `ppt/…` |
| `.xlsx` (openpyxl) | `docProps/app.xml`, `docProps/core.xml`, `xl/theme/theme1.xml` (offset 477), … | `xl/…` |
| `.odt`/`.odp`/`.ods` | `mimetype` (first entry, **stored uncompressed**, at offset 38) | the entry's own bytes |

The zip local file header layout, used to walk part names without inflating anything:

```
offset  0  4  signature   b"PK\x03\x04"
offset  6  2  flags
offset  8  2  compression method
offset 18  4  compressed size
offset 22  4  uncompressed size
offset 26  2  file name length
offset 28  2  extra field length
offset 30  n  file name
           e  extra field
              compressed data (compressed size bytes)
```

So the next header sits at `offset + 30 + nlen + elen + csize`.

### python-docx 1.2.0

```python
doc = docx.Document(io.BytesIO(data))
doc.element.body.iterchildren()      # document order; qn("w:p") and qn("w:tbl") interleaved
Paragraph(child, doc).style.name     # 'Heading 1', 'Heading 2', 'Normal'
Paragraph(child, doc).text
Table(child, doc).rows               # row.cells -> cell.text
doc.comments                         # iterable of Comment
comment.comment_id, comment.author, comment.text     # 0, 'Reviewer', 'A reviewer note'
doc.add_comment(runs=p.runs, text=..., author=..., initials=...)   # fixture side only
doc.core_properties.title / .author
```

`doc.paragraphs` does **not** include paragraphs inside tables, and does not tell you where a table sat. The `body.iterchildren()` walk above is the only way to get true document order, and document order is what `represent` promises.

### python-pptx 1.0

```python
prs = pptx.Presentation(io.BytesIO(data))
prs.slides                                   # ordered
slide.shapes.title                           # may be None
slide.shapes                                 # ordered; shape.has_text_frame, shape.text_frame.text
shape.shape_type == MSO_SHAPE_TYPE.PICTURE   # int 13
shape.image.blob / .content_type / .ext      # b'\x89PNG…', 'image/png', 'png'
slide.has_notes_slide                        # False when no notes were ever added
slide.notes_slide.notes_text_frame.text
```

`slide.notes_slide` **creates** a notes slide as a side effect if none exists — always guard with `has_notes_slide` first, or `describe` mutates the parsed package and the notes-count fact becomes whatever order the tests ran in.

### openpyxl 3.1

```python
wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True)
wb.sheetnames                        # ['Data', 'Notes']
ws.max_row, ws.max_column            # 50, 3
ws.calculate_dimension()             # 'A1:C50'    -- NOTE: ReadOnlyWorksheet has NO `.dimensions`
wb.close()

openpyxl.load_workbook(..., data_only=True)   # cached values; formula cells -> the cached <v>
openpyxl.load_workbook(...)                   # formulas as strings, '=B2*2'
```

**The formula/value trap.** openpyxl never computes formulas and writes **no cached values**. A workbook openpyxl produced therefore yields `None` for *every* formula cell under `data_only=True`. Measured:

| load | row 2 of the fixture |
| --- | --- |
| default | `['r1', 1, '=B2*2']` |
| `data_only=True`, workbook as openpyxl wrote it | `['r1', 1, None]` |
| `data_only=True`, after injecting `<v>2</v>` | `['r1', 1, 2]` |

Two consequences, both load-bearing:
- The **fixture** must inject cached values into the sheet XML to exercise the value path honestly (Task 4 does this).
- The **handler** must report a `Degradation` when a formula cell has a formula but no cached value, rather than rendering a blank. A blank cell and an uncomputed formula are different facts, and this is the same "scanned page is not an empty page" distinction Spec 4 already drew.

### ODF, hand-built

A minimal, valid, sniffable ODF package (verified round-trip through `zipfile` + `lxml`):

```
mimetype           <- MUST be the first entry and MUST be ZIP_STORED
content.xml
META-INF/manifest.xml
```

`content.xml` for text uses `text:h` (with `text:outline-level`) for headings and `text:p` for paragraphs, under `office:document-content/office:body/office:text`.

---

## File Structure

**Created:**

| File | Responsibility |
| --- | --- |
| `src/readeverything/adapters/ooxml.py` | Zip part access with no I/O: walk local headers in a head, classify an office container, read one part from full bytes. |
| `src/readeverything/adapters/odf.py` | ODF text extraction with lxml: paragraphs/headings, slides, sheets. |
| `src/readeverything/handlers/office_word.py` | `OfficeWordHandler`: heading outline, document-order body, tables, comments. |
| `src/readeverything/handlers/office_slides.py` | `OfficeSlidesHandler`: slide outline, per-slide text plus labelled notes, embedded media. |
| `src/readeverything/handlers/office_sheets.py` | `OfficeSheetsHandler`: sheet outline, delimited cells, formulas versus values. |
| `tests/fixtures_office.py` | All six document types generated at test time. No committed binaries. |
| `tests/unit/handlers/test_office_word.py` | Word handler unit tests + compliance. |
| `tests/unit/handlers/test_office_slides.py` | Slides handler unit tests + compliance. |
| `tests/unit/handlers/test_office_sheets.py` | Sheets handler unit tests + compliance. |
| `tests/unit/adapters/test_ooxml.py` | Part-name walking and container classification. |
| `tests/unit/adapters/test_odf.py` | ODF extraction. |
| `tests/unit/test_office_fixtures.py` | The fixtures are what they claim to be. |
| `tests/integration/test_office.py` | The §1.1 acceptance scenario. |
| `tests/live/test_office_vision.py` | `describe_slide_image` against a real model, `live`-marked. |

**Modified (keep every one minimal and additive — Spec 8 shares two of them):**

| File | Change |
| --- | --- |
| `src/readeverything/domain/locators.py` | Add `CellRange`; add it to the `Locator` union. |
| `src/readeverything/adapters/detection.py` | One `if` block calling `office_mimetype`, plus its import. |
| `src/readeverything/composition.py` | One `_optional_office_handlers` function, one splat in the handler list. |
| `pyproject.toml` | `office` extra; three mypy overrides. |
| `tests/unit/test_dependencies_stay_confined.py` | `docx`, `pptx`, `openpyxl`, `lxml` homes. |
| `tests/unit/domain/test_locators.py` | `CellRange` validation. |
| `tests/unit/adapters/test_detection.py` | A `.docx` is not `application/zip`. |
| `tests/integration/conftest.py` | An `office_root` fixture. |
| `README.md` | Three table rows, the `office` extra, and delete the "no handlers yet" sentence. |

**Layering note.** The handlers import `adapters/ooxml.py` and `adapters/odf.py` directly. That is legal under the layered contract (`handlers` sits above `adapters`) and does not violate `ports/handler.py`'s rule, which forbids a handler *touching a filesystem, shelling out, or reading the environment*. Both adapters are pure `bytes -> structure` functions: no I/O, no subprocess, no environment. Inventing a port for "read a zip member" would be ceremony around a pure function. Bytes still arrive only through the injected `SourceReader`.

---

## Task 1: `CellRange`, the sixth locator

**Files:**
- Modify: `src/readeverything/domain/locators.py`
- Test: `tests/unit/domain/test_locators.py`

**Interfaces:**
- Produces: `CellRange(sheet: str, row: int, col: int, rows: int = 1, cols: int = 1)`, frozen, slots, added to `type Locator`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/domain/test_locators.py`:

```python
def test_a_cell_range_rejects_a_negative_origin() -> None:
    """0-indexed internally, so -1 is not "one before A1"; it is nothing."""
    with pytest.raises(ValueError, match="row"):
        CellRange(sheet="Data", row=-1, col=0)
    with pytest.raises(ValueError, match="col"):
        CellRange(sheet="Data", row=0, col=-1)


def test_a_cell_range_rejects_a_non_positive_extent() -> None:
    """A zero-row block addresses no cell, which is not a citation."""
    with pytest.raises(ValueError, match="rows"):
        CellRange(sheet="Data", row=0, col=0, rows=0)
    with pytest.raises(ValueError, match="cols"):
        CellRange(sheet="Data", row=0, col=0, cols=0)


def test_a_cell_range_rejects_a_blank_sheet_name() -> None:
    """A sheet name is how the citation is resolved back. Without one the
    locator points at a workbook rather than at a place in it."""
    with pytest.raises(ValueError, match="sheet"):
        CellRange(sheet="   ", row=0, col=0)


def test_a_single_cell_is_the_default_extent() -> None:
    cell = CellRange(sheet="Data", row=3, col=2)
    assert (cell.rows, cell.cols) == (1, 1)


def test_a_cell_range_is_hashable_and_compares_by_value() -> None:
    """Every other locator is; `Segment` and `LocatorSegment` hold them in
    frozen dataclasses and tests compare them directly."""
    assert CellRange(sheet="Data", row=0, col=0) == CellRange(sheet="Data", row=0, col=0)
    assert len({CellRange(sheet="D", row=0, col=0), CellRange(sheet="D", row=0, col=0)}) == 1


def test_cell_range_is_in_the_locator_union() -> None:
    """A locator the union does not name cannot be returned by a handler
    without mypy --strict rejecting it at every call site."""
    assert CellRange in get_args(Locator.__value__)
```

Add to that file's imports: `from typing import get_args` and `CellRange`, `Locator` from `readeverything.domain.locators`.

- [ ] **Step 2: Run to verify it fails**

```bash
uv run --all-extras pytest tests/unit/domain/test_locators.py -x -q
```
Expected: FAIL — `ImportError: cannot import name 'CellRange'`.

- [ ] **Step 3: Implement**

In `src/readeverything/domain/locators.py`, after `BBox` and before the `Locator` union:

```python
@dataclass(frozen=True, slots=True)
class CellRange:
    """A rectangular block of cells in a named sheet, 0-indexed internally.

    None of the five older locators addresses a cell. `CharSpan` into rendered
    text is not it: the rendering is an artifact of the sheets handler's own
    delimiter choice, so a citation into it stops meaning anything the moment
    that delimiter changes. A cell's address does not depend on how the cell
    was printed.

    0-indexed here and A1 outside. A1 notation is presentation — it is
    1-indexed, its columns are base-26 letters, and it belongs to the handler
    that talks to users about spreadsheets. The domain counts from zero like
    everything else it addresses.
    """

    sheet: str
    row: int
    col: int
    rows: int = 1
    cols: int = 1

    def __post_init__(self) -> None:
        if not self.sheet.strip():
            raise ValueError("sheet must not be blank")
        if self.row < 0:
            raise ValueError(f"row must not be negative, got {self.row}")
        if self.col < 0:
            raise ValueError(f"col must not be negative, got {self.col}")
        if self.rows < 1:
            raise ValueError(f"rows must be at least 1, got {self.rows}")
        if self.cols < 1:
            raise ValueError(f"cols must be at least 1, got {self.cols}")
```

And extend the union on the last line:

```python
type Locator = TimeSpan | PageRef | BBox | CharSpan | ByteRange | CellRange
```

- [ ] **Step 4: Run the tests**

```bash
uv run --all-extras pytest tests/unit/domain/test_locators.py -x -q
uv run --all-extras mypy src/readeverything/domain/locators.py
```
Expected: PASS, clean.

- [ ] **Step 5: Commit**

```bash
uv run --all-extras ruff format src/readeverything/domain/locators.py tests/unit/domain/test_locators.py
uv run --all-extras ruff check src/readeverything/domain/locators.py tests/unit/domain/test_locators.py
git add src/readeverything/domain/locators.py tests/unit/domain/test_locators.py
git commit -m "feat(domain): address a cell, because no existing locator can"
```

---

## Task 2: The `office` extra, and `adapters/ooxml.py`

**Files:**
- Create: `src/readeverything/adapters/ooxml.py`
- Test: `tests/unit/adapters/test_ooxml.py`
- Modify: `pyproject.toml`, `tests/unit/test_dependencies_stay_confined.py`

**Interfaces:**
- Produces:
  - `WORD_MIME`, `SLIDES_MIME`, `SHEETS_MIME`, `ODF_TEXT_MIME`, `ODF_SLIDES_MIME`, `ODF_SHEETS_MIME` — `str` constants.
  - `OFFICE_MIMETYPES: frozenset[str]` — all six.
  - `zip_part_names(head: bytes) -> tuple[str, ...]`
  - `office_mimetype(head: bytes) -> str | None`
  - `read_part(data: bytes, name: str) -> bytes | None`
  - `part_names(data: bytes) -> tuple[str, ...]`

- [ ] **Step 1: Add the extra and the mypy overrides**

In `pyproject.toml`, under `[project.optional-dependencies]`, after the `documents` line:

```toml
# Kept separate from `documents` so a caller who only wants PDFs does not
# acquire four more libraries. lxml is here for ODF, which has no maintained
# reader: odfpy is unmaintained and the text extraction is a few hundred lines.
office = [
    "python-docx>=1.1",
    "python-pptx>=1.0",
    "openpyxl>=3.1",
    "lxml>=5.0",
]
```

And after the existing `faster_whisper` override block, following its comment style exactly:

```toml
[[tool.mypy.overrides]]
# python-docx ships no py.typed; confined to one handler (see
# tests/unit/test_dependencies_stay_confined.py), so this is scoped there
# rather than globally.
module = "docx.*"
ignore_missing_imports = true

[[tool.mypy.overrides]]
# python-pptx ships no py.typed; confined to one handler.
module = "pptx.*"
ignore_missing_imports = true

[[tool.mypy.overrides]]
# openpyxl ships no py.typed; confined to one handler.
module = "openpyxl.*"
ignore_missing_imports = true
```

Note the `.*` suffixes: the handlers import submodules (`docx.table`, `docx.oxml.ns`, `pptx.util`), and a bare `module = "docx"` would not cover them.

Do **not** add an `lxml` override — `lxml-stubs` is not installed, but `lxml` ships its own `py.typed`-adjacent stubs in recent releases; if mypy complains in Task 5, add the override there with the same comment shape.

In `tests/unit/test_dependencies_stay_confined.py`, add to `CONFINED`:

```python
    # The three OOXML readers, each confined to the one handler that speaks its
    # document model. There is no shared "office" module importing all three: a
    # caller who installs the extra for spreadsheets should not have a Word
    # parser loaded, and a single module would make that impossible.
    "docx": {"handlers/office_word.py"},
    "pptx": {"handlers/office_slides.py"},
    "openpyxl": {"handlers/office_sheets.py"},
    # ODF has no maintained reader, so `adapters/odf.py` walks the flat XML
    # parts itself. lxml is that walk and nothing else touches it.
    "lxml": {"adapters/odf.py"},
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/adapters/test_ooxml.py`:

```python
"""The container sniffer, which is what stops a .docx being a hex dump."""

from __future__ import annotations

import io
import zipfile

from readeverything.adapters.ooxml import (
    ODF_TEXT_MIME,
    SHEETS_MIME,
    SLIDES_MIME,
    WORD_MIME,
    office_mimetype,
    part_names,
    read_part,
    zip_part_names,
)
from tests.fixtures_office import docx_bytes, odt_bytes, pptx_bytes, xlsx_bytes


def test_a_docx_is_recognised_from_its_head() -> None:
    assert office_mimetype(docx_bytes()[:4096]) == WORD_MIME


def test_a_pptx_is_recognised_from_its_head() -> None:
    assert office_mimetype(pptx_bytes()[:4096]) == SLIDES_MIME


def test_an_xlsx_is_recognised_even_though_content_types_is_written_last() -> None:
    """The measurement this whole module exists for.

    openpyxl writes `[Content_Types].xml` as the LAST zip entry, so it is
    unreachable from any bounded head. Classifying by the `xl/` part-name
    prefix is what makes a spreadsheet detectable at all.
    """
    head = xlsx_bytes()[:4096]
    assert b"[Content_Types].xml" not in head
    assert office_mimetype(head) == SHEETS_MIME


def test_an_odt_is_recognised_from_its_stored_mimetype_entry() -> None:
    """ODF stores `mimetype` first and uncompressed precisely so it can be
    sniffed. Reading it is the whole rule."""
    assert office_mimetype(odt_bytes()[:4096]) == ODF_TEXT_MIME


def test_a_plain_zip_is_not_claimed_as_an_office_document() -> None:
    """A `.zip` and a `.jar` must stay archives; Spec 8 descends into them."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.txt", "hello")
        archive.writestr("data/rows.csv", "a,b\n1,2\n")
    assert office_mimetype(buffer.getvalue()[:4096]) is None


def test_bytes_that_are_not_a_zip_are_not_claimed() -> None:
    assert office_mimetype(b"%PDF-1.7\nnot a zip at all") is None
    assert office_mimetype(b"") is None


def test_a_truncated_head_yields_no_answer_rather_than_a_wrong_one() -> None:
    """Better `None` — which leaves the caller on its existing fallbacks —
    than a guess from a name that was cut in half."""
    assert office_mimetype(docx_bytes()[:35]) is None


def test_part_names_walking_stops_at_the_end_of_the_head() -> None:
    """The walk must terminate on a partial header rather than reading past
    the buffer or looping."""
    names = zip_part_names(xlsx_bytes()[:600])
    assert names == ("docProps/app.xml", "docProps/core.xml")


def test_read_part_returns_a_named_member_of_the_whole_package() -> None:
    body = read_part(docx_bytes(), "word/document.xml")
    assert body is not None
    assert b"<w:document" in body


def test_read_part_returns_none_for_a_member_that_is_not_there() -> None:
    assert read_part(docx_bytes(), "word/nonexistent.xml") is None


def test_read_part_returns_none_for_bytes_that_are_not_a_zip() -> None:
    """No exception: every caller of this is on a handler's degrade path."""
    assert read_part(b"not a zip", "anything") is None


def test_part_names_lists_the_whole_package_not_just_the_head() -> None:
    assert "word/document.xml" in part_names(docx_bytes())
```

- [ ] **Step 3: Run to verify they fail**

```bash
uv run --all-extras pytest tests/unit/adapters/test_ooxml.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: readeverything.adapters.ooxml` (and `tests.fixtures_office`, which Task 4 creates; that is fine, this task's tests go green only after Task 4 — run them again there).

**Ordering note for the executor:** this task's tests import `tests.fixtures_office`, which Task 4 builds. Implement Task 2's module now, and expect its tests to stay red until Task 4 lands. Task 4's step 4 re-runs them. If you prefer strictly-green commits, do Task 4 before Task 2; nothing else depends on the order.

- [ ] **Step 4: Implement**

Create `src/readeverything/adapters/ooxml.py`:

```python
"""Reading an office package's zip container, without a zip library's help.

Two jobs that look alike and are not.

`office_mimetype` classifies a container from its FIRST 4096 BYTES, because
that is all `Perception` ever hands the detector (`pipeline/perception.py`).
`zipfile` cannot help: a zip's central directory lives at the END of the file,
so a bounded head has no index at all. What a head does have is a run of local
file headers, each naming one part, and the part names are enough — an OOXML
package puts its family's parts under `word/`, `ppt/` or `xl/`, and an ODF
package stores a `mimetype` entry first and uncompressed for exactly this
purpose.

Spec §3 proposed reading `[Content_Types].xml` instead. Measured, that cannot
work: openpyxl writes `[Content_Types].xml` as the LAST entry of a workbook, so
it is unreachable from any bounded head, and a rule built on it would detect
Word and PowerPoint while silently failing every Excel file. The override it
carries is also `…spreadsheetml.sheet.main+xml` rather than the document
mimetype, so it would need unpicking even where it is reachable.

`read_part` and `part_names` are the other job: they get the WHOLE bytes and
may use `zipfile` normally.

No I/O here, and no environment: every function takes bytes and returns data.
That is what lets the handlers import this module directly without breaking
`ports/handler.py`'s rule that a handler never touches a filesystem.
"""

from __future__ import annotations

import io
import struct
import zipfile

WORD_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
SLIDES_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
SHEETS_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
ODF_TEXT_MIME = "application/vnd.oasis.opendocument.text"
ODF_SLIDES_MIME = "application/vnd.oasis.opendocument.presentation"
ODF_SHEETS_MIME = "application/vnd.oasis.opendocument.spreadsheet"

#: Every mimetype this spec's three handlers claim. `composition.py` and the
#: README table are the consumers; keeping one tuple means adding a format
#: cannot leave one of them behind.
OFFICE_MIMETYPES = frozenset(
    {
        WORD_MIME,
        SLIDES_MIME,
        SHEETS_MIME,
        ODF_TEXT_MIME,
        ODF_SLIDES_MIME,
        ODF_SHEETS_MIME,
    }
)

#: The part-name prefix each OOXML family stores its own parts under. Order is
#: irrelevant — a package holds exactly one of these trees.
_OOXML_PREFIXES = {"word/": WORD_MIME, "ppt/": SLIDES_MIME, "xl/": SHEETS_MIME}

_ODF_MIMETYPES = frozenset({ODF_TEXT_MIME, ODF_SLIDES_MIME, ODF_SHEETS_MIME})

#: Local file header: b"PK\x03\x04", then 26 more bytes before the name.
_LOCAL_HEADER = b"PK\x03\x04"
_LOCAL_HEADER_SIZE = 30
#: Where the ODF `mimetype` entry's bytes begin: 30-byte header plus the
#: 8-character name, with no extra field. Not assumed — `zip_part_names`
#: computes it — but recorded because it is why ODF is sniffable at all.
_ODF_MIMETYPE_ENTRY = "mimetype"

#: Set in a local header's flag word when the sizes are written AFTER the data,
#: in a trailing descriptor. The header's own size field is then zero and the
#: walk cannot find the next header, so it stops rather than looping forever.
_STREAMED_SIZES = 0x08


def _entries(head: bytes) -> list[tuple[str, int, int, int]]:
    """`(name, method, data_offset, compressed_size)` for every complete header.

    Stops at the first thing that is not a complete local file header: the end
    of the buffer, a truncated name, a central-directory signature, or an entry
    whose size was streamed. Never reads past `head` and never loops: every
    iteration advances by at least `_LOCAL_HEADER_SIZE`.
    """
    found: list[tuple[str, int, int, int]] = []
    offset = 0
    while offset + _LOCAL_HEADER_SIZE <= len(head):
        if head[offset : offset + 4] != _LOCAL_HEADER:
            break
        flags, method = struct.unpack_from("<HH", head, offset + 6)
        compressed, _uncompressed = struct.unpack_from("<II", head, offset + 18)
        name_length, extra_length = struct.unpack_from("<HH", head, offset + 26)
        name_start = offset + _LOCAL_HEADER_SIZE
        name_end = name_start + name_length
        if name_end > len(head):
            # The name is cut in half. Half a name is not evidence.
            break
        try:
            name = head[name_start:name_end].decode("utf-8")
        except UnicodeDecodeError:
            break
        data_offset = name_end + extra_length
        found.append((name, method, data_offset, compressed))
        if flags & _STREAMED_SIZES and compressed == 0:
            # The size lives in a descriptor after the data, so where the next
            # header starts is unknowable from here.
            break
        offset = data_offset + compressed
    return found


def zip_part_names(head: bytes) -> tuple[str, ...]:
    """Part names readable from the local file headers inside `head`.

    A prefix of the package's contents, in stored order — never the whole
    listing, because the head is bounded. Use `part_names` when you hold the
    whole file.
    """
    return tuple(name for name, _method, _offset, _size in _entries(head))


def office_mimetype(head: bytes) -> str | None:
    """The specific office mimetype these leading bytes describe, or None.

    None means "this is not an office document as far as the head can tell",
    which leaves the caller on whatever it would otherwise have concluded. A
    plain `.zip` and a `.jar` must land here, because Spec 8 descends into
    those and must not descend into a `.docx`.

    Residual risk, recorded rather than hidden: a plain zip whose first entries
    happen to sit under a top-level `word/`, `ppt/` or `xl/` directory is read
    as an office document. The handler then fails to parse it and degrades with
    an honest report, which is the library's contract for a misdetection — and
    is strictly better than the hex dump it gets today.
    """
    if not head.startswith(_LOCAL_HEADER):
        return None
    entries = _entries(head)
    if not entries:
        return None
    name, method, data_offset, compressed = entries[0]
    if name == _ODF_MIMETYPE_ENTRY and method == zipfile.ZIP_STORED:
        declared = head[data_offset : data_offset + compressed]
        if len(declared) == compressed:
            value = declared.decode("ascii", errors="replace")
            if value in _ODF_MIMETYPES:
                return value
    for part, mime in ((n, m) for n, *_ in entries for p, m in _OOXML_PREFIXES.items() if n.startswith(p) for n in (n,) for p in (p,)):
        return mime
    return None


def read_part(data: bytes, name: str) -> bytes | None:
    """One member of a whole package, or None if it is not there.

    Never raises. Every caller is on a handler's path, and a handler degrades.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as package:
            return package.read(name)
    except (KeyError, OSError, zipfile.BadZipFile):
        return None


def part_names(data: bytes) -> tuple[str, ...]:
    """Every member of a whole package, or an empty tuple if it is unreadable."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as package:
            return tuple(package.namelist())
    except (OSError, zipfile.BadZipFile):
        return ()
```

**Implementer's note:** the comprehension in `office_mimetype`'s `for part, mime in …` line above is deliberately left as the one thing to write yourself — write it as a plain readable loop:

```python
    for name, _method, _offset, _size in entries:
        for prefix, mime in _OOXML_PREFIXES.items():
            if name.startswith(prefix):
                return mime
    return None
```

- [ ] **Step 5: Run mypy and ruff**

```bash
uv run --all-extras mypy src/readeverything/adapters/ooxml.py
uv run --all-extras ruff check src/readeverything/adapters/ooxml.py
uv run --all-extras pytest tests/unit/test_dependencies_stay_confined.py -x -q
```

The confinement test will FAIL on `test_the_confinement_table_is_live` until Tasks 5-8 create the files the new entries name. That is expected; it goes green at Task 8. Do not weaken the table to make it pass early.

- [ ] **Step 6: Commit**

```bash
uv run --all-extras ruff format src/readeverything/adapters/ooxml.py tests/unit/adapters/test_ooxml.py
git add src/readeverything/adapters/ooxml.py tests/unit/adapters/test_ooxml.py pyproject.toml tests/unit/test_dependencies_stay_confined.py
git commit -m "feat(adapters): classify an office container from its first 4096 bytes"
```

---

## Task 3: Detection stops calling a .docx a zip

**Files:**
- Modify: `src/readeverything/adapters/detection.py`
- Test: `tests/unit/adapters/test_detection.py`

**Interfaces:**
- Consumes: `office_mimetype` from Task 2.
- Produces: no new names. `PuremagicDetector.detect` gains one branch.

**Lane note:** Spec 8's agent also edits this file. Keep the change to exactly one import and one `if` block, both additive, so the merge is a no-conflict insertion.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/adapters/test_detection.py`:

```python
async def test_a_docx_is_not_reported_as_a_zip() -> None:
    """A `.docx` IS a zip, and puremagic says so. Reporting that dispatches an
    organisation's policy document to a hex dump."""
    mime = await PuremagicDetector().detect("policy.docx", docx_bytes()[:4096])
    assert str(mime) == WORD_MIME


async def test_a_spreadsheet_is_told_apart_from_a_document_by_content() -> None:
    """puremagic returns the SAME four candidates for all three OOXML families,
    every one at 0.40 confidence — below `_SIGNATURE_FLOOR`, and topped by
    `wordprocessingml.document` regardless of what the file is. Taking its
    answer would label every spreadsheet a Word document."""
    detector = PuremagicDetector()
    word = await detector.detect("a.docx", docx_bytes()[:4096])
    sheet = await detector.detect("b.xlsx", xlsx_bytes()[:4096])
    slides = await detector.detect("c.pptx", pptx_bytes()[:4096])
    assert {str(word), str(sheet), str(slides)} == {WORD_MIME, SHEETS_MIME, SLIDES_MIME}


async def test_an_office_document_is_detected_from_content_not_its_name() -> None:
    """The spec's rule: detection must not be extension-driven. A deck named
    `.bin` is still a deck."""
    mime = await PuremagicDetector().detect("mystery.bin", pptx_bytes()[:4096])
    assert str(mime) == SLIDES_MIME


async def test_an_odt_is_detected_from_its_stored_mimetype_entry() -> None:
    mime = await PuremagicDetector().detect("notes.odt", odt_bytes()[:4096])
    assert str(mime) == ODF_TEXT_MIME


async def test_a_plain_zip_is_still_a_plain_zip() -> None:
    """Spec 8 descends into archives. If this refinement claimed every zip,
    archive descent would have nothing left to descend into."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.txt", "hello")
    mime = await PuremagicDetector().detect("bundle.zip", buffer.getvalue()[:4096])
    assert str(mime) != WORD_MIME
    assert "opendocument" not in str(mime)
```

Add the imports this file needs: `io`, `zipfile`, the six mime constants from `readeverything.adapters.ooxml`, and the four fixture builders from `tests.fixtures_office`.

- [ ] **Step 2: Run to verify they fail**

```bash
uv run --all-extras pytest tests/unit/adapters/test_detection.py -x -q
```
Expected: FAIL — a `.docx` comes back as the extension guess only, and `mystery.bin` comes back as `application/octet-stream`.

- [ ] **Step 3: Implement**

Add the import at the top of `src/readeverything/adapters/detection.py`, after the `puremagic` import:

```python
from readeverything.adapters.ooxml import office_mimetype
```

Extend the module docstring's numbered list with a new step between 0 and 1:

```
0.5. A zip container that is really an office document, classified by its part
   names. See `adapters/ooxml.office_mimetype`.
```

And insert this block in `detect`, immediately after the `_is_iso_bmff` block and before the `for match in matches:` loop:

```python
        # An office document is a zip, and saying so is true and useless: it
        # sends an organisation's policy document to the hex-dump fallback.
        # Checked before the signature loop for the same reason ISO BMFF is:
        # puremagic's answer here is not a weaker version of this one, it is
        # noise. Measured, it returns the identical four candidates for a
        # .docx, a .pptx and an .xlsx — every one at 0.40 confidence, topped by
        # `wordprocessingml.document` whatever the file actually is.
        #
        # This is also what keeps Spec 8's archive descent honest: a `.docx`
        # gets a specific mimetype that a handler claims, so `walk` treats it
        # as a document, while a plain `.zip` falls through this branch
        # untouched and stays an archive.
        office = office_mimetype(head)
        if office is not None:
            try:
                return MimeType.parse(office)
            except ValueError:
                pass
```

- [ ] **Step 4: Run the tests**

```bash
uv run --all-extras pytest tests/unit/adapters/test_detection.py tests/unit/adapters/test_ooxml.py -x -q
uv run --all-extras mypy src/readeverything/adapters/detection.py
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uv run --all-extras ruff format src/readeverything/adapters/detection.py tests/unit/adapters/test_detection.py
git add src/readeverything/adapters/detection.py tests/unit/adapters/test_detection.py
git commit -m "feat(adapters): a .docx is a document, not a zip"
```

---

## Task 4: Office fixtures, generated not committed

**Files:**
- Create: `tests/fixtures_office.py`, `tests/unit/test_office_fixtures.py`
- Modify: `tests/unit/test_dependencies_stay_confined.py`

**Interfaces:**
- Produces, importable from `tests/fixtures_office.py`:
  - `docx_bytes(*, headings=…, comment=None, tracked=False) -> bytes`
  - `pptx_bytes(*, slides=…, notes=…, picture_on=()) -> bytes`
  - `xlsx_bytes(*, sheets=…, formulas=False, cached=False) -> bytes`
  - `odt_bytes(blocks=…) -> bytes`, `odp_bytes(slides=…) -> bytes`, `ods_bytes(sheets=…) -> bytes`
  - `big_xlsx(rows: int) -> bytes` — for budget-truncation tests.

- [ ] **Step 1: Write the fixture module**

Create `tests/fixtures_office.py`:

```python
"""Office documents generated at test time rather than committed as binaries.

Same reasoning as `tests/fixtures_pdf.py`: a committed binary rots, nobody can
read a diff of one, and a corrupted byte looks identical to an intentional
edit. Generating them keeps the input to every office test readable.

Two of these do something `python-docx` and `openpyxl` cannot, by rewriting one
part inside the finished zip:

* **Tracked changes.** python-docx has no API for `w:ins`/`w:del`, and the
  Word handler reports their presence as a fact. Injecting the element is the
  only way to have a document that carries one.
* **Cached formula values.** openpyxl never computes a formula and writes no
  cached `<v>`, so `data_only=True` returns `None` for every formula cell in
  any workbook it produced. A test of "represent shows the value" against such
  a workbook would assert that a blank is a blank. Injecting the `<v>` Excel
  itself would have written is what makes the value path real.

ODF is hand-built. There is no maintained writer, the format is a three-entry
zip, and building it here means the test knows exactly what it fed the reader.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Sequence

import docx
import openpyxl
import pptx
from pptx.util import Inches

#: ODF namespaces, only the ones the fixtures actually emit.
_OFFICE_NS = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
_TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
_TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_DRAW_NS = "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"

_MANIFEST = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<manifest:manifest xmlns:manifest='
    '"urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.3"/>'
)


def _rewrite_part(data: bytes, name: str, edit) -> bytes:  # type: ignore[no-untyped-def]
    """The package with one part passed through `edit`, everything else copied.

    Entry ORDER is preserved, which matters: `adapters/ooxml.office_mimetype`
    reads local headers in stored order, and an ODF package is only sniffable
    because `mimetype` is first. A naive rebuild that reordered entries would
    make the detection tests pass or fail for reasons unrelated to detection.
    """
    source = zipfile.ZipFile(io.BytesIO(data))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            blob = source.read(item.filename)
            if item.filename == name:
                blob = edit(blob)
            target.writestr(item, blob)
    return buffer.getvalue()


def docx_bytes(
    *,
    headings: Sequence[tuple[int, str, str]] = (
        (1, "Alpha", "The body of the alpha section."),
        (2, "Bravo", "The body of the bravo section."),
        (1, "Charlie", "The body of the charlie section."),
    ),
    table: Sequence[Sequence[str]] | None = (("region", "total"), ("north", "12")),
    comment: str | None = None,
    tracked: bool = False,
) -> bytes:
    """A Word document with headings, body paragraphs and optionally a table.

    The table is placed after the FIRST heading rather than at the end, so a
    handler that appends tables instead of reading them in document order
    produces visibly wrong text.
    """
    document = docx.Document()
    for index, (level, title, body) in enumerate(headings):
        document.add_heading(title, level)
        paragraph = document.add_paragraph(body)
        if index == 0:
            if table is not None:
                built = document.add_table(rows=len(table), cols=len(table[0]))
                for row_index, row in enumerate(table):
                    for cell_index, value in enumerate(row):
                        built.cell(row_index, cell_index).text = value
            if comment is not None:
                document.add_comment(
                    runs=paragraph.runs, text=comment, author="Reviewer", initials="RV"
                )
    buffer = io.BytesIO()
    document.save(buffer)
    data = buffer.getvalue()
    if tracked:
        data = _rewrite_part(data, "word/document.xml", _insert_tracked_change)
    return data


def _insert_tracked_change(xml: bytes) -> bytes:
    """Wrap the first run in a `w:ins`, the way Word records an insertion.

    python-docx has no API for this and the Word handler reports tracked
    changes as a card fact, so without this there is no document that exercises
    the true branch.
    """
    opening = re.search(rb"<w:r>", xml)
    if opening is None:
        return xml
    closing = xml.index(b"</w:r>", opening.end()) + len(b"</w:r>")
    run = xml[opening.start() : closing]
    inserted = (
        b'<w:ins w:id="901" w:author="Reviewer" w:date="2026-08-15T00:00:00Z">' + run + b"</w:ins>"
    )
    return xml[: opening.start()] + inserted + xml[closing:]


def pptx_bytes(
    *,
    titles: Sequence[str] = ("Opening position", "The numbers", "What we decided"),
    body: str = "First point\nSecond point",
    notes: Sequence[str | None] = (None, "The number is soft; the trend is not.", None),
    picture_on: Sequence[int] = (),
) -> bytes:
    """A deck. `notes[i]` is slide i+1's speaker notes, or None for no notes.

    `picture_on` holds 1-indexed slide numbers that get an embedded PNG.
    """
    presentation = pptx.Presentation()
    picture = _png_bytes()
    for index, title in enumerate(titles):
        layout = presentation.slide_layouts[1 if index == 0 else 5]
        slide = presentation.slides.add_slide(layout)
        if slide.shapes.title is not None:
            slide.shapes.title.text = title
        if index == 0:
            slide.placeholders[1].text = body
        if index + 1 in picture_on:
            slide.shapes.add_picture(io.BytesIO(picture), Inches(1), Inches(2))
        note = notes[index] if index < len(notes) else None
        if note is not None:
            slide.notes_slide.notes_text_frame.text = note
    buffer = io.BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


def _png_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 120, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def xlsx_bytes(*, formulas: bool = False, cached: bool = False) -> bytes:
    """A two-sheet workbook. `formulas` adds a `=B*2` column; `cached` gives
    those formulas the value Excel would have stored alongside them."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Data"
    sheet.append(["region", "units", "doubled"])
    for row, region in enumerate(("north", "south", "east"), start=2):
        sheet.append([region, row, None])
        if formulas:
            sheet[f"C{row}"] = f"=B{row}*2"
    notes = workbook.create_sheet("Notes")
    notes["A1"] = "Units are thousands."
    buffer = io.BytesIO()
    workbook.save(buffer)
    data = buffer.getvalue()
    if formulas and cached:
        data = _rewrite_part(data, "xl/worksheets/sheet1.xml", _cache_formula_values)
    return data


def _cache_formula_values(xml: bytes) -> bytes:
    """Give every formula the cached `<v>` Excel writes and openpyxl does not.

    Without this, `data_only=True` returns None for every formula cell of any
    openpyxl-written workbook, and a test of the value path asserts nothing.
    The value is a constant, not the formula's real result: the point is that a
    cached value EXISTS and is what the value path reports, not that this
    fixture can do arithmetic.
    """
    return re.sub(rb"(<f>[^<]*</f>)", rb"\1<v>99</v>", xml)


def big_xlsx(rows: int) -> bytes:
    """One wide sheet, for proving a budget truncates and says so."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Wide"
    sheet.append(["region", "units", "note"])
    for row in range(rows):
        sheet.append([f"region-{row}", row, "a note long enough to add up"])
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _odf_package(mimetype: str, content: str) -> bytes:
    """A three-entry ODF package, `mimetype` first and STORED.

    First and uncompressed is not a stylistic choice: it is what the ODF spec
    requires so the type can be sniffed from a file's opening bytes, and it is
    exactly what `adapters/ooxml.office_mimetype` relies on.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as package:
        stored = zipfile.ZipInfo("mimetype")
        stored.compress_type = zipfile.ZIP_STORED
        package.writestr(stored, mimetype)
        package.writestr("content.xml", content)
        package.writestr("META-INF/manifest.xml", _MANIFEST)
    return buffer.getvalue()


def odt_bytes(
    blocks: Sequence[tuple[str, str]] = (
        ("h", "Alpha"),
        ("p", "The body of the alpha section."),
        ("h", "Bravo"),
        ("p", "The body of the bravo section."),
    ),
) -> bytes:
    """An ODF text document. Each block is `("h", text)` or `("p", text)`."""
    body = "".join(
        f'<text:h text:outline-level="1">{text}</text:h>'
        if kind == "h"
        else f"<text:p>{text}</text:p>"
        for kind, text in blocks
    )
    content = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<office:document-content xmlns:office="{_OFFICE_NS}" xmlns:text="{_TEXT_NS}">'
        f"<office:body><office:text>{body}</office:text></office:body>"
        f"</office:document-content>"
    )
    return _odf_package("application/vnd.oasis.opendocument.text", content)


def odp_bytes(
    slides: Sequence[tuple[str, Sequence[str]]] = (
        ("Opening position", ("First point", "Second point")),
        ("What we decided", ("Ship it",)),
    ),
) -> bytes:
    """An ODF presentation, one `draw:page` per slide."""
    pages = "".join(
        f'<draw:page draw:name="page{index}">'
        + f'<draw:frame><draw:text-box><text:p>{title}</text:p></draw:text-box></draw:frame>'
        + "".join(
            f"<draw:frame><draw:text-box><text:p>{line}</text:p></draw:text-box></draw:frame>"
            for line in lines
        )
        + "</draw:page>"
        for index, (title, lines) in enumerate(slides, start=1)
    )
    content = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<office:document-content xmlns:office="{_OFFICE_NS}" xmlns:text="{_TEXT_NS}" '
        f'xmlns:draw="{_DRAW_NS}">'
        f"<office:body><office:presentation>{pages}</office:presentation></office:body>"
        f"</office:document-content>"
    )
    return _odf_package("application/vnd.oasis.opendocument.presentation", content)


def ods_bytes(
    sheets: Sequence[tuple[str, Sequence[Sequence[str]]]] = (
        ("Data", (("region", "units"), ("north", "2"), ("south", "3"))),
        ("Notes", (("Units are thousands.",),)),
    ),
) -> bytes:
    """An ODF spreadsheet, one `table:table` per sheet."""
    tables = "".join(
        f'<table:table table:name="{name}">'
        + "".join(
            "<table:table-row>"
            + "".join(
                f"<table:table-cell><text:p>{cell}</text:p></table:table-cell>" for cell in row
            )
            + "</table:table-row>"
            for row in rows
        )
        + "</table:table>"
        for name, rows in sheets
    )
    content = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<office:document-content xmlns:office="{_OFFICE_NS}" xmlns:text="{_TEXT_NS}" '
        f'xmlns:table="{_TABLE_NS}">'
        f"<office:body><office:spreadsheet>{tables}</office:spreadsheet></office:body>"
        f"</office:document-content>"
    )
    return _odf_package("application/vnd.oasis.opendocument.spreadsheet", content)
```

- [ ] **Step 2: Write tests that the fixtures are what they claim**

Create `tests/unit/test_office_fixtures.py`. These guard the fixtures themselves: if `cached=True` silently stops injecting values, every formula-versus-value test becomes a test that `None == None`.

```python
"""The fixtures are guarded, because a broken fixture makes a test vacuous."""

from __future__ import annotations

import io
import zipfile

import docx
import openpyxl
import pptx

from tests.fixtures_office import (
    big_xlsx,
    docx_bytes,
    odp_bytes,
    ods_bytes,
    odt_bytes,
    pptx_bytes,
    xlsx_bytes,
)


def test_the_cached_workbook_has_values_and_the_uncached_one_does_not() -> None:
    """openpyxl computes nothing and writes no cached value, so without the
    injection `data_only=True` returns None for every formula cell — and a
    test of the value path would be asserting that a blank is a blank."""
    bare = openpyxl.load_workbook(io.BytesIO(xlsx_bytes(formulas=True)), data_only=True)
    assert [c.value for c in bare["Data"][2]][2] is None

    cached = openpyxl.load_workbook(
        io.BytesIO(xlsx_bytes(formulas=True, cached=True)), data_only=True
    )
    assert [c.value for c in cached["Data"][2]][2] == 99


def test_the_formula_workbook_still_reads_back_as_formulas() -> None:
    """Injecting a cached value must not destroy the `<f>` beside it."""
    book = openpyxl.load_workbook(io.BytesIO(xlsx_bytes(formulas=True, cached=True)))
    assert [c.value for c in book["Data"][2]][2] == "=B2*2"


def test_the_tracked_fixture_carries_a_tracked_change_and_the_plain_one_does_not() -> None:
    plain = zipfile.ZipFile(io.BytesIO(docx_bytes())).read("word/document.xml")
    tracked = zipfile.ZipFile(io.BytesIO(docx_bytes(tracked=True))).read("word/document.xml")
    assert b"<w:ins" not in plain
    assert b"<w:ins" in tracked


def test_the_tracked_fixture_is_still_a_readable_document() -> None:
    """A rewritten part that python-docx cannot parse would make every tracked
    changes test fail for the wrong reason."""
    document = docx.Document(io.BytesIO(docx_bytes(tracked=True)))
    assert any("alpha" in p.text.lower() for p in document.paragraphs)


def test_the_commented_fixture_carries_exactly_one_comment() -> None:
    document = docx.Document(io.BytesIO(docx_bytes(comment="Check this number.")))
    comments = list(document.comments)
    assert len(comments) == 1
    assert comments[0].text == "Check this number."


def test_the_table_sits_between_headings_not_at_the_end() -> None:
    """A handler that appends tables rather than reading document order must
    produce visibly wrong text, so the fixture must not put the table last."""
    from docx.oxml.ns import qn

    document = docx.Document(io.BytesIO(docx_bytes()))
    tags = [child.tag for child in document.element.body.iterchildren()]
    assert qn("w:tbl") in tags
    assert tags.index(qn("w:tbl")) < len(tags) - 2


def test_notes_are_present_on_exactly_the_slides_that_asked_for_them() -> None:
    deck = pptx.Presentation(io.BytesIO(pptx_bytes()))
    assert [s.has_notes_slide for s in deck.slides] == [False, True, False]


def test_a_picture_lands_only_on_the_requested_slide() -> None:
    deck = pptx.Presentation(io.BytesIO(pptx_bytes(picture_on=(2,))))
    pictures = [
        sum(1 for shape in slide.shapes if shape.shape_type == 13) for slide in deck.slides
    ]
    assert pictures == [0, 1, 0]


def test_every_odf_package_stores_its_mimetype_first_and_uncompressed() -> None:
    """The ODF spec requires it and `office_mimetype` depends on it. A fixture
    that compressed the entry would make the ODF detection tests pass or fail
    for reasons unrelated to detection."""
    for data in (odt_bytes(), odp_bytes(), ods_bytes()):
        package = zipfile.ZipFile(io.BytesIO(data))
        assert package.namelist()[0] == "mimetype"
        assert package.getinfo("mimetype").compress_type == zipfile.ZIP_STORED


def test_a_big_workbook_is_actually_big() -> None:
    book = openpyxl.load_workbook(io.BytesIO(big_xlsx(500)), read_only=True)
    assert book["Wide"].max_row == 501
    book.close()
```

- [ ] **Step 3: Confine the fixture-only imports**

The three writers are now imported from `tests/fixtures_office.py` as well as from the handlers. `CONFINED` only governs `src/`, so nothing changes there — but add a mirror rule beside `REPORTLAB_CONFINED_TEST_FILE` in `tests/unit/test_dependencies_stay_confined.py`:

```python
#: The three OOXML writers generate office fixtures at test time so no binary
#: is committed (see tests/fixtures_office.py). Within `tests/` they are
#: confined to the fixture module and to the fixture module's own guard test,
#: so every other test imports the fixture functions rather than the writers.
OFFICE_WRITER_TEST_FILES = {"fixtures_office.py", "test_office_fixtures.py"}
```

and the test:

```python
def test_the_office_writers_are_confined_to_the_fixture_module() -> None:
    violations: list[str] = []
    for path in TESTS.rglob("*.py"):
        relative = str(path.relative_to(TESTS))
        roots = _imported_roots(ast.parse(path.read_text()))
        if roots & {"docx", "pptx", "openpyxl"} and relative not in OFFICE_WRITER_TEST_FILES:
            violations.append(relative)
    assert not violations, f"office writers imported outside the fixture module: {violations}"
```

- [ ] **Step 4: Run everything this unblocks**

```bash
uv run --all-extras pytest tests/unit/test_office_fixtures.py tests/unit/adapters/test_ooxml.py tests/unit/adapters/test_detection.py -x -q
```
Expected: PASS. Tasks 2 and 3's tests go green here.

- [ ] **Step 5: Commit**

```bash
uv run --all-extras ruff format tests/fixtures_office.py tests/unit/test_office_fixtures.py tests/unit/test_dependencies_stay_confined.py
git add tests/fixtures_office.py tests/unit/test_office_fixtures.py tests/unit/test_dependencies_stay_confined.py
git commit -m "test: generate office fixtures rather than committing binaries"
```

---

## Task 5: `adapters/odf.py`

**Files:**
- Create: `src/readeverything/adapters/odf.py`
- Test: `tests/unit/adapters/test_odf.py`

**Interfaces:**
- Consumes: `read_part` from Task 2.
- Produces:
  - `@dataclass(frozen=True, slots=True) class OdfBlock: level: int; text: str` — `level=0` for a body paragraph, `1..9` for a heading.
  - `odf_blocks(data: bytes) -> tuple[OdfBlock, ...]`
  - `odf_slides(data: bytes) -> tuple[tuple[str, ...], ...]` — per slide, its text runs in order.
  - `odf_sheets(data: bytes) -> tuple[tuple[str, tuple[tuple[str, ...], ...]], ...]` — per sheet, its name and rows.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/adapters/test_odf.py`:

```python
"""ODF read with lxml, because odfpy is unmaintained."""

from __future__ import annotations

from readeverything.adapters.odf import odf_blocks, odf_sheets, odf_slides
from tests.fixtures_office import odp_bytes, ods_bytes, odt_bytes


def test_headings_and_paragraphs_come_back_in_document_order() -> None:
    blocks = odf_blocks(odt_bytes())
    assert [(b.level, b.text) for b in blocks] == [
        (1, "Alpha"),
        (0, "The body of the alpha section."),
        (1, "Bravo"),
        (0, "The body of the bravo section."),
    ]


def test_a_heading_carries_its_outline_level() -> None:
    """A section outline is a tree. Flattening every heading to one level would
    make a sub-section look like a peer of the section containing it."""
    blocks = odf_blocks(odt_bytes(blocks=(("h", "Top"), ("p", "Body"))))
    assert blocks[0].level == 1
    assert blocks[1].level == 0


def test_bytes_that_are_not_an_odf_package_yield_nothing_rather_than_raising() -> None:
    """Every caller is a handler, and a handler degrades."""
    assert odf_blocks(b"not a zip") == ()
    assert odf_slides(b"not a zip") == ()
    assert odf_sheets(b"not a zip") == ()


def test_slides_come_back_one_tuple_per_page() -> None:
    slides = odf_slides(odp_bytes())
    assert len(slides) == 2
    assert slides[0][0] == "Opening position"
    assert "First point" in slides[0]


def test_sheets_come_back_named_and_in_row_order() -> None:
    sheets = odf_sheets(ods_bytes())
    assert [name for name, _rows in sheets] == ["Data", "Notes"]
    assert sheets[0][1][0] == ("region", "units")
    assert sheets[0][1][1] == ("north", "2")


def test_an_empty_document_yields_an_empty_tuple_not_a_fabricated_block() -> None:
    assert odf_blocks(odt_bytes(blocks=())) == ()
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run --all-extras pytest tests/unit/adapters/test_odf.py -x -q
```
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

Create `src/readeverything/adapters/odf.py`:

```python
"""ODF text extraction, walked directly rather than through a library.

odfpy is unmaintained, and the part of ODF this library needs — the text in
document order — is a walk over one flat XML part. Depending on an unmaintained
package to avoid two hundred lines is the trade that ages worst.

Namespace-aware by local name: ODF documents in the wild bind the same
namespaces to different prefixes, and matching on `text:p` as a literal string
works until the first file written by something other than the tool you tested
against.

No I/O and no environment here, like `ooxml.py`: bytes in, data out.
"""

from __future__ import annotations

from dataclasses import dataclass

from lxml import etree

from readeverything.adapters.ooxml import read_part

#: The single part every ODF package keeps its body in.
_CONTENT_PART = "content.xml"

_TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
_TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_DRAW_NS = "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"

_OUTLINE_LEVEL = f"{{{_TEXT_NS}}}outline-level"
_TABLE_NAME = f"{{{_TABLE_NS}}}name"
_DRAW_PAGE = f"{{{_DRAW_NS}}}page"
_TABLE_TABLE = f"{{{_TABLE_NS}}}table"
_TABLE_ROW = f"{{{_TABLE_NS}}}table-row"
_TABLE_CELL = f"{{{_TABLE_NS}}}table-cell"
_TEXT_H = f"{{{_TEXT_NS}}}h"
_TEXT_P = f"{{{_TEXT_NS}}}p"

#: What an unlabelled heading level defaults to. ODF makes `outline-level`
#: optional and a heading without one is still a heading.
_DEFAULT_HEADING_LEVEL = 1

#: A body paragraph's level. Zero rather than None so `OdfBlock.level` is one
#: comparable number and callers never branch on a missing value.
_BODY_LEVEL = 0


@dataclass(frozen=True, slots=True)
class OdfBlock:
    """One paragraph or heading. `level` is 0 for body text, 1-9 for a heading."""

    level: int
    text: str


def _content(data: bytes) -> etree._Element | None:
    """The parsed `content.xml`, or None for anything unreadable.

    Never raises. `resolve_entities=False` and `no_network=True` are not
    optional: this parses a file the caller found on disk, and an XML parser
    that resolves external entities on untrusted input is an exfiltration
    primitive.
    """
    part = read_part(data, _CONTENT_PART)
    if part is None:
        return None
    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
    try:
        return etree.fromstring(part, parser=parser)
    except etree.XMLSyntaxError:
        return None


def _text_of(element: etree._Element) -> str:
    """Every descendant's text, joined and whitespace-normalised.

    `itertext` rather than `.text`, because ODF splits a paragraph across
    `text:span` children the moment any of it is styled — reading `.text` alone
    silently drops every bold word in the document.
    """
    return " ".join("".join(element.itertext()).split())


def odf_blocks(data: bytes) -> tuple[OdfBlock, ...]:
    """Headings and paragraphs of an `.odt`, in document order."""
    root = _content(data)
    if root is None:
        return ()
    blocks: list[OdfBlock] = []
    for element in root.iter(_TEXT_H, _TEXT_P):
        text = _text_of(element)
        if element.tag == _TEXT_H:
            raw = element.get(_OUTLINE_LEVEL)
            try:
                level = int(raw) if raw is not None else _DEFAULT_HEADING_LEVEL
            except ValueError:
                level = _DEFAULT_HEADING_LEVEL
            blocks.append(OdfBlock(level=max(1, level), text=text))
        else:
            blocks.append(OdfBlock(level=_BODY_LEVEL, text=text))
    return tuple(blocks)


def odf_slides(data: bytes) -> tuple[tuple[str, ...], ...]:
    """Each `draw:page`'s text runs, in order. The first is the title by
    convention, which is how ODP stores it and how the slides handler reads it."""
    root = _content(data)
    if root is None:
        return ()
    slides: list[tuple[str, ...]] = []
    for page in root.iter(_DRAW_PAGE):
        runs = [text for element in page.iter(_TEXT_P) if (text := _text_of(element))]
        slides.append(tuple(runs))
    return tuple(slides)


def odf_sheets(data: bytes) -> tuple[tuple[str, tuple[tuple[str, ...], ...]], ...]:
    """Each `table:table`'s name and rows of cell text."""
    root = _content(data)
    if root is None:
        return ()
    sheets: list[tuple[str, tuple[tuple[str, ...], ...]]] = []
    for table in root.iter(_TABLE_TABLE):
        rows = tuple(
            tuple(_text_of(cell) for cell in row.iter(_TABLE_CELL))
            for row in table.iter(_TABLE_ROW)
        )
        sheets.append((table.get(_TABLE_NAME) or "", rows))
    return tuple(sheets)
```

- [ ] **Step 4: Run the tests**

```bash
uv run --all-extras pytest tests/unit/adapters/test_odf.py -x -q
uv run --all-extras mypy src/readeverything/adapters/odf.py
```

If mypy reports missing stubs for `lxml`, add an override to `pyproject.toml` in the same shape as the three from Task 2, with a comment naming why.

- [ ] **Step 5: Commit**

```bash
uv run --all-extras ruff format src/readeverything/adapters/odf.py tests/unit/adapters/test_odf.py
git add src/readeverything/adapters/odf.py tests/unit/adapters/test_odf.py pyproject.toml
git commit -m "feat(adapters): read ODF's flat XML directly, since odfpy is unmaintained"
```

---

## Task 6: `OfficeWordHandler`

**Files:**
- Create: `src/readeverything/handlers/office_word.py`, `tests/unit/handlers/test_office_word.py`

**Interfaces:**
- Consumes: `SourceReader`, `Observer`, `odf_blocks`/`OdfBlock`, `CellRange` is NOT used here.
- Produces:
  - `OfficeWordHandler(*, source: SourceReader, observer: Observer | None = None)`
  - ClassVars: `mime_patterns = (WORD_MIME, ODF_TEXT_MIME)`, `priority = 0`, `handler_id = "office_word"`, `handler_version = 1`.
  - Params models: `ReadSectionParams(index: int = 0)`, `ReadRangeParams(start: int = 0, end: int = 4096)`, `ListCommentsParams()`, `ReadTableParams(index: int = 0)`.
  - `SECTION_SEPARATOR = "\n\n"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/handlers/test_office_word.py`:

```python
"""The Word handler: a table of contents, and text that knows its section."""

from __future__ import annotations

import pytest

from readeverything.domain.identity import ContentHash, MimeType, SourceRef
from readeverything.domain.locators import CharSpan
from readeverything.domain.rendition import Budget, TextContent
from readeverything.handlers.office_word import (
    OfficeWordHandler,
    ReadRangeParams,
    ReadSectionParams,
    ReadTableParams,
)
from readeverything.testing.fakes import InMemorySource
from readeverything.testing.handler_compliance import MediaHandlerCompliance
from tests.fixtures_office import docx_bytes, odt_bytes

URI = "policy.docx"


def _handler(content: bytes) -> OfficeWordHandler:
    return OfficeWordHandler(source=InMemorySource({URI: content, "somewhere/else": content}))


def _ref(content: bytes) -> SourceRef:
    return SourceRef(
        uri=URI,
        mime=MimeType.parse("application/vnd.openxmlformats-officedocument."
                            "wordprocessingml.document"),
        content_hash=ContentHash("0" * 64),
        size_bytes=len(content),
    )


class TestWordCompliance(MediaHandlerCompliance):
    @pytest.fixture
    def content(self) -> bytes:
        return docx_bytes(comment="Check this number.")

    @pytest.fixture
    def handler(self, content: bytes) -> OfficeWordHandler:
        return OfficeWordHandler(
            source=InMemorySource({"compliance-subject": content, "somewhere/else": content})
        )


async def test_the_outline_is_a_table_of_contents() -> None:
    """What an agent needs to decide where to look, without reading the body."""
    content = docx_bytes()
    card = await _handler(content).describe(_ref(content))
    assert [segment.label for segment in card.outline] == ["Alpha", "Bravo", "Charlie"]


async def test_every_outline_segment_points_into_the_represented_text() -> None:
    """An outline whose locator does not address the text it summarises is a
    table of contents with the wrong page numbers."""
    content = docx_bytes()
    handler = _handler(content)
    card = await handler.describe(_ref(content))
    rendered = await handler.represent(_ref(content), Budget(max_chars=None))
    for segment in card.outline:
        assert isinstance(segment.locator, CharSpan)
        assert segment.label in rendered.text[segment.locator.start : segment.locator.end]


async def test_the_card_counts_paragraphs_headings_and_words() -> None:
    content = docx_bytes()
    card = await _handler(content).describe(_ref(content))
    assert card.facts["heading_count"] == 3
    assert card.facts["paragraph_count"] >= 3
    assert card.facts["word_count"] > 0


async def test_tracked_changes_are_reported_as_a_fact_both_ways() -> None:
    """Both directions, because a fact that is always "yes" tests nothing."""
    plain = docx_bytes()
    tracked = docx_bytes(tracked=True)
    assert (await _handler(plain).describe(_ref(plain))).facts["tracked_changes"] == "no"
    assert (await _handler(tracked).describe(_ref(tracked))).facts["tracked_changes"] == "yes"


async def test_the_card_counts_comments() -> None:
    content = docx_bytes(comment="Check this number.")
    card = await _handler(content).describe(_ref(content))
    assert card.facts["comment_count"] == 1


async def test_every_character_resolves_to_the_section_it_came_from() -> None:
    """The property the handler exists to provide, asserted across a boundary
    — an off-by-one in the segment starts passes any test that samples one
    section's middle."""
    content = docx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    first = rendered.text.index("alpha section")
    second = rendered.text.index("bravo section")
    assert rendered.locator_map.resolve(first) != rendered.locator_map.resolve(second)


async def test_barriers_sit_at_heading_boundaries() -> None:
    """One barrier per heading after the first: a chunker must not merge the
    end of one section with the start of the next."""
    content = docx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert len(rendered.barriers) == 2
    for barrier in rendered.barriers:
        assert rendered.locator_map.resolve(barrier) != rendered.locator_map.resolve(barrier - 1)


async def test_a_table_is_rendered_in_document_order_not_appended() -> None:
    """A table is frequently the answer, and where it sat is part of what it
    means. The fixture puts it after the first heading, so a handler that
    appends tables puts `north` after `charlie` and fails here."""
    content = docx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert rendered.text.index("north") < rendered.text.index("charlie section")


async def test_a_table_renders_as_pipe_delimited_rows() -> None:
    content = docx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert "region | total" in rendered.text


async def test_read_section_returns_that_section_located_at_its_span() -> None:
    content = docx_bytes()
    rendition = await _handler(content).invoke(_ref(content), "read_section",
                                               ReadSectionParams(index=1))
    assert isinstance(rendition.content, TextContent)
    assert "Bravo" in rendition.content.text
    assert isinstance(rendition.locator, CharSpan)


async def test_asking_for_a_section_past_the_end_degrades_rather_than_raising() -> None:
    content = docx_bytes()
    rendition = await _handler(content).invoke(_ref(content), "read_section",
                                               ReadSectionParams(index=99))
    assert rendition.degraded


async def test_list_comments_returns_the_comment_text_and_its_author() -> None:
    content = docx_bytes(comment="Check this number.")
    rendition = await _handler(content).invoke(_ref(content), "list_comments", ListCommentsParams())
    assert isinstance(rendition.content, TextContent)
    assert "Check this number." in rendition.content.text
    assert "Reviewer" in rendition.content.text


async def test_a_document_with_no_comments_says_so_rather_than_returning_nothing() -> None:
    """Empty output and "there are none" are different answers, and only one of
    them tells an agent to stop looking."""
    content = docx_bytes()
    rendition = await _handler(content).invoke(_ref(content), "list_comments", ListCommentsParams())
    assert isinstance(rendition.content, TextContent)
    assert rendition.content.text.strip()


async def test_read_table_returns_one_table_by_index() -> None:
    content = docx_bytes()
    rendition = await _handler(content).invoke(_ref(content), "read_table", ReadTableParams(index=0))
    assert isinstance(rendition.content, TextContent)
    assert "north" in rendition.content.text


async def test_an_odt_reads_through_the_same_handler() -> None:
    content = odt_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert "alpha section" in rendered.text
    card = await _handler(content).describe(_ref(content))
    assert [segment.label for segment in card.outline] == ["Alpha", "Bravo"]


async def test_an_unreadable_document_degrades_rather_than_raising() -> None:
    content = b"this is not a word document at all"
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert rendered.degradations
    assert rendered.text


async def test_a_document_with_no_headings_still_maps_every_character() -> None:
    """`LocatorMap` demands total, gapless, zero-start coverage. A document
    with no heading has no section to attribute text to, and a handler that
    emits no segment produces a `Rendered` that will not construct."""
    content = docx_bytes(headings=(), table=None)
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert rendered.locator_map.length == len(rendered.text)


async def test_a_budget_truncates_and_says_so() -> None:
    content = docx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=20))
    assert len(rendered.text) <= 20
    assert any("truncated" in d.what for d in rendered.degradations)
```

Add `ListCommentsParams` to the import block at the top.

- [ ] **Step 2: Run to verify they fail**

```bash
uv run --all-extras pytest tests/unit/handlers/test_office_word.py -x -q
```
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

Create `src/readeverything/handlers/office_word.py`. Structure it exactly as `handlers/pdf.py` is structured: module docstring arguing the decisions, guarded import naming the extra, params models, then the handler with `requires`/`affordances`/`describe`/`invoke`/`represent`, a private `_represent`, and a `_fit` that applies the budget while pruning the map and the barriers.

The guarded import, copying `pdf.py`'s shape:

```python
try:
    import docx
    from docx.document import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph
except ImportError as exc:  # pragma: no cover - exercised via a patched sys.modules
    raise ImportError(
        "readeverything's Word support needs python-docx, which ships in the "
        "'office' extra: pip install 'readeverything[office]'. "
        "The composition root omits Word handling when python-docx is absent, so "
        "reaching this means the handler was imported directly."
    ) from exc
```

Load-bearing requirements, every one:

- **Document order comes from `document.element.body.iterchildren()`**, matching `qn("w:p")` and `qn("w:tbl")`. `document.paragraphs` skips table content and loses table position; the test above fails a handler that uses it.
- **A section is a heading and everything under it until the next heading.** A document's leading text before any heading is its own section, labelled `"(untitled)"` — a `LocatorMap` must start at offset 0, so there is no such thing as text belonging to no section.
- **Every section owns at least one character.** Join sections with `SECTION_SEPARATOR = "\n\n"` and include the separator in the section's own `LocatorSegment`, exactly as `pdf.py`'s `PAGE_SEPARATOR` comment explains. A heading with an empty body would otherwise contribute a zero-width `CharSpan` and `CharSpan.__post_init__` raises.
- **Barriers are the offsets where sections 2..N begin** — one per heading boundary, so heading count minus one when the document starts with a heading.
- **Tables render pipe-delimited**, `" | ".join(cell.text for cell in row.cells)`, one row per line, in document order.
- **Tracked changes** are `"yes"` when `word/document.xml` contains a `w:ins` or `w:del` element. Detect it on the parsed body (`body.iter(qn("w:ins"))`), not by substring-searching raw XML — the fixture injects real elements and a substring search would also match a `w:insideH` border tag.
- **Comments** come from `document.comments`; `comment.author` and `comment.text` are the fields. Guard with `getattr(document, "comments", ())` so an older python-docx degrades rather than raising, and report `comment_count` as `0` in that case.
- **ODF arrives at the same handler.** Branch on the container: if `office_mimetype(data)` is `ODF_TEXT_MIME`, build the section list from `odf_blocks(data)` (`level > 0` is a heading) instead of python-docx. Both branches feed the same section-flattening code, so there is one `represent` and one `LocatorMap` builder.
- **The handler never raises.** An unopenable document returns a `Rendered` carrying a one-line summary and a `Degradation`, located by `ByteRange(0, max(1, ref.size_bytes))` — copy `pdf.py`'s `_nothing_to_read` and `_unreadable` verbatim in shape and reasoning.
- **`describe` parses the document.** Unlike PDF there is no cheaper probe that can answer heading count, and the parse costs no model call and no subprocess. Say so in `describe`'s docstring rather than leaving a reader to wonder why there is no `MediaProbe` here.
- **`card.kind` is `MediaKind.BINARY`**, matching `pdf.py`: these mimetypes reach the handler at the registry's exact-mimetype step, long before the kind step.
- **Narrate `represent`** with `OperationStarted`/`OperationFinished` around it, `_OPERATION = "represent"`, exactly as `pdf.py` does.
- **`requires()` returns `frozenset()`.** Reading a Word document needs no model and no binary.

Affordances, all `DetailLevel.SEGMENT`, all `requires=frozenset()`:

| name | description |
| --- | --- |
| `read_section` | "Return one section of the document: a heading and the text under it." |
| `read_range` | "Return a character range of the document's flattened text." |
| `list_comments` | "List every comment in the document, with its author." |
| `read_table` | "Return one table as pipe-delimited rows." |

- [ ] **Step 4: Run the tests**

```bash
uv run --all-extras pytest tests/unit/handlers/test_office_word.py -x -q
uv run --all-extras mypy src/readeverything/handlers/office_word.py
uv run --all-extras ruff check src/readeverything/handlers/office_word.py
```

- [ ] **Step 5: Commit**

```bash
uv run --all-extras ruff format src/readeverything/handlers/office_word.py tests/unit/handlers/test_office_word.py
git add src/readeverything/handlers/office_word.py tests/unit/handlers/test_office_word.py
git commit -m "feat(handlers): read a Word document as a table of contents and sections"
```

---

## Task 7: `OfficeSlidesHandler`

**Files:**
- Create: `src/readeverything/handlers/office_slides.py`, `tests/unit/handlers/test_office_slides.py`

**Interfaces:**
- Produces:
  - `OfficeSlidesHandler(*, source: SourceReader, vision: VisionModel | None = None, observer: Observer | None = None)`
  - ClassVars: `mime_patterns = (SLIDES_MIME, ODF_SLIDES_MIME)`, `priority = 0`, `handler_id = "office_slides"`, `handler_version = 1`.
  - Params: `ReadSlideParams(page: int = 1)`, `ListMediaParams()`, `DescribeSlideImageParams(page: int = 1, index: int = 0, question: str)`.
  - `NOTES_HEADING = "Speaker notes:"`, `SLIDE_SEPARATOR = "\n"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/handlers/test_office_slides.py`:

```python
"""The slides handler: the most natural barrier in any format the library reads."""

from __future__ import annotations

import pytest

from readeverything.domain.capability import Capability
from readeverything.domain.identity import ContentHash, MimeType, SourceRef
from readeverything.domain.locators import ByteRange, PageRef
from readeverything.domain.rendition import Budget, TextContent
from readeverything.handlers.office_slides import (
    NOTES_HEADING,
    DescribeSlideImageParams,
    ListMediaParams,
    OfficeSlidesHandler,
    ReadSlideParams,
)
from readeverything.testing.fakes import FakeVisionModel, InMemorySource
from readeverything.testing.handler_compliance import MediaHandlerCompliance
from tests.fixtures_office import odp_bytes, pptx_bytes

URI = "deck.pptx"
SLIDES_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


def _handler(content: bytes, *, vision: object | None = None) -> OfficeSlidesHandler:
    return OfficeSlidesHandler(
        source=InMemorySource({URI: content, "somewhere/else": content}), vision=vision
    )


def _ref(content: bytes) -> SourceRef:
    return SourceRef(
        uri=URI,
        mime=MimeType.parse(SLIDES_MIME),
        content_hash=ContentHash("0" * 64),
        size_bytes=len(content),
    )


class TestSlidesCompliance(MediaHandlerCompliance):
    @pytest.fixture
    def content(self) -> bytes:
        return pptx_bytes(picture_on=(2,))

    @pytest.fixture
    def handler(self, content: bytes) -> OfficeSlidesHandler:
        return OfficeSlidesHandler(
            source=InMemorySource({"compliance-subject": content, "somewhere/else": content}),
            vision=FakeVisionModel(),
        )


async def test_the_outline_is_one_segment_per_slide_labelled_by_title() -> None:
    content = pptx_bytes()
    card = await _handler(content).describe(_ref(content))
    assert [s.label for s in card.outline] == [
        "Opening position",
        "The numbers",
        "What we decided",
    ]
    assert [s.locator for s in card.outline] == [PageRef(1), PageRef(2), PageRef(3)]


async def test_the_card_reports_slide_count_notes_and_media() -> None:
    content = pptx_bytes(picture_on=(2,))
    card = await _handler(content).describe(_ref(content))
    assert card.facts["slide_count"] == 3
    assert card.facts["notes_present"] == "yes"
    assert card.facts["media_count"] == 1


async def test_a_deck_with_no_notes_says_no_rather_than_omitting_the_fact() -> None:
    """A missing fact and a "no" read the same to a model that only sees the
    card. Only one of them is a claim the handler actually checked."""
    content = pptx_bytes(notes=(None, None, None))
    card = await _handler(content).describe(_ref(content))
    assert card.facts["notes_present"] == "no"


async def test_describing_a_deck_does_not_create_notes_slides() -> None:
    """`slide.notes_slide` CREATES one as a side effect. Touching it unguarded
    makes the notes fact depend on whether anything looked at the deck first."""
    content = pptx_bytes(notes=(None, None, None))
    handler = _handler(content)
    await handler.describe(_ref(content))
    card = await handler.describe(_ref(content))
    assert card.facts["notes_present"] == "no"


async def test_every_character_resolves_to_the_slide_it_came_from() -> None:
    content = pptx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    for number, word in ((1, "Opening position"), (2, "The numbers"), (3, "What we decided")):
        offset = rendered.text.index(word)
        assert rendered.locator_map.resolve(offset) == PageRef(number)
        assert rendered.locator_map.resolve(offset + len(word) - 1) == PageRef(number)


async def test_there_is_one_barrier_per_slide_break() -> None:
    content = pptx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert len(rendered.barriers) == 2
    for barrier in rendered.barriers:
        assert rendered.locator_map.resolve(barrier) != rendered.locator_map.resolve(barrier - 1)


async def test_speaker_notes_are_included_and_labelled() -> None:
    """They routinely hold the reasoning the slide only asserts. Labelling is
    what stops a model attributing a presenter's aside to the slide itself."""
    content = pptx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert "The number is soft" in rendered.text
    marker = rendered.text.index(NOTES_HEADING)
    assert rendered.text.index("The number is soft") > marker


async def test_a_note_resolves_to_the_slide_that_carries_it() -> None:
    """A note attributed to the wrong slide is worse than a missing note."""
    content = pptx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    offset = rendered.text.index("The number is soft")
    assert rendered.locator_map.resolve(offset) == PageRef(2)


async def test_read_slide_returns_that_slide_including_its_notes() -> None:
    content = pptx_bytes()
    rendition = await _handler(content).invoke(_ref(content), "read_slide", ReadSlideParams(page=2))
    assert rendition.locator == PageRef(2)
    assert isinstance(rendition.content, TextContent)
    assert "The numbers" in rendition.content.text
    assert "The number is soft" in rendition.content.text


async def test_asking_for_a_slide_past_the_end_degrades_rather_than_raising() -> None:
    content = pptx_bytes()
    rendition = await _handler(content).invoke(_ref(content), "read_slide", ReadSlideParams(page=99))
    assert rendition.degraded
    assert isinstance(rendition.locator, ByteRange)


async def test_list_media_reports_each_embedded_image_with_its_slide() -> None:
    content = pptx_bytes(picture_on=(2,))
    rendition = await _handler(content).invoke(_ref(content), "list_media", ListMediaParams())
    assert isinstance(rendition.content, TextContent)
    assert "image/png" in rendition.content.text
    assert "2" in rendition.content.text


async def test_describe_slide_image_is_absent_without_a_vision_model() -> None:
    """Negotiation, not a runtime apology: the affordance must not appear."""
    names = {a.name for a in _handler(pptx_bytes()).affordances()}
    assert "describe_slide_image" not in names


async def test_describe_slide_image_appears_with_a_vision_model() -> None:
    handler = _handler(pptx_bytes(), vision=FakeVisionModel())
    affordance = next(a for a in handler.affordances() if a.name == "describe_slide_image")
    assert affordance.requires == frozenset({Capability.VISION})


async def test_describe_slide_image_reaches_the_embedded_picture() -> None:
    content = pptx_bytes(picture_on=(2,))
    rendition = await _handler(content, vision=FakeVisionModel()).invoke(
        _ref(content),
        "describe_slide_image",
        DescribeSlideImageParams(page=2, index=0, question="What is shown?"),
    )
    assert rendition.locator == PageRef(2)
    assert isinstance(rendition.content, TextContent)


async def test_asking_about_an_image_that_is_not_there_degrades() -> None:
    content = pptx_bytes()
    rendition = await _handler(content, vision=FakeVisionModel()).invoke(
        _ref(content),
        "describe_slide_image",
        DescribeSlideImageParams(page=1, index=0, question="What is shown?"),
    )
    assert rendition.degraded


async def test_an_odp_reads_through_the_same_handler() -> None:
    content = odp_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert "Opening position" in rendered.text
    assert len(rendered.barriers) == 1


async def test_an_unreadable_deck_degrades_rather_than_raising() -> None:
    content = b"not a presentation"
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert rendered.degradations
    assert rendered.text


async def test_a_slide_with_no_text_still_owns_a_character() -> None:
    """`CharSpan` rejects a zero-width span, so an empty slide between two full
    ones is what breaks the map."""
    content = pptx_bytes(titles=("Alpha", "", "Charlie"), body="", notes=(None, None, None))
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert rendered.locator_map.length == len(rendered.text)
    assert len(rendered.locator_map.segments) == 3
```

Check `readeverything.testing.fakes` for the actual fake vision model's name before writing these — if it is not `FakeVisionModel`, use whatever `tests/unit/handlers/test_image_handler.py` uses and keep the rest unchanged.

- [ ] **Step 2: Run to verify they fail**

```bash
uv run --all-extras pytest tests/unit/handlers/test_office_slides.py -x -q
```
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

Create `src/readeverything/handlers/office_slides.py`, same shape as Task 6.

Guarded import:

```python
try:
    import pptx
except ImportError as exc:  # pragma: no cover - exercised via a patched sys.modules
    raise ImportError(
        "readeverything's slide support needs python-pptx, which ships in the "
        "'office' extra: pip install 'readeverything[office]'. "
        "The composition root omits slide handling when python-pptx is absent, so "
        "reaching this means the handler was imported directly."
    ) from exc
```

Load-bearing requirements:

- **`PageRef(slide_number)`, 1-indexed.** python-pptx's slide list is 0-indexed; this is the likeliest off-by-one in the task, and the test asserts each title's last character as well as its first.
- **Per slide, in this order: title, then body text in placeholder order, then notes under `NOTES_HEADING`.** The heading is what stops a presenter's aside being read as something the slide claimed.
- **Never touch `slide.notes_slide` without checking `slide.has_notes_slide` first.** Accessing it CREATES a notes slide, which changes the deck the next call describes. There is a test for exactly this.
- **Every slide owns at least one character.** `SLIDE_SEPARATOR = "\n"` included in the slide's own `LocatorSegment`, same reasoning as `pdf.py`'s `PAGE_SEPARATOR` — an empty slide in the middle of a deck is what breaks the map.
- **Barriers at every slide boundary**: slide count minus one.
- **`list_media`** walks `slide.shapes`, keeps shapes whose `shape_type` is `MSO_SHAPE_TYPE.PICTURE`, and reports slide number, index within the slide, `image.content_type` and `len(image.blob)`. It does not return the bytes: a card-adjacent listing that returned megabytes of PNG would defeat progressive disclosure.
- **`describe_slide_image(page, index, question)`** fetches `shape.image.blob` and `shape.image.content_type` and hands them to `self._vision.describe(blob, content_type, question)`. Locator is `PageRef(page)`. Registered only when `self._vision is not None`, `requires=frozenset({Capability.VISION})`, `DetailLevel.DEEP` — copy how `pdf.py` gates `ask_about_image`. On a missing slide, a missing image, a vision failure or an empty answer, degrade with a `PageRef` (or `ByteRange` when no slide was ever observed) rather than raising, exactly as `pdf.py`'s `_ask_about_page` does.
- **ODF** via `odf_slides(data)` when `office_mimetype(data) == ODF_SLIDES_MIME`: first run is the title, the rest is the body, and there are no notes. Both branches feed one flattening.
- **`describe` must not open a vision model, must not read image bytes** — `media_count` comes from counting picture shapes, which does not touch `.blob`.

- [ ] **Step 4: Run the tests**

```bash
uv run --all-extras pytest tests/unit/handlers/test_office_slides.py -x -q
uv run --all-extras mypy src/readeverything/handlers/office_slides.py
uv run --all-extras ruff check src/readeverything/handlers/office_slides.py
```

- [ ] **Step 5: Commit**

```bash
uv run --all-extras ruff format src/readeverything/handlers/office_slides.py tests/unit/handlers/test_office_slides.py
git add src/readeverything/handlers/office_slides.py tests/unit/handlers/test_office_slides.py
git commit -m "feat(handlers): read a deck, speaker notes labelled and attributed"
```

---

## Task 8: `OfficeSheetsHandler`

**Files:**
- Create: `src/readeverything/handlers/office_sheets.py`, `tests/unit/handlers/test_office_sheets.py`

**Interfaces:**
- Produces:
  - `OfficeSheetsHandler(*, source: SourceReader, observer: Observer | None = None)`
  - ClassVars: `mime_patterns = (SHEETS_MIME, ODF_SHEETS_MIME)`, `priority = 0`, `handler_id = "office_sheets"`, `handler_version = 1`.
  - Params: `ReadSheetParams(name: str = "", offset: int = 0, limit: int = 100)`, `ReadCellsParams(name: str = "", a1_range: str = "A1", formulas: bool = False)`, `ListSheetsParams()`.
  - `CELL_DELIMITER = " | "`, `SHEET_SEPARATOR = "\n"`.
  - Module functions `parse_a1(a1_range: str, sheet: str) -> CellRange | None` and `to_a1(row: int, col: int) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/handlers/test_office_sheets.py`:

```python
"""The sheets handler: `CellRange`'s only producer, and the formula/value split."""

from __future__ import annotations

import pytest

from readeverything.domain.identity import ContentHash, MimeType, SourceRef
from readeverything.domain.locators import ByteRange, CellRange
from readeverything.domain.rendition import Budget, TextContent
from readeverything.handlers.office_sheets import (
    ListSheetsParams,
    OfficeSheetsHandler,
    ReadCellsParams,
    ReadSheetParams,
    parse_a1,
    to_a1,
)
from readeverything.testing.fakes import InMemorySource
from readeverything.testing.handler_compliance import MediaHandlerCompliance
from tests.fixtures_office import big_xlsx, ods_bytes, xlsx_bytes

URI = "book.xlsx"
SHEETS_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _handler(content: bytes) -> OfficeSheetsHandler:
    return OfficeSheetsHandler(source=InMemorySource({URI: content, "somewhere/else": content}))


def _ref(content: bytes) -> SourceRef:
    return SourceRef(
        uri=URI,
        mime=MimeType.parse(SHEETS_MIME),
        content_hash=ContentHash("0" * 64),
        size_bytes=len(content),
    )


class TestSheetsCompliance(MediaHandlerCompliance):
    @pytest.fixture
    def content(self) -> bytes:
        return xlsx_bytes(formulas=True, cached=True)

    @pytest.fixture
    def handler(self, content: bytes) -> OfficeSheetsHandler:
        return OfficeSheetsHandler(
            source=InMemorySource({"compliance-subject": content, "somewhere/else": content})
        )


def test_a1_round_trips_through_the_zero_indexed_domain() -> None:
    """A1 is 1-indexed and base-26; `CellRange` is 0-indexed. The conversion is
    where an off-by-one becomes a citation pointing at the wrong row."""
    assert parse_a1("A1", "Data") == CellRange(sheet="Data", row=0, col=0)
    assert parse_a1("B3", "Data") == CellRange(sheet="Data", row=2, col=1)
    assert parse_a1("A1:C3", "Data") == CellRange(sheet="Data", row=0, col=0, rows=3, cols=3)
    assert to_a1(0, 0) == "A1"
    assert to_a1(2, 1) == "B3"


def test_a1_handles_multi_letter_columns() -> None:
    """Column AA is 27, not 11. Base-26 with no zero digit is the trap."""
    assert parse_a1("AA1", "Data") == CellRange(sheet="Data", row=0, col=26)
    assert to_a1(0, 26) == "AA1"


def test_an_unparseable_a1_range_yields_none_rather_than_a_wrong_cell() -> None:
    assert parse_a1("not a range", "Data") is None
    assert parse_a1("", "Data") is None
    assert parse_a1("1A", "Data") is None


async def test_the_card_names_every_sheet_and_its_used_range() -> None:
    content = xlsx_bytes()
    card = await _handler(content).describe(_ref(content))
    assert [s.label for s in card.outline] == ["Data", "Notes"]
    assert card.facts["sheet_count"] == 2
    assert card.facts["sheet.Data.used_range"] == "A1:C4"
    assert card.facts["sheet.Data.rows"] == 4
    assert card.facts["sheet.Data.columns"] == 3


async def test_every_outline_segment_carries_a_cell_range() -> None:
    """`CellRange`'s reason for existing: a sheet is addressed as cells, not as
    a character offset into this handler's chosen delimiter."""
    content = xlsx_bytes()
    card = await _handler(content).describe(_ref(content))
    assert all(isinstance(s.locator, CellRange) for s in card.outline)
    assert card.outline[0].locator == CellRange(sheet="Data", row=0, col=0, rows=4, cols=3)


async def test_every_character_resolves_to_the_sheet_it_came_from() -> None:
    content = xlsx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    offset = rendered.text.index("Units are thousands.")
    locator = rendered.locator_map.resolve(offset)
    assert isinstance(locator, CellRange)
    assert locator.sheet == "Notes"


async def test_there_is_a_barrier_at_every_sheet_boundary() -> None:
    content = xlsx_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert len(rendered.barriers) == 1
    barrier = rendered.barriers[0]
    assert rendered.locator_map.resolve(barrier) != rendered.locator_map.resolve(barrier - 1)


async def test_represent_shows_the_value_not_the_formula() -> None:
    """That is what the sheet MEANS. An auditor wants the formula and asks for
    it; a reader wants the number."""
    content = xlsx_bytes(formulas=True, cached=True)
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert "99" in rendered.text
    assert "=B2*2" not in rendered.text


async def test_read_cells_shows_the_formula_when_asked() -> None:
    """Reporting only one of the two is how a spreadsheet lies to a reader."""
    content = xlsx_bytes(formulas=True, cached=True)
    rendition = await _handler(content).invoke(
        _ref(content), "read_cells", ReadCellsParams(name="Data", a1_range="C2", formulas=True)
    )
    assert isinstance(rendition.content, TextContent)
    assert "=B2*2" in rendition.content.text
    assert rendition.locator == CellRange(sheet="Data", row=1, col=2)


async def test_read_cells_shows_the_value_by_default() -> None:
    content = xlsx_bytes(formulas=True, cached=True)
    rendition = await _handler(content).invoke(
        _ref(content), "read_cells", ReadCellsParams(name="Data", a1_range="C2")
    )
    assert isinstance(rendition.content, TextContent)
    assert "99" in rendition.content.text


async def test_a_formula_with_no_cached_value_is_reported_rather_than_shown_blank() -> None:
    """openpyxl computes nothing and many writers store no cached value. A
    blank cell and an uncomputed formula are different facts, and rendering
    both as blank is the spreadsheet's version of "this document is empty"."""
    content = xlsx_bytes(formulas=True, cached=False)
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert any("formula" in d.what.lower() for d in rendered.degradations)
    assert any("cached" in d.detail.lower() for d in rendered.degradations)


async def test_a_workbook_without_formulas_reports_no_formula_degradation() -> None:
    """A degradation that is always present tells a reader nothing."""
    content = xlsx_bytes(formulas=False)
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert not any("formula" in d.what.lower() for d in rendered.degradations)


async def test_read_sheet_pages_through_rows() -> None:
    content = big_xlsx(50)
    rendition = await _handler(content).invoke(
        _ref(content), "read_sheet", ReadSheetParams(name="Wide", offset=10, limit=5)
    )
    assert isinstance(rendition.content, TextContent)
    assert "region-9" in rendition.content.text
    assert "region-20" not in rendition.content.text
    assert isinstance(rendition.locator, CellRange)
    assert rendition.locator.row == 10


async def test_asking_for_a_sheet_that_is_not_there_degrades_rather_than_raising() -> None:
    content = xlsx_bytes()
    rendition = await _handler(content).invoke(
        _ref(content), "read_sheet", ReadSheetParams(name="Nope")
    )
    assert rendition.degraded
    assert isinstance(rendition.locator, ByteRange)


async def test_list_sheets_names_them_with_their_shapes() -> None:
    content = xlsx_bytes()
    rendition = await _handler(content).invoke(_ref(content), "list_sheets", ListSheetsParams())
    assert isinstance(rendition.content, TextContent)
    assert "Data" in rendition.content.text
    assert "Notes" in rendition.content.text


async def test_a_large_sheet_is_truncated_with_an_explicit_degradation() -> None:
    """A million-row sheet must be cut and must SAY it was cut."""
    content = big_xlsx(2000)
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=500))
    assert len(rendered.text) <= 500
    assert any("truncated" in d.what for d in rendered.degradations)


async def test_an_ods_reads_through_the_same_handler() -> None:
    content = ods_bytes()
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert "north" in rendered.text
    card = await _handler(content).describe(_ref(content))
    assert [s.label for s in card.outline] == ["Data", "Notes"]


async def test_an_unreadable_workbook_degrades_rather_than_raising() -> None:
    content = b"not a workbook"
    rendered = await _handler(content).represent(_ref(content), Budget(max_chars=None))
    assert rendered.degradations
    assert rendered.text


async def test_an_empty_sheet_still_owns_a_character() -> None:
    rendered = await _handler(xlsx_bytes()).represent(_ref(xlsx_bytes()), Budget(max_chars=None))
    assert rendered.locator_map.length == len(rendered.text)
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run --all-extras pytest tests/unit/handlers/test_office_sheets.py -x -q
```
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

Create `src/readeverything/handlers/office_sheets.py`, same shape as Tasks 6 and 7.

Guarded import:

```python
try:
    import openpyxl
except ImportError as exc:  # pragma: no cover - exercised via a patched sys.modules
    raise ImportError(
        "readeverything's spreadsheet support needs openpyxl, which ships in the "
        "'office' extra: pip install 'readeverything[office]'. "
        "The composition root omits spreadsheet handling when openpyxl is absent, so "
        "reaching this means the handler was imported directly."
    ) from exc
```

Load-bearing requirements:

- **`describe` uses `load_workbook(..., read_only=True)` and closes it.** A million-row workbook must not be fully materialised to answer "what sheets are there". `ReadOnlyWorksheet` has **no `.dimensions`** — use `ws.calculate_dimension()`, `ws.max_row`, `ws.max_column`. Always `wb.close()`; a read-only workbook holds an open zip handle.
- **`represent` opens the workbook TWICE**: once `data_only=True` for values and once plain for formulas. That is the only way to know a cell has a formula whose cached value is missing, and knowing that is what the formula degradation reports. Two opens of an in-memory buffer is cheap; guessing is not.
- **`represent` renders the value.** A formula cell with a cached value renders that value. A formula cell **without** one renders the formula text and contributes to a single `Degradation` — one per workbook, not one per cell, matching how `pdf.py` reports scanned pages:
  ```
  what:   "formulas without cached values"
  detail: "N cell(s) hold a formula with no cached value (cell Data!C2, …); the
           workbook was written by a tool that does not compute formulas, so the
           formula text is shown in place of a result"
  ```
- **Every sheet is a `LocatorSegment` carrying a `CellRange`** covering its used range. Per-cell segments would be enormous and are not what a citation needs; per-sheet matches how `pdf.py` maps per page.
- **Every sheet owns at least one character.** `SHEET_SEPARATOR = "\n"` included in the sheet's own segment; an empty sheet between two full ones is what breaks the map.
- **Barriers at every sheet boundary**: sheet count minus one.
- **`parse_a1` / `to_a1` live in this module**, not in `domain/locators.py`. A1 is presentation. Column letters are base-26 with no zero digit: `A`=0, `Z`=25, `AA`=26. Return `None` from `parse_a1` on anything unparseable so the caller degrades rather than citing a cell nobody asked for.
- **`read_sheet(name, offset, limit)`** returns `limit` rows starting at 0-indexed `offset`, located by `CellRange(sheet=name, row=offset, col=0, rows=<rows actually returned>, cols=<columns>)`. `rows` must be what was returned, not what was asked for, or the locator claims cells that were never read.
- **`read_cells(name, a1_range, formulas=False)`** returns the block, delimited, located by the parsed `CellRange`.
- **ODF** via `odf_sheets(data)` when `office_mimetype(data) == ODF_SHEETS_MIME`. ODF cells carry no formula/value distinction in this reader, so the formula degradation never fires for an `.ods` — which is honest, since nothing was hidden.
- **The handler never raises.** Same `_unreadable`/`_nothing_to_read`/`_fit` shapes as `pdf.py`.

- [ ] **Step 4: Run the tests**

```bash
uv run --all-extras pytest tests/unit/handlers/test_office_sheets.py -x -q
uv run --all-extras mypy src/readeverything/handlers/office_sheets.py
uv run --all-extras pytest tests/unit/test_dependencies_stay_confined.py -x -q
```
The confinement table goes green here: all four homes now exist and import what they claim.

- [ ] **Step 5: Commit**

```bash
uv run --all-extras ruff format src/readeverything/handlers/office_sheets.py tests/unit/handlers/test_office_sheets.py
git add src/readeverything/handlers/office_sheets.py tests/unit/handlers/test_office_sheets.py
git commit -m "feat(handlers): read a workbook, with formulas and values both reachable"
```

---

## Task 9: Composition, the integration scenario, and the README

**Files:**
- Modify: `src/readeverything/composition.py`, `tests/integration/conftest.py`, `README.md`
- Create: `tests/integration/test_office.py`
- Test: `tests/unit/test_composition.py`

**Interfaces:**
- Consumes: the three handlers from Tasks 6-8.
- Produces: `_optional_office_handlers(source, vision, observer) -> list[MediaHandler]` in `composition.py`.

**Lane note:** Spec 8 also edits `composition.py`. One new function and one new splat line, both additive. Do not reorder the existing handler list; `BinaryHandler` must stay last.

- [ ] **Step 1: Add the integration fixture**

In `tests/integration/conftest.py`, add beside `documents_root`:

```python
@pytest.fixture
def office_root(tmp_path: Path) -> Path:
    """A directory holding one of each office family — the §1.1 scenario."""
    (tmp_path / "policy.docx").write_bytes(docx_bytes(comment="Check this number."))
    (tmp_path / "deck.pptx").write_bytes(pptx_bytes(picture_on=(2,)))
    (tmp_path / "book.xlsx").write_bytes(xlsx_bytes(formulas=True, cached=True))
    (tmp_path / "notes.odt").write_bytes(odt_bytes())
    (tmp_path / "slides.odp").write_bytes(odp_bytes())
    (tmp_path / "sheet.ods").write_bytes(ods_bytes())
    return tmp_path
```

with the matching import from `tests.fixtures_office`.

- [ ] **Step 2: Write the failing tests**

Create `tests/integration/test_office.py`:

```python
"""Spec §1.1's acceptance scenario, through the real composition root."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from readeverything.composition import build_perception
from readeverything.domain.locators import CellRange, CharSpan, PageRef
from readeverything.domain.rendition import Budget

pytestmark = pytest.mark.integration


async def test_inspect_reports_each_family_without_parsing_the_whole_body(
    office_root: Path,
) -> None:
    perception = await build_perception(office_root, probe_binaries=False)

    word = await perception.inspect("policy.docx")
    assert word.facts["heading_count"] == 3

    deck = await perception.inspect("deck.pptx")
    assert deck.facts["slide_count"] == 3
    assert [s.label for s in deck.outline][0] == "Opening position"

    book = await perception.inspect("book.xlsx")
    assert book.facts["sheet_count"] == 2
    assert book.facts["sheet.Data.used_range"] == "A1:C4"


async def test_no_office_document_falls_through_to_the_hex_dump(office_root: Path) -> None:
    """The README's complaint, closed. A hex dump of a `.docx` is a hex dump of
    a zip container and the agent learns nothing."""
    perception = await build_perception(office_root, probe_binaries=False)
    for name in ("policy.docx", "deck.pptx", "book.xlsx", "notes.odt", "slides.odp", "sheet.ods"):
        card = await perception.inspect(name)
        assert "hexdump" not in card.affordance_names(), name


async def test_every_character_resolves_to_its_slide_heading_or_sheet(
    office_root: Path,
) -> None:
    perception = await build_perception(office_root, probe_binaries=False)

    deck = await perception.represent("deck.pptx", Budget(max_chars=None))
    assert isinstance(deck.locator_map.resolve(0), PageRef)
    assert deck.barriers

    word = await perception.represent("policy.docx", Budget(max_chars=None))
    assert isinstance(word.locator_map.resolve(0), CharSpan)
    assert word.barriers

    book = await perception.represent("book.xlsx", Budget(max_chars=None))
    assert isinstance(book.locator_map.resolve(0), CellRange)
    assert book.barriers


async def test_asking_for_slide_two_returns_slide_two_with_its_notes(
    office_root: Path,
) -> None:
    perception = await build_perception(office_root, probe_binaries=False)
    rendition = await perception.invoke("deck.pptx", "read_slide", {"page": 2})
    assert rendition.locator == PageRef(2)
    assert "The number is soft" in rendition.content.text


async def test_a_sheet_reads_as_text_with_formulas_reachable(office_root: Path) -> None:
    perception = await build_perception(office_root, probe_binaries=False)
    value = await perception.invoke("book.xlsx", "read_cells", {"name": "Data", "a1_range": "C2"})
    formula = await perception.invoke(
        "book.xlsx", "read_cells", {"name": "Data", "a1_range": "C2", "formulas": True}
    )
    assert "=B2*2" in formula.content.text
    assert "=B2*2" not in value.content.text


async def test_the_odf_equivalents_work_the_same_way(office_root: Path) -> None:
    perception = await build_perception(office_root, probe_binaries=False)
    assert (await perception.inspect("notes.odt")).outline
    assert (await perception.inspect("slides.odp")).outline
    assert (await perception.inspect("sheet.ods")).outline


async def test_a_base_install_without_the_office_extra_falls_back_to_binary(
    office_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Narrower, not broken — the same contract Pillow and pypdfium2 already
    have."""
    monkeypatch.setitem(sys.modules, "docx", None)
    monkeypatch.setitem(sys.modules, "pptx", None)
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    perception = await build_perception(office_root, probe_binaries=False)
    card = await perception.inspect("policy.docx")
    assert "hexdump" in card.affordance_names()
```

Check `Perception`'s actual method names (`inspect`, `represent`, `invoke`) and `invoke`'s parameter shape against `tests/integration/test_documents.py` before writing, and match it.

- [ ] **Step 3: Run to verify they fail**

```bash
uv run --all-extras pytest tests/integration/test_office.py -x -q
```
Expected: FAIL — every office file reports `hexdump`.

- [ ] **Step 4: Implement the composition wiring**

In `src/readeverything/composition.py`, add after `_optional_pdf_handler`:

```python
def _optional_office_handlers(
    source: SourceReader, vision: VisionModel | None, observer: Observer | None
) -> list[MediaHandler]:
    """The three office handlers, each present only if its own reader imports.

    Guarded exactly like `_optional_image_handler` and `_optional_pdf_handler`,
    but THREE separate guards rather than one: the `office` extra installs
    python-docx, python-pptx and openpyxl together, and an environment that has
    only one of them should still read that one family. A single try/except
    around all three would make one missing package cost the other two.
    """
    handlers: list[MediaHandler] = []
    try:
        from readeverything.handlers.office_word import OfficeWordHandler
    except ImportError:
        pass
    else:
        handlers.append(OfficeWordHandler(source=source, observer=observer))
    try:
        from readeverything.handlers.office_slides import OfficeSlidesHandler
    except ImportError:
        pass
    else:
        handlers.append(OfficeSlidesHandler(source=source, vision=vision, observer=observer))
    try:
        from readeverything.handlers.office_sheets import OfficeSheetsHandler
    except ImportError:
        pass
    else:
        handlers.append(OfficeSheetsHandler(source=source, observer=observer))
    return handlers
```

and one line in `build_perception`'s handler list, after the PDF splat and before the video splat:

```python
        *_optional_office_handlers(source, vision, observer),
```

- [ ] **Step 5: Run the tests**

```bash
uv run --all-extras pytest tests/integration/test_office.py tests/unit/test_composition.py -x -q
```

- [ ] **Step 6: Update the README**

In the "What's supported today" table, add three rows after the PDF row:

```markdown
| Word (`.docx`, `.odt`) | `binary` | `read_section`, `read_range`, `list_comments`, `read_table` | `office` extra (python-docx, lxml) |
| Slides (`.pptx`, `.odp`) | `binary` | `read_slide`, `list_media`; `describe_slide_image` when a vision model is supplied | `office` extra (python-pptx, lxml); a vision model for `describe_slide_image` |
| Spreadsheets (`.xlsx`, `.ods`) | `binary` | `read_sheet`, `read_cells`, `list_sheets` | `office` extra (openpyxl, lxml) |
```

Replace the "Office documents and archives have no handlers yet…" paragraph with:

```markdown
Archives have no handler yet — a `.zip` or a `.tar` falls through to the binary
fallback above (a hex dump), not a dedicated representation.

Legacy `.doc`, `.ppt` and `.xls` are out of scope: they are OLE2 compound
files, a different container format entirely, and their pure-Python support is
poor. They fall through to the hex dump.
```

And add to the Extras block:

```bash
pip install "readeverything[office]"     # python-docx, python-pptx, openpyxl, lxml
```

- [ ] **Step 7: Commit**

```bash
uv run --all-extras ruff format src/readeverything/composition.py tests/integration/
git add src/readeverything/composition.py tests/integration/conftest.py tests/integration/test_office.py README.md
git commit -m "feat(composition): register the office handlers, and close the README's gap"
```

---

## Task 10: Live validation of `describe_slide_image`

**Files:**
- Create: `tests/live/test_office_vision.py`

**Interfaces:** consumes the live-test config in `tests/live/conftest.py`.

**Note for the executor:** these are `live`-marked and deselected by default (`addopts = "-m 'not live and not accuracy'"`). Do NOT run them without telling the human partner first — they need to stop other inference on that server.

- [ ] **Step 1: Write the test**

Create `tests/live/test_office_vision.py`:

```python
"""`describe_slide_image` against a real model rather than a fake.

Structure only, never the model's words: that an answer came back, that it is
not an echo of the question, and that it is located at the slide the image was
embedded in.
"""

from __future__ import annotations

import pytest

from readeverything.domain.identity import ContentHash, MimeType, SourceRef
from readeverything.domain.locators import PageRef
from readeverything.domain.rendition import TextContent
from readeverything.handlers.office_slides import DescribeSlideImageParams, OfficeSlidesHandler
from readeverything.testing.fakes import InMemorySource
from tests.fixtures_office import pptx_bytes

pytestmark = pytest.mark.live


async def test_a_real_model_describes_a_picture_embedded_in_a_slide(live_vision) -> None:  # type: ignore[no-untyped-def]
    content = pptx_bytes(picture_on=(2,))
    handler = OfficeSlidesHandler(
        source=InMemorySource({"deck.pptx": content}), vision=live_vision
    )
    ref = SourceRef(
        uri="deck.pptx",
        mime=MimeType.parse(
            "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        ),
        content_hash=ContentHash("0" * 64),
        size_bytes=len(content),
    )
    rendition = await handler.invoke(
        ref,
        "describe_slide_image",
        DescribeSlideImageParams(page=2, index=0, question="What colour is this image?"),
    )
    assert isinstance(rendition.content, TextContent)
    assert rendition.content.text.strip()
    assert rendition.content.text.strip() != "What colour is this image?"
    assert rendition.locator == PageRef(2)
    assert not rendition.degraded
```

Check the actual fixture name in `tests/live/conftest.py` (it may be `live_vision` or something else) and match it.

- [ ] **Step 2: Ask the human partner before running**

```bash
uv run --all-extras pytest tests/live/test_office_vision.py -m live -x -q
```

- [ ] **Step 3: Commit**

```bash
uv run --all-extras ruff format tests/live/test_office_vision.py
git add tests/live/test_office_vision.py
git commit -m "test(live): describe a picture embedded in a real deck"
```

---

## Plan Self-Review

**Spec coverage.** §1.1 acceptance → Task 9's integration tests, clause by clause. §2 independence from Spec 8 → nothing archive-aware is built; Task 3's `test_a_plain_zip_is_still_a_plain_zip` is the guard that keeps Spec 8's descent rule satisfiable. §3 detection → Tasks 2 and 3, with the deviation argued in Measured Facts. §4 three handlers one reader → Tasks 2, 5, 6, 7, 8. §4.1 Word → Task 6. §4.2 Slides → Task 7. §4.3 Sheets → Task 8. §5 cell locator → Task 1. §6 dependencies → Task 2 (extra, overrides, confinement). §7 testing → Tasks 4 (fixtures), 6-8 (unit + compliance), 1 (locator validation), 3 (detection), 9 (integration), 10 (live). §8 non-goals → no OLE2, no layout, no charts beyond `list_media`, no revision reconstruction, no writing; the README edit in Task 9 states the OLE2 exclusion publicly.

**Deviations from the spec, all argued above and needing the reviewer's nod:**

1. **§3's `[Content_Types].xml` rule is replaced by part-name classification.** Measured: openpyxl writes that part LAST, so it is unreachable within the 4096-byte head the detector receives; a rule built on it detects Word and PowerPoint and silently fails every Excel file. ODF still uses its `mimetype` entry exactly as §3 says.
2. **§4's `adapters/ooxml.py` does not resolve relationships or extract media.** python-docx and python-pptx already do both, correctly; a second implementation inside this library would be a worse one. `ooxml.py` keeps the two jobs those libraries cannot do — head-bounded classification, and part access for ODF.
3. **`describe` parses the document rather than probing it.** There is no `MediaProbe` equivalent that can answer heading count or sheet dimensions, and the parse costs no model call and no subprocess. Sheets uses `read_only=True` so a huge workbook is not materialised.
4. **A new `Degradation` the spec does not name:** formulas with no cached value. It falls directly out of §4.3's "reporting only one of the two is how a spreadsheet lies to a reader" and out of the measurement that openpyxl-written workbooks have no cached values at all.

**Known risks a reviewer should hold me to.**
- Task 7's `slide.notes_slide` side effect is the subtlest bug available here: reading it without `has_notes_slide` mutates the parsed deck and makes the notes fact order-dependent. There is a test, and it only fails on the *second* `describe` call — a single-call test would not catch it.
- Task 8's A1 conversion is 1-indexed base-26 against a 0-indexed domain. Both directions are tested, including `AA`, which is where base-26-with-no-zero breaks naive code.
- The "every unit owns at least one character" rule recurs in all three handlers and is stated three times deliberately. An implementer who skips it in one handler gets a `CharSpan` ValueError only on a document with an empty section/slide/sheet — and each handler's test suite has exactly one such case.
- Task 2's tests depend on Task 4's fixtures, so Task 2 commits red. The ordering note says so; a reviewer seeing a red commit there should check the note rather than the code.
- `office_mimetype` can misread a plain zip whose first entries sit under `word/`, `ppt/` or `xl/`. Recorded in the function's docstring; the consequence is an honest degradation instead of a hex dump, which is not a regression.
