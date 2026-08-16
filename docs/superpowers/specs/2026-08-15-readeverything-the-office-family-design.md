# readeverything: The Office Family

**Date:** 2026-08-15
**Status:** Approved for planning
**Predecessors:** Spec 1 (perception core), Spec 4 (the document family)
**Sibling:** Spec 8 (descending into containers) — independent; see §2
**Successor:** Spec 10 (faithful rendering)

---

## 1. Why this, and why now

The README says it plainly:

> Office documents and archives have no handlers yet — files of those kinds
> fall through to the binary fallback above (a hex dump), not a dedicated
> representation.

A hex dump of a `.docx` is a hex dump of a zip container. The agent learns
nothing. And unlike the exotic formats it is reasonable to punt on, these three
families are where organisational knowledge actually lives: the decision is in
the deck, the numbers are in the spreadsheet, the policy is in the Word file.

The structural argument follows Spec 4's shape. That spec earned `PageRef` its
first producer by reading PDFs. A slide deck is the *other* natural producer —
and unlike a PDF, its pagination is semantic rather than typographic: slide 7
is a thing an author made, not a place the text happened to break. Spreadsheets
push further, on a locator the library does not yet have (§5).

### 1.1 Acceptance

> Point the library at a directory holding a `.docx`, a `.pptx`, and an
> `.xlsx`. `inspect` reports each one's kind and shape — heading count, slide
> count, sheet names and used ranges — without parsing the whole document body.
> `represent` returns text whose every character resolves to the slide, heading
> section, or sheet it came from, with barriers at slide and sheet boundaries.
> Asking for slide 4 returns slide 4, including its speaker notes. Asking for a
> sheet returns its cells as text a model can read, with formulas visible
> rather than silently replaced by cached values. An image embedded in a deck is
> reachable and describable when a vision model is present. The `.odt`/`.odp`/
> `.ods` equivalents work the same way.

---

## 2. Independence from Spec 8

This spec and Spec 8 are built concurrently, in separate worktrees. They do not
share a line of code.

That is not a coincidence to be careful about; it is the port boundary working
as designed. These handlers read bytes through `SourceReader` and are forbidden
from touching a filesystem or knowing what a path means. **Whether a `.docx`
arrived loose on disk or as a member of a tarball is invisible to every line of
this spec.** Spec 8 makes archive members reachable; this spec makes office
documents readable; the composition of the two is free and untested by either
side until integration.

The single shared artifact is Spec 8 §2's URI grammar, which this spec's test
fixtures use to name a document inside an archive in exactly one integration
test, added at integration time and not before.

**A conflicting note:** a `.docx` *is* a zip file, and Spec 8 teaches `walk` to
descend into zips. Left alone, that would list `report.docx!word/document.xml`
as a source. It must not. The rule, owned by Spec 8 and recorded here because
this spec is why it exists: **`walk` descends into a container only when no
handler claims the container's own mimetype at a higher priority than the
archive handler.** OOXML and ODF mimetypes are detected specifically (they are
not `application/zip`), these handlers claim them, and so office documents are
documents rather than folders. A `.jar` or a plain `.zip` still descends.

---

## 3. Detection

Detection must not be extension-driven, and `puremagic` reports most OOXML as
`application/zip` because that is what the bytes are.

`adapters/detection.py` gains a refinement step: when the detected mimetype is
`application/zip`, peek at the container's `[Content_Types].xml` (OOXML) or
`mimetype` entry (ODF, which is stored uncompressed as the first entry
precisely so it can be sniffed) and report the specific type:

| Family | Mimetype |
| --- | --- |
| Word | `application/vnd.openxmlformats-officedocument.wordprocessingml.document` |
| Slides | `application/vnd.openxmlformats-officedocument.presentationml.presentation` |
| Sheets | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` |
| ODF | `application/vnd.oasis.opendocument.{text,presentation,spreadsheet}` |

Legacy `.doc`/`.ppt`/`.xls` (OLE2 compound files) are **out of scope** and fall
through to the binary handler. They are a different container format entirely,
their pure-Python support is poor, and Spec 10's renderer is the honest answer
for them.

---

## 4. Three handlers, one reader

`adapters/ooxml.py` holds what the three families share: opening the zip,
resolving relationships (`_rels/`), reading part XML, and extracting embedded
media. The handlers hold what differs, which is most of the interesting part.

Splitting into three files rather than one `office.py` is deliberate: the PDF
handler is already the largest file in `handlers/`, and a single module covering
three document models would exceed it several times over. Each handler is
independently readable and independently testable.

### 4.1 `handlers/office_word.py`

- **Card.** `facts`: paragraph count, word count, heading count, whether it has
  tracked changes or comments. `outline`: one `Segment` per heading, locator
  `CharSpan`, label the heading text — a document's table of contents, which is
  exactly what an agent needs to decide where to look.
- **`represent`.** Body text in document order, with a `LocatorMap` mapping
  every character to its heading section. `barriers` at heading boundaries.
- **Affordances.** `read_section(index)`, `read_range(start, end)`,
  `list_comments()`, `read_table(index)`.
- Tables render as pipe-delimited text rather than being skipped, because a
  table is frequently the answer.

### 4.2 `handlers/office_slides.py`

- **Card.** `facts`: slide count, whether notes are present, embedded media
  count. `outline`: one `Segment` per slide, locator `PageRef(slide)`, label the
  slide title.
- **`represent`.** Per slide: title, body text in placeholder order, then
  speaker notes under a marked heading. `barriers` at every slide boundary —
  the most natural barrier in any format the library reads.
- **Affordances.** `read_slide(page)`, `list_media()`, and
  `describe_slide_image(page, index)` gated on `VISION`, which reaches an
  embedded picture and hands it to the existing vision path.
- **Speaker notes are included in `represent` and labelled.** They routinely
  hold the reasoning the slide only asserts. Labelling them is what stops a
  model attributing a presenter's aside to the slide itself.

### 4.3 `handlers/office_sheets.py`

- **Card.** `facts`: sheet count, per-sheet used range and row/column counts.
  `outline`: one `Segment` per sheet, label the sheet name.
- **`represent`.** Each sheet as delimited text with a header row, bounded by
  budget — a million-row sheet is truncated with an explicit `Degradation`
  rather than silently.
- **Affordances.** `read_sheet(name, offset, limit)`, `read_cells(name, a1_range)`,
  `list_sheets()`.
- **Formulas.** `openpyxl` can read either the formula or the last cached value.
  `represent` shows the **value**, because that is what the sheet means; a
  `read_cells(..., formulas=True)` parameter shows formulas, because that is
  what an auditor needs. Reporting only one of the two is how a spreadsheet lies
  to a reader.

---

## 5. The cell locator

None of the five existing locators addresses a cell. `CharSpan` into rendered
text is not it — the rendering is an artifact of this handler's formatting
choices, so a citation into it does not survive a change of delimiter.

Add to `domain/locators.py`:

```python
@dataclass(frozen=True, slots=True)
class CellRange:
    """A rectangular block of cells in a named sheet, 0-indexed internally."""
    sheet: str
    row: int
    col: int
    rows: int = 1
    cols: int = 1
```

Validated like its siblings (non-negative origin, positive extent) and added to
the `Locator` union. This is the one domain change in this spec, and it is the
same move Spec 4 made: a format arrives that the vocabulary cannot address, so
the vocabulary grows — rather than the format being flattened into a locator
that does not mean what it says.

A1 notation is a presentation concern and lives in the handler, not here.

---

## 6. Dependencies

A new `office` extra, kept separate from `documents` so a caller who only wants
PDFs does not acquire four libraries:

```toml
office = [
    "python-docx>=1.1",
    "python-pptx>=1.0",
    "openpyxl>=3.1",
    "lxml>=5.0",
]
```

ODF is read by `adapters/odf.py` using `lxml` directly against the flat XML
parts; `odfpy` is unmaintained and the formats' text extraction is a few hundred
lines.

Each handler guards its import exactly as `handlers/pdf.py` guards `pypdfium2`
— raising an `ImportError` naming the extra — and `composition.py` omits the
handler when the import is unavailable. This is settled house style and the
plan should follow `pdf.py` literally.

`tests/unit/test_dependencies_stay_confined.py` gains the new modules, so a
stray `import docx` outside its adapter fails the suite.

---

## 7. Testing

- **Fixtures.** `tests/fixtures_office.py`, generating documents at test time
  with the same libraries, mirroring how `tests/fixtures_pdf.py` uses reportlab.
  No committed binaries.
- **Unit, per handler.** The `handler_compliance` suite from
  `readeverything.testing` plus per-family behavior: heading outlines, notes
  labelling, formula-versus-value, budget truncation reporting a `Degradation`.
- **Unit, `test_locators.py`.** `CellRange` validation.
- **Unit, detection.** A `.docx` is not reported as `application/zip`.
- **Integration, `tests/integration/test_office.py`.** The §1.1 acceptance
  scenario.
- **Live.** Only `describe_slide_image`, marked `live`, matching how the image
  and PDF specs treat vision.

---

## 8. What this deliberately does not do

- **No legacy OLE2 formats** (§3).
- **No layout fidelity.** Text in document order, not text positioned on a page.
  That is Spec 10's job and the reason Spec 10 exists.
- **No charts.** A chart in a deck or sheet is reported as a fact and, when it
  has an embedded image, is reachable through the media affordance. Parsing
  chart XML into data is a spec of its own.
- **No revision history.** Tracked changes are reported as a *fact* (present or
  absent) and comments are readable, but reconstructing the document at a prior
  revision is out of scope.
- **No writing.**
