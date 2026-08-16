# readeverything: Faithful Rendering

**Date:** 2026-08-15
**Status:** Approved for planning
**Predecessors:** Spec 4 (the document family), Spec 9 (the office family)

---

## 1. Why this, and why now

Spec 9 reads office documents structurally: text in document order, slides,
sheets, notes. That is the right default and it covers most questions. It
cannot answer one class of question at all.

> "Which quarter's bar is taller?"
> "Is the disclaimer inside the box or below it?"
> "What does this diagram show?"

A slide is a *visual* artifact. Its meaning is frequently in arrangement,
emphasis, and imagery that no text extraction recovers. The library already
knows how to answer questions about pictures — `describe_image`, `ocr`,
`ask_about_image` all exist and work — and it already knows how to turn a
paginated document into page images, because `handlers/pdf.py` does exactly
that with `page_image`.

The missing piece is a way to *get* a faithful page image from a format
pypdfium2 cannot open. That is one adapter and one port.

The structural argument: this is the fourth capability to be negotiated rather
than required. `VISION` unlocks `describe_image` and `ocr_page`; a transcriber
unlocks `read_span`; an ffmpeg binary unlocks video. Each followed the same
shape, and the shape is now well-proven enough that adding a fourth is a small,
predictable change rather than a new idea. A library that grows a fourth
instance of an idiom it has already used three times is a library whose
architecture is holding.

### 1.1 Acceptance

> On a machine with no `soffice` binary, everything in Spec 9 works exactly as
> it does today and no rendering affordance appears anywhere — no tool that
> exists and returns an apology. Install LibreOffice, change nothing else, and
> `page_image` appears on the slide, word, and spreadsheet handlers, and on
> legacy `.doc`/`.ppt`/`.xls` files which gain a card and readable text where
> they previously got a hex dump. Asking to describe slide 4 renders slide 4
> and hands it to the vision path, and the answer cites the slide.

---

## 2. The capability

Add to `domain/capability.py`:

```python
DOCUMENT_RENDER = "document_render"
```

Probed the way ffmpeg is probed today — by asking the adapter, at composition
time, whether its binary is present and runnable — so the answer is settled
before any affordance is published. `adapters/model_probe.py` and
`adapters/probing.py` already establish this pattern; the plan should follow
whichever of them ffmpeg uses, literally.

---

## 3. The port

`ports/rendering.py`:

```python
@runtime_checkable
class DocumentRenderer(Protocol):
    def claims(self, mime: MimeType) -> bool: ...

    async def page_count(self, path: str) -> int: ...

    async def render_page(
        self, path: str, page: int, *, dpi: int = 150
    ) -> bytes:
        """A PNG of `page`, 1-indexed as a reader would count."""
        ...
```

Path-in, bytes-out. A renderer is an external process and external processes
take paths — the same acknowledgement `SourceReader.local_path` already makes,
and the reason that method exists. Notably this means **rendering a slide deck
inside a tarball works**, because Spec 8's `NestedSource.local_path`
materialises the member. Neither spec did anything to arrange that.

---

## 4. The adapter

`adapters/soffice_renderer.py`.

LibreOffice's automation surface is genuinely awkward and the design must
respect that rather than pretend otherwise:

- **One conversion, not one per page.** `soffice --convert-to pdf` produces a
  PDF of the whole document; page images then come from **pypdfium2**, which
  this repository already depends on and already knows how to drive. So the
  adapter is a converter, and `handlers/pdf.py`'s existing rendering code is
  reused rather than duplicated. This also means `render_page` after the first
  call is fast.
- **The converted PDF is an artifact.** It goes in the content-addressed
  artifact store under the source's content hash plus the adapter's version, so
  a document is converted once per machine, ever. This is the single most
  important performance decision in the spec: conversion is seconds, and
  without caching every slide render would pay it.
- **`soffice` is not concurrency-safe** across processes sharing a user profile.
  The adapter creates a private profile directory per instance
  (`-env:UserInstallation=file:///…`) and serialises conversions through the
  existing `Limiter` port under `DOCUMENT_RENDER`. `SemaphoreLimiter` already
  does this for `VISION`; the same mechanism, a different key.
- **Timeouts are mandatory and bounded.** A malformed document can hang
  `soffice` indefinitely. A conversion exceeding its deadline is killed and
  raises `RenditionFailedError`; it never blocks a perception forever.
- **No network, no macros.** Conversion runs with macro execution disabled.
  A document from an untrusted directory must not be able to execute on
  conversion, and this is the only place in the library where that is even
  possible.

---

## 5. What it unlocks

With `DOCUMENT_RENDER` present:

| Handler | Gains |
| --- | --- |
| `office_slides` | `page_image(page, dpi)`; `describe_slide(page)` when `VISION` too |
| `office_word` | `page_image(page, dpi)` — and pagination, which the structural reader genuinely does not have |
| `office_sheets` | `page_image(page, dpi)`, as the print layout |
| **legacy** `.doc` / `.ppt` / `.xls` | A card, `represent` text via the converted PDF, and page images — the whole family Spec 9 §3 deliberately declined |

That last row is the one that changes what the library can do rather than how
well it does it. Spec 9 punted on OLE2 formats because pure-Python support for
them is poor; this is the honest answer, and it arrives as a capability rather
than as a dependency everyone pays for.

`handlers/office_legacy.py` claims the OLE2 mimetypes and **requires**
`DOCUMENT_RENDER` from `requires()` — unlike the other three, which merely gain
affordances. Without the binary it is not registered and those files keep
falling through to the hex dump, which is the current behavior and therefore no
regression.

Every added affordance is `DEEP` and never runs during `inspect` or
`represent`. Spec 4 §11 settled this argument for OCR and the reasoning is
identical: a four-hundred-slide deck must not convert itself because someone
listed a directory.

---

## 6. Composition

```python
async def build_perception(
    root, *,
    renderer: DocumentRenderer | None = None,   # None = probe for soffice
    ...
) -> Perception
```

`None` probes and uses `soffice` if present; passing an explicit renderer allows
a caller to point at their own converter; passing `NullRenderer()` disables
rendering even on a machine that has LibreOffice — because a caller who wants
determinism in CI must be able to get it without uninstalling software.

---

## 7. Testing

- **Unit.** The adapter against a fake subprocess runner: profile isolation,
  timeout kill, macro flags, artifact caching hit and miss. No binary needed.
- **Unit.** Capability negotiation — with `DOCUMENT_RENDER` absent, no rendering
  affordance appears on any card. This is the test that protects §1.1's first
  sentence and it matters more than the rendering itself.
- **Integration, `test_real_binaries.py`.** Skipped without `soffice`, matching
  exactly how ffmpeg is treated there today.
- **Live.** `describe_slide`, marked `live`.
- **No committed rendered images and no pixel comparisons.** LibreOffice's
  output shifts between versions, and a golden-image test would fail on a
  rendering improvement. Assertions are on dimensions, page count, and
  non-blankness.

---

## 8. What this deliberately does not do

- **No bundled binary.** ~500MB, and not pip-installable. The port exists so a
  caller can supply something else.
- **No headless-browser renderer** for HTML. Different spec, different port
  implementation, same interface.
- **No fidelity claim.** The library reports that a page image came from a
  converter, and LibreOffice's rendering of a PowerPoint is a *rendering*, not
  the thing itself. Fonts substitute. This is recorded in the rendition's
  provenance rather than glossed.
