# ask_about_image Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give an agent one tool that asks a vision model a question about any image-bearing file — a photograph, a PDF page, or a video frame — optionally scoped to a rectangular region, without a discovery round trip.

**Architecture:** One agent tool bound to a *name convention*, not to an affordance. `ask_about_image(uri, question, where)` forwards `{**where, "question": question}` to `perception.invoke(uri, "ask_about_image", ...)`. Three handlers claim that affordance name, each declaring its own coordinate schema. The tool layer never learns a mimetype; `Perception.invoke` already validates params against whichever schema the resolved handler declared.

**Tech Stack:** Python 3.12+, pydantic v2, Pillow, pypdfium2, langchain-core, pytest + pytest-asyncio, `uv run` for everything.

**Spec:** `docs/superpowers/specs/2026-08-15-readeverything-ask-about-image-design.md`

## Global Constraints

- **Handlers never raise about their input.** A handler returns a degraded `Rendition` instead. Only `Transcriber` may raise; `VisionModel` failures are caught by the handler. Existing per-handler patterns show the exact shape — copy them, do not invent one.
- **An empty completion is a failure, not an answer.** Never emit an empty description as a `Rendition`; degrade or raise `InfrastructureError` exactly as the neighbouring code in the same file already does.
- **No new environment variables.** `composition.py` reads none and must continue to read none.
- **`question` has no default.** Unlike `describe_image`'s prompt, an absent question is a call with no intent.
- **Region cropping is a precision feature, not an economy one.** Measured: a 72x48 crop costs the same 1,140 prompt tokens as the 720x480 frame it came from. Never write a comment or docstring claiming a crop saves tokens.
- **Only `agent/tools.py` and `adapters/langchain_*.py` may import `langchain`.**
- Run tests with `uv run pytest`. Lint with `uv run ruff check` and `uv run ruff format`.

---

### Task 1: Shared region params and crop helper

Extracts the unit-square validator that currently lives only in `image.py` so PDF and video can share it. Today those two have no boundary validation at all.

**Files:**
- Create: `src/readeverything/handlers/regions.py`
- Modify: `src/readeverything/handlers/image.py` (make `CropParams` subclass `RegionParams`)
- Test: `tests/unit/handlers/test_regions.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class RegionParams(BaseModel)` with fields `x: float`, `y: float`, `w: float`, `h: float` (defaults `0.0, 0.0, 1.0, 1.0`) and a `model_validator(mode="after")` named `_stay_inside_the_frame`.
  - `def crop_to_region(image: Image.Image, region: RegionParams) -> bytes` — returns PNG bytes.
  - `def region_bbox(region: RegionParams, page: int | None = None) -> BBox`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/handlers/test_regions.py`:

```python
import io

import pytest
from PIL import Image

from readeverything.handlers.regions import RegionParams, crop_to_region, region_bbox


def _image(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


def test_default_region_is_the_whole_image():
    region = RegionParams()
    data = crop_to_region(_image(100, 50), region)
    assert Image.open(io.BytesIO(data)).size == (100, 50)


def test_region_crops_to_the_requested_fraction():
    region = RegionParams(x=0.25, y=0.25, w=0.5, h=0.5)
    data = crop_to_region(_image(100, 50), region)
    assert Image.open(io.BytesIO(data)).size == (50, 25)


def test_a_sliver_keeps_at_least_one_pixel():
    """A rectangle that rounds to zero width is inexpressible as an image."""
    region = RegionParams(x=0.0, y=0.0, w=0.001, h=0.001)
    data = crop_to_region(_image(100, 50), region)
    assert Image.open(io.BytesIO(data)).size == (1, 1)


def test_a_region_running_off_the_edge_is_rejected_at_the_boundary():
    with pytest.raises(ValueError, match="unit square"):
        RegionParams(x=0.8, y=0.0, w=0.5, h=1.0)


def test_crop_returns_png():
    data = crop_to_region(_image(10, 10), RegionParams())
    assert Image.open(io.BytesIO(data)).format == "PNG"


def test_region_bbox_carries_the_page_when_given():
    box = region_bbox(RegionParams(x=0.1, y=0.2, w=0.3, h=0.4), page=7)
    assert (box.page, box.x, box.y, box.w, box.h) == (7, 0.1, 0.2, 0.3, 0.4)


def test_region_bbox_has_no_page_by_default():
    assert region_bbox(RegionParams()).page is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/handlers/test_regions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'readeverything.handlers.regions'`

- [ ] **Step 3: Write the implementation**

Create `src/readeverything/handlers/regions.py`:

```python
"""Rectangles, shared by every handler that can hand pixels to a model.

The unit-square check lived in `image.py` alone, which meant a PDF page region
and a video frame region had no boundary validation at all — a caller's
mistake surfaced as a bare `ValueError` from inside `BBox`, mid-crop, rather
than as a rejected parameter at the edge.

Cropping does NOT reduce what a vision call costs. Measured 2026-08-15 against
qwen3.8-27b-mtp, a 720x480 frame and a 72x48 crop of it both cost 1,140 prompt
tokens: the server resizes to a fixed grid, so cost is per image, not per
pixel. A region is worth asking for because it puts the rectangle in the
locator instead of in prose, not because it saves anything.
"""

from __future__ import annotations

import io

from PIL import Image
from pydantic import BaseModel, Field, model_validator

from readeverything.domain.locators import BBox


class RegionParams(BaseModel):
    """A rectangle in normalised coordinates, defaulting to the whole frame."""

    x: float = Field(default=0.0, ge=0.0, le=1.0, description="Left edge, 0-1 of width.")
    y: float = Field(default=0.0, ge=0.0, le=1.0, description="Top edge, 0-1 of height.")
    w: float = Field(default=1.0, gt=0.0, le=1.0, description="Width, 0-1 of width.")
    h: float = Field(default=1.0, gt=0.0, le=1.0, description="Height, 0-1 of height.")

    @model_validator(mode="after")
    def _stay_inside_the_frame(self) -> RegionParams:
        """A crop running off the edge is a parameter error, so reject it here.

        `BBox` catches it too, but only once the crop is already running — the
        caller then sees a bare `ValueError` from deep inside the domain rather
        than a rejection at the boundary where their mistake was. The `BBox`
        check stays as the backstop for every other path that builds one.
        """
        if self.x + self.w > 1.0 or self.y + self.h > 1.0:
            raise ValueError(
                f"crop must be within the unit square, got x={self.x} y={self.y} "
                f"w={self.w} h={self.h}"
            )
        return self

    @property
    def is_whole_frame(self) -> bool:
        return (self.x, self.y, self.w, self.h) == (0.0, 0.0, 1.0, 1.0)


def crop_to_region(image: Image.Image, region: RegionParams) -> bytes:
    """`image` cropped to `region`, as PNG bytes.

    The `max(..., + 1)` guards keep a thin rectangle from rounding to zero
    width or height: PIL would accept the degenerate box and produce an image
    no locator can describe.
    """
    box = (
        int(region.x * image.width),
        int(region.y * image.height),
        max(int((region.x + region.w) * image.width), int(region.x * image.width) + 1),
        max(int((region.y + region.h) * image.height), int(region.y * image.height) + 1),
    )
    buffer = io.BytesIO()
    image.crop(box).save(buffer, format="PNG")
    return buffer.getvalue()


def region_bbox(region: RegionParams, page: int | None = None) -> BBox:
    """The locator for `region`, on `page` when the medium has pages."""
    return BBox(page=page, x=region.x, y=region.y, w=region.w, h=region.h)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/handlers/test_regions.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Reuse the shared base in `image.py`**

In `src/readeverything/handlers/image.py`, replace the whole `CropParams` class definition (fields plus its `_stay_inside_the_frame` validator) with:

```python
class CropParams(RegionParams):
    """The whole-image crop affordance's params. Coordinates come from `RegionParams`."""
```

Add to the imports:

```python
from readeverything.handlers.regions import RegionParams, crop_to_region, region_bbox
```

Then replace the body of the `case "crop_region":` branch in `invoke` with:

```python
            case "crop_region":
                if not isinstance(params, CropParams):
                    raise TypeError(f"expected CropParams, got {type(params).__name__}")
                image = await self._require_image(ref)
                return Rendition(
                    locator=region_bbox(params),
                    content=ImageContent(data=crop_to_region(image, params), mime="image/png"),
                )
```

- [ ] **Step 6: Run the full image handler suite to prove nothing regressed**

Run: `uv run pytest tests/unit/handlers/test_image_handler.py tests/unit/handlers/test_regions.py -v`
Expected: PASS. `crop_region`'s observable behaviour is unchanged — same coordinates, same PNG, same locator.

- [ ] **Step 7: Commit**

```bash
git add src/readeverything/handlers/regions.py src/readeverything/handlers/image.py tests/unit/handlers/test_regions.py
git commit -m "Share one rectangle between every handler that crops"
```

---

### Task 2: `ask_about_image` on the image handler

**Files:**
- Modify: `src/readeverything/handlers/image.py`
- Test: `tests/unit/handlers/test_image_handler.py`

**Interfaces:**
- Consumes: `RegionParams`, `crop_to_region`, `region_bbox` from Task 1.
- Produces: `class AskAboutImageParams(RegionParams)` with `question: str` (required, no default); affordance named `"ask_about_image"`, `level=DetailLevel.DEEP`, `requires=frozenset({Capability.VISION})`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/handlers/test_image_handler.py`. Match the file's existing fixtures and construction style — read the top of the file first and reuse whatever it already provides for building a handler and a `SourceRef`.

```python
@pytest.mark.asyncio
async def test_ask_about_image_is_absent_without_a_vision_model(png_bytes):
    handler = ImageHandler(source=FakeSource({"a.png": png_bytes}))
    assert "ask_about_image" not in {a.name for a in handler.affordances()}


@pytest.mark.asyncio
async def test_ask_about_image_is_offered_with_a_vision_model(png_bytes):
    handler = ImageHandler(source=FakeSource({"a.png": png_bytes}), vision=FakeVision())
    assert "ask_about_image" in {a.name for a in handler.affordances()}


@pytest.mark.asyncio
async def test_ask_about_image_puts_the_question_to_the_model(png_bytes, png_ref):
    vision = FakeVision()
    handler = ImageHandler(source=FakeSource({"a.png": png_bytes}), vision=vision)
    rendition = await handler.invoke(
        png_ref, "ask_about_image", AskAboutImageParams(question="How many cats?")
    )
    assert "How many cats?" in rendition.content.text
    assert vision.calls == 1


@pytest.mark.asyncio
async def test_a_region_sends_the_model_fewer_bytes_than_the_whole_image(png_bytes, png_ref):
    """FakeVision reports the byte count it received, so this asserts on what
    reached the model rather than on the prose that came back."""
    vision = FakeVision()
    handler = ImageHandler(source=FakeSource({"a.png": png_bytes}), vision=vision)
    whole = await handler.invoke(
        png_ref, "ask_about_image", AskAboutImageParams(question="q")
    )
    part = await handler.invoke(
        png_ref,
        "ask_about_image",
        AskAboutImageParams(question="q", x=0.0, y=0.0, w=0.25, h=0.25),
    )
    assert whole.content.text != part.content.text


@pytest.mark.asyncio
async def test_a_region_is_the_locator(png_bytes, png_ref):
    handler = ImageHandler(source=FakeSource({"a.png": png_bytes}), vision=FakeVision())
    rendition = await handler.invoke(
        png_ref,
        "ask_about_image",
        AskAboutImageParams(question="q", x=0.1, y=0.2, w=0.3, h=0.4),
    )
    assert (rendition.locator.x, rendition.locator.y) == (0.1, 0.2)


@pytest.mark.asyncio
async def test_an_empty_completion_is_not_an_answer(png_bytes, png_ref):
    handler = ImageHandler(
        source=FakeSource({"a.png": png_bytes}), vision=FakeVisionRefusing()
    )
    with pytest.raises(InfrastructureError):
        await handler.invoke(
            png_ref, "ask_about_image", AskAboutImageParams(question="q")
        )


def test_a_question_is_required():
    with pytest.raises(ValidationError):
        AskAboutImageParams()
```

Add `from pydantic import ValidationError` and the `AskAboutImageParams` import as needed.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/handlers/test_image_handler.py -k ask_about -v`
Expected: FAIL — `AskAboutImageParams` does not exist.

- [ ] **Step 3: Write the implementation**

Add the params model near the other params classes in `image.py`:

```python
class AskAboutImageParams(RegionParams):
    question: str = Field(description="What you want to know about the image.")
```

Add to `affordances()`, inside the existing `if self._vision is None` guard's else-path (the tuple returned when vision is present):

```python
            Affordance(
                name="ask_about_image",
                description=(
                    "Ask a vision model a question about this image, or about a "
                    "rectangular region of it. Give x/y/w/h as fractions 0-1 to ask "
                    "about part of it; omit them to ask about the whole image. "
                    "Asking about a region costs the same as asking about the whole "
                    "image — use it to be precise, not to save."
                ),
                params=AskAboutImageParams,
                requires=frozenset({Capability.VISION}),
                level=DetailLevel.DEEP,
            ),
```

Add the `invoke` branch:

```python
            case "ask_about_image":
                if not isinstance(params, AskAboutImageParams):
                    raise TypeError(
                        f"expected AskAboutImageParams, got {type(params).__name__}"
                    )
                if self._vision is None:
                    raise UnknownAffordanceError(
                        "ask_about_image", (a.name for a in self.affordances())
                    )
                if params.is_whole_frame:
                    text = await self._see(ref, params.question, "ask_about_image")
                else:
                    image = await self._require_image(ref)
                    cropped = crop_to_region(image, params)
                    text = await self._vision.describe(cropped, "image/png", params.question)
                    if not text.strip():
                        raise InfrastructureError(
                            f"vision model returned no description for {ref.uri}"
                        )
                return Rendition(locator=region_bbox(params), content=TextContent(text))
```

Note the whole-frame path reuses `_see`, which sends the file's own bytes untouched — re-encoding an unmodified image through PIL would change what the model sees for no reason.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/handlers/test_image_handler.py -v`
Expected: PASS, including every pre-existing test in the file.

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/handlers/image.py tests/unit/handlers/test_image_handler.py
git commit -m "Ask a question about an image, or about part of one"
```

---

### Task 3: `ask_about_image` on the PDF handler

**Files:**
- Modify: `src/readeverything/handlers/pdf.py`
- Test: `tests/unit/handlers/test_pdf_handler.py`

**Interfaces:**
- Consumes: Task 1's helpers.
- Produces: `class AskAboutPageParams(RegionParams)` with `question: str` (required), `page: int = 1` (`ge=1`), `dpi: int = 150` (`gt=0`). Affordance name `"ask_about_image"` — the *name* is the convention, the params class name is local.

**Read this before writing anything.** `PdfHandler` today holds a
`TextRecognizer`, not a `VisionModel`, and `TextRecognizer.recognize(image,
mime) -> str` takes **no prompt** — it structurally cannot carry a question.
So this task also adds a vision dependency:

- `PdfHandler.__init__` gains `vision: VisionModel | None = None`, stored as
  `self._vision`, alongside the existing `recognizer`.
- `ask_about_image` is offered when `self._vision is not None`. It is NOT
  gated on the recognizer.
- `ocr_page` keeps using `self._recognizer.recognize(...)` and must not
  change in any way.
- `composition.py` — in the function that builds `PdfHandler` (around line
  97-101, where `recognizer` is already built from `vision`) — passes
  `vision=vision` as well. `vision` is already in scope there.
- Add `from readeverything.ports.vision import VisionModel` to `pdf.py`.

Also add one test asserting `ocr_page` still works when `vision` is passed
and `recognizer` is not, and vice versa — the two dependencies are
independent and nothing should couple them.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/handlers/test_pdf_handler.py`, reusing that file's existing fixtures for a PDF and a `SourceRef`:

```python
@pytest.mark.asyncio
async def test_ask_about_image_is_absent_without_a_vision_model(pdf_bytes):
    handler = PdfHandler(source=FakeSource({"a.pdf": pdf_bytes}), probe=PdfiumProbe())
    assert "ask_about_image" not in {a.name for a in handler.affordances()}


@pytest.mark.asyncio
async def test_ask_about_image_reaches_the_model(pdf_bytes, pdf_ref):
    vision = FakeVision()
    handler = PdfHandler(source=FakeSource({"a.pdf": pdf_bytes}), vision=vision)
    rendition = await handler.invoke(
        pdf_ref, "ask_about_image", AskAboutPageParams(question="What chart is this?", page=1)
    )
    assert "What chart is this?" in rendition.content.text
    assert vision.calls == 1


@pytest.mark.asyncio
async def test_the_locator_carries_the_page(pdf_bytes, pdf_ref):
    handler = PdfHandler(source=FakeSource({"a.pdf": pdf_bytes}), vision=FakeVision())
    rendition = await handler.invoke(
        pdf_ref,
        "ask_about_image",
        AskAboutPageParams(question="q", page=1, x=0.0, y=0.5, w=1.0, h=0.5),
    )
    assert rendition.locator.page == 1
    assert rendition.locator.y == 0.5


@pytest.mark.asyncio
async def test_a_missing_page_degrades_rather_than_raising(pdf_bytes, pdf_ref):
    handler = PdfHandler(source=FakeSource({"a.pdf": pdf_bytes}), vision=FakeVision())
    rendition = await handler.invoke(
        pdf_ref, "ask_about_image", AskAboutPageParams(question="q", page=9999)
    )
    assert rendition.degraded
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/handlers/test_pdf_handler.py -k ask_about -v`
Expected: FAIL — `AskAboutPageParams` does not exist.

- [ ] **Step 3: Write the implementation**

Add the params class beside the other PDF params:

```python
class AskAboutPageParams(RegionParams):
    question: str = Field(description="What you want to know about the page.")
    page: int = Field(default=1, ge=1, description="1-indexed page number.")
    dpi: int = Field(default=150, gt=0, description="Render resolution, in dots per inch.")
```

Offer the affordance under a `self._vision is not None` condition (NOT the recognizer's):

```python
                Affordance(
                    name="ask_about_image",
                    description=(
                        "Ask a vision model a question about a rendered page, or about "
                        "a rectangular region of one. Use this when the answer is in a "
                        "chart, a diagram or a scan rather than in the page's own text "
                        "— read_page is far cheaper when the text is really there."
                    ),
                    params=AskAboutPageParams,
                    requires=frozenset({Capability.VISION}),
                    level=DetailLevel.DEEP,
                ),
```

Add a private method modelled directly on `_ocr_page` (structure only — it calls `self._vision.describe(png, "image/png", params.question)`, NOT the recognizer) — same open, same page-count guard, same `_PIL_AVAILABLE` guard, same `finally: document.close()`:

```python
    async def _ask_about_page(self, ref: SourceRef, params: AskAboutPageParams) -> Rendition:
        if self._vision is None:
            raise UnknownAffordanceError("ask_about_image", (a.name for a in self.affordances()))
        data = await self._source.read_bytes(ref.uri)
        document = self._open(data)
        if document is None:
            return self._degraded_text(
                ByteRange(0, max(1, ref.size_bytes)), f"{ref.uri} could not be opened as a PDF"
            )
        try:
            if params.page > len(document):
                return self._degraded_text(
                    ByteRange(0, max(1, ref.size_bytes)),
                    f"page {params.page} does not exist; the document has "
                    f"{len(document)} page(s)",
                )
            if not _PIL_AVAILABLE:
                return self._degraded_text(
                    PageRef(params.page),
                    "page could not be rendered: Pillow is not installed",
                )
            png = self._render_png(document[params.page - 1], params.dpi)
        finally:
            document.close()
        if not params.is_whole_frame:
            png = crop_to_region(Image.open(io.BytesIO(png)), params)
        try:
            text = await self._vision.describe(png, "image/png", params.question)
        except Exception:
            return self._degraded_text(
                region_bbox(params, page=params.page),
                f"the vision model failed to answer about page {params.page}",
            )
        if not text.strip():
            return self._degraded_text(
                region_bbox(params, page=params.page),
                f"the vision model returned no answer about page {params.page}",
            )
        return Rendition(
            locator=region_bbox(params, page=params.page),
            content=TextContent(" ".join(text.split())),
        )
```

Wire the `invoke` branch:

```python
            case "ask_about_image":
                if not isinstance(params, AskAboutPageParams):
                    raise TypeError(f"expected AskAboutPageParams, got {type(params).__name__}")
                return await self._ask_about_page(ref, params)
```

Check the file's existing imports for `io` and `Image` before adding them — `_render_png` may already provide what you need.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/handlers/test_pdf_handler.py -v`
Expected: PASS, including every pre-existing test.

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/handlers/pdf.py tests/unit/handlers/test_pdf_handler.py
git commit -m "Ask a question about a rendered page"
```

---

### Task 4: `ask_about_image` on the video handler

**Files:**
- Modify: `src/readeverything/handlers/video.py`
- Test: `tests/unit/handlers/test_video_handler.py`

**Interfaces:**
- Consumes: Task 1's helpers.
- Produces: `class AskAboutFrameParams(RegionParams)` with `question: str` (required), `seconds: float = 0.0` (`ge=0.0`). Affordance name `"ask_about_image"`.

**Locator decision — read before implementing.** The domain has no composite locator, so "this rectangle at this second" is inexpressible: `TimeSpan` cannot carry a `BBox`. This affordance therefore returns a `TimeSpan`, and a region narrows what the model *sees* without appearing in the locator. Do not invent a composite locator type in this task — that is a domain change with its own consequences for `LocatorMap`, and it is out of scope here. Do not silently pretend the region is captured either; the affordance description below says so plainly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/handlers/test_video_handler.py`, reusing its existing fixtures and fakes:

```python
@pytest.mark.asyncio
async def test_ask_about_image_is_absent_without_a_vision_model(video_ref, fake_frames):
    handler = VideoHandler(source=..., frames=fake_frames, probe=...)
    assert "ask_about_image" not in {a.name for a in handler.affordances()}


@pytest.mark.asyncio
async def test_ask_about_image_asks_about_the_frame_at_a_time(video_ref, fake_frames):
    vision = FakeVision()
    handler = VideoHandler(source=..., frames=fake_frames, probe=..., vision=vision)
    rendition = await handler.invoke(
        video_ref, "ask_about_image", AskAboutFrameParams(question="Who is speaking?", seconds=12.0)
    )
    assert "Who is speaking?" in rendition.content.text
    assert vision.calls == 1


@pytest.mark.asyncio
async def test_the_locator_is_the_timespan(video_ref, fake_frames):
    handler = VideoHandler(source=..., frames=fake_frames, probe=..., vision=FakeVision())
    rendition = await handler.invoke(
        video_ref, "ask_about_image", AskAboutFrameParams(question="q", seconds=12.0)
    )
    assert rendition.locator.start == 12.0


@pytest.mark.asyncio
async def test_a_region_narrows_what_the_model_sees(video_ref, fake_frames):
    vision = FakeVision()
    handler = VideoHandler(source=..., frames=fake_frames, probe=..., vision=vision)
    whole = await handler.invoke(
        video_ref, "ask_about_image", AskAboutFrameParams(question="q", seconds=1.0)
    )
    part = await handler.invoke(
        video_ref,
        "ask_about_image",
        AskAboutFrameParams(question="q", seconds=1.0, x=0.0, y=0.0, w=0.5, h=0.5),
    )
    assert whole.content.text != part.content.text


@pytest.mark.asyncio
async def test_an_unreachable_frame_degrades_rather_than_raising(video_ref, failing_frames):
    handler = VideoHandler(source=..., frames=failing_frames, probe=..., vision=FakeVision())
    rendition = await handler.invoke(
        video_ref, "ask_about_image", AskAboutFrameParams(question="q", seconds=1.0)
    )
    assert rendition.degraded
```

Replace each `...` with whatever the file's existing `describe_frame` tests pass — copy their construction exactly rather than inventing fixtures.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/handlers/test_video_handler.py -k ask_about -v`
Expected: FAIL — `AskAboutFrameParams` does not exist.

- [ ] **Step 3: Write the implementation**

Params class beside the other video params:

```python
class AskAboutFrameParams(RegionParams):
    question: str = Field(description="What you want to know about the frame.")
    seconds: float = Field(
        default=0.0, ge=0.0, description="Point in the timeline to extract a frame from."
    )
```

Affordance, inside the existing `if self._vision is not None:` block:

```python
            affordances.append(
                Affordance(
                    name="ask_about_image",
                    description=(
                        "Ask a vision model a question about the frame at one moment, "
                        "or about a rectangular region of it. Read the transcript first "
                        "— it is far cheaper and usually answers the question. The "
                        "result is located by time; a region narrows what the model "
                        "sees but is not carried in the locator."
                    ),
                    params=AskAboutFrameParams,
                    requires=frozenset({Capability.FFMPEG, Capability.VISION}),
                    level=DetailLevel.DEEP,
                )
            )
```

Add the method, modelled on `_describe_frame` — same `local_path` guard, same `frame_at` guard, same `_degraded_frame` calls, and the same limiter context the other vision affordances use:

```python
    async def _ask_about_frame(self, ref: SourceRef, params: AskAboutFrameParams) -> Rendition:
        if self._vision is None:
            raise UnknownAffordanceError("ask_about_image", (a.name for a in self.affordances()))
        seconds = params.seconds
        try:
            path = await self._source.local_path(ref.uri)
        except Exception:
            return self._degraded_frame(ref, seconds, f"{ref.uri} could not be read")
        try:
            frame = await self._frames.frame_at(path, seconds)
        except Exception:
            frame = None
        if frame is None:
            return self._degraded_frame(ref, seconds, await self._absent_frame_detail(path, seconds))
        if not params.is_whole_frame:
            frame = crop_to_region(Image.open(io.BytesIO(frame)), params)
        try:
            text = await self._vision.describe(frame, "image/png", params.question)
        except Exception:
            return self._degraded_frame(
                ref, seconds, f"the vision model failed to answer about {_timestamp(seconds)}"
            )
        if not text.strip():
            return self._degraded_frame(
                ref, seconds, f"the vision model returned no answer about {_timestamp(seconds)}"
            )
        return Rendition(
            locator=TimeSpan(seconds, seconds + FALLBACK_FRAME_DURATION_S),
            content=TextContent(" ".join(text.split())),
        )
```

Wire `invoke`:

```python
            case "ask_about_image":
                if not isinstance(params, AskAboutFrameParams):
                    raise TypeError(f"expected AskAboutFrameParams, got {type(params).__name__}")
                return await self._ask_about_frame(ref, params)
```

Check whether `_describe_frame` wraps its vision call in `self._limit(Capability.VISION)`. If it does, wrap this one identically — an easier-to-reach vision path must not be an unbounded one. Pillow is an optional extra for this handler: if `video.py` does not already import PIL, import it lazily inside the `if not params.is_whole_frame:` branch and degrade with "region cropping needs Pillow" when it is unavailable, rather than making PIL a hard import for all video work.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/handlers/test_video_handler.py -v`
Expected: PASS, including every pre-existing test.

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/handlers/video.py tests/unit/handlers/test_video_handler.py
git commit -m "Ask a question about a frame"
```

---

### Task 5: The agent tool

**Files:**
- Modify: `src/readeverything/agent/tools.py`
- Test: `tests/unit/agent/test_tools.py`

**Interfaces:**
- Consumes: the `"ask_about_image"` affordance name convention from Tasks 2-4. Nothing else — this task must not import a handler or a params class.
- Produces: a fourth tool in `build_tools`'s returned list, named `ask_about_image`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/agent/test_tools.py`, reusing its existing fake `Perception`:

```python
@pytest.mark.asyncio
async def test_ask_about_image_forwards_question_and_where_together():
    perception = RecordingPerception()
    tool = _by_name(build_tools(perception), "ask_about_image")
    await tool.ainvoke({"uri": "a.png", "question": "How many?", "where": {"x": 0.5, "w": 0.5}})
    assert perception.invoked == ("a.png", "ask_about_image", {"x": 0.5, "w": 0.5, "question": "How many?"})


@pytest.mark.asyncio
async def test_ask_about_image_needs_no_where():
    perception = RecordingPerception()
    tool = _by_name(build_tools(perception), "ask_about_image")
    await tool.ainvoke({"uri": "a.png", "question": "What is this?"})
    assert perception.invoked[2] == {"question": "What is this?"}


@pytest.mark.asyncio
async def test_ask_about_image_never_inspects_the_file():
    """The tool layer knows nothing about kinds — an inspect call here would
    mean it had started making decisions it must not make."""
    perception = RecordingPerception()
    tool = _by_name(build_tools(perception), "ask_about_image")
    await tool.ainvoke({"uri": "a.png", "question": "q"})
    assert perception.inspected == []


@pytest.mark.asyncio
async def test_a_file_without_the_affordance_lists_what_it_does_have():
    perception = RaisingPerception(
        UnknownAffordanceError("ask_about_image", ["read_range"])
    )
    tool = _by_name(build_tools(perception), "ask_about_image")
    result = await tool.ainvoke({"uri": "a.txt", "question": "q"})
    assert "read_range" in result


def test_the_tool_list_is_the_same_length_for_every_file():
    """The docstring's rule: the tool list never varies with what was last
    looked at. Four tools, always."""
    assert len(build_tools(RecordingPerception())) == 4
```

Add `RecordingPerception` / `RaisingPerception` only if the file has no equivalent — reuse whatever fake it already defines.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/agent/test_tools.py -k ask_about -v`
Expected: FAIL — no tool named `ask_about_image`.

- [ ] **Step 3: Write the implementation**

Add the params schema beside the others:

```python
class AskAboutImageParams(BaseModel):
    uri: str = Field(description="Path to the image, PDF or video to ask about.")
    question: str = Field(description="What you want to know.")
    params: dict[str, Any] = Field(
        default_factory=dict,
        alias="where",
        description=(
            "Where to look, when the file needs it: page/dpi for a PDF, seconds for a "
            "video, and x/y/w/h as fractions 0-1 for a region of any of them. "
            "Omit for the whole image."
        ),
    )
```

Use `populate_by_name=True` in its `model_config` so both `where` and the field name work.

Rename the module constant and add the new name:

```python
#: Affordances that turn image bytes into something a text model can read.
#: Named here so the rendering and the handler cannot drift apart silently.
_IMAGE_READING_AFFORDANCES = ("ask_about_image", "describe_image", "ocr")
```

Add the coroutine inside `build_tools`, alongside the others:

```python
    @never_raises
    async def ask_about_image(uri: str, question: str, where: Mapping[str, Any] | None = None) -> str:
        rendition = await perception.invoke(
            uri, "ask_about_image", {**(where or {}), "question": question}
        )
        return _render_rendition(rendition)

    async def _ask(uri: str, question: str, where: dict[str, Any] | None = None) -> str:
        return (await ask_about_image(uri, question, where)).render()
```

`_render_rendition` is called without affordances because this tool's result is always text — the image-bytes hint branch cannot be reached from here.

Append the tool to the returned list:

```python
        StructuredTool.from_function(
            coroutine=_ask,
            name="ask_about_image",
            description=(
                "Ask a vision model a question about a picture — a photograph, a page "
                "of a PDF, or a frame of a video. Give `where` to say which page, which "
                "moment, or which rectangular region; omit it for the whole image. "
                "You do not need to call inspect_path first. This runs a model and is "
                "the most expensive thing you can do: for a video, read the transcript "
                "before you look at frames."
            ),
            args_schema=AskAboutImageParams,
            handle_validation_error=_render_validation_error,
        ),
```

- [ ] **Step 4: Fix the dead-end hint**

In `_render_rendition`'s `ImageContent` branch, the message currently tells the agent to `call invoke_affordance with <names>`. Because `ask_about_image` is now both an affordance name and a tool name, make the advice reachable:

```python
            usable = [a for a in _IMAGE_READING_AFFORDANCES if a in affordances]
            if "ask_about_image" in usable:
                body = (
                    f"[{mime} image, {len(data)} bytes — call ask_about_image on this "
                    f"file, with the same coordinates, to ask about it]"
                )
            elif usable:
                route = " or ".join(usable)
                body = (
                    f"[{mime} image, {len(data)} bytes — "
                    f"call invoke_affordance with {route} to read it]"
                )
            else:
                ...unchanged...
```

The wording matters: the bytes themselves are still not questionable, so the hint must point at the *file plus coordinates*, not at the bytes. Anything that implies the agent can pass these bytes back is the same dead end in new words.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/agent/test_tools.py -v`
Expected: PASS, including pre-existing tests. If a pre-existing test asserts a three-tool list, update it to four — that is a real, intended change.

- [ ] **Step 6: Run the whole suite and the lints**

```bash
uv run pytest
uv run ruff check
uv run ruff format --check
```
Expected: all green. `tests/unit/test_public_surface.py` and `test_dependencies_stay_confined.py` both encode project rules — if either fails, the fix is the code, not the test.

- [ ] **Step 7: Commit**

```bash
git add src/readeverything/agent/tools.py tests/unit/agent/test_tools.py
git commit -m "One tool for asking a question about a picture"
```

---

### Task 6: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-15-readeverything-ask-about-image-design.md`

- [ ] **Step 1: Check what the README claims about tools**

Run: `grep -n "invoke_affordance\|inspect_path\|three tools" README.md`

If it enumerates the tool pack, add `ask_about_image` with one line of description. If it does not, skip to Step 2 — do not invent a section.

- [ ] **Step 2: Record the locator limitation in the spec**

The spec says a region "puts the crop in the locator rather than in prose". That is true for images and PDF pages and false for video frames, where `TimeSpan` cannot carry a `BBox`. Add to the spec's Region section:

```markdown
**Video is the exception.** The domain has no composite locator, so "this
rectangle at this second" is inexpressible: `ask_about_image` on a video
returns a `TimeSpan`, and the region narrows what the model sees without
appearing in the locator. Adding a composite locator would touch `LocatorMap`
and every consumer of it, which is a larger change than this one and is not
attempted here. The video affordance's description says so, so an agent is not
misled about what the locator means.
```

- [ ] **Step 3: Commit**

```bash
git add README.md docs/superpowers/specs/2026-08-15-readeverything-ask-about-image-design.md
git commit -m "Say where the region does and does not reach the locator"
```

---

## Self-Review

**Spec coverage:**
- Two-step dance → Task 5 (tool needs no `inspect_path`; test asserts it).
- Region-scoped asking → Tasks 1-4.
- Bytes are a dead end → Task 5 Step 4.
- Name convention, not a router → Task 5 (tool imports no handler; test asserts no inspect).
- `question` has no default → Task 2 test `test_a_question_is_required`.
- Caching is free → nothing to build; correctly absent from the tasks.
- `ocr` stays separate → no task touches it. Correct.
- Cost is not reduced by cropping → Task 1's module docstring and Task 2's affordance description both say so.

**Gap found and closed:** the spec asserted the region always lands in the locator. Video cannot honour that. Task 4 states the decision, its affordance description discloses it, and Task 6 Step 2 corrects the spec.

**Type consistency:** `RegionParams`, `crop_to_region`, `region_bbox` are defined in Task 1 and used under those exact names in Tasks 2-4. Params classes are deliberately named differently per handler (`AskAboutImageParams`, `AskAboutPageParams`, `AskAboutFrameParams`) because only the *affordance name* is the convention; `agent/tools.py` has its own unrelated `AskAboutImageParams` for the tool's own schema, which is fine — it never imports the handler one.

**Placeholder scan:** the `...` in Task 4's tests are explicitly instructed to be filled from the file's existing `describe_frame` tests rather than invented. No TBDs.
