"""`page_image`, shared by the three office handlers.

A deck, a Word document and a spreadsheet all become page images the same way:
hand the converter a real path, get a PNG back, and say plainly that what came
back is a *rendering*. Three copies of that would be three places for the
honesty to drift, which is the failure mode that matters here — the image is
the easy part.

`handlers/regions.py` is the precedent: shared handler-level machinery lives in
its own module rather than in whichever handler happened to need it first.

Every affordance built here is `DEEP` and never runs during `inspect` or
`represent`. Spec 4 §11 settled that argument for OCR and the reasoning is
identical: a four-hundred-slide deck must not convert itself because someone
listed a directory.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from readeverything.domain.affordance import Affordance, DetailLevel
from readeverything.domain.capability import Capability
from readeverything.domain.identity import SourceRef
from readeverything.domain.locators import ByteRange, PageRef
from readeverything.domain.rendition import (
    Degradation,
    ImageContent,
    Rendition,
    TextContent,
)
from readeverything.ports.rendering import DocumentRenderer
from readeverything.ports.source import SourceReader


class PageImageParams(BaseModel):
    page: int = Field(default=1, ge=1, description="1-indexed page number to render.")
    dpi: int = Field(default=150, gt=0, description="Render resolution, in dots per inch.")


def page_image_affordance(unit: str) -> Affordance:
    """The declaration, worded for the unit the format actually has.

    "Render slide 4" and "render page 4" are different requests to a reader,
    and a deck has slides. The parameter is still called `page` — one schema
    across every handler is worth more to a model than three near-identical
    ones — but what it counts is named in the description.
    """
    return Affordance(
        name="page_image",
        description=(
            f"Render one {unit} as a PNG image, for a vision tool to read. Use this "
            f"when the answer is in the arrangement, a chart or a diagram rather "
            f"than in the text — reading the text is far cheaper when the answer "
            f"is really there. The image is produced by converting the document, "
            f"so it is a faithful rendering rather than the original file."
        ),
        params=PageImageParams,
        requires=frozenset({Capability.DOCUMENT_RENDER}),
        level=DetailLevel.DEEP,
    )


def rendering_provenance(renderer: DocumentRenderer) -> Degradation:
    """What this image is, said out loud.

    A converter's output is a *rendering*: fonts substitute when the document's
    are not installed, and layout engines differ. An agent reading type off a
    slide and reporting the font would otherwise be reporting the converter's
    substitute as the author's choice. Named rather than described generically,
    because "some converter" is not something a reader can weigh.
    """
    return Degradation(
        what="rendered by a converter",
        detail=(
            f"this image is {renderer.revision}'s rendering of the document rather "
            f"than the document itself; fonts may have been substituted and layout "
            f"may differ from the original application's"
        ),
    )


async def render_page_image(
    *,
    renderer: DocumentRenderer,
    source: SourceReader,
    ref: SourceRef,
    params: PageImageParams,
    unit: str,
) -> Rendition:
    """One page as a PNG, or a readable explanation of why not.

    Never raises. A handler degrades rather than raising about its input, so an
    agent guessing a slide number past the end of a deck gets something it can
    read and correct rather than a traceback.

    `local_path` rather than `read_bytes`, because a converter is an external
    process and external processes take paths. That is also, at no extra cost,
    what makes rendering a deck INSIDE a tarball work: `NestedSource.local_path`
    materialises the member, and the converter never learns it was nested.
    """
    try:
        path = await source.local_path(ref.uri)
    except Exception:
        return _degraded(ref, f"{ref.uri} could not be made available to the converter as a file")
    try:
        png = await renderer.render_page(path, params.page, dpi=params.dpi)
    except Exception:
        return _degraded(ref, f"{unit} {params.page} could not be rendered")
    return Rendition(
        locator=PageRef(params.page),
        content=ImageContent(data=png, mime="image/png"),
        degradations=(rendering_provenance(renderer),),
    )


def _degraded(ref: SourceRef, detail: str) -> Rendition:
    """Located by `ByteRange`, not `PageRef`.

    No page was ever rendered, and claiming one would point a citation at
    something this handler never saw.
    """
    return Rendition(
        locator=ByteRange(0, max(1, ref.size_bytes)),
        content=TextContent(detail),
        degraded=True,
    )
