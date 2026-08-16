"""The shared `page_image` affordance, and what it says about itself.

Three handlers publish this and none of them owns it: a deck, a Word document
and a spreadsheet all become page images the same way, and three copies would
be three places for the honesty in the degradation to drift.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from readeverything.domain.capability import Capability
from readeverything.domain.errors import RenditionFailedError
from readeverything.domain.identity import ContentHash, MimeType, SourceRef
from readeverything.domain.locators import ByteRange, PageRef
from readeverything.domain.rendition import ImageContent, TextContent
from readeverything.handlers.page_images import (
    PageImageParams,
    page_image_affordance,
    render_page_image,
)


class _Renderer:
    revision = "fake/1"

    def __init__(self, *, fails: bool = False) -> None:
        self.calls: list[tuple[str, int, int]] = []
        self._fails = fails

    def claims(self, mime: MimeType) -> bool:
        return True

    async def page_count(self, path: str) -> int:
        return 2

    async def page_text(self, path: str, page: int) -> str:
        return f"page {page}"

    async def render_page(self, path: str, page: int, *, dpi: int = 150) -> bytes:
        self.calls.append((path, page, dpi))
        if self._fails:
            raise RenditionFailedError("the converter said no")
        return b"\x89PNG fake"


class _Source:
    """`local_path` is the only method under test; the rest satisfy the port."""

    def __init__(self, path: str) -> None:
        self._path = path

    async def read_bytes(self, uri: str) -> bytes:  # pragma: no cover - unused here
        return b""

    async def read_range(self, uri: str, start: int, end: int) -> bytes:  # pragma: no cover
        return b""

    def stream(self, uri: str, *, chunk_size: int = 1 << 20) -> AsyncIterator[bytes]:
        raise NotImplementedError  # pragma: no cover

    async def local_path(self, uri: str) -> str:
        return self._path


def _ref() -> SourceRef:
    return SourceRef(
        uri="deck.pptx",
        mime=MimeType.parse("application/vnd.ms-powerpoint"),
        size_bytes=100,
        content_hash=ContentHash("0" * 64),
    )


# --- the affordance declaration -------------------------------------------


def test_the_affordance_requires_document_render() -> None:
    """Without this, the registry would publish `page_image` on a machine with
    no converter and the tool would exist and apologise."""
    assert page_image_affordance("slide").requires == frozenset({Capability.DOCUMENT_RENDER})


def test_the_affordance_is_deep_and_never_runs_during_a_card() -> None:
    """A four-hundred-slide deck must not convert itself because someone
    listed a directory. Spec 4 settled this for OCR; the reasoning is the same.
    """
    assert page_image_affordance("slide").level.value == "deep"


def test_the_affordance_names_the_unit_the_format_actually_has() -> None:
    """ "Render slide 4" and "render page 4" are different requests to a reader,
    and a deck has slides."""
    assert "slide" in page_image_affordance("slide").description
    assert "page" in page_image_affordance("page").description


# --- rendering ------------------------------------------------------------


async def test_a_rendered_page_comes_back_as_image_content(tmp_path: Path) -> None:
    renderer = _Renderer()
    rendition = await render_page_image(
        renderer=renderer,
        source=_Source(str(tmp_path / "deck.pptx")),
        ref=_ref(),
        params=PageImageParams(page=2, dpi=120),
        unit="slide",
    )

    assert isinstance(rendition.content, ImageContent)
    assert rendition.content.mime == "image/png"
    assert rendition.locator == PageRef(2)


async def test_the_renderer_is_given_a_real_path_not_a_uri(tmp_path: Path) -> None:
    """A converter is an external process and takes a path. `local_path` is
    also what makes rendering a deck INSIDE a tarball work, because
    `NestedSource` materialises the member there."""
    path = str(tmp_path / "materialised.pptx")
    renderer = _Renderer()
    await render_page_image(
        renderer=renderer,
        source=_Source(path),
        ref=_ref(),
        params=PageImageParams(page=1, dpi=150),
        unit="slide",
    )

    assert renderer.calls == [(path, 1, 150)]


async def test_a_rendering_says_it_is_a_rendering(tmp_path: Path) -> None:
    """The honesty requirement, and it is not decoration.

    LibreOffice's rendering of a PowerPoint is a rendering: fonts substitute
    and layout engines differ. An agent reading type off a slide and reporting
    the font would otherwise be reporting LibreOffice's substitute as the
    author's choice.
    """
    rendition = await render_page_image(
        renderer=_Renderer(),
        source=_Source(str(tmp_path / "deck.pptx")),
        ref=_ref(),
        params=PageImageParams(page=1),
        unit="slide",
    )

    (note,) = rendition.degradations
    assert "rendering" in note.detail
    assert "font" in note.detail
    assert "fake/1" in note.detail, "the note names what produced it"


async def test_a_rendering_is_not_marked_degraded(tmp_path: Path) -> None:
    """It is a good image. `degraded` would tell an agent to discount it, and
    that is a different claim from "this came from a converter"."""
    rendition = await render_page_image(
        renderer=_Renderer(),
        source=_Source(str(tmp_path / "deck.pptx")),
        ref=_ref(),
        params=PageImageParams(page=1),
        unit="slide",
    )
    assert not rendition.degraded


async def test_a_converter_failure_degrades_rather_than_raising(tmp_path: Path) -> None:
    """A handler never raises about its input. An agent guessing a slide number
    past the end gets something it can read and correct."""
    rendition = await render_page_image(
        renderer=_Renderer(fails=True),
        source=_Source(str(tmp_path / "deck.pptx")),
        ref=_ref(),
        params=PageImageParams(page=99),
        unit="slide",
    )

    assert rendition.degraded
    assert isinstance(rendition.content, TextContent)
    assert "slide 99" in rendition.content.text
    assert rendition.locator == ByteRange(0, 100)


async def test_a_source_that_cannot_be_materialised_degrades(tmp_path: Path) -> None:
    class _Unmaterialisable(_Source):
        async def local_path(self, uri: str) -> str:
            raise OSError("no such file")

    rendition = await render_page_image(
        renderer=_Renderer(),
        source=_Unmaterialisable("unused"),
        ref=_ref(),
        params=PageImageParams(page=1),
        unit="slide",
    )
    assert rendition.degraded


def test_the_params_reject_a_zero_page() -> None:
    """1-indexed as a reader counts, and the schema is what tells a model so."""
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        PageImageParams(page=0)
