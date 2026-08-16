"""Capability negotiation for `page_image`, across all three office handlers.

This file matters more than the rendering does. The design's first promise is
that on a machine with no converter, everything works exactly as it did and no
rendering affordance appears anywhere — no tool that exists and returns an
apology. That is a property of the *absence* of a capability, so it cannot be
demonstrated by anything that renders successfully.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from readeverything.adapters.local_source import LocalFileSource
from readeverything.adapters.ooxml import SHEETS_MIME, SLIDES_MIME, WORD_MIME
from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.domain.errors import RenditionFailedError
from readeverything.domain.identity import ContentHash, MimeType, SourceRef
from readeverything.domain.locators import PageRef
from readeverything.domain.rendition import ImageContent, TextContent
from readeverything.handlers.office_sheets import OfficeSheetsHandler
from readeverything.handlers.office_slides import OfficeSlidesHandler
from readeverything.handlers.office_word import OfficeWordHandler
from readeverything.handlers.page_images import PageImageParams
from readeverything.ports.handler import MediaHandler
from readeverything.ports.rendering import DocumentRenderer
from readeverything.registry.registry import MimeTypeRegistry
from readeverything.testing.fakes import FakeSource, FakeVision, FakeVisionRefusing
from tests.fixtures_office import docx_bytes, pptx_bytes, xlsx_bytes


class _Renderer:
    revision = "fake/1"

    def claims(self, mime: MimeType) -> bool:
        return True

    async def page_count(self, path: str) -> int:
        return 3

    async def page_text(self, path: str, page: int) -> str:
        return f"page {page}"

    async def render_page(self, path: str, page: int, *, dpi: int = 150) -> bytes:
        return b"\x89PNG fake " + str(page).encode()


def _names(handler: MediaHandler) -> set[str]:
    return {a.name for a in handler.affordances()}


def _visible(handler: MediaHandler, capabilities: CapabilitySet) -> set[str]:
    registry = MimeTypeRegistry(handlers=[handler], capabilities=capabilities)
    return {a.name for a in registry.available_affordances(handler)}


WITH_RENDERING = CapabilitySet.of({Capability.DOCUMENT_RENDER: "soffice 25.8"})
WITHOUT_ANYTHING = CapabilitySet.empty()


def _handlers(renderer: DocumentRenderer | None) -> list[MediaHandler]:
    source = FakeSource({})
    return [
        OfficeWordHandler(source=source, renderer=renderer),
        OfficeSlidesHandler(source=source, renderer=renderer),
        OfficeSheetsHandler(source=source, renderer=renderer),
    ]


# --- the promise ----------------------------------------------------------


@pytest.mark.parametrize("index", [0, 1, 2])
def test_with_no_converter_wired_no_rendering_affordance_is_declared(index: int) -> None:
    """The handler does not even offer it. Nothing downstream has to filter."""
    assert "page_image" not in _names(_handlers(renderer=None)[index])


@pytest.mark.parametrize("index", [0, 1, 2])
def test_with_a_converter_wired_but_the_capability_absent_it_is_not_published(
    index: int,
) -> None:
    """The second, independent line of defence, and the one that actually
    protects the promise.

    A composition wires the converter unconditionally — exactly as it wires
    ffmpeg, because a binary is not an import there is anything to guard — and
    lets capability negotiation decide. So a machine with no soffice reaches
    HERE with a renderer object in hand, and the affordance's own `requires`
    is the only thing standing between an agent and a tool that exists and
    apologises.
    """
    handler = _handlers(renderer=_Renderer())[index]
    assert "page_image" in _names(handler), "declared..."
    assert "page_image" not in _visible(handler, WITHOUT_ANYTHING), "...but not published"


@pytest.mark.parametrize("index", [0, 1, 2])
def test_with_the_capability_present_it_is_published(index: int) -> None:
    handler = _handlers(renderer=_Renderer())[index]
    assert "page_image" in _visible(handler, WITH_RENDERING)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_everything_that_worked_without_a_converter_still_works_with_one(
    index: int,
) -> None:
    """ "Change nothing else" means nothing else changed. `page_image` is the
    only difference between the two handlers."""
    without = _names(_handlers(renderer=None)[index])
    with_it = _names(_handlers(renderer=_Renderer())[index])
    assert with_it - without == {"page_image"}


def test_the_slide_handler_does_not_confuse_vision_with_rendering() -> None:
    """Two separate capabilities. A vision model with no converter can still
    describe an embedded picture, and a converter with no vision model can
    still hand back a slide image for something else to read."""
    source = FakeSource({})
    vision_only = OfficeSlidesHandler(source=source, vision=FakeVision())
    render_only = OfficeSlidesHandler(source=source, renderer=_Renderer())

    assert "describe_slide_image" in _names(vision_only)
    assert "page_image" not in _names(vision_only)
    assert "page_image" in _names(render_only)
    assert "describe_slide_image" not in _names(render_only)


def test_rendering_is_deep_so_a_directory_listing_never_converts_anything() -> None:
    for handler in _handlers(renderer=_Renderer()):
        affordance = next(a for a in handler.affordances() if a.name == "page_image")
        assert affordance.level.value == "deep"


# --- it actually renders --------------------------------------------------


@pytest.mark.parametrize(
    ("index", "name", "build", "mime"),
    [
        (0, "doc.docx", docx_bytes, WORD_MIME),
        (1, "deck.pptx", pptx_bytes, SLIDES_MIME),
        (2, "book.xlsx", xlsx_bytes, SHEETS_MIME),
    ],
    ids=["word", "slides", "sheets"],
)
async def test_each_handler_renders_a_page_through_the_injected_converter(
    index: int, name: str, build: Callable[[], bytes], mime: str, tmp_path: Path
) -> None:
    content = build()
    (tmp_path / name).write_bytes(content)
    source = LocalFileSource(root=tmp_path)
    built: list[MediaHandler] = [
        OfficeWordHandler(source=source, renderer=_Renderer()),
        OfficeSlidesHandler(source=source, renderer=_Renderer()),
        OfficeSheetsHandler(source=source, renderer=_Renderer()),
    ]
    handler = built[index]
    ref = SourceRef(
        uri=name,
        mime=MimeType.parse(mime),
        content_hash=ContentHash("0" * 64),
        size_bytes=len(content),
    )

    rendition = await handler.invoke(ref, "page_image", PageImageParams(page=2, dpi=96))

    assert isinstance(rendition.content, ImageContent)
    assert rendition.content.data == b"\x89PNG fake 2"
    assert rendition.locator == PageRef(2)
    assert rendition.degradations, "a rendering must say that it is one"


# --- describe_slide -------------------------------------------------------
#
# The one-call version of "render slide 4 and ask about it". Two capabilities
# at once, which is what makes it worth its own affordance rather than leaving
# a caller to chain page_image into ask_about_image -- the same argument that
# produced `ask_about_image` in 0.2.0, one medium along.


def test_describe_slide_needs_both_capabilities() -> None:
    source = FakeSource({})
    render_only = OfficeSlidesHandler(source=source, renderer=_Renderer())
    vision_only = OfficeSlidesHandler(source=source, vision=FakeVision())
    both = OfficeSlidesHandler(source=source, vision=FakeVision(), renderer=_Renderer())

    assert "describe_slide" not in _names(render_only)
    assert "describe_slide" not in _names(vision_only)
    assert "describe_slide" in _names(both)


def test_describe_slide_declares_both_requirements() -> None:
    """Either one missing must un-publish it. Declaring only one would let a
    machine with a converter and no model offer a tool that cannot answer."""
    handler = OfficeSlidesHandler(source=FakeSource({}), vision=FakeVision(), renderer=_Renderer())
    affordance = next(a for a in handler.affordances() if a.name == "describe_slide")
    assert affordance.requires == frozenset({Capability.DOCUMENT_RENDER, Capability.VISION})


@pytest.mark.parametrize(
    "capabilities",
    [
        CapabilitySet.empty(),
        CapabilitySet.of({Capability.VISION: "m"}),
        CapabilitySet.of({Capability.DOCUMENT_RENDER: "soffice"}),
    ],
    ids=["neither", "vision only", "rendering only"],
)
def test_describe_slide_is_not_published_without_both(capabilities: CapabilitySet) -> None:
    handler = OfficeSlidesHandler(source=FakeSource({}), vision=FakeVision(), renderer=_Renderer())
    assert "describe_slide" not in _visible(handler, capabilities)


def test_describe_slide_is_published_with_both() -> None:
    handler = OfficeSlidesHandler(source=FakeSource({}), vision=FakeVision(), renderer=_Renderer())
    capabilities = CapabilitySet.of({Capability.VISION: "m", Capability.DOCUMENT_RENDER: "soffice"})
    assert "describe_slide" in _visible(handler, capabilities)


async def test_describe_slide_renders_the_slide_and_asks_about_it(tmp_path: Path) -> None:
    """The acceptance sentence's last clause: the answer cites the slide."""
    from readeverything.handlers.office_slides import DescribeSlideParams

    (tmp_path / "deck.pptx").write_bytes(pptx_bytes())
    handler = OfficeSlidesHandler(
        source=LocalFileSource(root=tmp_path),
        vision=FakeVision(),
        renderer=_Renderer(),
    )
    ref = SourceRef(
        uri="deck.pptx",
        mime=MimeType.parse(SLIDES_MIME),
        content_hash=ContentHash("0" * 64),
        size_bytes=1,
    )

    rendition = await handler.invoke(
        ref, "describe_slide", DescribeSlideParams(page=3, question="what is on it?")
    )

    assert isinstance(rendition.content, TextContent)
    assert rendition.content.text.strip()
    assert rendition.locator == PageRef(3)
    assert not rendition.degraded


async def test_describe_slide_says_the_model_looked_at_a_rendering(tmp_path: Path) -> None:
    """The model did not see the deck; it saw a converted picture of one slide.
    An answer about typography is an answer about the converter's fonts."""
    from readeverything.handlers.office_slides import DescribeSlideParams

    (tmp_path / "deck.pptx").write_bytes(pptx_bytes())
    handler = OfficeSlidesHandler(
        source=LocalFileSource(root=tmp_path),
        vision=FakeVision(),
        renderer=_Renderer(),
    )
    ref = SourceRef(
        uri="deck.pptx",
        mime=MimeType.parse(SLIDES_MIME),
        content_hash=ContentHash("0" * 64),
        size_bytes=1,
    )

    rendition = await handler.invoke(
        ref, "describe_slide", DescribeSlideParams(page=1, question="what is on it?")
    )
    assert any("font" in d.detail for d in rendition.degradations)


async def test_describe_slide_degrades_when_the_slide_cannot_be_rendered(
    tmp_path: Path,
) -> None:
    from readeverything.handlers.office_slides import DescribeSlideParams

    class _Failing(_Renderer):
        async def render_page(self, path: str, page: int, *, dpi: int = 150) -> bytes:
            raise RenditionFailedError("no")

    (tmp_path / "deck.pptx").write_bytes(pptx_bytes())
    handler = OfficeSlidesHandler(
        source=LocalFileSource(root=tmp_path),
        vision=FakeVision(),
        renderer=_Failing(),
    )
    ref = SourceRef(
        uri="deck.pptx",
        mime=MimeType.parse(SLIDES_MIME),
        content_hash=ContentHash("0" * 64),
        size_bytes=1,
    )

    rendition = await handler.invoke(
        ref, "describe_slide", DescribeSlideParams(page=9, question="what is on it?")
    )
    assert rendition.degraded


async def test_describe_slide_degrades_when_the_model_refuses(tmp_path: Path) -> None:
    """A vision failure must not raise out of a handler, and must not be
    reported as a fact about the slide."""
    from readeverything.handlers.office_slides import DescribeSlideParams

    (tmp_path / "deck.pptx").write_bytes(pptx_bytes())
    handler = OfficeSlidesHandler(
        source=LocalFileSource(root=tmp_path),
        vision=FakeVisionRefusing(),
        renderer=_Renderer(),
    )
    ref = SourceRef(
        uri="deck.pptx",
        mime=MimeType.parse(SLIDES_MIME),
        content_hash=ContentHash("0" * 64),
        size_bytes=1,
    )

    rendition = await handler.invoke(
        ref, "describe_slide", DescribeSlideParams(page=1, question="what is on it?")
    )
    assert rendition.degraded
