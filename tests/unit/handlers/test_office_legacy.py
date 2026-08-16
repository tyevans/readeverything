"""Legacy OLE2 documents, read entirely through a converter.

Spec 9 declined this family because pure-Python OLE2 support is poor. It
arrives here as a capability rather than as a dependency everyone pays for: no
converter, no handler, and the files keep falling through to the hex dump —
which is the current behaviour and therefore no regression.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.domain.errors import RenditionFailedError, UnknownAffordanceError
from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
from readeverything.domain.locators import ByteRange, PageRef
from readeverything.domain.rendition import Budget, ImageContent, TextContent
from readeverything.handlers.binary import BinaryHandler
from readeverything.handlers.office_legacy import (
    LEGACY_MIMETYPES,
    OfficeLegacyHandler,
)
from readeverything.handlers.page_images import PageImageParams
from readeverything.registry.registry import MimeTypeRegistry
from readeverything.testing.fakes import FakeSource

URI = "old.doc"
LEGACY_MIME = "application/msword"


class _Renderer:
    revision = "fake/1"

    def __init__(self, *, pages: tuple[str, ...] = ("first page", "second page")) -> None:
        self._pages = pages
        self.failing = False

    def claims(self, mime: MimeType) -> bool:
        return str(mime) in LEGACY_MIMETYPES

    async def page_count(self, path: str) -> int:
        if self.failing:
            raise RenditionFailedError("the converter said no")
        return len(self._pages)

    async def page_text(self, path: str, page: int) -> str:
        if self.failing:
            raise RenditionFailedError("the converter said no")
        return self._pages[page - 1]

    async def render_page(self, path: str, page: int, *, dpi: int = 150) -> bytes:
        if self.failing:
            raise RenditionFailedError("the converter said no")
        return b"\x89PNG page " + str(page).encode()


class _Source(FakeSource):
    """`FakeSource` plus a real `local_path`, which a converter needs."""

    def __init__(self, path: Path) -> None:
        super().__init__({URI: path.read_bytes()})
        self._path = str(path)

    async def local_path(self, uri: str) -> str:
        return self._path

    def stream(self, uri: str, *, chunk_size: int = 1 << 20) -> AsyncIterator[bytes]:
        return super().stream(uri, chunk_size=chunk_size)


@pytest.fixture
def source(tmp_path: Path) -> _Source:
    path = tmp_path / URI
    path.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 500)
    return _Source(path)


def _ref() -> SourceRef:
    return SourceRef(
        uri=URI,
        mime=MimeType.parse(LEGACY_MIME),
        content_hash=ContentHash("0" * 64),
        size_bytes=508,
    )


# --- negotiation ----------------------------------------------------------


def test_the_handler_requires_a_converter_outright(source: _Source) -> None:
    """Unlike the other three, which merely gain an affordance. Without a
    converter there is nothing this handler can do at all — not a card, not a
    word — so it must not be registered rather than registered and empty."""
    handler = OfficeLegacyHandler(source=source, renderer=_Renderer())
    assert handler.requires() == frozenset({Capability.DOCUMENT_RENDER})


def test_without_the_capability_the_handler_is_dropped_and_the_hex_dump_wins(
    source: _Source,
) -> None:
    """The no-regression claim, stated as a test. A legacy file on a machine
    with no converter behaves exactly as it does today."""
    registry = MimeTypeRegistry(
        handlers=[
            OfficeLegacyHandler(source=source, renderer=_Renderer()),
            BinaryHandler(source=source),
        ],
        capabilities=CapabilitySet.empty(),
    )
    assert isinstance(registry.resolve(MimeType.parse(LEGACY_MIME)), BinaryHandler)


def test_with_the_capability_the_handler_wins(source: _Source) -> None:
    registry = MimeTypeRegistry(
        handlers=[
            OfficeLegacyHandler(source=source, renderer=_Renderer()),
            BinaryHandler(source=source),
        ],
        capabilities=CapabilitySet.of({Capability.DOCUMENT_RENDER: "soffice 25.8"}),
    )
    assert isinstance(registry.resolve(MimeType.parse(LEGACY_MIME)), OfficeLegacyHandler)


def test_it_claims_all_three_legacy_mimetypes(source: _Source) -> None:
    """Only `application/msword` is reachable through this library's detector —
    an OLE2 header is identical across Word, Excel and PowerPoint, and puremagic
    reports msword for all three above the signature floor. The other two are
    claimed because a caller with a real OLE2 directory walker plugs in a better
    detector and this is then already correct, at no cost today.
    """
    assert set(OfficeLegacyHandler.mime_patterns) == set(LEGACY_MIMETYPES)


# --- the card -------------------------------------------------------------


async def test_the_card_reports_what_the_conversion_observed(source: _Source) -> None:
    card = await OfficeLegacyHandler(source=source, renderer=_Renderer()).describe(_ref())
    assert card.facts["readable"] == "yes"
    assert card.facts["page_count"] == 2
    assert card.kind is MediaKind.BINARY


async def test_the_card_does_not_claim_which_office_application_made_the_file(
    source: _Source,
) -> None:
    """The regression this guards is a future "helpful" change.

    `application/msword` reaches this handler for a `.ppt` and an `.xls` too,
    because the compound-file header does not distinguish them. Mapping that
    mimetype to a Word-shaped card would state something known to be false
    about two thirds of the family. What the handler actually observed is the
    converted document's page count, so that is what the card says.
    """
    card = await OfficeLegacyHandler(source=source, renderer=_Renderer()).describe(_ref())
    words = " ".join(
        [*map(str, card.facts.values()), *card.facts, *(s.label for s in card.outline)]
    )
    for claim in ("word", "excel", "powerpoint", "spreadsheet", "slide", "deck"):
        assert claim not in words.lower(), f"the card asserts {claim!r} from a mimetype"


async def test_the_card_outlines_one_segment_per_page(source: _Source) -> None:
    card = await OfficeLegacyHandler(source=source, renderer=_Renderer()).describe(_ref())
    assert [segment.locator for segment in card.outline] == [PageRef(1), PageRef(2)]


async def test_a_document_the_converter_cannot_open_gets_an_honest_card(
    source: _Source,
) -> None:
    renderer = _Renderer()
    renderer.failing = True
    card = await OfficeLegacyHandler(source=source, renderer=renderer).describe(_ref())
    assert card.facts["readable"] == "no"
    assert card.outline == ()


# --- represent ------------------------------------------------------------


async def test_represent_reads_the_text_of_the_converted_document(
    source: _Source,
) -> None:
    """The row that changes what the library can do. These files got a hex dump
    before; now they get words."""
    rendered = await OfficeLegacyHandler(source=source, renderer=_Renderer()).represent(
        _ref(), Budget(max_chars=None)
    )
    assert "first page" in rendered.text
    assert "second page" in rendered.text


async def test_represent_maps_every_character_to_the_page_it_came_from(
    source: _Source,
) -> None:
    rendered = await OfficeLegacyHandler(source=source, renderer=_Renderer()).represent(
        _ref(), Budget(max_chars=None)
    )
    assert rendered.locator_map.resolve(0) == PageRef(1)
    assert rendered.locator_map.resolve(rendered.text.index("second")) == PageRef(2)


async def test_represent_puts_a_barrier_at_every_page_break(source: _Source) -> None:
    rendered = await OfficeLegacyHandler(source=source, renderer=_Renderer()).represent(
        _ref(), Budget(max_chars=None)
    )
    assert len(rendered.barriers) == 1


async def test_represent_says_the_text_came_through_a_converter(source: _Source) -> None:
    """§8's honesty requirement applied to text, not only to images. This is
    not the file's own text: it is what LibreOffice's importer made of it."""
    rendered = await OfficeLegacyHandler(source=source, renderer=_Renderer()).represent(
        _ref(), Budget(max_chars=None)
    )
    assert any("convert" in d.what or "convert" in d.detail for d in rendered.degradations)


async def test_a_page_that_converted_to_no_text_is_reported_not_silently_empty(
    source: _Source,
) -> None:
    """A blank stretch in the middle of a document must not read as "the
    document ends here"."""
    renderer = _Renderer(pages=("first page", "", "third page"))
    rendered = await OfficeLegacyHandler(source=source, renderer=renderer).represent(
        _ref(), Budget(max_chars=None)
    )
    assert "page 2" in rendered.text
    assert "third page" in rendered.text


async def test_represent_degrades_rather_than_raising_when_conversion_fails(
    source: _Source,
) -> None:
    renderer = _Renderer()
    renderer.failing = True
    rendered = await OfficeLegacyHandler(source=source, renderer=renderer).represent(
        _ref(), Budget(max_chars=None)
    )
    assert rendered.degradations
    assert rendered.locator_map.resolve(0) == ByteRange(0, 508)


async def test_represent_honours_a_budget(source: _Source) -> None:
    rendered = await OfficeLegacyHandler(source=source, renderer=_Renderer()).represent(
        _ref(), Budget(max_chars=6)
    )
    assert len(rendered.text) == 6
    assert any("truncated" in d.what for d in rendered.degradations)


# --- affordances ----------------------------------------------------------


async def test_read_page_returns_one_page(source: _Source) -> None:
    handler = OfficeLegacyHandler(source=source, renderer=_Renderer())
    from readeverything.handlers.office_legacy import ReadPageParams

    rendition = await handler.invoke(_ref(), "read_page", ReadPageParams(page=2))
    assert isinstance(rendition.content, TextContent)
    assert "second page" in rendition.content.text
    assert rendition.locator == PageRef(2)


async def test_read_page_past_the_end_degrades_rather_than_raising(source: _Source) -> None:
    handler = OfficeLegacyHandler(source=source, renderer=_Renderer())
    from readeverything.handlers.office_legacy import ReadPageParams

    rendition = await handler.invoke(_ref(), "read_page", ReadPageParams(page=99))
    assert rendition.degraded


async def test_page_image_renders_through_the_converter(source: _Source) -> None:
    handler = OfficeLegacyHandler(source=source, renderer=_Renderer())
    rendition = await handler.invoke(_ref(), "page_image", PageImageParams(page=2))
    assert isinstance(rendition.content, ImageContent)
    assert rendition.content.data == b"\x89PNG page 2"
    assert rendition.degradations, "a rendering must say that it is one"


async def test_an_unknown_affordance_is_refused(source: _Source) -> None:
    handler = OfficeLegacyHandler(source=source, renderer=_Renderer())
    with pytest.raises(UnknownAffordanceError):
        await handler.invoke(_ref(), "read_slide", PageImageParams())
