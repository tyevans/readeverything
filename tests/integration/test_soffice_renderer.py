"""`SofficeRenderer` against a real LibreOffice.

The unit tests use shell-script stand-ins, which is the right way to test the
adapter's contract — but a converter is a component whose correctness cannot be
established against a stand-in that only copies a file. These are what prove
`--convert-to pdf` actually produces the pages this library then renders, and
that the private profile LibreOffice is handed is one it accepts.

Skipped without `soffice`, exactly as the ffmpeg-backed tests are skipped
without ffmpeg. No golden images and no pixel comparisons: LibreOffice's
output shifts between versions, and a golden-image test would fail on a
rendering improvement. The assertions are dimensions, page count, and
non-blankness.
"""

from __future__ import annotations

import asyncio
import io
import shutil
from pathlib import Path

import pytest
from PIL import Image

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.soffice_renderer import SofficeRenderer
from readeverything.domain.errors import RenditionFailedError
from tests.fixtures_office import docx_bytes, odp_bytes, pptx_bytes, xlsx_bytes

pytestmark = pytest.mark.integration


def _skip_without_soffice() -> None:
    if shutil.which("soffice") is None:
        pytest.skip("soffice not available")


@pytest.fixture
def renderer(tmp_path: Path) -> SofficeRenderer:
    _skip_without_soffice()
    return SofficeRenderer(
        artifacts=InMemoryArtifactStore(),
        profile_root=tmp_path / "profile",
        timeout_s=180.0,
    )


def _write(tmp_path: Path, name: str, data: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


@pytest.mark.slow
async def test_a_deck_converts_and_reports_one_page_per_slide(
    renderer: SofficeRenderer, tmp_path: Path
) -> None:
    deck = _write(tmp_path, "deck.pptx", pptx_bytes(titles=("one", "two", "three")))
    assert await renderer.page_count(deck) == 3


@pytest.mark.slow
async def test_a_word_document_gains_the_pagination_the_reader_does_not_have(
    renderer: SofficeRenderer, tmp_path: Path
) -> None:
    """The structural Word reader has no pages at all. Conversion is where
    they come from, which is the whole reason rendering earns its place for
    this format rather than only for decks."""
    document = _write(tmp_path, "doc.docx", docx_bytes())
    assert await renderer.page_count(document) >= 1


@pytest.mark.slow
async def test_a_spreadsheet_converts_to_its_print_layout(
    renderer: SofficeRenderer, tmp_path: Path
) -> None:
    book = _write(tmp_path, "book.xlsx", xlsx_bytes())
    assert await renderer.page_count(book) >= 1


@pytest.mark.slow
async def test_an_odf_document_converts_too(renderer: SofficeRenderer, tmp_path: Path) -> None:
    """ODF is LibreOffice's own format; a converter that could not read it
    would be a wiring mistake rather than a limitation.

    The fixture comes from LibreOffice rather than from `tests/fixtures_office`
    for a measured reason: see the next test.
    """
    modern = _write(tmp_path, "modern.docx", docx_bytes())
    odf = await _convert_to(modern, "odt", tmp_path)
    assert await renderer.page_count(odf) >= 1


@pytest.mark.slow
async def test_this_repos_hand_built_odf_is_rejected_and_the_rejection_is_seen(
    renderer: SofficeRenderer, tmp_path: Path
) -> None:
    """The exit-status trap, confirmed against the real binary.

    `tests/fixtures_office.odp_bytes` is a deliberately minimal three-entry
    zip — enough for `adapters/odf.py`, which walks one flat XML part, and not
    enough for LibreOffice, which wants styles, meta and a populated manifest.
    Measured on 25.8.7.3:

        exit 0, stdout empty, stderr "Error: source file could not be loaded",
        no PDF written

    Exit ZERO. An adapter checking `returncode` would call that a success and
    hand a caller a zero-page document. Checking the produced file is what
    makes this a failure, and this test is what keeps that check honest — it
    is the only place the real binary demonstrates the trap.
    """
    deck = _write(tmp_path, "deck.odp", odp_bytes())
    with pytest.raises(RenditionFailedError, match="produced no PDF"):
        await renderer.page_count(deck)


@pytest.mark.slow
async def test_a_rendered_slide_is_a_png_of_the_expected_shape(
    renderer: SofficeRenderer, tmp_path: Path
) -> None:
    """Dimensions, not pixels. A default python-pptx deck is 10x7.5 inches, so
    at 72dpi it is about 720x540 — asserted as an aspect ratio and a floor,
    since a LibreOffice upgrade may round differently."""
    deck = _write(tmp_path, "deck.pptx", pptx_bytes())
    png = await renderer.render_page(deck, 1, dpi=72)

    assert png.startswith(b"\x89PNG")
    image = Image.open(io.BytesIO(png))
    assert image.width > 200 and image.height > 150
    assert image.width > image.height, "a slide is landscape"


@pytest.mark.slow
async def test_doubling_the_dpi_doubles_the_pixels(
    renderer: SofficeRenderer, tmp_path: Path
) -> None:
    deck = _write(tmp_path, "deck.pptx", pptx_bytes())
    small = Image.open(io.BytesIO(await renderer.render_page(deck, 1, dpi=72)))
    large = Image.open(io.BytesIO(await renderer.render_page(deck, 1, dpi=144)))

    assert large.width == pytest.approx(small.width * 2, abs=3)
    assert large.height == pytest.approx(small.height * 2, abs=3)


@pytest.mark.slow
async def test_a_rendered_slide_is_not_blank(renderer: SofficeRenderer, tmp_path: Path) -> None:
    """The cheapest honest check that something was actually drawn. A slide
    that converted to a white rectangle would satisfy every dimension
    assertion above, and would be exactly the failure that matters.
    """
    deck = _write(tmp_path, "deck.pptx", pptx_bytes(titles=("A very visible title",)))
    image = Image.open(io.BytesIO(await renderer.render_page(deck, 1, dpi=72)))

    assert len(image.convert("L").getcolors(maxcolors=1 << 16) or []) > 1


@pytest.mark.slow
async def test_a_legacy_ole2_document_converts(renderer: SofficeRenderer, tmp_path: Path) -> None:
    """The row Spec 9 declined. There is no pure-Python writer for OLE2, so
    the fixture is produced by LibreOffice itself — which is honest here,
    because it is LibreOffice that will read it back."""
    modern = _write(tmp_path, "modern.docx", docx_bytes())
    legacy = await _convert_to(modern, "doc", tmp_path)
    assert await renderer.page_count(legacy) >= 1


@pytest.mark.slow
async def test_a_second_call_does_not_reconvert(renderer: SofficeRenderer, tmp_path: Path) -> None:
    """Not a mock's call count: the wall clock. Conversion is seconds and a
    cache hit is milliseconds, so the difference is not subtle."""
    deck = _write(tmp_path, "deck.pptx", pptx_bytes())
    loop = asyncio.get_running_loop()

    started = loop.time()
    await renderer.page_count(deck)
    cold = loop.time() - started

    started = loop.time()
    await renderer.page_count(deck)
    warm = loop.time() - started

    assert warm < cold / 4, f"cold {cold:.2f}s, warm {warm:.2f}s: the PDF was not cached"


@pytest.mark.slow
async def test_a_file_that_is_not_a_document_fails_rather_than_hanging(
    renderer: SofficeRenderer, tmp_path: Path
) -> None:
    junk = _write(tmp_path, "junk.pptx", b"\x00\x01\x02 this is not a document at all")
    with pytest.raises(RenditionFailedError):
        await renderer.page_count(junk)


async def _convert_to(path: str, extension: str, tmp_path: Path) -> str:
    """A legacy fixture, made by the only tool on this machine that can.

    Deliberately NOT going through `SofficeRenderer`: that adapter only ever
    asks for PDF, and a fixture builder borrowing it would couple the fixture
    to the thing under test.
    """
    outdir = tmp_path / "legacy"
    outdir.mkdir(exist_ok=True)
    process = await asyncio.create_subprocess_exec(
        "soffice",
        f"-env:UserInstallation={(tmp_path / 'fixture-profile').resolve().as_uri()}",
        "--headless",
        "--convert-to",
        extension,
        "--outdir",
        str(outdir),
        path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await asyncio.wait_for(process.communicate(), timeout=180.0)
    produced = sorted(outdir.glob(f"*.{extension}"))
    assert produced, f"soffice produced no .{extension} fixture"
    return str(produced[0])
