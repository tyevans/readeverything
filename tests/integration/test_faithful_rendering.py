"""The acceptance sentence, end to end, against real LibreOffice.

    Install LibreOffice, change nothing else, and `page_image` appears on the
    slide, word and spreadsheet handlers, and legacy `.doc`/`.ppt`/`.xls` files
    gain a card and readable text where they previously got a hex dump.

Every other test in this feature holds a piece of that. This holds the whole
sentence, through `build_perception` with nothing configured, so the wiring is
proved rather than assumed.

Skipped without `soffice`. The mirror-image half — the machine with NO
converter — is not skippable and must not be, so it is expressed with
`NullRenderer` and lives in `tests/unit/test_composition.py`, where it runs on
every machine including this one.
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import pytest
from PIL import Image

from readeverything.composition import build_perception
from readeverything.domain.capability import Capability
from readeverything.domain.identity import MimeType
from readeverything.domain.rendition import Budget, ImageContent, TextContent
from readeverything.handlers.binary import BinaryHandler
from readeverything.handlers.office_legacy import OfficeLegacyHandler
from tests.fixtures_office import docx_bytes, pptx_bytes, xlsx_bytes

pytestmark = [pytest.mark.integration, pytest.mark.slow]

LEGACY_MIME = "application/msword"


@pytest.fixture(autouse=True)
def _needs_soffice() -> None:
    if shutil.which("soffice") is None:
        pytest.skip("soffice not available")


@pytest.fixture
def library(tmp_path: Path) -> Path:
    (tmp_path / "deck.pptx").write_bytes(pptx_bytes(titles=("Opening", "Numbers", "Decided")))
    (tmp_path / "memo.docx").write_bytes(docx_bytes())
    (tmp_path / "book.xlsx").write_bytes(xlsx_bytes())
    return tmp_path


async def test_installing_libreoffice_is_the_only_thing_a_caller_has_to_do(
    library: Path,
) -> None:
    """Nothing configured. `page_image` appears on all three."""
    perception = await build_perception(library)

    for name in ("deck.pptx", "memo.docx", "book.xlsx"):
        names = {a.name for a in (await perception.inspect(name)).affordances}
        assert "page_image" in names, name


async def test_the_probe_finds_soffice_and_records_a_version(library: Path) -> None:
    perception = await build_perception(library)
    revision = perception.registry.capabilities.revisions[Capability.DOCUMENT_RENDER]
    assert "LibreOffice" in revision, revision


async def test_asking_for_slide_two_renders_slide_two(library: Path) -> None:
    """Not a golden image. It is a PNG, it is landscape, and it is not blank —
    which is the whole of what can be asserted about a rendering whose engine
    changes between versions."""
    perception = await build_perception(library)

    rendition = await perception.invoke("deck.pptx", "page_image", {"page": 2, "dpi": 72})

    assert isinstance(rendition.content, ImageContent)
    image = Image.open(io.BytesIO(rendition.content.data))
    assert image.width > image.height
    assert len(image.convert("L").getcolors(maxcolors=1 << 16) or []) > 1


async def test_a_rendered_page_says_it_came_from_a_converter(library: Path) -> None:
    perception = await build_perception(library)
    rendition = await perception.invoke("deck.pptx", "page_image", {"page": 1, "dpi": 72})
    assert any("font" in d.detail for d in rendition.degradations)


async def test_the_second_render_of_a_document_does_not_reconvert(library: Path) -> None:
    """One conversion per document, not one per page. Wall clock, not a mock."""
    import time

    perception = await build_perception(library)

    started = time.perf_counter()
    await perception.invoke("deck.pptx", "page_image", {"page": 1, "dpi": 72})
    cold = time.perf_counter() - started

    started = time.perf_counter()
    await perception.invoke("deck.pptx", "page_image", {"page": 3, "dpi": 72})
    warm = time.perf_counter() - started

    assert warm < cold / 2, f"cold {cold:.2f}s, warm {warm:.2f}s"


# --- the legacy family ----------------------------------------------------


@pytest.fixture
async def legacy_doc(tmp_path: Path) -> Path:
    """A real OLE2 `.doc`, produced by the only tool here that can make one."""
    import asyncio

    (tmp_path / "modern.docx").write_bytes(docx_bytes())
    process = await asyncio.create_subprocess_exec(
        "soffice",
        f"-env:UserInstallation={(tmp_path / 'fixture-profile').resolve().as_uri()}",
        "--headless",
        "--convert-to",
        "doc",
        "--outdir",
        str(tmp_path),
        str(tmp_path / "modern.docx"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await asyncio.wait_for(process.communicate(), timeout=180.0)
    (tmp_path / "modern.docx").unlink()
    produced = tmp_path / "modern.doc"
    assert produced.exists(), "soffice produced no .doc fixture"
    return produced


async def test_a_legacy_document_reaches_the_legacy_handler(legacy_doc: Path) -> None:
    perception = await build_perception(legacy_doc.parent)
    handler = perception.registry.resolve(MimeType.parse(LEGACY_MIME))
    assert isinstance(handler, OfficeLegacyHandler)


async def test_a_legacy_document_gains_a_card_where_it_had_a_hex_dump(
    legacy_doc: Path,
) -> None:
    perception = await build_perception(legacy_doc.parent)
    card = await perception.inspect(legacy_doc.name)

    assert card.facts["readable"] == "yes"
    assert int(str(card.facts["page_count"])) >= 1
    assert {"read_page", "page_image"} == {a.name for a in card.affordances}


async def test_a_legacy_document_gains_readable_text(legacy_doc: Path) -> None:
    """The row that changes what the library can do. `docx_bytes()` writes a
    heading called "Bravo"; before this, reading that file produced a hex dump.
    """
    perception = await build_perception(legacy_doc.parent)
    rendered = await perception.represent(legacy_doc.name, Budget(max_chars=None))
    assert "Bravo" in rendered.text


async def test_a_legacy_documents_text_says_it_came_through_a_converter(
    legacy_doc: Path,
) -> None:
    perception = await build_perception(legacy_doc.parent)
    rendered = await perception.represent(legacy_doc.name, Budget(max_chars=None))
    assert any("convert" in d.what for d in rendered.degradations)


async def test_a_legacy_document_renders_a_page(legacy_doc: Path) -> None:
    perception = await build_perception(legacy_doc.parent)
    rendition = await perception.invoke(legacy_doc.name, "page_image", {"page": 1, "dpi": 72})
    assert isinstance(rendition.content, ImageContent)
    assert rendition.content.data.startswith(b"\x89PNG")


async def test_read_page_returns_the_pages_words(legacy_doc: Path) -> None:
    perception = await build_perception(legacy_doc.parent)
    rendition = await perception.invoke(legacy_doc.name, "read_page", {"page": 1})
    assert isinstance(rendition.content, TextContent)
    assert "Bravo" in rendition.content.text


async def test_turning_rendering_off_puts_the_legacy_file_back_on_the_hex_dump(
    legacy_doc: Path,
) -> None:
    """The no-regression claim, on the machine that has LibreOffice."""
    from readeverything.adapters.null_renderer import NullRenderer

    perception = await build_perception(legacy_doc.parent, renderer=NullRenderer())
    handler = perception.registry.resolve(MimeType.parse(LEGACY_MIME))
    assert isinstance(handler, BinaryHandler)
