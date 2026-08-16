"""Spec 9 §1.1's acceptance scenario, through the real composition root."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from readeverything.agent.tools import build_tools
from readeverything.composition import build_perception
from readeverything.domain.locators import CellRange, CharSpan, PageRef
from readeverything.domain.rendition import Budget, TextContent

pytestmark = pytest.mark.integration

#: Every module `_optional_office_handlers` imports inside its `try`. Evicting
#: the third-party package alone proves nothing — see
#: `test_documents.py::test_a_base_install_without_pypdfium2_falls_back_to_binary`
#: for the full argument. In short: a guarded import served from `sys.modules`
#: never re-executes its own `import docx`, so the handler registers regardless
#: of what the guard does.
_OFFICE_MODULES = (
    "readeverything.handlers.office_word",
    "readeverything.handlers.office_slides",
    "readeverything.handlers.office_sheets",
)
_OFFICE_PACKAGES = ("docx", "pptx", "openpyxl")


def _text(rendition: object) -> str:
    content = rendition.content  # type: ignore[attr-defined]
    assert isinstance(content, TextContent)
    return content.text


async def test_inspect_reports_each_family_and_its_shape(office_root: Path) -> None:
    """The §1.1 acceptance clause: kind and shape, per family."""
    perception = await build_perception(office_root, probe_binaries=False)

    word = await perception.inspect("policy.docx")
    assert word.facts["heading_count"] == 3
    assert word.facts["comment_count"] == 1

    deck = await perception.inspect("deck.pptx")
    assert deck.facts["slide_count"] == 3
    assert next(s.label for s in deck.outline) == "Opening position"

    book = await perception.inspect("book.xlsx")
    assert book.facts["sheet_count"] == 2
    assert book.facts["sheet.Data.used_range"] == "A1:C4"


async def test_no_office_document_falls_through_to_the_hex_dump(office_root: Path) -> None:
    """The README's complaint, closed. A hex dump of a `.docx` is a hex dump of
    a zip container, and the agent learns nothing from it.
    """
    perception = await build_perception(office_root, probe_binaries=False)
    for name in ("policy.docx", "deck.pptx", "book.xlsx", "notes.odt", "slides.odp", "sheet.ods"):
        card = await perception.inspect(name)
        assert "hexdump" not in card.affordance_names(), name


async def test_every_character_resolves_to_its_slide_heading_or_sheet(
    office_root: Path,
) -> None:
    """The three locator vocabularies this spec puts to work, one per family."""
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
    assert "The number is soft" in _text(rendition)


async def test_a_sheet_reads_as_text_with_formulas_reachable(office_root: Path) -> None:
    """Both halves, because reporting only one is how a spreadsheet lies."""
    perception = await build_perception(office_root, probe_binaries=False)
    value = await perception.invoke("book.xlsx", "read_cells", {"name": "Data", "a1_range": "C2"})
    formula = await perception.invoke(
        "book.xlsx", "read_cells", {"name": "Data", "a1_range": "C2", "formulas": True}
    )
    assert "=B2*2" in _text(formula)
    assert "=B2*2" not in _text(value)


async def test_the_odf_equivalents_work_the_same_way(office_root: Path) -> None:
    perception = await build_perception(office_root, probe_binaries=False)
    assert [s.label for s in (await perception.inspect("notes.odt")).outline] == ["Alpha", "Bravo"]
    assert (await perception.inspect("slides.odp")).facts["slide_count"] == 2
    assert (await perception.inspect("sheet.ods")).facts["sheet_count"] == 2


async def test_an_office_document_is_detected_by_content_not_by_its_name(
    office_root: Path,
) -> None:
    """The spec's detection rule, end to end: a deck named `.bin` is a deck."""
    (office_root / "mystery.bin").write_bytes((office_root / "deck.pptx").read_bytes())
    perception = await build_perception(office_root, probe_binaries=False)
    card = await perception.inspect("mystery.bin")
    assert card.facts["slide_count"] == 3


async def test_the_agent_can_ask_a_deck_for_a_slide(office_root: Path) -> None:
    """End to end through the tools an agent actually holds."""
    perception = await build_perception(office_root, probe_binaries=False)
    tools = {t.name: t for t in build_tools(perception)}
    result = await tools["invoke_affordance"].ainvoke(
        {"uri": "deck.pptx", "affordance": "read_slide", "params": {"page": 3}}
    )
    assert "What we decided" in result


async def test_a_base_install_without_the_office_extra_falls_back_to_binary(
    office_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Narrower, not broken — the same contract Pillow and pypdfium2 have."""
    for module in _OFFICE_MODULES:
        monkeypatch.delitem(sys.modules, module, raising=False)
    for package in _OFFICE_PACKAGES:
        monkeypatch.setitem(sys.modules, package, None)
    perception = await build_perception(office_root, probe_binaries=False)
    card = await perception.inspect("policy.docx")
    assert card.kind == "binary"
    assert "hexdump" in card.affordance_names()
    registered = {type(h).__name__ for h in perception.registry.handlers}
    assert not registered & {
        "OfficeWordHandler",
        "OfficeSlidesHandler",
        "OfficeSheetsHandler",
    }


async def test_one_missing_reader_does_not_cost_the_other_two(
    office_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `office` extra installs three packages, but an environment holding
    only some of them should still read the families it can. One try/except
    around all three would make one absence cost the other two.
    """
    monkeypatch.delitem(sys.modules, "readeverything.handlers.office_word", raising=False)
    monkeypatch.setitem(sys.modules, "docx", None)
    perception = await build_perception(office_root, probe_binaries=False)
    assert (await perception.inspect("policy.docx")).kind == "binary"
    assert (await perception.inspect("deck.pptx")).facts["slide_count"] == 3
    assert (await perception.inspect("book.xlsx")).facts["sheet_count"] == 2
