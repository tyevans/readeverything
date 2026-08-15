"""PDFs, wired through the composition root and the tools an agent actually holds."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from readeverything.agent.tools import build_tools
from readeverything.composition import build_perception

pytestmark = pytest.mark.integration


async def test_a_pdf_is_handled_when_the_documents_extra_is_present(
    documents_root: Path,
) -> None:
    """A PDF's card is `binary` kind (R3): what identifies it is its mime,
    its page count, and its affordances, not a new `MediaKind` member.
    """
    perception = await build_perception(documents_root, probe_binaries=False)
    card = await perception.inspect("report.pdf")
    assert card.ref.mime.type == "application"
    assert card.ref.mime.subtype == "pdf"
    assert card.facts["page_count"] == 3
    assert "read_page" in {a.name for a in card.affordances}


async def test_a_base_install_without_pypdfium2_falls_back_to_binary(
    documents_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Narrower, not broken — the same contract Pillow already has.

    Whether `pypdfium2` was already imported elsewhere in this process (by
    another test module) determines whether blocking it here actually changes
    anything — the same caveat that applies to the analogous Pillow test in
    `tests/unit/test_composition.py`. What this asserts is that the build
    still succeeds either way; it is not a reliable way to prove the handler
    was omitted in a full test run.
    """
    monkeypatch.setitem(sys.modules, "pypdfium2", None)
    perception = await build_perception(documents_root, probe_binaries=False)
    card = await perception.inspect("report.pdf")
    assert card.kind == "binary"


async def test_the_agent_can_ask_a_pdf_for_a_page(documents_root: Path) -> None:
    """End to end through the tools an agent actually holds."""
    perception = await build_perception(documents_root, probe_binaries=False)
    tools = {t.name: t for t in build_tools(perception)}
    result = await tools["invoke_affordance"].ainvoke(
        {"uri": "report.pdf", "affordance": "read_page", "params": {"page": 2}}
    )
    assert "Section two" in result
