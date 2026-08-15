"""The README's base-install example, transcribed into a test.

This test does not parse or execute the README itself — it is a hand-kept
transcription of the example that must be updated whenever the example
changes. A README example with no test at all is a bug report with good
formatting; this is the weaker but still useful guarantee that the
transcription runs and its assertions hold.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from readeverything import build_perception, build_tools

pytestmark = pytest.mark.integration


async def test_the_readme_example_runs(tmp_path: Path) -> None:
    """This test is the reason the example in the README can be trusted, and it
    is why the example must stay small enough to assert on.
    """
    (tmp_path / "notes.txt").write_text("the quick brown fox")
    perception = await build_perception(tmp_path, probe_binaries=False)
    card = await perception.inspect("notes.txt")
    tools = build_tools(perception)
    assert card.kind == "text"
    assert len(tools) == 3
