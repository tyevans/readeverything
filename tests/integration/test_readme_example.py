"""The README's base-install example, kept honest by a test that runs it.

A README example that does not run is a bug report with good formatting.
"""

from __future__ import annotations

from pathlib import Path

from readeverything import build_perception, build_tools


async def test_the_readme_example_runs(tmp_path: Path) -> None:
    """This test is the reason the example in the README can be trusted, and it
    is why the example must stay small enough to assert on.
    """
    (tmp_path / "notes.txt").write_text("the quick brown fox")
    perception = await build_perception(tmp_path)
    card = await perception.inspect("notes.txt")
    tools = build_tools(perception)
    assert card.kind == "text"
    assert len(tools) == 3
