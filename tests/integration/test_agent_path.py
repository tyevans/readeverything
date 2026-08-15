"""The full agent path: composition, tools, and image handling together."""

from __future__ import annotations

import json
from pathlib import Path

from readeverything.agent.tools import build_tools
from readeverything.composition import build_perception
from readeverything.testing.fakes import FakeVision


async def test_an_agent_can_see_a_directory_of_mixed_files(media_root: Path) -> None:
    """The acceptance sentence, end to end, through the tools an agent holds.

    Every component here is real except the model. This is the first test in
    the project where the tool pack meets image handling at all.
    """
    perception = await build_perception(media_root, vision=FakeVision(), probe_binaries=False)
    tools = {tool.name: tool for tool in build_tools(perception)}
    assert set(tools) == {"inspect_path", "list_paths", "invoke_affordance"}

    listing = await tools["list_paths"].ainvoke({"uri": "."})
    assert "notes.txt" in listing and "photo.png" in listing

    card = json.loads(await tools["inspect_path"].ainvoke({"uri": "photo.png"}))
    assert card["kind"] == "image"
    assert "describe_image" in {a["name"] for a in card["affordances"]}

    described = await tools["invoke_affordance"].ainvoke(
        {"uri": "photo.png", "affordance": "describe_image", "params": {}}
    )
    assert "located at" in described


async def test_a_tool_call_against_a_missing_file_returns_rather_than_raises(
    media_root: Path,
) -> None:
    """The tool pack never raises. An agent gets a result it can read and retry."""
    perception = await build_perception(media_root, probe_binaries=False)
    tools = {tool.name: tool for tool in build_tools(perception)}
    result = await tools["inspect_path"].ainvoke({"uri": "nope.txt"})
    assert "ERROR" in result
