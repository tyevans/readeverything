"""The README's `deepagents` composition, exercised by a test.

The README documents `create_deep_agent(tools=build_tools(perception))` as
the way to hand this library's tools to an agent. That claim is only true if
the composition actually type-checks and constructs against the real
`deepagents` package — not against a description of it.

This does not invoke the agent: that would need a live model. The claim under
test is narrower and cheaper — that `build_tools(perception)` returns objects
`create_deep_agent` accepts, and that the resulting graph exposes exactly the
three tools this library contributes, alongside `deepagents`' own.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("deepagents")

pytestmark = pytest.mark.integration

from deepagents import create_deep_agent  # noqa: E402
from langgraph.prebuilt.tool_node import ToolNode  # noqa: E402

from readeverything import build_perception, build_tools  # noqa: E402


async def test_build_tools_composes_with_create_deep_agent(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("the quick brown fox")
    perception = await build_perception(tmp_path, probe_binaries=False)
    tools = build_tools(perception)

    # `model` is a string here rather than a live client: `create_deep_agent`
    # does not connect to anything at construction time, and this test's
    # claim is only that construction succeeds, not that inference works.
    agent = create_deep_agent(model="anthropic:claude-opus-5", tools=tools)

    tools_node = agent.get_graph().nodes["tools"].data
    assert isinstance(tools_node, ToolNode)
    registered = set(tools_node.tools_by_name)
    assert {"inspect_path", "list_paths", "invoke_affordance"} <= registered
