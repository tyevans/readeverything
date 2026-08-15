from pathlib import Path

import pytest

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.detection import PuremagicDetector
from readeverything.adapters.hashing import ContentHasher
from readeverything.adapters.local_source import LocalFileSource
from readeverything.agent.tools import build_tools
from readeverything.domain.capability import CapabilitySet
from readeverything.handlers.binary import BinaryHandler
from readeverything.handlers.text import TextHandler
from readeverything.pipeline.perception import Perception
from readeverything.registry.registry import MimeTypeRegistry


@pytest.fixture
def perception(tmp_path: Path) -> Perception:
    (tmp_path / "notes.txt").write_bytes(b"alpha\nbeta\n")
    source = LocalFileSource(root=tmp_path)
    return Perception(
        source=source,
        detector=PuremagicDetector(),
        hasher=ContentHasher(source=source),
        registry=MimeTypeRegistry(
            handlers=(TextHandler(source=source), BinaryHandler(source=source)),
            capabilities=CapabilitySet.empty(),
        ),
        artifacts=InMemoryArtifactStore(),
    )


def test_the_pack_offers_the_three_core_tools(perception: Perception) -> None:
    names = {tool.name for tool in build_tools(perception)}
    assert {"inspect_path", "list_paths", "invoke_affordance"} <= names


def test_every_tool_has_a_description(perception: Perception) -> None:
    """The description is the model's only guidance; a blank one blinds it."""
    for tool in build_tools(perception):
        assert tool.description.strip()


async def test_inspect_path_returns_a_rendered_card(perception: Perception) -> None:
    tool = next(t for t in build_tools(perception) if t.name == "inspect_path")
    output = await tool.ainvoke({"uri": "notes.txt"})
    assert "text/plain" in output
    assert "read_range" in output


async def test_a_missing_file_returns_an_error_string_not_an_exception(
    perception: Perception,
) -> None:
    tool = next(t for t in build_tools(perception) if t.name == "inspect_path")
    output = await tool.ainvoke({"uri": "absent.txt"})
    assert "ERROR" in output


async def test_invoke_affordance_round_trips(perception: Perception) -> None:
    tool = next(t for t in build_tools(perception) if t.name == "invoke_affordance")
    output = await tool.ainvoke(
        {"uri": "notes.txt", "affordance": "read_range", "params": {"start": 0, "end": 5}}
    )
    assert "alpha" in output


async def test_invoking_an_unavailable_affordance_returns_an_error(
    perception: Perception,
) -> None:
    tool = next(t for t in build_tools(perception) if t.name == "invoke_affordance")
    output = await tool.ainvoke({"uri": "notes.txt", "affordance": "hexdump", "params": {}})
    assert "ERROR" in output
    assert "read_range" in output  # the error names what IS available


async def test_a_missing_required_argument_returns_an_error_string(
    perception: Perception,
) -> None:
    """Model-authored arguments are untrusted input; they must not raise."""
    tool = next(t for t in build_tools(perception) if t.name == "inspect_path")
    output = await tool.ainvoke({})
    assert "ERROR" in output


async def test_a_wrongly_typed_argument_returns_an_error_string(
    perception: Perception,
) -> None:
    tool = next(t for t in build_tools(perception) if t.name == "invoke_affordance")
    output = await tool.ainvoke(
        {"uri": "notes.txt", "affordance": "read_range", "params": "not-a-dict"}
    )
    assert "ERROR" in output
