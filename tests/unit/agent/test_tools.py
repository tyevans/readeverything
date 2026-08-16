from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.detection import PuremagicDetector
from readeverything.adapters.hashing import ContentHasher
from readeverything.adapters.local_source import LocalFileSource
from readeverything.agent.tools import _render_rendition, build_tools
from readeverything.domain.capability import CapabilitySet
from readeverything.domain.card import Card
from readeverything.domain.errors import UnknownAffordanceError
from readeverything.domain.locators import BBox, PageRef
from readeverything.domain.rendition import (
    Degradation,
    ImageContent,
    Rendition,
    TextContent,
)
from readeverything.handlers.binary import BinaryHandler
from readeverything.handlers.text import TextHandler
from readeverything.pipeline.perception import Perception
from readeverything.registry.registry import MimeTypeRegistry


def _by_name(tools: list[Any], name: str) -> Any:
    return next(t for t in tools if t.name == name)


class RecordingPerception:
    """A fake `Perception` that records what it was asked to invoke/inspect."""

    def __init__(self) -> None:
        self.invoked: tuple[str, str, dict[str, Any]] | None = None
        self.inspected: list[str] = []

    async def inspect(self, uri: str) -> Card:
        self.inspected.append(uri)
        raise AssertionError("inspect should not be called by ask_about_image")

    async def list(self, uri: str = ".") -> list[str]:
        raise NotImplementedError

    async def invoke(self, uri: str, name: str, params: Mapping[str, Any]) -> Rendition:
        self.invoked = (uri, name, dict(params))
        return Rendition(
            locator=BBox(page=None, x=0.0, y=0.0, w=1.0, h=1.0),
            content=TextContent(text="the answer"),
        )


class RaisingPerception:
    """A fake `Perception` whose `invoke` always raises a given error."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def inspect(self, uri: str) -> Card:
        raise NotImplementedError

    async def list(self, uri: str = ".") -> list[str]:
        raise NotImplementedError

    async def invoke(self, uri: str, name: str, params: Mapping[str, Any]) -> Rendition:
        raise self._error


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


def test_the_pack_offers_the_four_core_tools(perception: Perception) -> None:
    names = {tool.name for tool in build_tools(perception)}
    assert {"inspect_path", "list_paths", "invoke_affordance", "ask_about_image"} <= names


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


def test_image_content_names_an_affordance_that_exists() -> None:
    """Neither `describe_image` nor `ocr` is a tool name.

    The old text told the model to "pass to a vision tool to read it". There is
    no vision tool. Instructing a model toward a tool that does not exist is
    the same defect shape as a degradation describing a cause nothing checked —
    text asserting something nothing established — and it reaches the model.
    """
    rendition = Rendition(
        locator=BBox(page=None, x=0.0, y=0.0, w=1.0, h=1.0),
        content=ImageContent(data=b"\x89PNG", mime="image/png"),
    )
    rendered = _render_rendition(rendition, ("describe_image", "ocr"))
    assert "vision tool" not in rendered
    assert "describe_image" in rendered


def test_image_content_says_so_plainly_when_nothing_can_read_it() -> None:
    """With no vision capability the honest answer is that it cannot be read
    here — not a pointer at an affordance the registry filtered out."""
    rendition = Rendition(
        locator=BBox(page=None, x=0.0, y=0.0, w=1.0, h=1.0),
        content=ImageContent(data=b"\x89PNG", mime="image/png"),
    )
    rendered = _render_rendition(rendition, ())
    assert "vision tool" not in rendered
    assert "cannot be read" in rendered


async def test_a_wrongly_typed_argument_returns_an_error_string(
    perception: Perception,
) -> None:
    tool = next(t for t in build_tools(perception) if t.name == "invoke_affordance")
    output = await tool.ainvoke(
        {"uri": "notes.txt", "affordance": "read_range", "params": "not-a-dict"}
    )
    assert "ERROR" in output


@pytest.mark.asyncio
async def test_ask_about_image_forwards_question_and_where_together() -> None:
    perception = RecordingPerception()
    tool = _by_name(build_tools(cast(Perception, perception)), "ask_about_image")
    await tool.ainvoke({"uri": "a.png", "question": "How many?", "where": {"x": 0.5, "w": 0.5}})
    assert perception.invoked == (
        "a.png",
        "ask_about_image",
        {"x": 0.5, "w": 0.5, "question": "How many?"},
    )


@pytest.mark.asyncio
async def test_ask_about_image_needs_no_where() -> None:
    perception = RecordingPerception()
    tool = _by_name(build_tools(cast(Perception, perception)), "ask_about_image")
    await tool.ainvoke({"uri": "a.png", "question": "What is this?"})
    assert perception.invoked is not None
    assert perception.invoked[2] == {"question": "What is this?"}


@pytest.mark.asyncio
async def test_ask_about_image_never_inspects_the_file() -> None:
    """The tool layer knows nothing about kinds — an inspect call here would
    mean it had started making decisions it must not make."""
    perception = RecordingPerception()
    tool = _by_name(build_tools(cast(Perception, perception)), "ask_about_image")
    await tool.ainvoke({"uri": "a.png", "question": "q"})
    assert perception.inspected == []


@pytest.mark.asyncio
async def test_a_file_without_the_affordance_lists_what_it_does_have() -> None:
    perception = RaisingPerception(UnknownAffordanceError("ask_about_image", ["read_range"]))
    tool = _by_name(build_tools(cast(Perception, perception)), "ask_about_image")
    result = await tool.ainvoke({"uri": "a.txt", "question": "q"})
    assert "read_range" in result


def test_the_tool_list_is_the_same_length_for_every_file() -> None:
    """The docstring's rule: the tool list never varies with what was last
    looked at. Four tools, always."""
    assert len(build_tools(cast(Perception, RecordingPerception()))) == 4


def test_a_renditions_provenance_reaches_the_reader() -> None:
    """A field nobody renders is a field nobody reads.

    A page image converted through LibreOffice is a *rendering* of a document,
    not the document — fonts substitute. That fact belongs next to the image,
    where a model deciding whether the type looks wrong can see it, rather than
    in a dataclass the tool pack drops on the floor.
    """
    rendition = Rendition(
        locator=PageRef(4),
        content=TextContent("the fourth slide"),
        degradations=(
            Degradation(
                what="rendered by a converter",
                detail="fonts may have been substituted",
            ),
        ),
    )

    rendered = _render_rendition(rendition)

    assert "rendered by a converter" in rendered
    assert "fonts may have been substituted" in rendered


def test_a_rendition_with_nothing_to_report_says_nothing() -> None:
    """The overwhelmingly common case must not grow a noise line."""
    rendered = _render_rendition(Rendition(locator=PageRef(1), content=TextContent("plain")))
    assert rendered == "located at PageRef(page=1):\nplain"
