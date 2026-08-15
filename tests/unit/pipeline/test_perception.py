from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.detection import PuremagicDetector
from readeverything.adapters.hashing import ContentHasher
from readeverything.adapters.local_source import LocalFileSource
from readeverything.domain.affordance import Affordance, DetailLevel
from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.domain.errors import UnknownAffordanceError
from readeverything.domain.identity import MediaKind
from readeverything.domain.rendition import Budget, TextContent
from readeverything.handlers.binary import BinaryHandler
from readeverything.handlers.text import TextHandler
from readeverything.pipeline.perception import Perception
from readeverything.registry.registry import MimeTypeRegistry


@pytest.fixture
def perception(tmp_path: Path) -> Perception:
    (tmp_path / "notes.txt").write_bytes(b"alpha\nbeta\n")
    (tmp_path / "blob.bin").write_bytes(bytes(range(64)))
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


async def test_inspect_dispatches_text_to_the_text_handler(perception: Perception) -> None:
    card = await perception.inspect("notes.txt")
    assert card.kind is MediaKind.TEXT
    assert card.facts["lines"] == 2


async def test_inspect_falls_back_for_unknown_binary(perception: Perception) -> None:
    card = await perception.inspect("blob.bin")
    assert card.kind is MediaKind.BINARY


async def test_the_card_only_offers_available_affordances(perception: Perception) -> None:
    card = await perception.inspect("notes.txt")
    assert card.affordance_names() == ("read_range",)


async def test_invoke_routes_to_the_resolved_handler(perception: Perception) -> None:
    rendition = await perception.invoke("notes.txt", "read_range", {"start": 0, "end": 5})
    assert isinstance(rendition.content, TextContent)
    assert rendition.content.text == "alpha"


async def test_invoke_validates_params_against_the_declared_schema(
    perception: Perception,
) -> None:
    with pytest.raises(ValueError):
        await perception.invoke("notes.txt", "read_range", {"start": -5, "end": 1})


async def test_invoke_refuses_an_affordance_the_resolved_handler_does_not_declare(
    perception: Perception,
) -> None:
    """hexdump belongs to BinaryHandler; a text file never reaches it."""
    with pytest.raises(UnknownAffordanceError):
        await perception.invoke("notes.txt", "hexdump", {})


async def test_represent_returns_a_covering_map(perception: Perception) -> None:
    rendered = await perception.represent("notes.txt", Budget(max_chars=None))
    assert rendered.locator_map.length == len(rendered.text)


async def test_list_walks_the_tree(perception: Perception) -> None:
    assert sorted(await perception.list(".")) == ["blob.bin", "notes.txt"]


class _GatedParams(BaseModel):
    pass


class _GatedHandler(TextHandler):
    """A text handler with one extra affordance that requires VISION."""

    handler_id: ClassVar[str] = "gated"

    def affordances(self) -> tuple[Affordance, ...]:
        return (
            *super().affordances(),
            Affordance(
                name="describe_layout",
                description="Describe the visual layout of the text.",
                params=_GatedParams,
                requires=frozenset({Capability.VISION}),
                level=DetailLevel.DEEP,
            ),
        )


def _perception_with(capabilities: CapabilitySet, tmp_path: Path) -> Perception:
    (tmp_path / "notes.txt").write_bytes(b"alpha\nbeta\n")
    source = LocalFileSource(root=tmp_path)
    return Perception(
        source=source,
        detector=PuremagicDetector(),
        hasher=ContentHasher(source=source),
        registry=MimeTypeRegistry(
            handlers=(_GatedHandler(source=source), BinaryHandler(source=source)),
            capabilities=capabilities,
        ),
        artifacts=InMemoryArtifactStore(),
    )


async def test_a_capability_gated_affordance_is_hidden_without_the_capability(
    tmp_path: Path,
) -> None:
    """The agent never sees a tool this deployment cannot serve."""
    perception = _perception_with(CapabilitySet.empty(), tmp_path)
    card = await perception.inspect("notes.txt")
    assert "describe_layout" not in card.affordance_names()
    with pytest.raises(UnknownAffordanceError):
        await perception.invoke("notes.txt", "describe_layout", {})


async def test_a_capability_gated_affordance_appears_when_the_capability_is_present(
    tmp_path: Path,
) -> None:
    """...and does see it when the deployment can serve it."""
    perception = _perception_with(CapabilitySet.of({Capability.VISION: "fake-vision@1"}), tmp_path)
    card = await perception.inspect("notes.txt")
    assert "describe_layout" in card.affordance_names()
