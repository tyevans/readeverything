from pathlib import Path

import pytest

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.detection import PuremagicDetector
from readeverything.adapters.hashing import ContentHasher
from readeverything.adapters.local_source import LocalFileSource
from readeverything.domain.capability import CapabilitySet
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


async def test_invoke_refuses_an_unavailable_affordance(perception: Perception) -> None:
    from readeverything.domain.errors import UnknownAffordanceError

    with pytest.raises(UnknownAffordanceError):
        await perception.invoke("notes.txt", "hexdump", {})


async def test_represent_returns_a_covering_map(perception: Perception) -> None:
    rendered = await perception.represent("notes.txt", Budget(max_chars=None))
    assert rendered.locator_map.length == len(rendered.text)


async def test_list_walks_the_tree(perception: Perception) -> None:
    assert sorted(await perception.list(".")) == ["blob.bin", "notes.txt"]
