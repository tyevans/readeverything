"""`Perception.invoke` consults the artifact store, without guessing at unions."""

from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.detection import PuremagicDetector
from readeverything.adapters.hashing import ContentHasher
from readeverything.adapters.local_source import LocalFileSource
from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.domain.identity import SourceRef
from readeverything.domain.locators import ByteRange
from readeverything.domain.rendition import Rendition
from readeverything.handlers.binary import BinaryHandler
from readeverything.handlers.text import TextHandler
from readeverything.pipeline.perception import Perception
from readeverything.ports.handler import MediaHandler
from readeverything.ports.source import SourceReader
from readeverything.registry.registry import MimeTypeRegistry


class CountingTextHandler(TextHandler):
    """Wraps `TextHandler`, counting `invoke` calls, delegating the real work."""

    handler_id: ClassVar[str] = "counting-text"

    def __init__(self, *, source: SourceReader) -> None:
        super().__init__(source=source)
        self.invocations = 0

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition:
        self.invocations += 1
        return await super().invoke(ref, name, params)


class UncachedTextHandler(TextHandler):
    """Same as `TextHandler`, but opts out of the artifact cache."""

    handler_id: ClassVar[str] = "uncached-text"
    handler_version: ClassVar[int] = 0

    def __init__(self, *, source: SourceReader) -> None:
        super().__init__(source=source)
        self.invocations = 0

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition:
        self.invocations += 1
        return await super().invoke(ref, name, params)


def _perception(
    tmp_path: Path,
    *,
    handlers: tuple[MediaHandler, ...] | None = None,
    artifacts: InMemoryArtifactStore,
    capabilities: CapabilitySet | None = None,
) -> Perception:
    source = LocalFileSource(root=tmp_path)
    if handlers is None:
        handlers = (TextHandler(source=source), BinaryHandler(source=source))
    return Perception(
        source=source,
        detector=PuremagicDetector(),
        hasher=ContentHasher(source=source),
        registry=MimeTypeRegistry(
            handlers=handlers,
            capabilities=capabilities if capabilities is not None else CapabilitySet.empty(),
        ),
        artifacts=artifacts,
    )


@pytest.fixture(autouse=True)
def _files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"alpha\nbeta\n")
    (tmp_path / "a.bin").write_bytes(bytes(range(64)))


async def test_a_repeated_invoke_returns_an_identical_rendition(tmp_path: Path) -> None:
    """A hit and a miss must be indistinguishable in their result.

    This is the property that makes caching safe to have at all, and it is what
    the tests assert — not a speedup, which would be a timing assertion.
    """
    perception = _perception(tmp_path, artifacts=InMemoryArtifactStore())
    first = await perception.invoke("a.txt", "read_range", {"start": 0, "end": 5})
    second = await perception.invoke("a.txt", "read_range", {"start": 0, "end": 5})
    assert first == second


async def test_a_second_invoke_does_not_reenter_the_handler(tmp_path: Path) -> None:
    source = LocalFileSource(root=tmp_path)
    handler = CountingTextHandler(source=source)
    perception = _perception(tmp_path, handlers=(handler,), artifacts=InMemoryArtifactStore())
    await perception.invoke("a.txt", "read_range", {"start": 0, "end": 5})
    await perception.invoke("a.txt", "read_range", {"start": 0, "end": 5})
    assert handler.invocations == 1


async def test_different_params_do_not_share_an_artifact(tmp_path: Path) -> None:
    perception = _perception(tmp_path, artifacts=InMemoryArtifactStore())
    a = await perception.invoke("a.txt", "read_range", {"start": 0, "end": 5})
    b = await perception.invoke("a.txt", "read_range", {"start": 5, "end": 10})
    assert a != b


async def test_a_changed_capability_fingerprint_does_not_serve_the_old_artifact(
    tmp_path: Path,
) -> None:
    """Swapping the model must miss, or the index becomes a mixture.

    This is the component of the key that is easiest to forget and the one
    whose absence is silent.
    """
    store = InMemoryArtifactStore()
    first = await _perception(
        tmp_path,
        artifacts=store,
        capabilities=CapabilitySet.of({Capability.VISION: "model-a"}),
    ).invoke("a.txt", "read_range", {"start": 0, "end": 5})
    second = await _perception(
        tmp_path,
        artifacts=store,
        capabilities=CapabilitySet.of({Capability.VISION: "model-b"}),
    ).invoke("a.txt", "read_range", {"start": 0, "end": 5})
    assert first == second  # same input, same answer
    assert len(store.keys()) == 2  # but stored under two keys


async def test_a_handler_that_opts_out_is_not_cached(tmp_path: Path) -> None:
    """Cache participation is the handler's decision, not the pipeline's."""
    source = LocalFileSource(root=tmp_path)
    handler = UncachedTextHandler(source=source)
    perception = _perception(tmp_path, handlers=(handler,), artifacts=InMemoryArtifactStore())
    await perception.invoke("a.txt", "read_range", {"start": 0, "end": 5})
    await perception.invoke("a.txt", "read_range", {"start": 0, "end": 5})
    assert handler.invocations == 2


async def test_a_cached_hexdump_keeps_its_byte_range(tmp_path: Path) -> None:
    """The corruption case, asserted through the real pipeline rather than the codec.

    `BinaryHandler.hexdump` returns a ByteRange. If the codec ever regresses to
    shape-based union resolution, this fails here even if a codec unit test is
    deleted.
    """
    source = LocalFileSource(root=tmp_path)
    perception = _perception(
        tmp_path, handlers=(BinaryHandler(source=source),), artifacts=InMemoryArtifactStore()
    )
    fresh = await perception.invoke("a.bin", "hexdump", {"start": 0, "length": 16})
    cached = await perception.invoke("a.bin", "hexdump", {"start": 0, "length": 16})
    assert type(cached.locator) is ByteRange
    assert cached == fresh
