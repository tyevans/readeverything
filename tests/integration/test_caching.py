"""Caching, end to end: a file is hashed once, and a rewrite is seen."""

from __future__ import annotations

from pathlib import Path

import pytest

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.detection import PuremagicDetector
from readeverything.adapters.hashing import ContentHasher, StatMemo
from readeverything.adapters.local_source import LocalFileSource
from readeverything.composition import build_perception
from readeverything.domain.capability import CapabilitySet
from readeverything.domain.identity import ContentHash
from readeverything.handlers.binary import BinaryHandler
from readeverything.handlers.text import TextHandler
from readeverything.pipeline.perception import Perception
from readeverything.pipeline.resolution import ResolutionMemo
from readeverything.ports.hashing import ContentHashing
from readeverything.registry.registry import MimeTypeRegistry

pytestmark = pytest.mark.integration


class CountingHasher:
    """Wraps a real hasher and counts calls through it.

    This is the one place the tier's rule bends, and the bend is narrow on
    purpose: the hasher is real and does real work, and this only observes how
    often it is asked. Counting is not faking — a fake would answer instead of
    the adapter, and then the test would be about the fake.
    """

    def __init__(self, *, inner: ContentHashing) -> None:
        self._inner = inner
        self.calls = 0

    async def hash(self, uri: str) -> ContentHash:
        self.calls += 1
        return await self._inner.hash(uri)


async def test_a_second_look_at_a_file_hashes_it_once(media_root: Path) -> None:
    """One inspect plus one invoke used to hash the whole file twice.

    Asserted as a call count rather than a duration: timing assertions are
    flaky, and the claim being made is about repeated work, not about speed.
    """
    source = LocalFileSource(root=media_root)
    hasher = CountingHasher(inner=ContentHasher(source=source, memo=StatMemo()))
    perception = Perception(
        source=source,
        detector=PuremagicDetector(),
        hasher=hasher,
        registry=MimeTypeRegistry(
            handlers=[TextHandler(source=source), BinaryHandler(source=source)],
            capabilities=CapabilitySet.empty(),
        ),
        artifacts=InMemoryArtifactStore(),
        memo=ResolutionMemo(),
    )
    await perception.inspect("notes.txt")
    await perception.invoke("notes.txt", "read_range", {"start": 0, "end": 4})
    assert hasher.calls == 1


async def test_a_rewritten_file_is_seen_again(media_root: Path) -> None:
    """The failure mode that would make caching worse than no caching."""
    perception = await build_perception(media_root, probe_binaries=False)
    before = await perception.inspect("notes.txt")
    (media_root / "notes.txt").write_text("completely different content here")
    after = await perception.inspect("notes.txt")
    assert before.ref.content_hash != after.ref.content_hash


async def test_a_corrupted_cache_entry_is_a_miss_not_a_permanent_failure(
    media_root: Path,
) -> None:
    """A miss costs a recomputation; a guess costs a wrong answer — and a raise
    is worse than either, because a persistent store survives version changes
    and nothing evicts a bad entry from it.
    """
    store = InMemoryArtifactStore()
    perception = await build_perception(media_root, artifacts=store, probe_binaries=False)
    first = await perception.invoke("notes.txt", "read_range", {"start": 0, "end": 4})

    (key,) = store.keys()
    store._entries[key] = b"not json at all"  # deliberately corrupting the entry

    second = await perception.invoke("notes.txt", "read_range", {"start": 0, "end": 4})
    assert second.content == first.content
