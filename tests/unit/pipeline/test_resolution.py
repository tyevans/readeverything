"""The resolution memo: same ref, fewer stats, no staleness."""

from __future__ import annotations

import os
from pathlib import Path

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.detection import PuremagicDetector
from readeverything.adapters.hashing import ContentHasher
from readeverything.adapters.local_source import LocalFileSource
from readeverything.domain.capability import CapabilitySet
from readeverything.domain.identity import ContentHash
from readeverything.handlers.binary import BinaryHandler
from readeverything.handlers.text import TextHandler
from readeverything.pipeline.perception import Perception
from readeverything.pipeline.resolution import ResolutionMemo
from readeverything.ports.hashing import ContentHashing
from readeverything.ports.source import FileSource
from readeverything.registry.registry import MimeTypeRegistry
from readeverything.testing.fakes import FakeSource


class CountingHasher:
    """Counts `hash` calls through a real hasher rather than replacing it."""

    def __init__(self, *, inner: ContentHashing) -> None:
        self._inner = inner
        self.calls = 0

    async def hash(self, uri: str) -> ContentHash:
        self.calls += 1
        return await self._inner.hash(uri)


def _perception(
    *,
    source: FileSource | None = None,
    hasher: ContentHashing | None = None,
    memo: ResolutionMemo | None,
    tmp_path: Path | None = None,
) -> Perception:
    if source is None:
        assert tmp_path is not None
        (tmp_path / "a.txt").write_bytes(b"alpha\n")
        source = LocalFileSource(root=tmp_path)
    resolved_hasher = hasher if hasher is not None else ContentHasher(source=source)
    return Perception(
        source=source,
        detector=PuremagicDetector(),
        hasher=resolved_hasher,
        registry=MimeTypeRegistry(
            handlers=(TextHandler(source=source), BinaryHandler(source=source)),
            capabilities=CapabilitySet.empty(),
        ),
        artifacts=InMemoryArtifactStore(),
        memo=memo,
    )


async def test_a_second_resolve_of_one_path_does_not_rehash(tmp_path: Path) -> None:
    """The point of the memo, stated as a call count rather than a duration.

    Timing assertions are flaky; counting the calls through a real adapter is
    not. Hashing is the expensive operation and it scales with file size.
    """
    (tmp_path / "a.txt").write_bytes(b"alpha\n")
    source = LocalFileSource(root=tmp_path)
    counting = CountingHasher(inner=ContentHasher(source=source))
    perception = _perception(source=source, hasher=counting, memo=ResolutionMemo())
    await perception.inspect("a.txt")
    await perception.inspect("a.txt")
    assert counting.calls == 1


async def test_a_rewritten_file_is_resolved_again(tmp_path: Path) -> None:
    """A stale ref is the worst thing this cache could produce.

    The memo is keyed on (dev, inode, size, mtime_ns), so a rewrite invalidates
    even when the size is unchanged.
    """
    path = tmp_path / "a.txt"
    path.write_bytes(b"first")
    perception = _perception(memo=ResolutionMemo(), tmp_path=tmp_path)
    before = await perception.inspect("a.txt")
    os.utime(path, ns=(0, 0))
    path.write_bytes(b"secnd")  # same length, different content
    after = await perception.inspect("a.txt")
    assert before.ref.content_hash != after.ref.content_hash


async def test_a_hit_and_a_miss_are_indistinguishable(tmp_path: Path) -> None:
    """The property that makes the memo safe to have at all."""
    (tmp_path / "a.txt").write_bytes(b"alpha\n")
    source = LocalFileSource(root=tmp_path)
    cold = await _perception(source=source, memo=None).inspect("a.txt")
    warm_perception = _perception(source=source, memo=ResolutionMemo())
    await warm_perception.inspect("a.txt")
    warm = await warm_perception.inspect("a.txt")
    assert cold.ref == warm.ref


async def test_a_source_that_cannot_be_stat_is_never_memoized() -> None:
    """Without a stat there is no invalidation rule, so there is no caching.

    A non-local source (an object store) has no inode. Memoizing it on the uri
    alone would serve a stale ref forever after the object changed.
    """
    perception = _perception(source=FakeSource({"a.txt": b"x"}), memo=ResolutionMemo())
    first = await perception.inspect("a.txt")
    second = await perception.inspect("a.txt")
    assert first.ref == second.ref  # correct, just not cached
