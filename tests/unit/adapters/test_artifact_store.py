import asyncio
from pathlib import Path

import pytest

from readeverything.adapters.artifact_store import (
    FilesystemArtifactStore,
    InMemoryArtifactStore,
)


@pytest.fixture(params=["memory", "filesystem"])
def store(request: pytest.FixtureRequest, tmp_path: Path):  # type: ignore[no-untyped-def]
    if request.param == "memory":
        return InMemoryArtifactStore()
    return FilesystemArtifactStore(root=tmp_path)


async def test_a_miss_returns_none(store) -> None:  # type: ignore[no-untyped-def]
    assert await store.get("absent") is None


async def test_a_stored_artifact_round_trips(store) -> None:  # type: ignore[no-untyped-def]
    await store.put("k", b"value")
    assert await store.get("k") == b"value"


async def test_putting_an_existing_key_is_a_noop_not_an_error(store) -> None:  # type: ignore[no-untyped-def]
    """Entries are immutable; a concurrent re-derivation must not explode."""
    await store.put("k", b"first")
    await store.put("k", b"first")
    assert await store.get("k") == b"first"


async def test_binary_content_survives(store) -> None:  # type: ignore[no-untyped-def]
    await store.put("k", bytes(range(256)))
    assert await store.get("k") == bytes(range(256))


async def test_a_key_with_path_characters_is_stored_safely(tmp_path: Path) -> None:
    """A key must never be able to escape the store's root."""
    fs = FilesystemArtifactStore(root=tmp_path)
    await fs.put("../escape", b"x")
    assert await fs.get("../escape") == b"x"
    assert not (tmp_path.parent / "escape").exists()


async def test_concurrent_writers_do_not_share_a_temp_file(tmp_path: Path) -> None:
    """Two writers of the same key must not collide on one .partial file."""
    fs = FilesystemArtifactStore(root=tmp_path)
    await asyncio.gather(*(fs.put("k", b"value") for _ in range(8)))
    assert await fs.get("k") == b"value"
    assert not list(tmp_path.rglob("*.partial"))
