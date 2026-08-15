"""The laws every `ArtifactStore` must obey."""

from __future__ import annotations

import pytest


class ArtifactStoreCompliance:
    """Subclass and supply a `store` fixture."""

    @pytest.fixture
    def store(self) -> object:
        raise NotImplementedError("supply a `store` fixture")

    async def test_a_miss_returns_none(self, store) -> None:  # type: ignore[no-untyped-def]
        assert await store.get("absent-key") is None  # nosec B101 -- compliance-suite assertion, not a defensive check

    async def test_a_stored_artifact_round_trips(self, store) -> None:  # type: ignore[no-untyped-def]
        await store.put("k", b"value")
        assert await store.get("k") == b"value"  # nosec B101 -- compliance-suite assertion, not a defensive check

    async def test_entries_are_immutable(self, store) -> None:  # type: ignore[no-untyped-def]
        """A second put must not error and must not replace."""
        await store.put("k", b"first")
        await store.put("k", b"second")
        assert await store.get("k") == b"first"  # nosec B101 -- compliance-suite assertion, not a defensive check

    async def test_arbitrary_bytes_survive(self, store) -> None:  # type: ignore[no-untyped-def]
        await store.put("k", bytes(range(256)))
        assert await store.get("k") == bytes(range(256))  # nosec B101 -- compliance-suite assertion, not a defensive check
