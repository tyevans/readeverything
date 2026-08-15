"""The `ArtifactStore` law, run against both real stores.

The suite has shipped in the wheel since Plan 1 and no implementation had ever
subclassed it. An unexercised law is not a law that passes — it is one that has
never been given the chance to fail, which is the thing this project says it
does not do.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from readeverything.adapters.artifact_store import FilesystemArtifactStore, InMemoryArtifactStore
from readeverything.testing.artifact_compliance import ArtifactStoreCompliance


class TestInMemoryArtifactStoreCompliance(ArtifactStoreCompliance):
    @pytest.fixture
    def store(self) -> InMemoryArtifactStore:
        return InMemoryArtifactStore()


class TestFilesystemArtifactStoreCompliance(ArtifactStoreCompliance):
    @pytest.fixture
    def store(self, tmp_path: Path) -> FilesystemArtifactStore:
        return FilesystemArtifactStore(root=tmp_path)
