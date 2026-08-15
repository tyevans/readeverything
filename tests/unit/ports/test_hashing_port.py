# tests/unit/ports/test_hashing_port.py
from pathlib import Path

from readeverything.adapters.hashing import ContentHasher
from readeverything.adapters.local_source import LocalFileSource
from readeverything.domain.identity import ContentHash
from readeverything.ports.hashing import ContentHashing


class _PrecomputedHasher:
    """A caller supplying hashes from elsewhere — the case the port exists for."""

    def __init__(self, value: str) -> None:
        self._value = value

    async def hash(self, uri: str) -> ContentHash:
        return ContentHash(self._value)


def test_the_bundled_adapter_satisfies_the_port(tmp_path: Path) -> None:
    source = LocalFileSource(root=tmp_path)
    assert isinstance(ContentHasher(source=source), ContentHashing)


def test_an_unrelated_hasher_satisfies_the_port_without_inheriting() -> None:
    """Structural typing is the point: a caller must not have to subclass."""
    assert isinstance(_PrecomputedHasher("abc"), ContentHashing)


def test_an_object_without_hash_does_not_satisfy_the_port() -> None:
    assert not isinstance(object(), ContentHashing)
