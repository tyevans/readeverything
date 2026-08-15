from pathlib import Path

import pytest

from readeverything.adapters.local_source import LocalFileSource
from readeverything.domain.errors import SourceUnreadableError


async def test_reads_bytes(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"hello")
    assert await LocalFileSource(root=tmp_path).read_bytes("a.txt") == b"hello"


async def test_reads_a_range(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"hello world")
    assert await LocalFileSource(root=tmp_path).read_range("a.txt", 6, 11) == b"world"


async def test_streams_in_chunks(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"abcdef")
    chunks = [c async for c in LocalFileSource(root=tmp_path).stream("a.txt", chunk_size=2)]
    assert chunks == [b"ab", b"cd", b"ef"]


async def test_walk_returns_files_relative_to_the_root(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_bytes(b"x")
    (tmp_path / "b.txt").write_bytes(b"y")
    assert sorted(await LocalFileSource(root=tmp_path).walk(".")) == ["b.txt", "sub/a.txt"]


async def test_escaping_the_root_is_refused(tmp_path: Path) -> None:
    """A traversal must fail loudly, not read an unintended file."""
    with pytest.raises(SourceUnreadableError, match="outside the root"):
        await LocalFileSource(root=tmp_path).read_bytes("../etc/passwd")


async def test_a_missing_file_raises_the_domain_error(tmp_path: Path) -> None:
    with pytest.raises(SourceUnreadableError):
        await LocalFileSource(root=tmp_path).read_bytes("nope.txt")
