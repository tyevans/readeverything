from pathlib import Path

from readeverything.adapters.hashing import ContentHasher, StatMemo
from readeverything.adapters.local_source import LocalFileSource
from readeverything.domain.identity import ContentHash


async def test_identical_bytes_hash_identically(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"hello")
    (tmp_path / "b.txt").write_bytes(b"hello")
    hasher = ContentHasher(source=LocalFileSource(root=tmp_path))
    assert await hasher.hash("a.txt") == await hasher.hash("b.txt")


async def test_different_bytes_hash_differently(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"hello")
    (tmp_path / "b.txt").write_bytes(b"world")
    hasher = ContentHasher(source=LocalFileSource(root=tmp_path))
    assert await hasher.hash("a.txt") != await hasher.hash("b.txt")


async def test_the_memo_is_an_optimisation_only(tmp_path: Path) -> None:
    """A cold memo must produce the same answer as a warm one."""
    (tmp_path / "a.txt").write_bytes(b"hello")
    source = LocalFileSource(root=tmp_path)
    warm = ContentHasher(source=source, memo=StatMemo())
    first = await warm.hash("a.txt")
    second = await warm.hash("a.txt")
    cold = await ContentHasher(source=source).hash("a.txt")
    assert first == second == cold


async def test_editing_a_file_changes_its_hash_through_the_memo(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_bytes(b"hello")
    hasher = ContentHasher(source=LocalFileSource(root=tmp_path), memo=StatMemo())
    before = await hasher.hash("a.txt")
    path.write_bytes(b"hello there")
    assert await hasher.hash("a.txt") != before


def test_a_memo_for_a_vanished_file_misses_rather_than_raising(tmp_path: Path) -> None:
    """A file deleted between put and get must be a miss.

    The memo is an optimisation: a miss costs a rehash, never a wrong answer.
    Raising here would turn a stale optimisation into a failed read.
    """
    path = tmp_path / "gone.txt"
    path.write_bytes(b"data")
    memo = StatMemo()
    memo.put(path, ContentHash("abc"))
    path.unlink()
    assert memo.get(path) is None
