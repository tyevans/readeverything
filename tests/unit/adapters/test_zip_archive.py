"""The zip opener, against archives built in a tmpdir rather than committed."""

import zipfile
from pathlib import Path

import pytest

from readeverything.adapters.zip_archive import ZipArchiveOpener
from readeverything.domain.errors import SourceUnreadableError
from readeverything.domain.identity import MimeType


def _zip(tmp_path: Path, members: dict[str, bytes]) -> str:
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return str(path)


def test_claims_zip_and_not_tar() -> None:
    opener = ZipArchiveOpener()
    assert opener.claims(MimeType.parse("application/zip"))
    assert not opener.claims(MimeType.parse("application/x-tar"))


async def test_lists_entries_with_sizes(tmp_path: Path) -> None:
    path = _zip(tmp_path, {"a.txt": b"hello", "b.txt": b"x" * 100})
    entries = {e.path: e for e in await ZipArchiveOpener().entries(path)}
    assert entries["a.txt"].size_bytes == 5
    assert entries["b.txt"].size_bytes == 100
    assert entries["b.txt"].compressed_bytes < 100


async def test_entries_carry_a_byte_offset(tmp_path: Path) -> None:
    """A zip is seekable, so every member has a real place in the file."""
    path = _zip(tmp_path, {"a.txt": b"hello"})
    (entry,) = [e for e in await ZipArchiveOpener().entries(path) if not e.is_dir]
    assert entry.byte_offset is not None
    assert entry.byte_offset >= 0


async def test_directory_entries_are_marked(tmp_path: Path) -> None:
    path = tmp_path / "d.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sub/", b"")
        archive.writestr("sub/a.txt", b"hi")
    entries = {e.path: e for e in await ZipArchiveOpener().entries(str(path))}
    assert entries["sub/"].is_dir
    assert not entries["sub/a.txt"].is_dir


async def test_reads_a_member(tmp_path: Path) -> None:
    path = _zip(tmp_path, {"a.txt": b"hello world"})
    chunks = [c async for c in ZipArchiveOpener().open_member(path, "a.txt")]
    assert b"".join(chunks) == b"hello world"


async def test_a_missing_member_raises(tmp_path: Path) -> None:
    path = _zip(tmp_path, {"a.txt": b"hi"})
    with pytest.raises(SourceUnreadableError, match=r"nope\.txt"):
        [c async for c in ZipArchiveOpener().open_member(path, "nope.txt")]


async def test_an_absent_archive_raises_on_open_member(tmp_path: Path) -> None:
    with pytest.raises(SourceUnreadableError):
        [c async for c in ZipArchiveOpener().open_member(str(tmp_path / "gone.zip"), "a.txt")]


async def test_a_corrupt_archive_raises_rather_than_returning_nothing(tmp_path: Path) -> None:
    """Silence would look like an empty archive, which is a false claim."""
    path = tmp_path / "broken.zip"
    path.write_bytes(b"PK\x03\x04 and then garbage")
    with pytest.raises(SourceUnreadableError):
        await ZipArchiveOpener().entries(str(path))


async def test_a_symlink_member_is_reported(tmp_path: Path) -> None:
    """Reported as a fact here; refusing to FOLLOW it is `NestedSource`'s job."""
    path = tmp_path / "l.zip"
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (0xA1FF) << 16
        archive.writestr(info, b"/etc/passwd")
    (entry,) = await ZipArchiveOpener().entries(str(path))
    assert entry.is_symlink


async def test_a_dos_written_entry_is_not_mistaken_for_a_symlink(tmp_path: Path) -> None:
    """`external_attr` only carries mode bits when a unix host wrote it."""
    path = tmp_path / "dos.zip"
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("a.txt")
        info.create_system = 0
        info.external_attr = (0xA1FF) << 16
        archive.writestr(info, b"hi")
    (entry,) = await ZipArchiveOpener().entries(str(path))
    assert not entry.is_symlink


async def test_a_corrupt_member_fails_on_read_without_blinding_its_neighbours(
    tmp_path: Path,
) -> None:
    """Spec §1.1: one bad member must not cost the agent the other entries."""
    path = Path(_zip(tmp_path, {"good.txt": b"fine", "bad.txt": b"x" * 200}))
    raw = bytearray(path.read_bytes())
    # Corrupt the deflate stream of the second member without touching the
    # central directory, so listing still succeeds and only the read fails.
    marker = raw.rindex(b"bad.txt")
    raw[marker - 40 : marker - 30] = b"\x00" * 10
    path.write_bytes(bytes(raw))
    opener = ZipArchiveOpener()
    listed = [e.path for e in await opener.entries(str(path))]
    assert "good.txt" in listed and "bad.txt" in listed
    with pytest.raises(SourceUnreadableError):
        [c async for c in opener.open_member(str(path), "bad.txt")]
