"""The tar opener, including the solid-container materialisation."""

import io
import tarfile
from pathlib import Path

import pytest

from readeverything.adapters.tar_archive import TarArchiveOpener
from readeverything.domain.errors import ContainerLimitExceededError, SourceUnreadableError
from readeverything.domain.identity import MimeType


def _tar(tmp_path: Path, members: dict[str, bytes], *, mode: str = "w", name: str = "a.tar") -> str:
    path = tmp_path / name
    with tarfile.open(path, mode) as archive:  # type: ignore[call-overload]
        for member_name, data in members.items():
            info = tarfile.TarInfo(member_name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return str(path)


def test_claims_tar_family_and_not_zip() -> None:
    opener = TarArchiveOpener()
    assert opener.claims(MimeType.parse("application/x-tar"))
    assert opener.claims(MimeType.parse("application/gzip"))
    assert opener.claims(MimeType.parse("application/x-xz"))
    assert not opener.claims(MimeType.parse("application/zip"))


async def test_lists_entries(tmp_path: Path) -> None:
    path = _tar(tmp_path, {"a.txt": b"hello", "b.txt": b"xy"})
    entries = {e.path: e for e in await TarArchiveOpener().entries(path)}
    assert entries["a.txt"].size_bytes == 5
    assert entries["b.txt"].size_bytes == 2


async def test_an_uncompressed_tar_reports_byte_offsets(tmp_path: Path) -> None:
    path = _tar(tmp_path, {"a.txt": b"hello"})
    (entry,) = await TarArchiveOpener().entries(path)
    assert entry.byte_offset is not None


async def test_a_solid_tar_reports_no_byte_offsets(tmp_path: Path) -> None:
    """Offsets into a gzip stream are not offsets a caller can seek to."""
    path = _tar(tmp_path, {"a.txt": b"hello"}, mode="w:gz", name="a.tar.gz")
    (entry,) = await TarArchiveOpener().entries(path)
    assert entry.byte_offset is None


async def test_reads_a_member_from_a_plain_tar(tmp_path: Path) -> None:
    path = _tar(tmp_path, {"a.txt": b"hello world"})
    opener = TarArchiveOpener()
    assert b"".join([c async for c in opener.open_member(path, "a.txt")]) == b"hello world"
    assert not opener.materialised


async def test_reads_a_member_from_a_solid_archive(tmp_path: Path) -> None:
    path = _tar(tmp_path, {"a.txt": b"hello world"}, mode="w:gz", name="a.tar.gz")
    opener = TarArchiveOpener()
    try:
        chunks = [c async for c in opener.open_member(path, "a.txt")]
        assert b"".join(chunks) == b"hello world"
    finally:
        await opener.aclose()


async def test_a_solid_archive_is_decompressed_once(tmp_path: Path) -> None:
    """Three member reads must not mean three full decompressions."""
    path = _tar(
        tmp_path, {"a.txt": b"a", "b.txt": b"b", "c.txt": b"c"}, mode="w:gz", name="a.tar.gz"
    )
    opener = TarArchiveOpener()
    try:
        for member in ("a.txt", "b.txt", "c.txt"):
            assert [c async for c in opener.open_member(path, member)] == [member[0].encode()]
        assert len(opener.materialised) == 1
    finally:
        await opener.aclose()


async def test_aclose_removes_the_materialised_copy(tmp_path: Path) -> None:
    path = _tar(tmp_path, {"a.txt": b"hi"}, mode="w:gz", name="a.tar.gz")
    opener = TarArchiveOpener()
    [c async for c in opener.open_member(path, "a.txt")]
    (temp,) = opener.materialised.values()
    assert Path(temp).exists()
    await opener.aclose()
    assert not Path(temp).exists()


async def test_the_cache_evicts_least_recently_used_rather_than_failing(
    tmp_path: Path,
) -> None:
    """Spec §3.2: a directory of large tarballs should be SLOW, not unreadable.

    Failing at the bound would make it unreadable, so the bound evicts.
    """
    first = _tar(tmp_path, {"a.txt": b"x" * 4096}, mode="w:gz", name="one.tar.gz")
    second = _tar(tmp_path, {"b.txt": b"y" * 4096}, mode="w:gz", name="two.tar.gz")
    # Room for one decompressed tarball, not two.
    opener = TarArchiveOpener(max_materialised_bytes=20_000)
    try:
        assert b"".join([c async for c in opener.open_member(first, "a.txt")]) == b"x" * 4096
        assert b"".join([c async for c in opener.open_member(second, "b.txt")]) == b"y" * 4096
        assert list(opener.materialised) == [second]
        # And the evicted one is still READABLE, just paid for again.
        assert b"".join([c async for c in opener.open_member(first, "a.txt")]) == b"x" * 4096
        assert list(opener.materialised) == [first]
    finally:
        await opener.aclose()


async def test_one_archive_larger_than_the_whole_bound_is_refused(tmp_path: Path) -> None:
    """Eviction cannot help here: there is nothing left to evict."""
    path = _tar(tmp_path, {"a.txt": b"x" * 4096}, mode="w:gz", name="a.tar.gz")
    opener = TarArchiveOpener(max_materialised_bytes=16)
    try:
        with pytest.raises(ContainerLimitExceededError, match="materialis"):
            [c async for c in opener.open_member(path, "a.txt")]
    finally:
        await opener.aclose()


@pytest.mark.parametrize(
    ("mode", "name"),
    [("w:bz2", "a.tar.bz2"), ("w:xz", "a.tar.xz")],
)
async def test_reads_the_other_solid_compressions(tmp_path: Path, mode: str, name: str) -> None:
    """bzip2 and xz take the same solid path gzip does, via their own magic."""
    path = _tar(tmp_path, {"a.txt": b"hello world"}, mode=mode, name=name)
    opener = TarArchiveOpener()
    try:
        assert b"".join([c async for c in opener.open_member(path, "a.txt")]) == b"hello world"
        assert list(opener.materialised) == [path]
    finally:
        await opener.aclose()


async def test_a_missing_member_raises(tmp_path: Path) -> None:
    path = _tar(tmp_path, {"a.txt": b"hi"})
    with pytest.raises(SourceUnreadableError, match=r"nope\.txt"):
        [c async for c in TarArchiveOpener().open_member(path, "nope.txt")]


async def test_an_absent_archive_raises(tmp_path: Path) -> None:
    with pytest.raises(SourceUnreadableError):
        await TarArchiveOpener().entries(str(tmp_path / "gone.tar"))


async def test_a_directory_member_has_no_readable_content(tmp_path: Path) -> None:
    path = tmp_path / "d.tar"
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("sub")
        info.type = tarfile.DIRTYPE
        archive.addfile(info)
    with pytest.raises(SourceUnreadableError, match="no readable content"):
        [c async for c in TarArchiveOpener().open_member(str(path), "sub")]


async def test_a_symlink_member_is_reported_and_not_opened(tmp_path: Path) -> None:
    """A tarball can carry a link to /etc/passwd. Following it is the hole."""
    path = tmp_path / "l.tar"
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("passwd")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)
    (entry,) = await TarArchiveOpener().entries(str(path))
    assert entry.is_symlink
    with pytest.raises(SourceUnreadableError, match="symlink"):
        [c async for c in TarArchiveOpener().open_member(str(path), "passwd")]


async def test_a_corrupt_archive_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.tar.gz"
    path.write_bytes(b"\x1f\x8b and then garbage")
    with pytest.raises(SourceUnreadableError):
        await TarArchiveOpener().entries(str(path))
