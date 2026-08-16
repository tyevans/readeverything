"""`NestedSource`: delegation, resolution, limits and the two guards."""

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from readeverything.adapters.detection import PuremagicDetector
from readeverything.adapters.local_source import LocalFileSource
from readeverything.adapters.nested_source import CompositeOpener, NestedSource
from readeverything.adapters.tar_archive import TarArchiveOpener
from readeverything.adapters.zip_archive import ZipArchiveOpener
from readeverything.domain.errors import ContainerLimitExceededError, SourceUnreadableError
from readeverything.domain.identity import MimeType
from readeverything.ports.containers import ContainerLimits


def _openers() -> CompositeOpener:
    return CompositeOpener(openers=[ZipArchiveOpener(), TarArchiveOpener()])


def _nested(root: Path, limits: ContainerLimits | None = None) -> NestedSource:
    return NestedSource(
        LocalFileSource(root=root),
        limits=ContainerLimits() if limits is None else limits,
        archives=_openers(),
        detector=PuremagicDetector(),
    )


def _zip(root: Path, name: str, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(root / name, "w", zipfile.ZIP_DEFLATED) as archive:
        for member, data in members.items():
            archive.writestr(member, data)


def _targz(root: Path, name: str, members: dict[str, bytes]) -> None:
    with tarfile.open(root / name, "w:gz") as archive:
        for member, data in members.items():
            info = tarfile.TarInfo(member)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


async def test_a_single_segment_uri_delegates_verbatim(tmp_path: Path) -> None:
    """The compatibility story: loose files behave as if this were not installed."""
    (tmp_path / "a.txt").write_bytes(b"hello")
    source = _nested(tmp_path)
    assert await source.read_bytes("a.txt") == b"hello"
    assert await source.size("a.txt") == 5
    assert await source.exists("a.txt")
    assert await source.read_range("a.txt", 1, 3) == b"el"
    assert await source.local_path("a.txt") == str(tmp_path / "a.txt")
    assert [c async for c in source.stream("a.txt", chunk_size=2)] == [b"he", b"ll", b"o"]


async def test_reads_a_zip_member(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"hello world"})
    assert await _nested(tmp_path).read_bytes("a.zip!inner.txt") == b"hello world"


async def test_reads_a_member_two_containers_deep(tmp_path: Path) -> None:
    """The §1.1 shape, at the source layer."""
    inner = tmp_path / "build"
    inner.mkdir()
    _targz(inner, "nested.tar.gz", {"notes.txt": b"deep"})
    _zip(tmp_path, "docs.zip", {"nested.tar.gz": (inner / "nested.tar.gz").read_bytes()})
    source = _nested(tmp_path)
    try:
        assert await source.read_bytes("docs.zip!nested.tar.gz!notes.txt") == b"deep"
    finally:
        await source.aclose()


async def test_size_is_the_uncompressed_size(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"x" * 5000})
    assert await _nested(tmp_path).size("a.zip!inner.txt") == 5000


async def test_exists_is_true_for_a_present_member(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"hi"})
    source = _nested(tmp_path)
    assert await source.exists("a.zip!inner.txt")
    assert not await source.exists("a.zip!nope.txt")


async def test_exists_is_false_when_the_container_is_missing(tmp_path: Path) -> None:
    """`exists` answers a question; it does not raise about a bad guess."""
    assert not await _nested(tmp_path).exists("gone.zip!inner.txt")


async def test_read_range_slices_a_member(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"hello world"})
    assert await _nested(tmp_path).read_range("a.zip!inner.txt", 6, 11) == b"world"


async def test_stream_chunks_a_member(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"abcdef"})
    source = _nested(tmp_path)
    chunks = [c async for c in source.stream("a.zip!inner.txt", chunk_size=2)]
    assert chunks == [b"ab", b"cd", b"ef"]


async def test_local_path_materialises_a_member(tmp_path: Path) -> None:
    """The one place the cost is acknowledged rather than hidden."""
    _zip(tmp_path, "a.zip", {"inner.txt": b"hello"})
    source = _nested(tmp_path)
    try:
        path = Path(await source.local_path("a.zip!inner.txt"))
        assert path.read_bytes() == b"hello"
    finally:
        await source.aclose()


async def test_local_path_is_stable_across_calls(tmp_path: Path) -> None:
    """`stat_key` compares inodes; a new temp file per call would thrash."""
    _zip(tmp_path, "a.zip", {"inner.txt": b"hello"})
    source = _nested(tmp_path)
    try:
        first = await source.local_path("a.zip!inner.txt")
        assert await source.local_path("a.zip!inner.txt") == first
    finally:
        await source.aclose()


async def test_aclose_removes_materialised_members(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"hello"})
    source = _nested(tmp_path)
    path = Path(await source.local_path("a.zip!inner.txt"))
    await source.aclose()
    assert not path.exists()


async def test_a_missing_member_raises(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"hi"})
    with pytest.raises(SourceUnreadableError, match=r"nope\.txt"):
        await _nested(tmp_path).read_bytes("a.zip!nope.txt")


async def test_a_container_with_no_opener_raises(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"not an archive")
    with pytest.raises(SourceUnreadableError, match="not a container"):
        await _nested(tmp_path).read_bytes("a.txt!inner.txt")


async def test_depth_beyond_the_limit_is_refused(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"hi"})
    source = _nested(tmp_path, ContainerLimits(max_depth=0))
    with pytest.raises(ContainerLimitExceededError, match="max_depth"):
        await source.read_bytes("a.zip!inner.txt")


async def test_a_traversing_member_is_refused(tmp_path: Path) -> None:
    """A member path is never resolved against the filesystem, so the root
    guard in `LocalFileSource` cannot see this one. It needs its own."""
    _zip(tmp_path, "a.zip", {"inner.txt": b"hi"})
    with pytest.raises(SourceUnreadableError, match="traversal"):
        await _nested(tmp_path).read_bytes("a.zip!../../etc/passwd")


async def test_an_absolute_member_is_refused(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"hi"})
    with pytest.raises(SourceUnreadableError, match="absolute"):
        await _nested(tmp_path).read_bytes("a.zip!/etc/passwd")


async def test_a_windows_drive_member_is_refused(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"hi"})
    with pytest.raises(SourceUnreadableError, match="absolute"):
        await _nested(tmp_path).read_bytes("a.zip!C:/windows/system32/config/sam")


async def test_a_backslash_traversal_is_refused(tmp_path: Path) -> None:
    """A zip written on Windows separates with `\\`, and `..` still means `..`."""
    _zip(tmp_path, "a.zip", {"inner.txt": b"hi"})
    with pytest.raises(SourceUnreadableError, match="traversal"):
        await _nested(tmp_path).read_bytes("a.zip!..\\..\\etc\\passwd")


async def test_a_member_over_the_byte_limit_is_refused(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"x" * 4096})
    source = _nested(tmp_path, ContainerLimits(max_member_bytes=64))
    with pytest.raises(ContainerLimitExceededError, match="max_member_bytes"):
        await source.read_bytes("a.zip!inner.txt")


async def test_the_expansion_ratio_fires_mid_stream_on_a_bomb(tmp_path: Path) -> None:
    """The check that matters. A bomb lies in its header, so the guard runs
    against bytes ACTUALLY WRITTEN, not against a declared size."""
    _zip(tmp_path, "bomb.zip", {"payload": b"\0" * (1 << 22)})
    source = _nested(tmp_path, ContainerLimits(max_expansion_ratio=2.0))
    with pytest.raises(ContainerLimitExceededError, match="expansion"):
        await source.read_bytes("bomb.zip!payload")


async def test_a_container_with_too_many_members_is_refused(tmp_path: Path) -> None:
    _zip(tmp_path, "many.zip", {f"f{n}.txt": b"x" for n in range(20)})
    source = _nested(tmp_path, ContainerLimits(max_members=5))
    with pytest.raises(ContainerLimitExceededError, match="max_members"):
        await source.read_bytes("many.zip!f0.txt")


async def test_a_container_whose_total_exceeds_the_limit_is_refused(tmp_path: Path) -> None:
    _zip(tmp_path, "big.zip", {f"f{n}.txt": b"x" * 1000 for n in range(10)})
    source = _nested(tmp_path, ContainerLimits(max_total_bytes=500))
    with pytest.raises(ContainerLimitExceededError, match="max_total_bytes"):
        await source.read_bytes("big.zip!f0.txt")


async def test_a_symlink_member_of_a_tar_is_refused_on_read(tmp_path: Path) -> None:
    path = tmp_path / "l.tar"
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("passwd")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)
    with pytest.raises(SourceUnreadableError, match="symlink"):
        await _nested(tmp_path).read_bytes("l.tar!passwd")


def test_the_composite_reports_no_opener_for_an_unknown_mime() -> None:
    assert _openers().opener_for(MimeType.parse("text/plain")) is None


def test_the_composite_claims_what_its_openers_claim() -> None:
    composite = _openers()
    assert composite.claims(MimeType.parse("application/zip"))
    assert composite.claims(MimeType.parse("application/gzip"))
    assert not composite.claims(MimeType.parse("image/png"))


async def test_the_composite_refuses_to_be_used_as_a_single_opener() -> None:
    """It has no one format, so answering `entries` would be a guess."""
    composite = _openers()
    with pytest.raises(NotImplementedError):
        await composite.entries("anything")
    with pytest.raises(NotImplementedError):
        composite.open_member("anything", "member")
