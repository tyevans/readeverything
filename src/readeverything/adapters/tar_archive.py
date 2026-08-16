"""Tar, through stdlib `tarfile`, and the solid-container problem.

`.tar` is seekable: a header chain gives every member an offset, and a ranged
read is a real one. `.tar.gz`, `.tar.bz2` and `.tar.xz` are SOLID -- the
compression wraps the whole archive, so member n cannot be reached without
decompressing 0..n-1, and reading three members naively costs three full
decompressions of the same file.

So a solid archive is decompressed ONCE into a temp file and every subsequent
read of any member goes through that copy, for the lifetime of this opener.
That is a cache with a bound (`max_materialised_bytes`), and at the bound it
EVICTS LEAST-RECENTLY-USED rather than failing: failing would make a directory
of large tarballs unreadable, where evicting only makes it slow.

The single case eviction cannot rescue -- one archive larger than the entire
bound -- does raise, and that is not a loophole in the eviction rule. Eviction
exists to turn "unreadable" into "slow"; when there is nothing left to evict
there is no slower alternative to offer, so raising is the honest outcome.
Do not "fix" that raise back into an eviction: there would be nothing to evict.

The temp files live in a `TemporaryDirectory` this instance owns, so they are
removed by `aclose()` and, failing that, by the directory's own finalizer at
interpreter exit. Nothing here is left behind on a crash path.
"""

from __future__ import annotations

import asyncio
import bz2
import gzip
import lzma
import tarfile
import tempfile
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from pathlib import Path
from typing import IO, cast

from readeverything.domain.errors import ContainerLimitExceededError, SourceUnreadableError
from readeverything.domain.identity import MimeType
from readeverything.ports.containers import ArchiveEntry

#: Every spelling of the tar family this opener answers `claims` for.
#: `application/gzip` is included because a `.tar.gz` is the common shape and
#: detection reports the OUTER compression, which is the honest answer about
#: the bytes -- so the opener, not the detector, is what discovers a tar
#: inside.
_MIMES = frozenset(
    {
        "application/x-tar",
        "application/x-gtar",
        "application/gzip",
        "application/x-gzip",
        "application/x-bzip2",
        "application/x-xz",
    }
)

# The three openers below each hand their stream straight to a `with` in
# `_decompress`, which is what closes it. `cast` because gzip/bz2/lzma return
# their own file classes, which are `IO[bytes]` in every way except mypy's
# nominal check.


def _open_gzip(path: str) -> IO[bytes]:
    return cast("IO[bytes]", gzip.open(path, "rb"))


def _open_bzip2(path: str) -> IO[bytes]:
    return cast("IO[bytes]", bz2.open(path, "rb"))


def _open_xz(path: str) -> IO[bytes]:
    return cast("IO[bytes]", lzma.open(path, "rb"))


#: Magic prefix -> the stdlib decompressor that reads it. A `.tgz` and a
#: `.tar.gz` are the same thing under two names, and a `.tar` someone gzipped
#: without renaming is the case an extension check gets wrong in the expensive
#: direction -- so this is keyed on the bytes.
_DECOMPRESSORS: tuple[tuple[bytes, Callable[[str], IO[bytes]]], ...] = (
    (b"\x1f\x8b", _open_gzip),
    (b"BZh", _open_bzip2),
    (b"\xfd7zXZ\x00", _open_xz),
)

_CHUNK = 1 << 20


def _decompressor(path: str) -> Callable[[str], IO[bytes]] | None:
    """How to decompress `path`, or None because it is already a plain tar."""
    with open(path, "rb") as handle:
        head = handle.read(6)
    for magic, opener in _DECOMPRESSORS:
        if head.startswith(magic):
            return opener
    return None


class TarArchiveOpener:
    """Reads the tar family, materialising a solid archive at most once."""

    def __init__(self, *, max_materialised_bytes: int = 8 << 30) -> None:
        self._max_materialised_bytes = max_materialised_bytes
        self._temp = tempfile.TemporaryDirectory(prefix="readeverything-tar-")
        #: Insertion-ordered and re-inserted on use, so the FIRST key is always
        #: the least recently used one.
        self._materialised: OrderedDict[str, str] = OrderedDict()
        self._sizes: dict[str, int] = {}
        self._held = 0
        self._counter = 0
        self._lock = asyncio.Lock()

    @property
    def materialised(self) -> Mapping[str, str]:
        """Solid archives decompressed so far, keyed by their original path.

        Public because the "decompressed once" promise is otherwise
        unobservable, and an unobservable promise is one that quietly stops
        being true.
        """
        return self._materialised

    async def aclose(self) -> None:
        """Remove every decompressed copy this opener made."""
        await asyncio.to_thread(self._temp.cleanup)
        self._materialised.clear()
        self._sizes.clear()
        self._held = 0

    def claims(self, mime: MimeType) -> bool:
        return str(mime) in _MIMES

    def _safe_decompressor(self, path: str) -> Callable[[str], IO[bytes]] | None:
        try:
            return _decompressor(path)
        except OSError as exc:
            raise SourceUnreadableError(f"cannot read tar {path!r}: {exc}") from exc

    async def _seekable_path(self, path: str) -> str:
        """`path` itself when it is a plain tar, else a decompressed copy."""
        decompress = await asyncio.to_thread(self._safe_decompressor, path)
        if decompress is None:
            return path
        async with self._lock:
            # Re-checked inside the lock: two concurrent reads of the same
            # tarball must decompress it once between them, not once each.
            existing = self._materialised.get(path)
            if existing is not None:
                self._materialised.move_to_end(path)
                return existing
            self._counter += 1
            target = str(Path(self._temp.name) / f"{self._counter}.tar")
            written = await asyncio.to_thread(self._decompress, decompress, path, target)
            self._materialised[path] = target
            self._sizes[path] = written
            self._held += written
            await self._evict_to_fit(keep=path)
            return target

    async def _evict_to_fit(self, *, keep: str) -> None:
        """Drop least-recently-used copies until the cache is inside its bound.

        `keep` is never evicted: it is the copy the caller is about to read,
        and evicting it would mean decompressing the same file twice within
        one call. If it alone is over the bound, `_decompress` already raised.
        """
        while self._held > self._max_materialised_bytes and len(self._materialised) > 1:
            oldest = next(key for key in self._materialised if key != keep)
            target = self._materialised.pop(oldest)
            self._held -= self._sizes.pop(oldest, 0)
            await asyncio.to_thread(Path(target).unlink, True)

    def _decompress(self, decompress: Callable[[str], IO[bytes]], path: str, target: str) -> int:
        """Stream the archive out, bounded as it goes. Returns bytes written.

        Checked DURING the write rather than against a declared size, for the
        same reason the expansion guard is: a compressed file's header is the
        bomb's own paperwork.

        This is the ONE place the bound raises rather than evicting. See the
        module docstring: a single archive bigger than the whole cache cannot
        be made to fit by evicting anything, because nothing would be left.
        """
        try:
            with decompress(path) as stream, open(target, "wb") as out:
                written = 0
                while True:
                    chunk = stream.read(_CHUNK)
                    if not chunk:
                        return written
                    written += len(chunk)
                    if written > self._max_materialised_bytes:
                        raise ContainerLimitExceededError(
                            f"{path!r} exceeds max_materialised_bytes "
                            f"({self._max_materialised_bytes}) when decompressed"
                        )
                    out.write(chunk)
        except (OSError, EOFError, gzip.BadGzipFile, lzma.LZMAError) as exc:
            raise SourceUnreadableError(f"cannot read tar {path!r}: {exc}") from exc

    async def entries(self, path: str) -> Sequence[ArchiveEntry]:
        solid = await asyncio.to_thread(self._safe_decompressor, path) is not None

        def _read() -> list[ArchiveEntry]:
            with tarfile.open(path, "r:*") as archive:
                return [
                    ArchiveEntry(
                        path=info.name,
                        size_bytes=info.size,
                        # tar does not compress per member, so a member's
                        # compressed size IS its size. The expansion guard
                        # upstream therefore never fires on a plain tar, which
                        # is correct: a plain tar cannot be a bomb.
                        compressed_bytes=info.size,
                        is_dir=info.isdir(),
                        is_symlink=info.issym() or info.islnk(),
                        modified_epoch_s=float(info.mtime),
                        # An offset into a gzip stream is not an offset anyone
                        # can seek to, so a solid archive reports none. This is
                        # the single fact that tells a caller which shape it
                        # has, and therefore what three member reads will cost.
                        byte_offset=None if solid else info.offset_data,
                    )
                    for info in archive.getmembers()
                    if info.name
                ]

        try:
            return await asyncio.to_thread(_read)
        except (OSError, tarfile.TarError, EOFError, ValueError) as exc:
            raise SourceUnreadableError(f"cannot read tar {path!r}: {exc}") from exc

    async def open_member(self, path: str, member: str) -> AsyncIterator[bytes]:
        readable = await self._seekable_path(path)
        try:
            archive = await asyncio.to_thread(tarfile.open, readable, "r:*")
        except (OSError, tarfile.TarError, EOFError) as exc:
            raise SourceUnreadableError(f"cannot read tar {path!r}: {exc}") from exc
        try:
            try:
                info = await asyncio.to_thread(archive.getmember, member)
            except KeyError as exc:
                raise SourceUnreadableError(f"no member {member!r} in {path!r}") from exc
            if info.issym() or info.islnk():
                # A tarball can carry a link to /etc/passwd, and materialising
                # it would follow that link straight out of the root. Refusing
                # is the only defensible default for a library whose whole
                # sandboxing story is "nothing outside the root". The link is
                # still REPORTED by `entries`, with its target, as a fact.
                raise SourceUnreadableError(
                    f"{member!r} in {path!r} is a symlink to {info.linkname!r}; "
                    "links inside containers are reported but never followed"
                )
            handle = await asyncio.to_thread(archive.extractfile, info)
            if handle is None:
                raise SourceUnreadableError(f"{member!r} in {path!r} has no readable content")
            try:
                while True:
                    try:
                        chunk = await asyncio.to_thread(handle.read, _CHUNK)
                    except (OSError, tarfile.TarError, EOFError) as exc:
                        raise SourceUnreadableError(
                            f"cannot read {member!r} from {path!r}: {exc}"
                        ) from exc
                    if not chunk:
                        return
                    yield chunk
            finally:
                await asyncio.to_thread(handle.close)
        finally:
            await asyncio.to_thread(archive.close)
