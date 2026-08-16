"""Zip, through stdlib `zipfile`.

The seekable half of the story: a zip's central directory gives every member an
offset and its own compressed size, so listing costs one seek and a small read,
and a ranged read of a member is a genuine ranged read. Nothing here
materialises anything.

Every `zipfile` failure is converted to `SourceUnreadableError`. A corrupt
archive that returned an empty entry list instead would be indistinguishable
from an empty archive, and an agent would report "this release contains
nothing" about a file it failed to open.
"""

from __future__ import annotations

import asyncio
import zipfile
from collections.abc import AsyncIterator, Sequence

from readeverything.domain.errors import SourceUnreadableError
from readeverything.domain.identity import MimeType
from readeverything.ports.containers import ArchiveEntry

#: What this opener answers `claims` for. `.zip` reaches detection under
#: several spellings depending on which of puremagic or `mimetypes` answered.
_MIMES = frozenset({"application/zip", "application/x-zip-compressed"})

#: Zip stores mode bits in the top 16 of `external_attr`, and S_IFLNK is 0xA000.
_S_IFMT = 0xF000
_S_IFLNK = 0xA000

#: The `create_system` value meaning a unix host wrote this entry. Only then
#: does `external_attr` carry mode bits at all.
_UNIX = 3

_CHUNK = 1 << 20


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    """Whether the entry is a symlink, per its stored unix mode.

    Only files written on a unix host carry mode bits at all, which is what
    `create_system` says; reading `external_attr` without checking it would
    misread a DOS-written entry's attribute byte as a file type.
    """
    if info.create_system != _UNIX:
        return False
    return (info.external_attr >> 16) & _S_IFMT == _S_IFLNK


class ZipArchiveOpener:
    """Reads zip containers. Stateless, so one instance serves every archive."""

    def claims(self, mime: MimeType) -> bool:
        return str(mime) in _MIMES

    async def entries(self, path: str) -> Sequence[ArchiveEntry]:
        def _read() -> list[ArchiveEntry]:
            with zipfile.ZipFile(path) as archive:
                return [
                    ArchiveEntry(
                        path=info.filename,
                        size_bytes=info.file_size,
                        compressed_bytes=info.compress_size,
                        is_dir=info.is_dir(),
                        is_symlink=_is_symlink(info),
                        # `date_time` has no timezone and no sub-second part;
                        # it is reported as a fact, never used for freshness.
                        modified_epoch_s=None,
                        byte_offset=info.header_offset,
                    )
                    for info in archive.infolist()
                    if info.filename
                ]

        try:
            return await asyncio.to_thread(_read)
        except (OSError, zipfile.BadZipFile, ValueError) as exc:
            raise SourceUnreadableError(f"cannot read zip {path!r}: {exc}") from exc

    async def open_member(self, path: str, member: str) -> AsyncIterator[bytes]:
        try:
            archive = await asyncio.to_thread(zipfile.ZipFile, path)
        except (OSError, zipfile.BadZipFile) as exc:
            raise SourceUnreadableError(f"cannot read zip {path!r}: {exc}") from exc
        try:
            handle = await asyncio.to_thread(archive.open, member)
        except (KeyError, OSError, zipfile.BadZipFile) as exc:
            await asyncio.to_thread(archive.close)
            raise SourceUnreadableError(f"cannot read {member!r} from {path!r}: {exc}") from exc
        try:
            while True:
                try:
                    chunk = await asyncio.to_thread(handle.read, _CHUNK)
                except (OSError, zipfile.BadZipFile, EOFError, ValueError) as exc:
                    # A member whose deflate stream is damaged. This fires on
                    # READ, after `entries` already succeeded, which is exactly
                    # the §1.1 requirement: one bad member must not cost the
                    # agent its neighbours.
                    raise SourceUnreadableError(
                        f"cannot read {member!r} from {path!r}: {exc}"
                    ) from exc
                if not chunk:
                    return
                yield chunk
        finally:
            await asyncio.to_thread(handle.close)
            await asyncio.to_thread(archive.close)
