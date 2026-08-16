"""Reading something inside something else.

A DECORATOR over another `FileSource`, not a replacement for one. Every method
splits the uri and, **if there is exactly one segment, delegates verbatim to
`inner` and returns**. That is the whole compatibility story: a perception over
a directory of loose files behaves identically whether or not this is
installed, and every existing `LocalFileSource` test keeps exercising the real
code path rather than a lookalike.

For a multi-segment uri it resolves left to right -- open the outermost
container from `inner`, open each subsequent container from the member bytes of
the one before, answer the request against the final member.

The reason this is the source layer and not an archive handler with a
`read_entry` affordance: an affordance returning member bytes gives an agent
bytes, while a nested uri gives it a PERCEPTION -- a card, an outline, page
affordances, OCR, provenance. Every handler in this repository already reads
through `SourceReader` and is forbidden from touching a filesystem, so the PDF
handler reads a PDF inside a tarball inside a zip without one line of it
changing. This file is the adapter collecting on a bill the architecture
already paid.

Two guards live here that `LocalFileSource`'s root check cannot cover, because
a member path is never resolved against a filesystem at all: a `..` component
and an absolute member are both refused outright, and a member the container
declares as a symlink is reported but never followed.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator, Sequence
from pathlib import Path, PurePosixPath

from readeverything.domain.container_uri import join_uri, split_uri
from readeverything.domain.errors import ContainerLimitExceededError, SourceUnreadableError
from readeverything.domain.identity import MimeType
from readeverything.ports.containers import ArchiveEntry, ArchiveOpener, ContainerLimits
from readeverything.ports.detection import MimeDetector
from readeverything.ports.source import FileSource

#: How much of a container is read to detect its mimetype. Matches
#: `pipeline.perception._HEAD_BYTES`, so a container is typed by exactly the
#: bytes the pipeline would have typed it by.
_HEAD_BYTES = 4096


class CompositeOpener:
    """Dispatches to whichever opener claims the mimetype.

    The extension point the spec promises: a caller who wants `.7z` or `.rar`
    supplies their own `ArchiveOpener` here, and this repository never grows a
    dependency on either.
    """

    def __init__(self, *, openers: Sequence[ArchiveOpener]) -> None:
        self._openers = tuple(openers)

    def opener_for(self, mime: MimeType) -> ArchiveOpener | None:
        for opener in self._openers:
            if opener.claims(mime):
                return opener
        return None

    def claims(self, mime: MimeType) -> bool:
        return self.opener_for(mime) is not None

    async def entries(self, path: str) -> Sequence[ArchiveEntry]:
        """Never. A composite has no one format, so this would be a guess.

        Present because `ArchiveOpener` declares it and a composite is passed
        wherever one is expected; callers reach the real opener through
        `opener_for`.
        """
        raise NotImplementedError("dispatch through `opener_for`; a composite has no one format")

    def open_member(self, path: str, member: str) -> AsyncIterator[bytes]:
        """Never, for the same reason as `entries`."""
        raise NotImplementedError("dispatch through `opener_for`; a composite has no one format")

    async def aclose(self) -> None:
        for opener in self._openers:
            closer = getattr(opener, "aclose", None)
            if closer is not None:
                await closer()


def _checked_member(member: str, uri: str) -> str:
    """A member path, or a refusal.

    `LocalFileSource` guards its root with `resolve()` and a parent check.
    That guard cannot see any of this, because a member path is never resolved
    against the filesystem -- it is looked up in a container's own directory,
    which will happily hand back whatever the archive's author wrote there. So
    the check is textual and it is strict.
    """
    if member.startswith(("/", "\\")):
        raise SourceUnreadableError(f"member {member!r} of {uri!r} is absolute; refused")
    if len(member) >= 2 and member[1] == ":" and member[0].isalpha():
        raise SourceUnreadableError(
            f"member {member!r} of {uri!r} is absolute (a drive letter); refused"
        )
    # Backslashes are normalised before splitting because a zip written on
    # Windows separates with them, and `..\..\etc\passwd` is the same attack
    # as `../../etc/passwd` written in the other dialect.
    if ".." in PurePosixPath(member.replace("\\", "/")).parts:
        raise SourceUnreadableError(f"member {member!r} of {uri!r} contains a traversal; refused")
    return member


def _read_head(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read(_HEAD_BYTES)


class NestedSource:
    """A `FileSource` that can see inside containers."""

    def __init__(
        self,
        inner: FileSource,
        *,
        limits: ContainerLimits,
        archives: ArchiveOpener,
        detector: MimeDetector,
    ) -> None:
        self._inner = inner
        self._limits = limits
        self._archives = archives
        self._detector = detector
        self._temp = tempfile.TemporaryDirectory(prefix="readeverything-nested-")
        #: uri -> the temp file holding its bytes. Keyed on the full nested uri
        #: so `local_path` is STABLE: `pipeline.resolution.stat_key` compares
        #: inodes, and a fresh temp file per call would make every member look
        #: like a different file on every access.
        self._materialised: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        """Remove every temp file this source and its openers created."""
        await asyncio.to_thread(self._temp.cleanup)
        self._materialised.clear()
        closer = getattr(self._archives, "aclose", None)
        if closer is not None:
            await closer()

    # -- resolution ---------------------------------------------------------

    def _dispatch(self, mime: MimeType) -> ArchiveOpener | None:
        """The opener for `mime`, whether `archives` is a composite or not."""
        chooser = getattr(self._archives, "opener_for", None)
        if chooser is not None:
            opener: ArchiveOpener | None = chooser(mime)
            return opener
        return self._archives if self._archives.claims(mime) else None

    async def _opener_for_path(self, path: str, uri: str) -> ArchiveOpener:
        """The opener for the container at local `path`, or a refusal."""
        try:
            head = await asyncio.to_thread(_read_head, path)
        except OSError as exc:
            raise SourceUnreadableError(f"cannot read container {uri!r}: {exc}") from exc
        mime = await self._detector.detect(uri, head)
        opener = self._dispatch(mime)
        if opener is None:
            raise SourceUnreadableError(f"{uri!r} is not a container this source can open ({mime})")
        return opener

    async def _entries(self, opener: ArchiveOpener, path: str, uri: str) -> Sequence[ArchiveEntry]:
        entries = await opener.entries(path)
        if len(entries) > self._limits.max_members:
            raise ContainerLimitExceededError(
                f"{uri!r} declares {len(entries)} members, over max_members "
                f"({self._limits.max_members})"
            )
        total = sum(entry.size_bytes for entry in entries)
        if total > self._limits.max_total_bytes:
            raise ContainerLimitExceededError(
                f"{uri!r} expands to {total} bytes, over max_total_bytes "
                f"({self._limits.max_total_bytes})"
            )
        return entries

    def _entry(self, entries: Sequence[ArchiveEntry], member: str, uri: str) -> ArchiveEntry:
        for entry in entries:
            if entry.path == member or entry.path.rstrip("/") == member:
                return entry
        raise SourceUnreadableError(f"no member {member!r} in {uri!r}")

    async def _container_path(self, segments: Sequence[str]) -> str:
        """A local filesystem path for the container named by `segments`.

        The outermost comes straight from `inner`. Anything deeper has to be
        materialised, because an `ArchiveOpener` takes a path -- which is the
        same honest cost `ports/source.py` already names for `local_path`, and
        the same thing that lets ffmpeg and pypdfium2 work on archive members
        without changing.
        """
        if len(segments) == 1:
            return await self._inner.local_path(segments[0])
        return await self._materialise(join_uri(segments))

    async def _resolve(
        self, uri: str
    ) -> tuple[ArchiveOpener, str, Sequence[ArchiveEntry], str, str]:
        """`(opener, container path, entries, member, container uri)` for `uri`."""
        segments = split_uri(uri)
        depth = len(segments) - 1
        if depth > self._limits.max_depth:
            raise ContainerLimitExceededError(
                f"{uri!r} is {depth} container(s) deep, over max_depth ({self._limits.max_depth})"
            )
        member = _checked_member(segments[-1], uri)
        container_uri = join_uri(segments[:-1])
        path = await self._container_path(segments[:-1])
        opener = await self._opener_for_path(path, container_uri)
        entries = await self._entries(opener, path, container_uri)
        return opener, path, entries, member, container_uri

    async def _member_bytes(self, uri: str) -> bytes:
        """The decompressed bytes of the member `uri` names."""
        opener, path, entries, member, container_uri = await self._resolve(uri)
        entry = self._entry(entries, member, container_uri)
        if entry.is_symlink:
            raise SourceUnreadableError(
                f"member {member!r} of {container_uri!r} is a symlink; "
                "links inside containers are reported but never followed"
            )
        if entry.size_bytes > self._limits.max_member_bytes:
            raise ContainerLimitExceededError(
                f"member {member!r} of {container_uri!r} is {entry.size_bytes} bytes, "
                f"over max_member_bytes ({self._limits.max_member_bytes})"
            )
        return await self._drain(opener, path, entry, container_uri)

    async def _drain(
        self, opener: ArchiveOpener, path: str, entry: ArchiveEntry, container_uri: str
    ) -> bytes:
        """Decompress one member, guarding as the bytes actually arrive.

        Both guards run MID-STREAM. A zip bomb lies in its header, so checking
        `entry.size_bytes` alone would be reading the bomb's own paperwork:
        the length of what has actually been written is the only number here
        that cannot be forged.
        """
        ceiling = self._limits.max_member_bytes
        ratio_ceiling = (
            entry.compressed_bytes * self._limits.max_expansion_ratio
            if entry.compressed_bytes
            else None
        )
        buffer = bytearray()
        async for chunk in opener.open_member(path, entry.path):
            buffer += chunk
            if len(buffer) > ceiling:
                raise ContainerLimitExceededError(
                    f"member {entry.path!r} of {container_uri!r} exceeds max_member_bytes "
                    f"({ceiling}) while decompressing"
                )
            if ratio_ceiling is not None and len(buffer) > ratio_ceiling:
                raise ContainerLimitExceededError(
                    f"member {entry.path!r} of {container_uri!r} exceeds its expansion "
                    f"ratio limit ({self._limits.max_expansion_ratio}) while decompressing; "
                    f"{len(buffer)} bytes out of {entry.compressed_bytes} compressed"
                )
        return bytes(buffer)

    async def _materialise(self, uri: str) -> str:
        async with self._lock:
            existing = self._materialised.get(uri)
            if existing is not None:
                return existing
            data = await self._member_bytes(uri)
            target = Path(self._temp.name) / f"{len(self._materialised)}-{PurePosixPath(uri).name}"
            await asyncio.to_thread(target.write_bytes, data)
            self._materialised[uri] = str(target)
            return str(target)

    # -- the FileSource surface ---------------------------------------------

    async def exists(self, uri: str) -> bool:
        segments = split_uri(uri)
        if len(segments) == 1:
            return await self._inner.exists(uri)
        try:
            _, _, entries, member, _ = await self._resolve(uri)
        except SourceUnreadableError:
            # `exists` answers a question rather than raising about a guess,
            # matching `LocalFileSource.exists` on a missing path.
            return False
        return any(e.path == member or e.path.rstrip("/") == member for e in entries)

    async def size(self, uri: str) -> int:
        """The member's UNCOMPRESSED size, from the container's directory.

        The directory, not a decompression: this is the number a card reports,
        and a card must stay within probe cost.
        """
        segments = split_uri(uri)
        if len(segments) == 1:
            return await self._inner.size(uri)
        _, _, entries, member, container_uri = await self._resolve(uri)
        return self._entry(entries, member, container_uri).size_bytes

    async def read_bytes(self, uri: str) -> bytes:
        segments = split_uri(uri)
        if len(segments) == 1:
            return await self._inner.read_bytes(uri)
        return await self._member_bytes(uri)

    async def read_range(self, uri: str, start: int, end: int) -> bytes:
        """Bytes in `[start, end)` of the member.

        A zip member has a real offset, but reaching byte `start` of it still
        means inflating everything before it -- so this slices the whole
        member rather than pretending a ranged read of a DEFLATE stream is
        cheaper than it is. §3.2's honesty is about not hiding that cost.
        """
        segments = split_uri(uri)
        if len(segments) == 1:
            return await self._inner.read_range(uri, start, end)
        return (await self._member_bytes(uri))[start : max(start, end)]

    async def stream(self, uri: str, *, chunk_size: int = 1 << 20) -> AsyncIterator[bytes]:
        segments = split_uri(uri)
        if len(segments) == 1:
            async for chunk in self._inner.stream(uri, chunk_size=chunk_size):
                yield chunk
            return
        data = await self._member_bytes(uri)
        for offset in range(0, len(data), chunk_size):
            yield data[offset : offset + chunk_size]

    async def local_path(self, uri: str) -> str:
        segments = split_uri(uri)
        if len(segments) == 1:
            return await self._inner.local_path(uri)
        return await self._materialise(uri)

    async def walk(self, uri: str) -> Sequence[str]:
        return await self._inner.walk(uri)
