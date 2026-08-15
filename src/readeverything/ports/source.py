"""Reaching bytes, without knowing where they live.

Split into three capability slices because collaborators genuinely need
different ones: a handler reads and never lists, the pipeline stats and never
reads, a directory walk lists and never reads. Annotating the slimmest slice is
what keeps an adapter honest about what it is being asked for.

`uri` is opaque. A local path, an object-store key and an archive member
`"/a.zip!inner.txt"` are all just strings here; only the adapter interprets
them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class SourceStat(Protocol):
    async def exists(self, uri: str) -> bool: ...

    async def size(self, uri: str) -> int: ...


@runtime_checkable
class SourceReader(Protocol):
    async def read_bytes(self, uri: str) -> bytes:
        """The whole source. Prefer `stream` for anything that may be large."""
        ...

    def stream(self, uri: str, *, chunk_size: int = 1 << 20) -> AsyncIterator[bytes]: ...

    async def read_range(self, uri: str, start: int, end: int) -> bytes:
        """Bytes in `[start, end)`."""
        ...

    async def local_path(self, uri: str) -> str:
        """A real filesystem path for this source.

        External tools take paths, not streams, so a non-local adapter must
        materialise a temporary file. This is the one place that cost is
        acknowledged rather than hidden behind a stream nobody can use.
        """
        ...


@runtime_checkable
class SourceLister(Protocol):
    async def walk(self, uri: str) -> Sequence[str]:
        """Every source under `uri`, recursively. Directories are not returned."""
        ...


@runtime_checkable
class FileSource(SourceStat, SourceReader, SourceLister, Protocol):
    """Everything a fully-featured source adapter provides."""
