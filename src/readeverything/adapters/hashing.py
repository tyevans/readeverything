"""Content hashing, with a stat memo in front of it.

`blake2b` from the standard library rather than blake3: fast enough streamed,
and it costs no dependency in a package whose base install is deliberately
light.

The memo maps `(device, inode, size, mtime_ns)` to a hash. It is an
**optimisation only** — a miss costs a rehash, never a wrong answer — which is
why an edited file with an unchanged size still invalidates: `mtime_ns` is part
of the key.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from readeverything.domain.identity import ContentHash
from readeverything.ports.source import SourceReader

_CHUNK = 1 << 20


class StatMemo:
    """Remembers hashes by identity-and-mtime. Not required for correctness."""

    def __init__(self) -> None:
        self._entries: dict[tuple[int, int, int, int], ContentHash] = {}

    @staticmethod
    def _key(path: Path) -> tuple[int, int, int, int] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def get(self, path: Path) -> ContentHash | None:
        key = self._key(path)
        return None if key is None else self._entries.get(key)

    def put(self, path: Path, value: ContentHash) -> None:
        key = self._key(path)
        if key is not None:
            self._entries[key] = value


class ContentHasher:
    """Hashes a source's bytes, streamed."""

    def __init__(self, *, source: SourceReader, memo: StatMemo | None = None) -> None:
        self._source = source
        self._memo = memo

    async def hash(self, uri: str) -> ContentHash:
        path: Path | None = None
        if self._memo is not None:
            path = Path(await self._source.local_path(uri))
            cached = await asyncio.to_thread(self._memo.get, path)
            if cached is not None:
                return cached
        digest = hashlib.blake2b(digest_size=32)
        async for chunk in self._source.stream(uri, chunk_size=_CHUNK):
            digest.update(chunk)
        value = ContentHash(digest.hexdigest())
        if self._memo is not None and path is not None:
            await asyncio.to_thread(self._memo.put, path, value)
        return value
