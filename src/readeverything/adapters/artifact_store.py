"""Where derived artifacts live.

Two adapters, one contract. Entries are immutable, so `put` on an existing key
is a no-op rather than an error — two workers deriving the same artifact
concurrently is normal, not a conflict.

`FilesystemArtifactStore` hashes the key before using it as a filename. Keys
from `artifact_key` are already hex digests, but the store must not depend on
that: a caller passing a raw key with `../` in it would otherwise write outside
the root.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from uuid import uuid4


class InMemoryArtifactStore:
    """A process-lifetime store. Useful in tests and for short-lived agents."""

    def __init__(self) -> None:
        self._entries: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes | None:
        return self._entries.get(key)

    async def put(self, key: str, value: bytes) -> None:
        self._entries.setdefault(key, value)


class FilesystemArtifactStore:
    """A store backed by a directory, sharded two levels to keep dirs small."""

    def __init__(self, *, root: Path | str) -> None:
        self._root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        safe = hashlib.blake2b(key.encode("utf-8"), digest_size=32).hexdigest()
        return self._root / safe[:2] / safe[2:4] / safe

    async def get(self, key: str) -> bytes | None:
        path = self._path(key)

        def _read() -> bytes | None:
            try:
                return path.read_bytes()
            except FileNotFoundError:
                return None

        return await asyncio.to_thread(_read)

    async def put(self, key: str, value: bytes) -> None:
        path = self._path(key)

        def _write() -> None:
            if path.exists():
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename so a crash cannot leave a truncated artifact
            # that later reads as a valid cache hit. Each writer gets its own
            # temp file so concurrent writers of the same key never collide.
            temporary = path.with_suffix(f".{uuid4().hex}.partial")
            temporary.write_bytes(value)
            temporary.replace(path)

        await asyncio.to_thread(_write)
