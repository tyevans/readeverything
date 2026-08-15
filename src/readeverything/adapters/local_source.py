"""A `FileSource` over a local directory, and nothing outside it.

The root is a boundary, not a convenience. Every uri is resolved and checked
against it, so `"../../etc/passwd"` fails loudly rather than reading an
unintended file. This is the only sandboxing the library performs, and it is
here rather than in the domain because "what is a path" is adapter knowledge.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from readeverything.domain.errors import SourceUnreadableError


class LocalFileSource:
    """Reads files beneath a single root directory."""

    def __init__(self, *, root: Path | str) -> None:
        self._root = Path(root).resolve()

    def _resolve(self, uri: str) -> Path:
        candidate = (self._root / uri).resolve()
        if candidate != self._root and self._root not in candidate.parents:
            raise SourceUnreadableError(f"{uri!r} resolves outside the root {self._root}")
        return candidate

    async def exists(self, uri: str) -> bool:
        return await asyncio.to_thread(self._resolve(uri).is_file)

    async def size(self, uri: str) -> int:
        path = self._resolve(uri)
        try:
            return await asyncio.to_thread(lambda: path.stat().st_size)
        except OSError as exc:
            raise SourceUnreadableError(f"cannot stat {uri!r}: {exc}") from exc

    async def read_bytes(self, uri: str) -> bytes:
        path = self._resolve(uri)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except OSError as exc:
            raise SourceUnreadableError(f"cannot read {uri!r}: {exc}") from exc

    async def read_range(self, uri: str, start: int, end: int) -> bytes:
        path = self._resolve(uri)

        def _read() -> bytes:
            with path.open("rb") as handle:
                handle.seek(start)
                return handle.read(max(0, end - start))

        try:
            return await asyncio.to_thread(_read)
        except OSError as exc:
            raise SourceUnreadableError(f"cannot read {uri!r}: {exc}") from exc

    async def stream(self, uri: str, *, chunk_size: int = 1 << 20) -> AsyncIterator[bytes]:
        path = self._resolve(uri)
        handle = await asyncio.to_thread(path.open, "rb")
        try:
            while True:
                chunk = await asyncio.to_thread(handle.read, chunk_size)
                if not chunk:
                    return
                yield chunk
        except OSError as exc:
            raise SourceUnreadableError(f"cannot read {uri!r}: {exc}") from exc
        finally:
            await asyncio.to_thread(handle.close)

    async def local_path(self, uri: str) -> str:
        return str(self._resolve(uri))

    async def walk(self, uri: str) -> Sequence[str]:
        base = self._resolve(uri)

        def _walk() -> list[str]:
            return sorted(str(p.relative_to(self._root)) for p in base.rglob("*") if p.is_file())

        try:
            return await asyncio.to_thread(_walk)
        except OSError as exc:
            raise SourceUnreadableError(f"cannot walk {uri!r}: {exc}") from exc
