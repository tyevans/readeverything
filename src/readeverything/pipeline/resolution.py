"""Remembering what a path resolved to, for as long as the path stands still.

This is deliberately NOT the artifact store, and the difference is the whole
reason it is a separate file. The artifact store is content-addressed: its key
contains the content hash, so an entry can never go stale and it never needs
invalidating. This memo is keyed on a *path*, which is mutable, so it carries an
invalidation rule -- `(dev, inode, size, mtime_ns)`, the same rule `StatMemo`
already uses for hashes.

Conflating the two would give the content-addressed store a staleness protocol
it does not need and must not acquire.

A source that cannot be stat'd is never memoized. Without a stat there is no
invalidation rule, and caching on the uri alone would serve a stale ref forever
after a non-local object changed.
"""

from __future__ import annotations

from pathlib import Path

from readeverything.domain.identity import SourceRef
from readeverything.ports.source import SourceReader

type StatKey = tuple[int, int, int, int]


async def stat_key(source: SourceReader, uri: str) -> StatKey | None:
    """`(dev, inode, size, mtime_ns)` for `uri`, or None if it cannot be stat'd."""
    try:
        path = Path(await source.local_path(uri))
        stat = path.stat()
    except (OSError, NotImplementedError):
        return None
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


class ResolutionMemo:
    """Maps a uri to the `SourceRef` it produced, while its stat is unchanged."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[StatKey, SourceRef]] = {}

    def get(self, uri: str, key: StatKey | None) -> SourceRef | None:
        if key is None:
            return None
        entry = self._entries.get(uri)
        if entry is None or entry[0] != key:
            return None
        return entry[1]

    def put(self, uri: str, key: StatKey | None, ref: SourceRef) -> None:
        if key is None:
            return
        self._entries[uri] = (key, ref)
