"""Deciding what a source is.

Content is the authority and the filename is a tiebreak, never the reverse. An
extension is a claim by whoever named the file; the bytes are a fact.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from readeverything.domain.identity import MimeType


@runtime_checkable
class MimeDetector(Protocol):
    async def detect(self, uri: str, head: bytes) -> MimeType:
        """The mimetype of a source, given its first bytes and its uri."""
        ...
