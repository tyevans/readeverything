"""A container's audio track, extracted to a decodable form — or the answer
that there isn't one.

`None` means "no audio track", the same normal-answer convention as
`FrameExtractor.frame_at` in `ports/frames.py`: a handler must be able to ask
about anything (a silent video, a corrupt file, a missing path) and get an
answer it can act on, never an exception.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AudioExtractor(Protocol):
    """A container's audio track as WAV bytes, or `None` if it has none."""

    async def extract(self, path: str) -> bytes | None:
        """Mono 16kHz WAV bytes of `path`'s audio track, or `None`.

        `None` means "no audio track" — a normal answer, not an error. Never
        raises: a missing file, an unreadable one, a file with no audio
        stream, and a timeout all return `None`.
        """
        ...
