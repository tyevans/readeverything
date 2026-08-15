"""A single decoded video frame, as PNG bytes, at a point in time.

`StreamProbe` (`ports/streams.py`) answers structural facts without decoding.
This protocol crosses that line deliberately — a frame must be decoded to
exist at all — so it is a different, narrower question with a different
failure shape: there may simply be no frame at the requested time, and that is
not an error.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class FrameExtractor(Protocol):
    """One decoded frame as PNG bytes, or `None` if there is no frame there.

    `None` means "no frame at that time" — it is not an error signal. A
    handler must be able to ask about anything (a missing file, a corrupt
    file, a timestamp past the end, a negative timestamp) and get an answer
    it can act on, never an exception.
    """

    async def frame_at(self, path: str, seconds: float) -> bytes | None: ...
