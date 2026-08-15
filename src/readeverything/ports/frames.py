"""A single decoded video frame, as PNG bytes, at a point in time; and where
the content changes sharply enough to be a scene cut.

`StreamProbe` (`ports/streams.py`) answers structural facts without decoding.
This protocol crosses that line deliberately — a frame must be decoded to
exist at all — so it is a different, narrower question with a different
failure shape: there may simply be no frame at the requested time, and that is
not an error.

`scene_cuts` has the opposite failure shape from `frame_at`, deliberately: it
RAISES on a genuine detection failure, matching `StreamProbe`/`ffprobe`'s
convention rather than `frame_at`'s "never raises". "No cuts found" and
"detection failed" must be distinguishable outcomes — an empty tuple with no
exception is a real answer ("this content is uniform"), where an exception
means the detector itself did not run. Collapsing the two into a single
silent `()` would make a broken detector indistinguishable from an unedited
video.
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

    async def scene_cuts(self, path: str, threshold: float = 0.4) -> tuple[float, ...]:
        """Timestamps where the content changes sharply, or `()` if none do.

        Raises on a genuine detection failure — see the module docstring.
        """
        ...
