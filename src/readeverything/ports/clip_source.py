"""Cutting a bounded range out of a container.

Separate from `FrameExtractor` because a clip is not a frame and the
difference is the whole point: a frame is a moment and a clip is a stretch of
time with motion in it, which is what `watch_segment` exists to ask about.
Folding this into `FrameExtractor` would give that port two return shapes and
two reasons to fail.

A path rather than bytes, matching `StreamProbe`, `AudioExtractor` and
`FrameExtractor`: ffmpeg seeks a container, and handing it a buffer would mean
writing the buffer back to disk first.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ClipExtractor(Protocol):
    """A bounded range of a container as encoded video bytes, or `None`."""

    async def clip(self, path: str, start_s: float, end_s: float) -> bytes | None:
        """`path` between `start_s` and `end_s`, or `None` if there is no such range.

        `None` means "nothing there" — a normal answer, the same convention as
        `FrameExtractor.frame_at` and `AudioExtractor.extract`. A missing file,
        an unreadable one, a range past the end and an empty range all return
        `None` rather than raising. A handler must be able to ask about
        anything and get an answer it can act on.

        CALLERS BOUND THE DURATION BEFORE CALLING. This port will happily cut
        an hour, and an hour costs about 7.8 million prompt tokens at the rate
        `ports/clips.py` documents. The refusal belongs where the cost is known
        — in the handler, which has a cap — not here, where the extraction
        itself is cheap and the caller may want a long clip for something else
        entirely.
        """
        ...
