"""Reading the words a container already carries.

This is the cheapest port in the library, and the gap it fills is the widest.
A 37-minute lecture's caption track is about 8,700 tokens and one second of
ffmpeg; learning the same thing from pixels cost twelve model calls and five
minutes, and learning it from the model's own video path would cost 4.9
million tokens — nineteen times the context it would have to fit in.

So captions are not an optimisation. They are the first thing to try, and the
reason `video.py` has a precedence rule at all.

A `CaptionExtractor` reads a path rather than bytes, matching `StreamProbe`
and `AudioExtractor`: ffmpeg seeks a container header, and handing it a buffer
would mean writing the buffer back to disk first.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from readeverything.domain.rendition import TranscriptCue


@runtime_checkable
class CaptionExtractor(Protocol):
    """A container's caption cues, or `None` when it carries no readable ones."""

    async def extract(
        self, path: str, track: int | None = None
    ) -> tuple[TranscriptCue, ...] | None:
        """`path`'s caption cues, or `None` if there are no readable ones.

        `None` means "nothing to read" — a normal answer, the same convention
        as `FrameExtractor.frame_at` and `AudioExtractor.extract`. A missing
        file, an unreadable one, a container with no subtitles and a container
        whose only subtitles are bitmaps all return `None` rather than raising.
        A handler must be able to ask about anything and get an answer it can
        act on.

        `track` indexes the container's SUBTITLE streams — 0-based, the way
        ffmpeg's `-map 0:s:N` counts — not its streams overall. The
        distinction is not academic: in the reference file the readable track
        is subtitle 1 but stream 3. `None` means "the first one", which is
        wrong for that file, so callers that have probed should say which.

        Cues are returned in the order the track lists them, with
        `CueSource.CAPTIONED` set. Whether they are contiguous, overlapping or
        gapped is the track author's business; tiling them onto a timeline is
        `domain.timeline.tile`'s.
        """
        ...
