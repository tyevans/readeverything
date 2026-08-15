"""`ClipExtractor` over `ffmpeg`.

`-ss` goes BEFORE `-i`, which seeks the input rather than decoding up to the
start point and discarding it. On a 63-minute file the difference is seconds
versus minutes. It costs keyframe-accuracy — the cut may land slightly before
the requested start — which is why `watch_segment` reports the span it ASKED
for as its locator rather than one derived from the output: the requested range
is a fact about the question, and the delivered range is an approximation of
it that ffmpeg would have to be interrogated to learn.

`+faststart` is deliberately NOT used. It was measured on 2026-08-15 against
llama.cpp b10438: MOOV atom position made no difference to whether the server
could decode the clip, and a second pass to move it would be cargo cult. The
fragmented flags below serve a different purpose — they are what makes writing
to a pipe possible at all, since a non-fragmented mp4 needs to seek back to
patch its header and a pipe cannot seek.

Audio is dropped (`-an`). The clip is going to a vision model that cannot hear
it, and the transcript path already answers what was said — sending the audio
would be bytes paid for and discarded.

Security note, matching `ffprobe_streams.py`: the argument vector never
contains a shell string, `path` is one argv element, and the call is wrapped in
`asyncio.wait_for` with kill-and-reap so a crafted container cannot make ffmpeg
work indefinitely.
"""

from __future__ import annotations

import asyncio
import contextlib

#: What `FfmpegClip` produces, and what a `ClipModel` is told it is receiving.
CLIP_MIME = "video/mp4"

#: Boxes that only appear once a fragmented mp4 actually carries frames.
#: `moof` is a movie fragment header and `mdat` is its payload.
_FRAME_BOXES = (b"moof", b"mdat")


def _has_frames(data: bytes) -> bool:
    """Whether ffmpeg's output contains any video, rather than just a header.

    ASKING FOR A RANGE PAST THE END OF A FILE EXITS ZERO. ffmpeg writes a
    complete, valid, entirely frameless container — 807 bytes of `ftyp` and
    `moov` on the file this was measured against — and reports success,
    because muxing an empty stream is not an error to a muxer. Neither the
    exit code nor "did it write anything" distinguishes that from a real clip,
    and an empty container handed to a model is a paid-for call that can only
    answer about nothing.

    The fragmented muxer gives an exact content signal: no frames, no `moof`.
    Checked rather than probed because a probe is a second subprocess to learn
    something already present in the bytes.
    """
    return any(box in data for box in _FRAME_BOXES)


class FfmpegClip:
    """A bounded range of a container as mp4 bytes, or `None`."""

    def __init__(self, *, timeout_s: float = 300.0) -> None:
        self._timeout_s = timeout_s

    async def clip(self, path: str, start_s: float, end_s: float) -> bytes | None:
        if end_s <= start_s or start_s < 0:
            return None
        try:
            process = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-v",
                "error",
                # Before -i: seek the input rather than decoding and discarding.
                "-ss",
                f"{start_s:.3f}",
                "-i",
                path,
                "-t",
                f"{end_s - start_s:.3f}",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-f",
                "mp4",
                # A non-fragmented mp4 seeks back to patch its header, which a
                # pipe cannot do. These flags are what make stdout viable.
                "-movflags",
                "frag_keyframe+empty_moov",
                "pipe:1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError:
            return None

        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=self._timeout_s)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            return None

        if process.returncode != 0 or not _has_frames(stdout):
            # A range past the end of the file exits ZERO and writes a valid
            # container — just an empty one. Both that and a hard failure mean
            # "no such range", which is a normal answer.
            return None
        return stdout
