"""`FrameExtractor` over `ffmpeg`.

The verified command:

    ffmpeg -ss <T> -i <path> -frames:v 1 -f image2 -vcodec png -loglevel error -y -

`-ss` goes *before* `-i` so the seek happens at the demuxer, which is fast;
after `-i` it decodes from the start instead — frame-accurate but slow on
long files, and this library is cheap-first by design. `-f image2` (not
`image2pipe`) is required to get a single frame out on stdout.

Measured against real ffmpeg 6.1.1, not assumed:

    -ss 2.5  (in range)   -> exit 0, 16942 stdout bytes, 0 stderr bytes
    -ss 999  (past the end) -> exit 0,     0 stdout bytes, 0 stderr bytes

Identical exit status, no stderr either way — only the byte count tells them
apart. So a frame is validated by OUTPUT LENGTH, never by exit status. This is
a different convention than `FfprobeStreams`/a missing stream (Task 5), which
fails loudly with a non-zero exit and a message on stderr. The two checks are
not interchangeable; unifying "did ffmpeg produce a frame" into an exit-status
check reintroduces this exact silent failure, so don't.

Returning `None` rather than `b""` on failure matters for the same reason:
`b""` is a value a caller can pass along, and an empty PNG handed to a model
as "the frame at t=999" is an observation nothing made. `None` cannot be
mistaken for an image.

Security note, matching `ffprobe_streams.py`: the argument vector passed to
`asyncio.create_subprocess_exec` never contains a shell string, and `path` and
the timestamp are separate argv elements, never string-formatted together.
The whole call is wrapped in `asyncio.wait_for` with kill-and-reap.
"""

from __future__ import annotations

import asyncio
import contextlib


class FfmpegFrames:
    """One decoded video frame as PNG bytes, via `ffmpeg`."""

    def __init__(self, *, timeout_s: float = 10.0) -> None:
        self._timeout_s = timeout_s

    async def frame_at(self, path: str, seconds: float) -> bytes | None:
        """One frame as PNG bytes, or None if there is no frame at that time.

        Never raises: a missing file, unreadable bytes, a path that is not
        media, a negative timestamp, or a timeout all return None rather than
        propagating — a handler must be able to ask about anything.
        """
        if seconds < 0:
            return None
        try:
            process = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-ss",
                str(seconds),
                "-i",
                path,
                "-frames:v",
                "1",
                "-f",
                "image2",
                "-vcodec",
                "png",
                "-loglevel",
                "error",
                "-y",
                "-",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError:
            return None

        try:
            stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=self._timeout_s)
        except TimeoutError:
            # Kill and reap rather than leave a zombie; the child may have
            # exited in the gap, in which case `kill()` raises
            # `ProcessLookupError`, which is not an error here.
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            return None

        # Deliberately NOT checking process.returncode here. ffmpeg exits 0
        # whether or not a frame was produced when seeking past the end (or
        # against a non-media file, in practice) — output length is the only
        # signal. See module docstring: this is the measured trap, and
        # unifying this with an exit-status check brings it back.
        if not stdout:
            return None
        return stdout
