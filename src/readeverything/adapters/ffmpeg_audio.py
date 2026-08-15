"""`AudioExtractor` over `ffmpeg`.

The verified command:

    ffmpeg -i <path> -vn -map 0:a:0 -ac 1 -ar 16000 -f wav -loglevel error -y -

`-vn` drops any video, `-map 0:a:0` selects the first audio stream explicitly
(so ffmpeg fails rather than silently picking a stream nobody asked for),
`-ac 1 -ar 16000` downmixes to mono 16kHz — the input `faster-whisper` wants —
and `-f wav` writes a WAV container to stdout.

Measured against real ffmpeg 6.1.1, not assumed:

    a file with an audio stream    -> exit 0,   WAV bytes on stdout
    a file with NO audio stream    -> exit 234, "Output file does not contain
                                       any stream" on stderr, empty stdout

THE ASYMMETRY, and it is the point of this module. `ffmpeg_frames.py`
documents that seeking a frame past a video's end exits **0** with zero bytes
and no stderr — frames are therefore validated by OUTPUT LENGTH, and that
module's docstring warns against unifying that check with an exit-status
check. Audio is the opposite: a missing stream exits **non-zero** with a
message on stderr, so streams are validated by EXIT STATUS. Two different
failure conventions live in the same external tool. This module is one half
of that pair; `ffmpeg_frames.py` is the other, and each names the other so a
future reader who wants to unify them has to override an explicit warning
first, not rediscover the trap by breaking one direction silently.

This adapter also treats empty stdout as `None`, belt and braces alongside the
exit-status check — cheap, and it means a future ffmpeg version that changes
its exit code on this path still can't produce an empty WAV disguised as
audio.

Security note, matching `ffmpeg_frames.py`: the argument vector passed to
`asyncio.create_subprocess_exec` never contains a shell string, and `path` is
a single argv element, never string-formatted into anything. The whole call is
wrapped in `asyncio.wait_for` with kill-and-reap.
"""

from __future__ import annotations

import asyncio
import contextlib


class FfmpegAudio:
    """A container's audio track as mono 16kHz WAV bytes, via `ffmpeg`."""

    def __init__(self, *, timeout_s: float = 30.0) -> None:
        self._timeout_s = timeout_s

    async def extract(self, path: str) -> bytes | None:
        """WAV bytes of `path`'s audio track, or `None` if it has none.

        Never raises: a missing file, unreadable bytes, a path that is not
        media, a file with no audio stream, and a timeout all return `None`
        rather than propagating — a handler must be able to ask about
        anything.
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-i",
                path,
                "-vn",
                "-map",
                "0:a:0",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
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

        # The opposite convention from frame_at (see module docstring): a
        # missing audio stream exits non-zero here, so exit status is the
        # primary check. Empty output is checked too, belt and braces.
        if process.returncode != 0 or not stdout:
            return None
        return stdout
