"""`CaptionExtractor` over `ffmpeg`.

Converting to SRT rather than reading each native codec is deliberate: ffmpeg
normalises `mov_text`, `ass`, `webvtt` and the rest into one grammar, so this
module parses one format instead of six. The cost is a subprocess, which this
library is already paying for every frame and every probe.

Real caption tracks carry markup. Every cue in the reference file is wrapped
in `<font size="24">`, and a tag that reaches an index is a tag that comes
back out in a citation — the sort of defect nobody notices until someone reads
a quotation and finds HTML in it.

Everything that goes wrong returns `None`. That is the port's contract and it
is load-bearing here more than elsewhere, because the ordinary case for
failure is not corruption but absence: most videos have no caption track, and
asking is how you find out.

Security note, matching `ffprobe_streams.py`: the argument vector never
contains a shell string, `path` is one argv element, and the call is wrapped
in `asyncio.wait_for` with kill-and-reap so a crafted container cannot make
ffmpeg work indefinitely.
"""

from __future__ import annotations

import asyncio
import contextlib
import re

from readeverything.domain.locators import TimeSpan
from readeverything.domain.rendition import CueSource, TranscriptCue

#: SRT writes `00:00:00,868`; WebVTT writes `00:00:00.868`. Both reach this
#: parser — ffmpeg's `srt` muxer is not the only producer a caller might have
#: — so both separators are accepted rather than one being assumed.
_TIMING = re.compile(
    r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{1,3})"
)

#: `<font size="24">`, `<i>`, `<b>` and whatever else an authoring tool left
#: behind. Stripped rather than escaped: none of it is content.
_MARKUP = re.compile(r"<[^>]+>")

#: Blank-line-separated blocks, tolerating the trailing whitespace real files
#: carry on their separator lines.
_BLOCKS = re.compile(r"\n[ \t]*\n")


def _seconds(hours: str, minutes: str, secs: str, millis: str) -> float:
    """`00:01:02,345` as 62.345. The fraction is left-aligned because `,5`
    means half a second, not five milliseconds."""
    return int(hours) * 3600 + int(minutes) * 60 + int(secs) + int(millis.ljust(3, "0")) / 1000.0


def parse_srt(text: str) -> tuple[TranscriptCue, ...]:
    """SRT text as cues. Unusable blocks are dropped, never raised.

    A caption track is a best-effort artifact of whoever authored the disc,
    and refusing the whole file over one malformed block would throw away 847
    good cues to punish one. What is dropped:

    - blocks with no parseable timing line — there is no span to put them at;
    - blocks whose end is at or before their start, because `TimeSpan` forbids
      it and a zero-width cue claims a moment while saying nothing about it;
    - blocks whose body is empty once markup is stripped, for the same reason.

    Kept separate from `FfmpegCaptions` so the parsing — the part with all the
    edge cases — is testable without a subprocess.
    """
    cues: list[TranscriptCue] = []
    for block in _BLOCKS.split(text.strip()):
        match = _TIMING.search(block)
        if match is None:
            continue
        start = _seconds(*match.group(1, 2, 3, 4))
        end = _seconds(*match.group(5, 6, 7, 8))
        if end <= start:
            continue
        # Everything after the timing line is the body. SRT wraps a sentence
        # across lines to fit a player's screen, which is a decision about
        # display width rather than a sentence boundary, so the wrap is undone.
        body = " ".join(_MARKUP.sub("", block[match.end() :]).split())
        if not body:
            continue
        cues.append(
            TranscriptCue(
                span=TimeSpan(start_s=start, end_s=end),
                text=body,
                speaker=None,
                confidence=None,
                source=CueSource.CAPTIONED,
            )
        )
    return tuple(cues)


class FfmpegCaptions:
    """A container's caption track as cues, or `None` if it has none."""

    def __init__(self, *, timeout_s: float = 60.0) -> None:
        self._timeout_s = timeout_s

    async def extract(
        self, path: str, track: int | None = None
    ) -> tuple[TranscriptCue, ...] | None:
        index = 0 if track is None else track
        try:
            process = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-v",
                "error",
                "-i",
                path,
                "-map",
                f"0:s:{index}",
                "-c:s",
                "srt",
                "-f",
                "srt",
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

        if process.returncode != 0:
            # No such track, a bitmap track ffmpeg will not turn into text, an
            # unreadable file: all of them mean "nothing to read here", which
            # is a normal answer and not this adapter's business to escalate.
            return None
        cues = parse_srt(stdout.decode("utf-8", errors="replace"))
        # A track that existed but yielded nothing usable is, to a caller,
        # indistinguishable from no track — and saying so lets a handler fall
        # through to ASR rather than believe it has read the words.
        return cues or None
