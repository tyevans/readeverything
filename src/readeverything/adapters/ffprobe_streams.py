"""`StreamProbe` over `ffprobe`.

`ffprobe -show_format -show_streams -of json` reads the container header, not
the frames, so this stays cheap in the same sense `pdfium_probe.py` does for
page counts. The three traps in ffprobe's JSON are handled here and nowhere
else:

- `format.duration` is a **string**. Missing or unparseable is raised as
  `InfrastructureError`, never defaulted to `0.0` — a fabricated zero duration
  would make every `TimeSpan` a later timeline produces a lie.
- `format.format_name` is a **comma-joined list of candidate formats**
  (`"mov,mp4,m4a,3gp,3g2,mj2"`), not one name. The whole string is kept;
  picking one would assert an identification ffprobe declined to make.
- `r_frame_rate` is a **rational string** (`"10/1"`, or `"0/0"` for a
  non-video stream). Denominator zero returns `None` rather than raising — a
  later task divides by frame rate, and a fabricated zero there becomes a
  division by zero far from its cause.

Security note, matching `binary_probe.py`: the argument vector passed to
`asyncio.create_subprocess_exec` never contains a shell string, and `path` is
one argv element. `-analyzeduration`/`-probesize` are fixed at small values
and the whole call is wrapped in `asyncio.wait_for` with kill-and-reap, so a
crafted file cannot make ffprobe work indefinitely — this library hands
results to an agent, and a probed file is adversarial input.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from readeverything.domain.errors import InfrastructureError
from readeverything.ports.streams import TEXT_SUBTITLE_CODECS, MediaFacts, StreamInfo

#: Small fixed bounds so ffprobe reads only enough of the header to answer,
#: never the whole file.
_ANALYZEDURATION = "2000000"  # microseconds
_PROBESIZE = "5000000"  # bytes


def _parse_frame_rate(raw: str | None) -> float | None:
    if not raw or "/" not in raw:
        return None
    num_s, _, den_s = raw.partition("/")
    try:
        num, den = float(num_s), float(den_s)
    except ValueError:
        return None
    if den == 0:
        return None
    return num / den


def _parse_stream(raw: dict[str, Any]) -> StreamInfo | None:
    codec_type = raw.get("codec_type")
    if codec_type not in ("video", "audio", "subtitle"):
        return None
    codec = str(raw.get("codec_name", ""))
    tags = raw.get("tags")
    language = tags.get("language") if isinstance(tags, dict) else None
    return StreamInfo(
        kind=codec_type,
        codec=codec,
        width=raw.get("width"),
        height=raw.get("height"),
        frame_rate=_parse_frame_rate(raw.get("r_frame_rate")),
        sample_rate=int(raw["sample_rate"]) if raw.get("sample_rate") is not None else None,
        channels=raw.get("channels"),
        language=str(language) if language is not None else None,
        is_text=codec_type == "subtitle" and codec in TEXT_SUBTITLE_CODECS,
    )


def _parse_facts(raw: dict[str, Any]) -> MediaFacts:
    fmt = raw.get("format")
    if not isinstance(fmt, dict):
        raise InfrastructureError("ffprobe reported no `format` block")
    duration_raw: Any = fmt.get("duration")
    try:
        duration_s = float(duration_raw)
    except (TypeError, ValueError) as exc:
        raise InfrastructureError(f"ffprobe reported no usable duration: {duration_raw!r}") from exc
    container = str(fmt.get("format_name", ""))
    streams_raw = raw.get("streams")
    streams = tuple(
        info
        for item in (streams_raw if isinstance(streams_raw, list) else [])
        if (info := _parse_stream(item)) is not None
    )
    return MediaFacts(duration_s=duration_s, container=container, streams=streams)


class FfprobeStreams:
    """Duration, container candidates and stream shapes, without decoding."""

    def __init__(self, *, timeout_s: float = 5.0) -> None:
        self._timeout_s = timeout_s

    async def probe(self, path: str) -> MediaFacts:
        try:
            process = await asyncio.create_subprocess_exec(
                "ffprobe",
                "-v",
                "error",
                "-analyzeduration",
                _ANALYZEDURATION,
                "-probesize",
                _PROBESIZE,
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise InfrastructureError(f"could not start ffprobe: {exc}") from exc

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self._timeout_s)
        except TimeoutError as exc:
            # A malformed header must not make ffprobe work indefinitely. Kill
            # and reap rather than leave a zombie; the child may have exited
            # in the gap, in which case `kill()` raises `ProcessLookupError`,
            # which is not an error here.
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            raise InfrastructureError(f"ffprobe timed out probing {path!r}") from exc

        if process.returncode != 0:
            raise InfrastructureError(
                f"ffprobe failed on {path!r} (exit {process.returncode}): "
                f"{stderr.decode('utf-8', errors='replace').strip()}"
            )
        try:
            raw = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise InfrastructureError(f"ffprobe produced unparseable output for {path!r}") from exc
        return _parse_facts(raw)
