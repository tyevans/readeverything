"""Probing OS executables by running them.

Availability means *it ran and reported a version*, not that a file exists at a
path. A binary that is present and broken is not a capability, and finding that
out at composition time is much cheaper than finding it out inside a handler
three layers down.

Security note: the argument vector passed to the subprocess must never contain
a caller-influenced string (a uri, a filename, model output). The executable
name and version flag come only from module constants or explicit constructor
arguments. `asyncio.create_subprocess_exec` is used with an argument vector —
never a shell string — so there is no shell-injection surface here.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
from collections.abc import Mapping

from readeverything.domain.capability import Capability

#: The executable each capability is provided by, and the flag that makes THAT
#: executable print a version and exit. The flag belongs to the executable, not
#: to the probe: a single global `-version` reported exiftool absent while it
#: was installed, because exiftool's flag is `-ver`.
DEFAULT_EXECUTABLES: Mapping[Capability, tuple[str, str]] = {
    Capability.FFMPEG: ("ffmpeg", "-version"),
    Capability.EXIFTOOL: ("exiftool", "-ver"),
    Capability.LIBREOFFICE: ("libreoffice", "--version"),
    Capability.TESSERACT: ("tesseract", "--version"),
}

#: Prefixes that mean the tool talked to us rather than identified itself.
_NOT_A_VERSION = ("warning", "error", "usage")


def _as_revision(stdout: bytes) -> str | None:
    """The first line, if it plausibly identifies the tool.

    `libreoffice -version` prints "Warning: -version is deprecated…", and the
    probe recorded that as the revision. It then entered the capability
    fingerprint and therefore every artifact cache key. Nothing had established
    that the captured line was a version — this is the probe's own contract
    turned on itself, so it now checks.
    """
    lines = stdout.decode("utf-8", errors="replace").strip().splitlines()
    if not lines:
        return None
    first = lines[0].strip()
    if not first or first.lower().startswith(_NOT_A_VERSION):
        return None
    return first


class BinaryProbe:
    """Reports a capability available when its executable runs and answers."""

    def __init__(
        self,
        *,
        executables: Mapping[Capability, tuple[str, str]] | None = None,
        timeout_s: float = 5.0,
    ) -> None:
        self._executables = dict(DEFAULT_EXECUTABLES if executables is None else executables)
        self._timeout_s = timeout_s

    async def revision(self, capability: Capability) -> str | None:
        entry = self._executables.get(capability)
        if entry is None:
            return None
        name, version_flag = entry
        located = shutil.which(name)
        if located is None:
            return None
        try:
            process = await asyncio.create_subprocess_exec(
                located,
                version_flag,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError:
            return None
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=self._timeout_s)
        except TimeoutError:
            # A hung `--version` must not hang composition, which happens at
            # startup. Kill it and reap it rather than leave a zombie. The
            # child may have exited in the gap between the timeout firing and
            # this line, in which case `kill()` raises `ProcessLookupError` —
            # a probe never raises, so that race is not an error here.
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            return None
        except OSError:
            return None
        if process.returncode != 0:
            return None
        return _as_revision(stdout)
