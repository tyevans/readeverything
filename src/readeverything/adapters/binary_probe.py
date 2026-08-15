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

#: The executable each capability is provided by, and the flag that makes it
#: print a version and exit. Both are fixed here rather than caller-supplied:
#: this module runs subprocesses, and the argument vector must never contain
#: anything a caller can influence.
DEFAULT_EXECUTABLES: Mapping[Capability, str] = {
    Capability.FFMPEG: "ffmpeg",
    Capability.EXIFTOOL: "exiftool",
    Capability.LIBREOFFICE: "libreoffice",
    Capability.TESSERACT: "tesseract",
}


class BinaryProbe:
    """Reports a capability available when its executable runs and answers."""

    def __init__(
        self,
        *,
        executables: Mapping[Capability, str] | None = None,
        version_flag: str = "-version",
        timeout_s: float = 5.0,
    ) -> None:
        self._executables = dict(DEFAULT_EXECUTABLES if executables is None else executables)
        self._version_flag = version_flag
        self._timeout_s = timeout_s

    async def revision(self, capability: Capability) -> str | None:
        name = self._executables.get(capability)
        if name is None:
            return None
        located = shutil.which(name)
        if located is None:
            return None
        try:
            process = await asyncio.create_subprocess_exec(
                located,
                self._version_flag,
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
        lines = stdout.decode("utf-8", errors="replace").strip().splitlines()
        return lines[0].strip() if lines else None
