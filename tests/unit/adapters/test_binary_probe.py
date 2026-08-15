"""BinaryProbe: availability means it ran and answered, not that a path exists."""

from __future__ import annotations

from pathlib import Path

from readeverything.adapters.binary_probe import DEFAULT_EXECUTABLES, BinaryProbe
from readeverything.domain.capability import Capability


async def test_a_missing_binary_is_unavailable() -> None:
    probe = BinaryProbe(
        executables={Capability.FFMPEG: ("definitely-not-a-real-binary-xyz", "-version")}
    )
    assert await probe.revision(Capability.FFMPEG) is None


async def test_a_present_binary_reports_a_revision() -> None:
    """`echo` stands in for ffmpeg: present, runs, prints something.

    What is under test is the probe's contract — locate, run, capture — not
    ffmpeg's version string, which would make this test require ffmpeg.
    """
    probe = BinaryProbe(executables={Capability.FFMPEG: ("echo", "7.1")})
    revision = await probe.revision(Capability.FFMPEG)
    assert revision is not None and "7.1" in revision


async def test_a_binary_that_hangs_is_unavailable_rather_than_hanging(tmp_path: Path) -> None:
    """A hung `--version` must not hang composition.

    Composition happens at startup. A probe without a timeout turns one broken
    binary into an application that never finishes starting.
    """
    script = tmp_path / "hang"
    script.write_text("#!/bin/sh\nsleep 30\n")
    script.chmod(0o755)
    probe = BinaryProbe(executables={Capability.FFMPEG: (str(script), "-version")}, timeout_s=0.2)
    assert await probe.revision(Capability.FFMPEG) is None


async def test_a_probe_never_raises() -> None:
    """Under uncertainty the library offers less, never more."""
    probe = BinaryProbe(executables={})
    assert await probe.revision(Capability.TESSERACT) is None


async def test_each_capability_carries_its_own_version_flag() -> None:
    """One global flag was the bug.

    `exiftool -version` is not exiftool's version invocation — it wants `-ver`.
    With a single flag for every executable, exiftool was installed on the
    machine and reported absent, and "absent" is indistinguishable from "not
    installed".
    """
    assert DEFAULT_EXECUTABLES[Capability.EXIFTOOL] == ("exiftool", "-ver")
    assert DEFAULT_EXECUTABLES[Capability.FFMPEG] == ("ffmpeg", "-version")
    assert DEFAULT_EXECUTABLES[Capability.LIBREOFFICE] == ("libreoffice", "--version")


async def test_a_warning_is_not_a_version(tmp_path: Path) -> None:
    """`libreoffice -version` printed a deprecation warning, and the probe
    recorded it as the revision. That string then entered the capability
    fingerprint and therefore every artifact cache key — a warning became part
    of this library's cache identity.

    Nothing established that the captured line was a version. Under uncertainty
    the probe returns None, which its own contract already requires.
    """
    script = tmp_path / "warner"
    script.write_text(
        "#!/bin/sh\necho 'Warning: -version is deprecated.  Use --version instead.'\n"
    )
    script.chmod(0o755)
    probe = BinaryProbe(executables={Capability.FFMPEG: (str(script), "-version")})
    assert await probe.revision(Capability.FFMPEG) is None


async def test_an_error_line_is_not_a_version(tmp_path: Path) -> None:
    script = tmp_path / "errorer"
    script.write_text("#!/bin/sh\necho 'Error: no such option'\n")
    script.chmod(0o755)
    probe = BinaryProbe(executables={Capability.FFMPEG: (str(script), "-version")})
    assert await probe.revision(Capability.FFMPEG) is None


async def test_a_real_version_line_is_accepted(tmp_path: Path) -> None:
    """The rejection must not be so eager that it rejects genuine versions."""
    script = tmp_path / "versioner"
    script.write_text("#!/bin/sh\necho 'ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023'\n")
    script.chmod(0o755)
    probe = BinaryProbe(executables={Capability.FFMPEG: (str(script), "-version")})
    revision = await probe.revision(Capability.FFMPEG)
    assert revision is not None and "6.1.1" in revision
