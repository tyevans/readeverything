"""BinaryProbe: availability means it ran and answered, not that a path exists."""

from __future__ import annotations

from pathlib import Path

from readeverything.adapters.binary_probe import BinaryProbe
from readeverything.domain.capability import Capability


async def test_a_missing_binary_is_unavailable() -> None:
    probe = BinaryProbe(executables={Capability.FFMPEG: "definitely-not-a-real-binary-xyz"})
    assert await probe.revision(Capability.FFMPEG) is None


async def test_a_present_binary_reports_a_revision() -> None:
    """`echo` stands in for ffmpeg: present, runs, prints something.

    What is under test is the probe's contract — locate, run, capture — not
    ffmpeg's version string, which would make this test require ffmpeg.
    """
    probe = BinaryProbe(executables={Capability.FFMPEG: "echo"}, version_flag="7.1")
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
    probe = BinaryProbe(executables={Capability.FFMPEG: str(script)}, timeout_s=0.2)
    assert await probe.revision(Capability.FFMPEG) is None


async def test_a_probe_never_raises() -> None:
    """Under uncertainty the library offers less, never more."""
    probe = BinaryProbe(executables={})
    assert await probe.revision(Capability.TESSERACT) is None
