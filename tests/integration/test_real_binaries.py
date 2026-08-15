"""The probe, against the machine's real executables.

Its unit tests use shell-script stand-ins, which is the right way to test the
contract — but a probe is the one component whose correctness cannot be
established against a stand-in. These ran clean while exiftool was installed and
reported absent, and while libreoffice's recorded revision was a deprecation
warning.

This test cannot assert WHICH binaries exist, since that varies by machine. It
asserts the property that actually failed: presence on PATH and a discovered
revision must agree.
"""

from __future__ import annotations

import shutil

import pytest

from readeverything.adapters.binary_probe import DEFAULT_EXECUTABLES, BinaryProbe
from readeverything.domain.capability import Capability

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("capability", list(DEFAULT_EXECUTABLES))
async def test_presence_on_path_agrees_with_discovery(capability: Capability) -> None:
    executable, _flag = DEFAULT_EXECUTABLES[capability]
    on_path = shutil.which(executable) is not None
    revision = await BinaryProbe().revision(capability)

    if on_path:
        assert revision is not None, f"{executable} is on PATH but the probe reports it unavailable"
        assert not revision.lower().startswith(("warning", "error")), (
            f"{executable}'s revision is a diagnostic, not a version: {revision!r}"
        )
    else:
        assert revision is None
