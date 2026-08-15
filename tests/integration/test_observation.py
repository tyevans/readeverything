"""§1.1's sentence, answered end to end: which file took the time.

`Perception` does not fan out across a directory itself — the caller writes
that loop, over `list()` — so these tests write it too, and check that the
observer sees each file's start and finish and can attribute elapsed time per
`uri` without threading any state of its own.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from readeverything.composition import build_perception
from readeverything.domain.observation import OperationFinished
from readeverything.domain.rendition import Budget
from readeverything.testing.fakes import FakeVision, RecordingObserver

pytestmark = pytest.mark.integration


async def test_an_observer_sees_a_whole_directory_being_read(media_root: Path) -> None:
    recorder = RecordingObserver()
    perception = await build_perception(
        media_root, vision=FakeVision(), observer=recorder, probe_binaries=False
    )
    for uri in await perception.list("."):
        await perception.represent(uri, Budget(max_chars=None))

    finished = [e for e in recorder.events if isinstance(e, OperationFinished)]
    assert {e.ref.uri for e in finished} == set(await perception.list("."))
    assert all(e.elapsed_s >= 0.0 for e in finished)


async def test_observing_changes_nothing(media_root: Path) -> None:
    """Attaching an observer perturbs no rendition anywhere in the directory."""
    recorder = RecordingObserver()
    unobserved = await build_perception(media_root, vision=FakeVision(), probe_binaries=False)
    observed = await build_perception(
        media_root, vision=FakeVision(), observer=recorder, probe_binaries=False
    )
    for uri in await unobserved.list("."):
        before = await unobserved.represent(uri, Budget(max_chars=None))
        after = await observed.represent(uri, Budget(max_chars=None))
        assert before.text == after.text
    assert recorder.events, "the observed arm must actually have been observed"
