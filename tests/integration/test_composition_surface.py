"""The composition root's registered handlers, checked against the front door."""

from __future__ import annotations

from pathlib import Path

import pytest

from readeverything.composition import build_perception
from readeverything.testing.fakes import FakeVision

pytestmark = pytest.mark.integration


async def test_every_registered_handler_is_exported_from_the_front_door(
    media_root: Path,
) -> None:
    """Adding a handler must mean adding it in exactly one place.

    The set of handlers is READ from a real composition rather than restated
    here. A hardcoded list would pass for a handler nobody exported, because
    nobody would have added it to the list either — the test would agree with
    the mistake instead of catching it.
    """
    import readeverything

    perception = await build_perception(media_root, vision=FakeVision(), probe_binaries=False)
    registered = {type(h).__name__ for h in perception.registry.handlers}
    assert registered, "composition registered no handlers at all"
    assert registered <= set(readeverything.__all__), (
        f"registered but not exported: {sorted(registered - set(readeverything.__all__))}"
    )
