"""Capability negotiation, end to end: what an agent sees changes with the deployment."""

from __future__ import annotations

from pathlib import Path

from readeverything.composition import build_perception
from readeverything.testing.fakes import FakeVision


async def test_the_same_directory_offers_less_without_a_vision_model(media_root: Path) -> None:
    """One directory, two deployments, and the difference is visible to the agent."""
    with_vision = await build_perception(media_root, vision=FakeVision(), probe_binaries=False)
    without = await build_perception(media_root, probe_binaries=False)

    rich = {a.name for a in (await with_vision.inspect("photo.png")).affordances}
    plain = {a.name for a in (await without.inspect("photo.png")).affordances}

    assert plain < rich
    assert "crop_region" in plain  # needs no model, so it survives
    assert {"describe_image", "ocr"} & plain == set()


async def test_nothing_unavailable_is_ever_offered(media_root: Path) -> None:
    """The design goal, asserted directly: an agent never sees a tool it cannot use."""
    perception = await build_perception(media_root, probe_binaries=False)
    capabilities = perception.registry.capabilities
    for uri in await perception.list("."):
        for affordance in (await perception.inspect(uri)).affordances:
            assert affordance.is_available(capabilities)
