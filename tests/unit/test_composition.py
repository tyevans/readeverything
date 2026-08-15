"""The composition root: one function between a directory and a `Perception`."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from readeverything.composition import build_perception
from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.testing.fakes import FakeVision


def _png_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 4), (0, 128, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


async def test_a_root_is_the_only_thing_a_caller_must_supply(tmp_path: Path) -> None:
    """The acceptance sentence, as a test.

    Before this, a caller assembled ten objects in dependency order and had to
    know which handler classes existed to do it.
    """
    (tmp_path / "a.txt").write_text("hello")
    perception = await build_perception(tmp_path)
    card = await perception.inspect("a.txt")
    assert card.ref.uri == "a.txt"


async def test_a_base_install_without_pillow_still_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Handlers whose dependencies are absent are omitted, not fatal.

    The front door advertises `ImageHandler` on a base install and importing it
    raises ModuleNotFoundError: PIL. A composition root that propagated that
    would make the base install unusable rather than merely narrower.
    """
    monkeypatch.setitem(sys.modules, "PIL", None)
    (tmp_path / "a.txt").write_text("hello")
    perception = await build_perception(tmp_path)
    assert await perception.inspect("a.txt") is not None


async def test_explicit_capabilities_are_used_verbatim_and_nothing_is_probed(
    tmp_path: Path,
) -> None:
    """Tests must be able to declare any capability set without touching the machine."""
    declared = CapabilitySet.of({Capability.FFMPEG: "declared-not-probed"})
    perception = await build_perception(tmp_path, capabilities=declared, probe_binaries=False)
    assert perception.registry.capabilities == declared


async def test_a_vision_model_registers_the_affordances_that_need_it(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(_png_bytes())
    with_vision = await build_perception(tmp_path, vision=FakeVision(), probe_binaries=False)
    names = {a.name for a in (await with_vision.inspect("a.png")).affordances}
    assert {"describe_image", "ocr"} <= names


async def test_no_vision_model_means_those_affordances_are_not_offered(tmp_path: Path) -> None:
    """Negotiation working, not degradation. The agent never sees a tool it cannot use."""
    (tmp_path / "a.png").write_bytes(_png_bytes())
    without = await build_perception(tmp_path, probe_binaries=False)
    names = {a.name for a in (await without.inspect("a.png")).affordances}
    assert "describe_image" not in names
    assert "crop_region" in names


def test_the_composition_root_reads_no_environment() -> None:
    """The constraint that has held since Spec 1 §3, checked at the new top layer."""
    source = Path("src/readeverything/composition.py").read_text()
    assert "os.environ" not in source
    assert "getenv" not in source


def test_every_bundled_handler_is_exported_from_the_front_door() -> None:
    """Adding a handler must mean adding it in exactly one place.

    A handler registered in the composition root but missing from the front
    door — or the reverse — is a surface that lies about itself.
    """
    import readeverything

    registered = {"TextHandler", "BinaryHandler", "ImageHandler"}
    assert registered <= set(readeverything.__all__)
