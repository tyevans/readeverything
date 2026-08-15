"""The composition root: one function between a directory and a `Perception`."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from readeverything.composition import build_perception
from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.domain.errors import DomainError
from readeverything.handlers.audio import AudioHandler
from readeverything.handlers.video import VideoHandler
from readeverything.testing.fakes import CountingLimiter, FakeVision, RecordingObserver


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
    perception = await build_perception(tmp_path, probe_binaries=False)
    card = await perception.inspect("a.txt")
    assert card.ref.uri == "a.txt"


async def test_a_base_install_without_pillow_still_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Handlers whose dependencies are absent are omitted, not fatal.

    The front door advertises `ImageHandler` on a base install and importing it
    raises ModuleNotFoundError: PIL. A composition root that propagated that
    would make the base install unusable rather than merely narrower.

    Blocking `PIL` in `sys.modules` alone proves nothing: `_optional_image_handler`
    does `from readeverything.handlers.image import ImageHandler` inside its
    `try`, and if `readeverything.handlers.image` is already cached in
    `sys.modules` (true in any full-suite run, since other test modules import
    it), that import is served from cache and the module's own `import PIL`
    never re-executes — so `ImageHandler` registers anyway and the test passes
    vacuously regardless of what the guard does. Evicting the handler module
    (and blocking `PIL`) forces the guarded import to actually run, and
    asserting the handler's absence from the registry is the real claim; "it
    did not raise" is much weaker.
    """
    monkeypatch.delitem(sys.modules, "readeverything.handlers.image", raising=False)
    monkeypatch.setitem(sys.modules, "PIL", None)
    (tmp_path / "a.txt").write_text("hello")
    perception = await build_perception(tmp_path, probe_binaries=False)
    assert await perception.inspect("a.txt") is not None
    assert "ImageHandler" not in {type(h).__name__ for h in perception.registry.handlers}


async def test_explicit_capabilities_are_used_verbatim_and_nothing_is_probed(
    tmp_path: Path,
) -> None:
    """Tests must be able to declare any capability set without touching the machine."""
    declared = CapabilitySet.of({Capability.FFMPEG: "declared-not-probed"})
    perception = await build_perception(tmp_path, capabilities=declared, probe_binaries=False)
    assert perception.registry.capabilities == declared


async def test_a_vision_model_and_a_disagreeing_declared_revision_raise(
    tmp_path: Path,
) -> None:
    """A caller-declared VISION revision that doesn't match the injected model
    would otherwise let two different vision models share one cache key.
    """
    mismatched = CapabilitySet.of({Capability.VISION: "some-other-model@2"})
    with pytest.raises(DomainError):
        await build_perception(
            tmp_path, vision=FakeVision(), capabilities=mismatched, probe_binaries=False
        )


async def test_a_vision_model_and_an_agreeing_declared_revision_is_fine(
    tmp_path: Path,
) -> None:
    vision = FakeVision()
    agreeing = CapabilitySet.of({Capability.VISION: vision.model_id})
    perception = await build_perception(
        tmp_path, vision=vision, capabilities=agreeing, probe_binaries=False
    )
    assert perception.registry.capabilities == agreeing


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


async def test_probing_an_unused_capability_does_not_change_the_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`VideoHandler` is the first bundled handler to consume FFMPEG, but no
    bundled handler consumes EXIFTOOL/LIBREOFFICE/TESSERACT, so a `BinaryProbe`
    finding one of those must not invalidate the artifact cache — the
    fingerprint of a deployment must not depend on a capability nothing uses.
    """
    baseline = await build_perception(tmp_path, probe_binaries=False)
    baseline_fingerprint = baseline.registry.capabilities.fingerprint()

    class _FakeProbeThatFindsUnusedBinaries:
        async def revision(self, capability: Capability) -> str | None:
            if capability in (Capability.FFMPEG, Capability.VISION):
                return None
            return "some-version-string"

    monkeypatch.setattr("readeverything.composition.BinaryProbe", _FakeProbeThatFindsUnusedBinaries)
    with_probe = await build_perception(tmp_path, probe_binaries=True)
    assert with_probe.registry.capabilities.fingerprint() == baseline_fingerprint


def test_the_composition_root_reads_no_environment() -> None:
    """The constraint that has held since Spec 1 §3, checked at the new top layer."""
    source = Path("src/readeverything/composition.py").read_text()
    assert "os.environ" not in source
    assert "getenv" not in source


async def test_no_ffmpeg_means_no_video_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negotiation against a real OS dependency, for the first time.

    The handler REQUIRES ffmpeg, so without it the registry drops the handler
    entirely and video files fall to the binary fallback. Narrower, not broken.
    """
    perception = await build_perception(
        tmp_path, capabilities=CapabilitySet.empty(), probe_binaries=False
    )
    assert "VideoHandler" not in {type(h).__name__ for h in perception.registry.handlers}


async def test_an_observer_and_a_limiter_reach_the_media_handlers(tmp_path: Path) -> None:
    """Threaded, not just accepted: `build_perception` must not swallow either.

    Reads the handlers' private attributes rather than exercising a full
    read — this is `build_perception`'s job of wiring, not a behavioural test
    of either handler, which already has its own.
    """
    observer = RecordingObserver()
    limiter = CountingLimiter()
    perception = await build_perception(
        tmp_path,
        capabilities=CapabilitySet.of({Capability.FFMPEG: "1"}),
        observer=observer,
        limiter=limiter,
    )
    handlers = {type(h).__name__: h for h in perception.registry.handlers}
    video = handlers["VideoHandler"]
    audio = handlers["AudioHandler"]
    assert isinstance(video, VideoHandler)
    assert isinstance(audio, AudioHandler)
    assert video._observer is observer  # noqa: SLF001
    assert video._limiter is limiter  # noqa: SLF001
    assert audio._observer is observer  # noqa: SLF001


async def test_defaults_change_nothing(tmp_path: Path) -> None:
    """No observer, no limiter — today's behaviour exactly."""
    (tmp_path / "a.txt").write_text("hello")
    perception = await build_perception(tmp_path, probe_binaries=False)
    card = await perception.inspect("a.txt")
    assert card.ref.uri == "a.txt"
