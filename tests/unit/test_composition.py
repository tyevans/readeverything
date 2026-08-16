"""The composition root: one function between a directory and a `Perception`."""

from __future__ import annotations

import asyncio
import io
import sys
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from readeverything.adapters.ooxml import SLIDES_MIME
from readeverything.adapters.semaphore_limiter import DEFAULT_LIMITS, SemaphoreLimiter
from readeverything.composition import build_perception
from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.domain.errors import DomainError
from readeverything.domain.identity import ContentHash, MimeType, SourceRef
from readeverything.domain.locators import CharSpan, PartSpan
from readeverything.domain.rendition import Budget
from readeverything.handlers.audio import AudioHandler
from readeverything.handlers.video import VideoHandler
from readeverything.ports.limits import Limiter
from readeverything.ports.streams import MediaFacts, StreamInfo
from readeverything.testing.fakes import CountingLimiter, FakeVision, RecordingObserver
from tests.fixtures_epub import build_epub


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
    """`VideoHandler` consumes FFMPEG and the office handlers consume
    DOCUMENT_RENDER, but no bundled handler consumes EXIFTOOL/LIBREOFFICE/
    TESSERACT, so a `BinaryProbe` finding one of those must not invalidate the
    artifact cache — the fingerprint of a deployment must not depend on a
    capability nothing uses.

    LIBREOFFICE stays on that list even though rendering landed. It is the
    "that particular binary answered a version probe" capability; DOCUMENT_RENDER,
    which is what a renderer negotiates on, is a different member probed through
    a different executable.
    """
    baseline = await build_perception(tmp_path, probe_binaries=False)
    baseline_fingerprint = baseline.registry.capabilities.fingerprint()

    class _FakeProbeThatFindsUnusedBinaries:
        async def revision(self, capability: Capability) -> str | None:
            if capability in (
                Capability.FFMPEG,
                Capability.VISION,
                Capability.DOCUMENT_RENDER,
            ):
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
    assert video._observer is observer
    assert video._limiter is limiter
    assert audio._observer is observer


async def test_supplying_an_observer_and_a_limiter_changes_no_output(tmp_path: Path) -> None:
    """Narration and bounding are invisible in the result, and only there.

    The old version of this test built one perception with defaults and
    asserted its output looked right — it named no second program, so it
    could not have detected an observer or a limiter changing anything. Both
    arms are built here, and the injected arm is a real `RecordingObserver`
    and a real `CountingLimiter`, so the comparison has two sides.

    Note this is about OUTPUT. `build_perception` now installs a
    `SemaphoreLimiter` by default, so concurrency deliberately does differ
    between a caller's limiter and ours; what must not differ is a single
    character of what the read reports.
    """
    (tmp_path / "a.txt").write_text("hello")
    plain = await build_perception(tmp_path, probe_binaries=False)
    observer = RecordingObserver()
    instrumented = await build_perception(
        tmp_path,
        probe_binaries=False,
        observer=observer,
        limiter=CountingLimiter(),
    )
    quiet = await plain.represent("a.txt", Budget(max_chars=None))
    noisy = await instrumented.represent("a.txt", Budget(max_chars=None))
    assert quiet.text == noisy.text
    assert (await plain.inspect("a.txt")).ref.uri == (await instrumented.inspect("a.txt")).ref.uri
    # And the observer really was wired, so "identical" is not identical-
    # because-nothing-happened.
    assert observer.events


class _CountingProxy:
    """Wraps whatever limiter composition installed, recording peak in-flight.

    It delegates rather than replacing: the bound under test is the one
    `build_perception` chose for itself, not one this test supplied.
    """

    def __init__(self, inner: Limiter) -> None:
        self._inner = inner
        self.in_flight: dict[Capability, int] = {}
        self.peak: dict[Capability, int] = {}

    @asynccontextmanager
    async def limit(self, capability: Capability) -> AsyncIterator[None]:
        async with self._inner.limit(capability):
            self.in_flight[capability] = self.in_flight.get(capability, 0) + 1
            self.peak[capability] = max(self.peak.get(capability, 0), self.in_flight[capability])
            try:
                yield
            finally:
                self.in_flight[capability] -= 1


class _SlowFrames:
    """Slow enough that concurrent extractions actually overlap.

    An instantaneous fake would never show a peak above one, and a peak of one
    proves nothing about a bound.
    """

    async def frame_at(self, path: str, seconds: float) -> bytes | None:
        await asyncio.sleep(0.005)
        return b"frame"

    async def scene_cuts(self, path: str, threshold: float = 0.4) -> tuple[float, ...]:
        return ()


class _SlowVision:
    model_id = "slow-vision@1"

    async def describe(self, data: bytes, mime: str, prompt: str) -> str:
        await asyncio.sleep(0.005)
        return f"[{len(data)} bytes]"


class _StubProbe:
    async def probe(self, path: str) -> MediaFacts:
        return MediaFacts(
            duration_s=30.0,
            container="mp4",
            streams=(
                StreamInfo(
                    kind="video",
                    codec="h264",
                    width=16,
                    height=16,
                    frame_rate=1.0,
                    sample_rate=None,
                    channels=None,
                ),
            ),
        )


class _AnyPathSource:
    async def read_bytes(self, uri: str) -> bytes:
        return b""

    async def read_range(self, uri: str, start: int, end: int) -> bytes:
        return b""

    def stream(self, uri: str, *, chunk_size: int = 1 << 20):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def local_path(self, uri: str) -> str:
        return "/nonexistent.mp4"


def _video_ref() -> SourceRef:
    return SourceRef(
        uri="v.mp4",
        mime=MimeType.parse("video/mp4"),
        content_hash=ContentHash("f" * 64),
        size_bytes=4096,
    )


async def test_a_caller_who_injects_no_limiter_is_still_bounded(tmp_path: Path) -> None:
    """The default is a real bound, not the absence of one.

    `VideoHandler` fans out across every sampled moment at once. With no
    bound, a thirty-second video launches thirty ffmpeg subprocesses
    simultaneously, and the over-subscription that follows is caught by
    `_moment` and reported to the caller as "(no frame could be decoded at
    this moment)" — our load, misattributed to their file. So the composition
    root supplies `SemaphoreLimiter()` when the caller supplies nothing.

    Asserted by peak in-flight count against `DEFAULT_LIMITS`, measured
    through a proxy that WRAPS the limiter composition chose rather than
    replacing it — and asserted EQUAL to the bound, not merely under it, so a
    handler that had quietly gone sequential would fail this too.
    """
    perception = await build_perception(
        tmp_path,
        capabilities=CapabilitySet.of({Capability.FFMPEG: "1"}),
    )
    video = {type(h).__name__: h for h in perception.registry.handlers}["VideoHandler"]
    assert isinstance(video, VideoHandler)
    assert video._limiter is not None
    counting = _CountingProxy(video._limiter)
    video._limiter = counting
    video._source = _AnyPathSource()
    video._probe = _StubProbe()
    video._frames = _SlowFrames()
    video._vision = _SlowVision()
    video._interval_s = 1.0

    await video.represent(_video_ref(), Budget(max_chars=None))

    assert counting.peak[Capability.FFMPEG] == DEFAULT_LIMITS[Capability.FFMPEG]
    assert counting.peak[Capability.VISION] == DEFAULT_LIMITS[Capability.VISION]


async def test_an_explicitly_injected_limiter_replaces_the_default(tmp_path: Path) -> None:
    """Defaulting must not mean overriding.

    The control arm is a real, differently-configured limiter rather than
    `None`: `None` is what composition now fills in for itself, so passing it
    would compare the default to the default.
    """
    mine = SemaphoreLimiter({Capability.VISION: 1})
    perception = await build_perception(
        tmp_path,
        capabilities=CapabilitySet.of({Capability.FFMPEG: "1"}),
        limiter=mine,
    )
    video = {type(h).__name__: h for h in perception.registry.handlers}["VideoHandler"]
    assert isinstance(video, VideoHandler)
    assert video._limiter is mine


async def test_a_caption_extractor_is_wired_without_being_asked_for(tmp_path: Path) -> None:
    """Unlike a vision model or a transcriber, reading captions needs no
    configuration, no weights and no endpoint — and the handler only reaches
    for the extractor when a probe already said a readable track exists.

    A caller should not have to know to ask for the cheapest thing in the
    library. Before this was wired, an agent asked what a captioned lecture
    was about spent twelve vision calls and five minutes on a question the
    file itself answered in a second.
    """
    perception = await build_perception(
        tmp_path, capabilities=CapabilitySet.of({Capability.FFMPEG: "1"})
    )
    handlers = [h for h in perception.registry.handlers if isinstance(h, VideoHandler)]
    assert handlers, "no video handler was registered"
    assert handlers[0]._captions is not None


# --- rendering ------------------------------------------------------------


class _Renderer:
    """A caller's own converter, which is the point of the port existing."""

    revision = "my-converter/7"

    def claims(self, mime: MimeType) -> bool:
        return True

    async def page_count(self, path: str) -> int:
        return 1

    async def page_text(self, path: str, page: int) -> str:
        return "converted text"

    async def render_page(self, path: str, page: int, *, dpi: int = 150) -> bytes:
        return b"\x89PNG"


def _deck(tmp_path: Path) -> Path:
    from tests.fixtures_office import pptx_bytes

    path = tmp_path / "deck.pptx"
    path.write_bytes(pptx_bytes())
    return path


async def _deck_affordances(tmp_path: Path, **kwargs: object) -> set[str]:
    _deck(tmp_path)
    perception = await build_perception(tmp_path, **kwargs)  # type: ignore[arg-type]
    return {a.name for a in (await perception.inspect("deck.pptx")).affordances}


async def test_an_injected_renderer_publishes_rendering_without_probing(
    tmp_path: Path,
) -> None:
    """A caller pointing at their own converter must not need LibreOffice
    installed for the library to believe them."""
    names = await _deck_affordances(tmp_path, renderer=_Renderer(), probe_binaries=False)
    assert "page_image" in names


async def test_an_injected_renderer_declares_its_own_revision(tmp_path: Path) -> None:
    """It enters the capability fingerprint and therefore the artifact cache
    key. Two different converters must not share cached descriptions."""
    _deck(tmp_path)
    perception = await build_perception(tmp_path, renderer=_Renderer(), probe_binaries=False)
    assert (
        perception.registry.capabilities.revisions[Capability.DOCUMENT_RENDER] == "my-converter/7"
    )


async def test_a_null_renderer_turns_rendering_off_on_a_machine_that_has_soffice(
    tmp_path: Path,
) -> None:
    """Determinism in CI without uninstalling software.

    `probe_binaries=True` deliberately: this must hold on the machine that
    developed the feature, which has LibreOffice.
    """
    from readeverything.adapters.null_renderer import NullRenderer

    names = await _deck_affordances(tmp_path, renderer=NullRenderer(), probe_binaries=True)
    assert "page_image" not in names
    assert "read_slide" in names, "everything else still works"


async def test_a_null_renderer_does_not_declare_the_capability(tmp_path: Path) -> None:
    from readeverything.adapters.null_renderer import NullRenderer

    perception = await build_perception(tmp_path, renderer=NullRenderer(), probe_binaries=True)
    assert Capability.DOCUMENT_RENDER not in perception.registry.capabilities.revisions


async def test_a_null_renderer_leaves_legacy_files_on_the_hex_dump(tmp_path: Path) -> None:
    from readeverything.adapters.null_renderer import NullRenderer
    from readeverything.handlers.binary import BinaryHandler

    perception = await build_perception(tmp_path, renderer=NullRenderer(), probe_binaries=True)
    handler = perception.registry.resolve(MimeType.parse("application/msword"))
    assert isinstance(handler, BinaryHandler)


async def test_without_the_capability_no_rendering_affordance_appears(tmp_path: Path) -> None:
    """The design's first promise, at the composition root: everything works as
    it did and no rendering affordance appears anywhere."""
    names = await _deck_affordances(tmp_path, probe_binaries=False)
    assert "page_image" not in names
    assert {"read_slide", "list_media"} <= names


async def test_the_renderer_and_the_perception_share_one_artifact_store(
    tmp_path: Path,
) -> None:
    """A converted PDF is an artifact like any other. Two stores would mean a
    caller who configured a persistent cache still paid for every conversion on
    every process start."""
    from readeverything.adapters.artifact_store import InMemoryArtifactStore
    from readeverything.adapters.soffice_renderer import SofficeRenderer

    store = InMemoryArtifactStore()
    perception = await build_perception(tmp_path, artifacts=store, probe_binaries=False)
    handler = perception.registry.resolve(MimeType.parse(SLIDES_MIME))
    renderer = handler._renderer  # type: ignore[attr-defined]
    assert isinstance(renderer, SofficeRenderer)
    assert renderer._artifacts is store


async def test_an_html_file_is_read_as_prose_rather_than_markup(tmp_path: Path) -> None:
    """The sentence this handler exists for, through the real root.

    Before it, `text/html` matched `TextHandler` at `MatchRank.KIND` and a
    saved article came back as tags, scripts and stylesheets.
    """
    page = tmp_path / "article.html"
    page.write_text(
        "<html><head><title>A Title</title></head><body>"
        "<h1>The Heading</h1><p>The prose.</p>"
        "<script>var junk = 1;</script></body></html>"
    )
    perception = await build_perception(tmp_path)
    rendered = await perception.represent("article.html", Budget(max_chars=None))
    assert "The prose." in rendered.text
    assert "<h1>" not in rendered.text
    assert "var junk" not in rendered.text


async def test_an_html_files_citation_points_into_the_original_markup(tmp_path: Path) -> None:
    source = "<html><body><h1>The Heading</h1><p>A sentence worth quoting.</p></body></html>"
    page = tmp_path / "article.html"
    page.write_text(source)
    perception = await build_perception(tmp_path)
    rendered = await perception.represent("article.html", Budget(max_chars=None))
    offset = rendered.text.index("A sentence worth quoting.")
    locator = rendered.locator_map.resolve(offset)
    assert isinstance(locator, CharSpan)
    assert source[locator.start : locator.end].strip() == "A sentence worth quoting."


async def test_an_epub_is_read_as_a_book_rather_than_hex_dumped(tmp_path: Path) -> None:
    """The sentence this handler exists for, through the real root.

    An epub is deliberately not descended into as a folder, and before this
    handler nothing claimed the mimetype -- so a book came back as hex.
    """
    (tmp_path / "book.epub").write_bytes(build_epub())
    perception = await build_perception(tmp_path)
    card = await perception.inspect("book.epub")
    assert card.facts["title"] == "A Short Book"
    rendered = await perception.represent("book.epub", Budget(max_chars=None))
    assert "It began quietly." in rendered.text


async def test_a_quote_from_a_book_names_the_part_it_came_from(tmp_path: Path) -> None:
    """The citation, in a container: extract that part and the sentence is there."""
    (tmp_path / "book.epub").write_bytes(build_epub())
    perception = await build_perception(tmp_path)
    rendered = await perception.represent("book.epub", Budget(max_chars=None))
    locator = rendered.locator_map.resolve(rendered.text.index("It ended."))
    assert isinstance(locator, PartSpan)
    with zipfile.ZipFile(tmp_path / "book.epub") as book:
        part = book.read(locator.part).decode()
    assert "It ended." in part[locator.start : locator.end]
