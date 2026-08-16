"""One function between a directory and a working `Perception`.

Everything here is a convenience over the public constructors and nothing more.
If this module can do something a caller assembling the pieces by hand cannot,
that is a bug in the constructors, not a feature of this file — which is why it
takes the same arguments they do and holds no state of its own.

It reads no environment variables. Every input is an argument, so two
differently-configured instances can run in one process.
"""

from __future__ import annotations

from pathlib import Path

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.binary_probe import BinaryProbe
from readeverything.adapters.detection import PuremagicDetector
from readeverything.adapters.hashing import ContentHasher, StatMemo
from readeverything.adapters.local_source import LocalFileSource
from readeverything.adapters.model_probe import ModelProbe
from readeverything.adapters.nested_source import CompositeOpener, NestedSource
from readeverything.adapters.probing import discover
from readeverything.adapters.semaphore_limiter import SemaphoreLimiter
from readeverything.adapters.tar_archive import TarArchiveOpener
from readeverything.adapters.zip_archive import ZipArchiveOpener
from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.domain.errors import DomainError
from readeverything.handlers.archive import ArchiveHandler
from readeverything.handlers.binary import BinaryHandler
from readeverything.handlers.text import TextHandler
from readeverything.pipeline.perception import Perception
from readeverything.pipeline.resolution import ResolutionMemo
from readeverything.ports.artifacts import ArtifactStore
from readeverything.ports.captions import CaptionExtractor
from readeverything.ports.clips import ClipModel
from readeverything.ports.containers import ArchiveOpener, ContainerLimits
from readeverything.ports.handler import MediaHandler
from readeverything.ports.limits import Limiter
from readeverything.ports.observation import Observer
from readeverything.ports.probe import CapabilityProbe
from readeverything.ports.recognition import TextRecognizer
from readeverything.ports.source import FileSource, SourceReader
from readeverything.ports.transcription import Transcriber
from readeverything.ports.vision import VisionModel
from readeverything.registry.registry import MimeTypeRegistry


def _capabilities_handlers_can_use(handlers: list[MediaHandler]) -> frozenset[Capability]:
    """The union of what the registered handlers, and their affordances, need.

    Probing anything outside this set is pure cost: a probed capability that
    no handler ever consults still enters `CapabilitySet.fingerprint()`, so it
    still invalidates every cached artifact when it changes on the host,
    despite never affecting a single rendition. Restricting the probe set to
    what is actually wired keeps the cache honest about what it depends on.
    """
    needed: set[Capability] = set()
    for handler in handlers:
        needed |= handler.requires()
        for affordance in handler.affordances():
            needed |= affordance.requires
    return frozenset(needed)


def _optional_image_handler(
    source: SourceReader, vision: VisionModel | None, observer: Observer | None
) -> list[MediaHandler]:
    """`ImageHandler` when Pillow is importable, nothing when it is not.

    Pillow lives behind the `images` extra. A base install must yield a working
    `Perception` that handles text and binary — narrower, not broken — so the
    import failure is a registration decision here rather than an exception
    reaching the caller.
    """
    try:
        from readeverything.handlers.image import ImageHandler
    except ImportError:
        return []
    return [ImageHandler(source=source, vision=vision, observer=observer)]


def _optional_pdf_handler(
    source: SourceReader, vision: VisionModel | None, observer: Observer | None
) -> list[MediaHandler]:
    """`PdfHandler` when pypdfium2 is importable, nothing when it is not.

    pypdfium2 lives behind the `documents` extra, guarded exactly like
    `_optional_image_handler` guards Pillow: narrower, not broken, on a base
    install.

    The recogniser is built from `vision` when one was supplied, so OCR
    negotiates on the capability the caller already provided rather than
    needing a second knob.
    """
    try:
        from readeverything.adapters.pdfium_probe import PdfiumProbe
        from readeverything.adapters.vision_recognizer import VisionTextRecognizer
        from readeverything.handlers.pdf import PdfHandler
    except ImportError:
        return []
    recognizer: TextRecognizer | None = (
        VisionTextRecognizer(vision=vision) if vision is not None else None
    )
    return [
        PdfHandler(
            source=source,
            probe=PdfiumProbe(),
            recognizer=recognizer,
            vision=vision,
            observer=observer,
        )
    ]


def _optional_office_handlers(
    source: SourceReader, vision: VisionModel | None, observer: Observer | None
) -> list[MediaHandler]:
    """The three office handlers, each present only if its own reader imports.

    Guarded exactly like `_optional_image_handler` and `_optional_pdf_handler`,
    but with THREE separate guards rather than one. The `office` extra installs
    python-docx, python-pptx and openpyxl together, yet an environment holding
    only some of them should still read the families it can — a single
    try/except around all three would make one missing package silently cost
    the other two.

    Only the slides handler takes `vision`: a picture embedded in a deck is the
    one thing in these three formats a model has to look at. Word and
    spreadsheet content is text all the way down.
    """
    handlers: list[MediaHandler] = []
    try:
        from readeverything.handlers.office_word import OfficeWordHandler
    except ImportError:
        pass
    else:
        handlers.append(OfficeWordHandler(source=source, observer=observer))
    try:
        from readeverything.handlers.office_slides import OfficeSlidesHandler
    except ImportError:
        pass
    else:
        handlers.append(OfficeSlidesHandler(source=source, vision=vision, observer=observer))
    try:
        from readeverything.handlers.office_sheets import OfficeSheetsHandler
    except ImportError:
        pass
    else:
        handlers.append(OfficeSheetsHandler(source=source, observer=observer))
    return handlers


def _video_handler(
    source: SourceReader,
    vision: VisionModel | None,
    transcriber: Transcriber | None,
    captions: CaptionExtractor | None,
    watcher: ClipModel | None,
    observer: Observer | None,
    limiter: Limiter | None,
) -> list[MediaHandler]:
    """`VideoHandler`, unconditionally.

    Unlike `_optional_image_handler`/`_optional_pdf_handler`, there is no
    Python package to guard: ffmpeg is an OS binary, not an import. The
    registry already drops any handler whose `requires()` is unsatisfied, so
    this constructs the handler and lets capability filtering decide whether
    it survives — the same "tool exists but returns sorry" trap that guards
    an import here would otherwise reintroduce.
    """
    from readeverything.adapters.ffmpeg_audio import FfmpegAudio
    from readeverything.adapters.ffmpeg_captions import FfmpegCaptions
    from readeverything.adapters.ffmpeg_clip import FfmpegClip
    from readeverything.adapters.ffmpeg_frames import FfmpegFrames
    from readeverything.adapters.ffprobe_streams import FfprobeStreams
    from readeverything.handlers.video import VideoHandler

    # The extractor is wired unconditionally alongside the transcriber. It costs
    # nothing to construct, and `VideoHandler` never touches it without a
    # transcriber to hand its bytes to — so a caller who configured ASR gets a
    # transcript on the video timeline without a second knob to find.
    #
    # Captions are wired the same way and for a stronger reason: unlike a
    # vision model or a transcriber, reading them needs no configuration, no
    # weights and no endpoint, and the handler only touches the extractor when
    # the probe already said a readable track exists. A caller should not have
    # to know to ask for the cheapest thing in the library — and before this
    # was wired, an agent asked what a captioned lecture was about spent
    # twelve vision calls and five minutes on a question the file answered in
    # a second.
    return [
        VideoHandler(
            source=source,
            probe=FfprobeStreams(),
            frames=FfmpegFrames(),
            vision=vision,
            audio=FfmpegAudio(),
            transcriber=transcriber,
            captions=FfmpegCaptions() if captions is None else captions,
            # The extractor is wired unconditionally and the WATCHER is not,
            # matching how vision is treated: cutting a clip costs an ffmpeg
            # call a caller already pays for elsewhere, while watching one
            # needs an endpoint that accepts video — which ours did not until
            # 2026-08-15. Without a watcher the affordance simply does not
            # appear, which is negotiation rather than degradation.
            clips=FfmpegClip(),
            watcher=watcher,
            observer=observer,
            limiter=limiter,
        )
    ]


def _audio_handler(
    source: SourceReader,
    transcriber: Transcriber | None,
    observer: Observer | None,
) -> list[MediaHandler]:
    """`AudioHandler`, unconditionally.

    Symmetric with `_video_handler`: ffmpeg is an OS binary, not an import, so
    there is nothing to guard with a `try`. The registry drops this handler
    when `FFMPEG` is unsatisfied, exactly as it does for video.

    No `limiter` here: `AudioHandler` transcribes in one call, with nothing
    concurrent to bound — `VideoHandler` is the one that fans out across
    frames.
    """
    from readeverything.adapters.ffmpeg_audio import FfmpegAudio
    from readeverything.adapters.ffprobe_streams import FfprobeStreams
    from readeverything.handlers.audio import AudioHandler

    return [
        AudioHandler(
            source=source,
            probe=FfprobeStreams(),
            audio=FfmpegAudio(),
            transcriber=transcriber,
            observer=observer,
        )
    ]


#: `build_perception`'s default for `containers`: descent ON, with §3.3's
#: values. A module-level singleton rather than a call in the signature
#: because a call there is a mutable-default trap in general -- harmless for a
#: frozen dataclass, but not worth teaching a reader to accept the shape.
DESCEND_INTO_CONTAINERS = ContainerLimits()


def _source_and_openers(
    root: Path | str,
    containers: ContainerLimits | None,
    archives: ArchiveOpener | None,
) -> tuple[FileSource, ArchiveOpener]:
    """The source to read through, and the openers to describe archives with.

    `containers=None` yields today's behavior EXACTLY, including no extra
    opens during `walk`: the decorator is not constructed at all, rather than
    constructed and told to do nothing, so there is no new code path to
    regress. The openers are still built, because `ArchiveHandler` describes a
    container either way -- a card is a probe, not a descent, and turning
    descent off should not cost a caller the ability to see what is in a
    tarball.
    """
    source = LocalFileSource(root=root)
    openers: ArchiveOpener = (
        CompositeOpener(
            openers=[
                ZipArchiveOpener(),
                TarArchiveOpener(
                    max_materialised_bytes=(
                        ContainerLimits() if containers is None else containers
                    ).max_materialised_bytes
                ),
            ]
        )
        if archives is None
        else archives
    )
    if containers is None:
        return source, openers
    return (
        NestedSource(
            source,
            limits=containers,
            archives=openers,
            detector=PuremagicDetector(),
        ),
        openers,
    )


async def build_perception(
    root: Path | str,
    *,
    vision: VisionModel | None = None,
    transcriber: Transcriber | None = None,
    captions: CaptionExtractor | None = None,
    watcher: ClipModel | None = None,
    capabilities: CapabilitySet | None = None,
    artifacts: ArtifactStore | None = None,
    probe_binaries: bool = True,
    observer: Observer | None = None,
    limiter: Limiter | None = None,
    containers: ContainerLimits | None = DESCEND_INTO_CONTAINERS,
    archives: ArchiveOpener | None = None,
) -> Perception:
    """A `Perception` over `root`, with everything else defaulted.

    `capabilities` given explicitly is used verbatim and nothing is probed —
    a test must be able to declare any capability set without depending on what
    happens to be installed on the machine running it.

    When a `vision` model is also given and `capabilities` declares
    `Capability.VISION`, the two must agree on the model's revision. They are
    two independent inputs otherwise, and disagreement would mean the artifact
    cache key does not reliably identify which model produced a cached
    description — silently correcting one to match the other would hide that
    the caller declared something false, so this raises instead.

    `observer` defaults to `None`: a caller who wants no narration pays
    nothing and gets exactly today's code paths.

    `limiter` does NOT default to nothing. `VideoHandler` fans out across
    every sampled moment concurrently, so an unbounded default would launch
    one ffmpeg subprocess per moment at once and, on a busy machine, report
    its own over-subscription back to the caller as "(no frame could be
    decoded at this moment)" — a claim about their file that nothing
    established. So when no limiter is given, this constructs a
    `SemaphoreLimiter()` with `DEFAULT_LIMITS`: conservative, per-capability,
    and overridable. A caller passing an explicit limiter keeps theirs
    untouched, and a caller who genuinely wants no bound at all passes
    `SemaphoreLimiter({})` — an unconfigured capability is unbounded, so an
    empty configuration opts out of bounding entirely.

    `Perception` itself does not fan out across files; the caller writes that
    loop (over `list()`), so it already controls whatever concurrency it wants
    and bounding that here too would only fight it. The bound this default
    supplies is strictly within a single file's read.

    `containers` controls descent into archives. It defaults to
    `ContainerLimits()` -- descent ON, because a library whose promise is
    "read everything" should read the tarball. Passing `None` disables it and
    yields today's behavior exactly, including no extra opens during `walk`.
    `archives` overrides the bundled zip and tar openers, which is the
    extension point for `.7z` or `.rar` without this repository growing a
    dependency on either.
    """
    source, openers = _source_and_openers(root, containers, archives)
    limiter = SemaphoreLimiter() if limiter is None else limiter
    handlers: list[MediaHandler] = [
        TextHandler(source=source, observer=observer),
        *_optional_image_handler(source, vision, observer),
        *_optional_pdf_handler(source, vision, observer),
        *_optional_office_handlers(source, vision, observer),
        *_video_handler(source, vision, transcriber, captions, watcher, observer, limiter),
        *_audio_handler(source, transcriber, observer),
        ArchiveHandler(source=source, archives=openers, observer=observer),
        # The fallback claims "*", so it must be last: the registry breaks a
        # rank tie by registration order, and a fallback registered first would
        # shadow nothing but would rank ahead of an equally-specific match.
        BinaryHandler(source=source, observer=observer),
    ]
    if capabilities is None:
        probes: list[CapabilityProbe] = [ModelProbe(vision=vision, transcriber=transcriber)]
        if probe_binaries:
            probes.append(BinaryProbe())
        # `BinaryProbe` checks every OS binary it knows about
        # (FFMPEG/EXIFTOOL/LIBREOFFICE/TESSERACT); `VideoHandler` is now the
        # first bundled handler to reference one (FFMPEG), so restricting the
        # probe to what is actually used keeps the others a no-op here.
        capabilities = await discover(
            probes=probes, capabilities=_capabilities_handlers_can_use(handlers)
        )
    elif vision is not None and Capability.VISION in capabilities.revisions:
        declared = capabilities.revisions[Capability.VISION]
        if declared != vision.model_id:
            raise DomainError(
                f"capabilities declares Capability.VISION revision {declared!r}, "
                f"but the injected vision model reports model_id {vision.model_id!r}; "
                "these must agree or the artifact cache key would misdescribe "
                "which model produced a cached description"
            )
    return Perception(
        source=source,
        detector=PuremagicDetector(),
        hasher=ContentHasher(source=source, memo=StatMemo()),
        registry=MimeTypeRegistry(handlers=handlers, capabilities=capabilities),
        artifacts=InMemoryArtifactStore() if artifacts is None else artifacts,
        memo=ResolutionMemo(),
    )
