"""Each third-party client lives in exactly one place.

import-linter cannot see third-party imports, so it cannot enforce this — and
this is the rule that actually stops a langchain or ffmpeg leak into the
domain. The table is the spec's confinement table, and it must fail when a
module drifts out of its home.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "readeverything"
TESTS = ROOT / "tests"

#: top-level third-party module -> the only files that may import it
CONFINED: dict[str, set[str]] = {
    # The clip adapter is a second home for both: it speaks llama.cpp's
    # `input_video` dialect, which the vision adapter has no reason to know
    # about, and it imports vision_langchain's flattening rather than copying
    # it.
    "langchain_core": {
        "agent/tools.py",
        "adapters/vision_langchain.py",
        "adapters/clip_langchain.py",
    },
    "langchain_openai": {"adapters/vision_langchain.py", "adapters/clip_langchain.py"},
    "puremagic": {"adapters/detection.py"},
    "charset_normalizer": {"handlers/text.py"},
    # pdf.py's PIL import is TYPE_CHECKING-only, annotating `_render_pil`'s
    # return type: `pdfium`'s own `to_pil()` is what touches Pillow at
    # runtime, and `_PIL_AVAILABLE` (checked by name) is the real guard.
    # video.py's is a real, but lazy, runtime import confined to
    # `_ask_about_frame`'s region-cropping branch: `frame_at` hands back raw
    # PNG bytes rather than a decoded image, so cropping needs an actual
    # decode, guarded by the same `_PIL_AVAILABLE` name check.
    "PIL": {"handlers/image.py", "handlers/regions.py", "handlers/pdf.py", "handlers/video.py"},
    "faster_whisper": {"adapters/whisper_transcriber.py"},
    # The remote transcriber is the only place that speaks HTTP directly:
    # every other network-touching adapter goes through langchain's client.
    "httpx": {"adapters/remote_whisper_transcriber.py"},
    # pypdfium2 wraps Google's PDFium. Two homes: the probe adapter answers
    # cheap document facts, and the PDF handler extracts text — which is not a
    # probe's job, and no handler imports an adapter.
    "pypdfium2": {"adapters/pdfium_probe.py", "handlers/pdf.py"},
    # The three OOXML readers, each confined to the one handler that speaks its
    # document model. There is deliberately no shared "office" module importing
    # all three: a caller who installed the extra for spreadsheets should not
    # have a Word parser loaded, and one module would make that impossible.
    "docx": {"handlers/office_word.py"},
    # ODF has no maintained reader — odfpy is unmaintained — so `adapters/odf.py`
    # walks the flat XML parts itself. lxml is that walk and nothing else in the
    # library touches it.
    "lxml": {"adapters/odf.py"},
    "subprocess": set(),
    # asyncio's subprocess API is how binary_probe.py spawns external
    # executables; the other three files use asyncio only for async I/O, but
    # every current importer must be listed for this table to stay live.
    "asyncio": {
        "adapters/artifact_store.py",
        "adapters/hashing.py",
        "adapters/local_source.py",
        "adapters/binary_probe.py",
        "adapters/pdfium_probe.py",
        # ffprobe is spawned via create_subprocess_exec with an argv vector,
        # same pattern as binary_probe.py.
        "adapters/ffprobe_streams.py",
        # ffmpeg, same pattern, extracting a single frame.
        "adapters/ffmpeg_frames.py",
        # ffmpeg, same pattern, extracting the audio track.
        "adapters/ffmpeg_audio.py",
        # ffmpeg, same pattern, converting a caption track to SRT on stdout.
        "adapters/ffmpeg_captions.py",
        # ffmpeg, same pattern, cutting a bounded range to stdout.
        "adapters/ffmpeg_clip.py",
        # whisper's transcribe() is synchronous and CPU-bound; run in a
        # thread so it doesn't block the event loop.
        "adapters/whisper_transcriber.py",
        # asyncio.Semaphore bounds per-capability concurrency.
        "adapters/semaphore_limiter.py",
        # `asyncio.gather` fetches a video's sampled moments concurrently. No
        # subprocess is spawned here and no adapter is imported: the handler
        # awaits the injected extractor and vision ports, and gathering them is
        # a property of the work (independent, slow, per-moment) rather than of
        # any particular runtime.
        "handlers/video.py",
    },
    # shutil.which locates the executable a capability probe is about to run;
    # confined to the one adapter that probes binaries.
    "shutil": {"adapters/binary_probe.py"},
}

#: `deepagents` is exercised only by the integration test proving the README's
#: composition actually constructs. It is optional (the `agents` extra) and
#: must never leak into `src/` — CONFINED above enforces that already, since
#: `deepagents` names no home there. This is the mirror rule for `tests/`:
#: exactly one test file may import it, so a stray import elsewhere (which
#: would make the whole suite depend on the extra) fails loudly here instead.
DEEPAGENTS_CONFINED_TEST_FILE = "integration/test_deepagents_composition.py"

#: `reportlab` generates PDF fixtures at test time so no binary is committed
#: (see tests/fixtures_pdf.py). It is a dev-only dependency: nothing under
#: `src/` may import it, and within `tests/` it is confined to the one module
#: that builds fixtures, so every other test file imports the fixtures
#: functions rather than reportlab itself.
REPORTLAB_CONFINED_TEST_FILE = "fixtures_pdf.py"

#: The three OOXML writers generate office fixtures at test time so no binary is
#: committed (see tests/fixtures_office.py). Within `tests/` they are confined
#: to the fixture module and to the fixture module's own guard test — which must
#: read a generated document back with the same library that wrote it, or it
#: would be guarding nothing. Every other test imports the fixture functions.
OFFICE_WRITERS = frozenset({"docx", "pptx", "openpyxl"})
OFFICE_WRITER_TEST_FILES = frozenset({"fixtures_office.py", "unit/test_office_fixtures.py"})


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_each_dependency_is_confined_to_its_declared_home() -> None:
    violations: list[str] = []
    for path in SRC.rglob("*.py"):
        relative = str(path.relative_to(SRC))
        for root in _imported_roots(ast.parse(path.read_text())):
            if root in CONFINED and relative not in CONFINED[root]:
                violations.append(f"{relative} imports {root}")
    assert not violations, f"confinement violated: {violations}"


def test_deepagents_is_confined_to_the_one_composition_test() -> None:
    violations: list[str] = []
    for path in TESTS.rglob("*.py"):
        relative = str(path.relative_to(TESTS))
        if "deepagents" in _imported_roots(ast.parse(path.read_text())) and (
            relative != DEEPAGENTS_CONFINED_TEST_FILE
        ):
            violations.append(relative)
    assert not violations, f"deepagents imported outside its confined test: {violations}"


def test_reportlab_is_confined_to_the_fixture_module() -> None:
    violations: list[str] = []
    for path in TESTS.rglob("*.py"):
        relative = str(path.relative_to(TESTS))
        if "reportlab" in _imported_roots(ast.parse(path.read_text())) and (
            relative != REPORTLAB_CONFINED_TEST_FILE
        ):
            violations.append(relative)
    assert not violations, f"reportlab imported outside its confined fixture module: {violations}"


def test_the_office_writers_are_confined_to_the_fixture_module() -> None:
    violations: list[str] = []
    for path in TESTS.rglob("*.py"):
        relative = str(path.relative_to(TESTS))
        roots = _imported_roots(ast.parse(path.read_text()))
        if roots & OFFICE_WRITERS and relative not in OFFICE_WRITER_TEST_FILES:
            violations.append(relative)
    assert not violations, f"office writers imported outside the fixture module: {violations}"


def test_the_confinement_table_is_live() -> None:
    """An entry naming a file that no longer imports it is stale and must fail."""
    stale: list[str] = []
    for root, homes in CONFINED.items():
        for home in homes:
            path = SRC / home
            if not path.exists() or root not in _imported_roots(ast.parse(path.read_text())):
                stale.append(f"{home} no longer imports {root}")
    assert not stale, f"stale confinement entries: {stale}"
