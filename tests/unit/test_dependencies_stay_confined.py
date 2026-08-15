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
    "langchain_core": {"agent/tools.py", "adapters/vision_langchain.py"},
    "langchain_openai": {"adapters/vision_langchain.py"},
    "puremagic": {"adapters/detection.py"},
    "charset_normalizer": {"handlers/text.py"},
    "PIL": {"handlers/image.py"},
    # pypdfium2 wraps Google's PDFium; confined to the one adapter module so
    # the pdfium import lives in exactly one place, per task-2's brief.
    "pypdfium2": {"adapters/pdfium_probe.py"},
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


def test_the_confinement_table_is_live() -> None:
    """An entry naming a file that no longer imports it is stale and must fail."""
    stale: list[str] = []
    for root, homes in CONFINED.items():
        for home in homes:
            path = SRC / home
            if not path.exists() or root not in _imported_roots(ast.parse(path.read_text())):
                stale.append(f"{home} no longer imports {root}")
    assert not stale, f"stale confinement entries: {stale}"
