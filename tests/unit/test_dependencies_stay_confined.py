"""Each third-party client lives in exactly one place.

import-linter cannot see third-party imports, so it cannot enforce this — and
this is the rule that actually stops a langchain or ffmpeg leak into the
domain. The table is the spec's confinement table, and it must fail when a
module drifts out of its home.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "readeverything"

#: top-level third-party module -> the only files that may import it
CONFINED: dict[str, set[str]] = {
    "langchain_core": {"agent/tools.py"},
    "puremagic": {"adapters/detection.py"},
    "charset_normalizer": {"handlers/text.py"},
    "subprocess": set(),
}


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


def test_the_confinement_table_is_live() -> None:
    """An entry naming a file that no longer imports it is stale and must fail."""
    stale: list[str] = []
    for root, homes in CONFINED.items():
        for home in homes:
            path = SRC / home
            if not path.exists() or root not in _imported_roots(ast.parse(path.read_text())):
                stale.append(f"{home} no longer imports {root}")
    assert not stale, f"stale confinement entries: {stale}"
