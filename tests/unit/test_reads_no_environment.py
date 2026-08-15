"""The library reads no environment.

Configuration is constructor arguments, so a caller can run two differently
configured instances in one process and so tests cannot be affected by the
machine they run on. A single `os.getenv` would silently break both.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "readeverything"

FORBIDDEN_ATTRS = {("os", "environ"), ("os", "getenv")}
FORBIDDEN_MODULES = {"dotenv"}


def _offences(tree: ast.AST) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and (node.value.id, node.attr) in FORBIDDEN_ATTRS
        ):
            found.append(f"{node.value.id}.{node.attr}")
        if isinstance(node, ast.Import):
            found.extend(a.name for a in node.names if a.name.split(".")[0] in FORBIDDEN_MODULES)
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] in FORBIDDEN_MODULES
        ):
            found.append(node.module)
    return found


def test_no_module_reads_the_environment() -> None:
    offenders: dict[str, list[str]] = {}
    for path in SRC.rglob("*.py"):
        found = _offences(ast.parse(path.read_text()))
        if found:
            offenders[str(path.relative_to(SRC))] = found
    assert not offenders, f"environment reads found: {offenders}"
