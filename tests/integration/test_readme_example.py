"""The README's example, extracted from README.md and executed.

Not a transcription: this test reads the file, takes the one python fence
marked `<!-- readeverything:tested -->`, compiles it with top-level `await`
allowed, and runs it against a `tmp_path` the test injects as `root`. An
example that stops working — or an import that stops resolving from the front
door — fails here rather than in a reader's terminal.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration

README = Path(__file__).resolve().parents[2] / "README.md"

#: The marker a fence must carry to be the executed example. A marker rather
#: than "the first python block" because the README has several blocks, and
#: silently picking one of them is how an example rots while a test stays
#: green.
MARKER = "<!-- readeverything:tested -->"

_MARKED_BLOCK = re.compile(
    re.escape(MARKER) + r".*?```python\n(?P<code>.*?)```",
    re.DOTALL,
)


def _example() -> str:
    """The one marked python fence, or a failure that names the constraint."""
    text = README.read_text(encoding="utf-8")
    markers = text.count(MARKER)
    if markers != 1:
        raise AssertionError(
            f"exactly one python block in {README.name} may carry {MARKER!r} — "
            f"the test executes that block and nothing else; found {markers}"
        )
    match = _MARKED_BLOCK.search(text)
    if match is None:
        raise AssertionError(
            f"{MARKER!r} in {README.name} is not followed by a ```python fence; "
            "the marker must sit immediately above the block to execute"
        )
    return match.group("code")


async def test_the_readme_example_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """This test is the reason the example in the README can be trusted: it is
    the example, compiled and run, not a copy of it.
    """
    (tmp_path / "notes.txt").write_text("the quick brown fox")
    namespace: dict[str, Any] = {"root": tmp_path, "__name__": "readme_example"}

    code = compile(_example(), str(README), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    # `eval` of a top-level-await compilation returns the coroutine to await;
    # the source is this repo's README, not anything a caller supplies.
    await eval(code, namespace)

    assert namespace["card"].kind == "text"
    assert len(namespace["tools"]) == 4
    assert "notes.txt" in capsys.readouterr().out, (
        "the example's observer is what makes it an example about watching a read"
    )
