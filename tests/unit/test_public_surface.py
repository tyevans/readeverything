"""The public API is closed and loads no driver."""

import subprocess
import sys

import readeverything


def test_everything_in_all_is_reachable() -> None:
    for name in readeverything.__all__:
        assert getattr(readeverything, name) is not None


def test_all_is_sorted_and_unique() -> None:
    assert readeverything.__all__ == sorted(set(readeverything.__all__))


def test_importing_the_package_loads_no_optional_driver() -> None:
    """`import readeverything` must not pull in langchain."""
    code = "import readeverything, sys; print('langchain_core' in sys.modules)"
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert output.stdout.strip() == "False"
