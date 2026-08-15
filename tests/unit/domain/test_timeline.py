from itertools import pairwise

import pytest

from readeverything.domain.timeline import tile


def test_gapless_positive_width_and_zero_start() -> None:
    """The properties the earlier adversarial tests already covered."""
    bounds = tile([0.4, 0.9, 0.9, 5.0], duration_s=5.0, min_width_s=0.001)
    assert bounds[0][0] == 0.0
    for start, end in bounds:
        assert start < end
    for (_, end), (next_start, _) in pairwise(bounds):
        assert end == next_start


@pytest.mark.parametrize(
    ("starts", "duration_s", "min_width_s"),
    [
        pytest.param([0.0, 0.5], 2.0, 10.0, id="min_width_s larger than the duration"),
        pytest.param(
            [4.98, 4.99, 5.0, 5.001, 5.002],
            5.0,
            0.01,
            id="cues packed within min_width_s of the end",
        ),
        pytest.param([1.0] * 50, 5.0, 0.001, id="many duplicate timestamps"),
        pytest.param([0.0, 1.0, 2.5, 4.0], 5.0, 0.001, id="a normal case"),
    ],
)
def test_no_span_ever_claims_time_the_file_does_not_have(
    starts: list[float], duration_s: float, min_width_s: float
) -> None:
    """The property the earlier adversarial tests forgot to assert.

    They checked gapless, positive width and zero-start — all true, all
    satisfied by a timeline that described 10 to 20 seconds of a 2-second file.
    A timeline is OF a file; its last moment cannot end after the file does,
    beyond the one `min_width_s` of unavoidable overshoot `tile` documents.
    """
    bounds = tile(starts, duration_s=duration_s, min_width_s=min_width_s)
    assert len(bounds) == len(starts)
    for _, end in bounds:
        assert end <= duration_s + min_width_s + 1e-6


def test_the_normal_case_actually_reaches_the_duration() -> None:
    """Guards against the assertion above being vacuously true for typical input."""
    bounds = tile([0.0, 1.0, 2.5, 4.0], duration_s=5.0, min_width_s=0.001)
    assert bounds[-1][1] == pytest.approx(5.0)
