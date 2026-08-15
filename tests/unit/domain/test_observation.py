import asyncio

import pytest

from readeverything.domain.identity import ContentHash, MimeType, SourceRef
from readeverything.domain.observation import (
    Event,
    OperationFinished,
    OperationProgressed,
    OperationStarted,
)
from readeverything.ports.observation import emit
from readeverything.testing.fakes import RaisingObserver


def _ref(*, uri: str = "a.mp4") -> SourceRef:
    return SourceRef(
        uri=uri,
        mime=MimeType.parse("video/mp4"),
        content_hash=ContentHash("a" * 64),
        size_bytes=100,
    )


def test_progress_reports_what_is_done_and_what_is_known() -> None:
    """`total` is `int | None` because it is not always knowable.

    A video knows how many moments it will sample before it starts. A
    transcription does not know how many cues it will produce until it has
    produced them. Reporting a made-up total would be a number nothing
    measured — the field admits ignorance instead.
    """
    known = OperationProgressed(operation="represent", ref=_ref(), done=3, total=40)
    unknown = OperationProgressed(operation="represent", ref=_ref(), done=3, total=None)
    assert known.total == 40
    assert unknown.total is None


def test_a_progressed_event_rejects_a_negative_count() -> None:
    with pytest.raises(ValueError):
        OperationProgressed(operation="represent", ref=_ref(), done=-1, total=None)


def test_done_may_not_exceed_a_known_total() -> None:
    """ "7 of 5 complete" is a claim about work nobody scheduled."""
    with pytest.raises(ValueError):
        OperationProgressed(operation="represent", ref=_ref(), done=7, total=5)


def test_elapsed_may_not_be_negative() -> None:
    with pytest.raises(ValueError):
        OperationFinished(operation="represent", ref=_ref(), elapsed_s=-0.1)


def test_emit_contains_an_observer_that_raises() -> None:
    """THE RULE THAT MATTERS MOST. A read must not fail because the progress
    reporting failed. The library has promised since its first spec that a
    handler never raises about its input; an injected callback is now part of
    that surface.
    """
    emit(RaisingObserver(), OperationStarted(operation="represent", ref=_ref()))
    # reaching here is the assertion


def test_emit_with_no_observer_does_nothing() -> None:
    emit(None, OperationStarted(operation="represent", ref=_ref()))


class _InterruptingObserver:
    """Raises `KeyboardInterrupt` — a stop signal, not an observer failure."""

    def observe(self, event: Event) -> None:
        raise KeyboardInterrupt


class _CancelledObserver:
    """Raises `asyncio.CancelledError`, which is a `BaseException` in 3.8+."""

    def observe(self, event: Event) -> None:
        raise asyncio.CancelledError


@pytest.mark.parametrize(
    ("observer", "escaping"),
    [
        (_InterruptingObserver(), KeyboardInterrupt),
        (_CancelledObserver(), asyncio.CancelledError),
    ],
)
def test_emit_does_not_contain_a_stop_signal(observer: object, escaping: type[BaseException]) -> None:
    """The other half of the rule, and the half nothing was holding.

    `emit` suppresses `Exception`, not `BaseException`, and until this test
    existed that distinction was asserted by nobody: widening the suppression
    to `BaseException` left the entire suite green. `KeyboardInterrupt` and
    `asyncio.CancelledError` are not the observer failing, they are a request
    to stop — swallowing one would mean a Ctrl-C that lands during a progress
    callback is discarded, or a cancelled read continues because cancellation
    arrived at an unlucky instant.
    """
    with pytest.raises(escaping):
        emit(observer, OperationStarted(operation="represent", ref=_ref()))  # type: ignore[arg-type]


def test_emit_still_contains_an_ordinary_exception_from_the_same_observer_shape() -> None:
    """Paired with the test above so the pair pins a boundary rather than a side.

    A `suppress(BaseException)` mutation passes this one and fails that one; a
    `suppress(())` mutation does the reverse. Neither test alone can tell you
    where the line is.
    """
    emit(RaisingObserver(), OperationStarted(operation="represent", ref=_ref()))
    # reaching here is the assertion
