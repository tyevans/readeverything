import pytest

from readeverything.domain.identity import ContentHash, MimeType, SourceRef
from readeverything.domain.observation import (
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
