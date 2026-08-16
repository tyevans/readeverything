import pytest

from readeverything.domain.locator_map import LocatorMap, LocatorSegment
from readeverything.domain.locators import CharSpan, PageRef, TimeSpan
from readeverything.domain.rendition import (
    Budget,
    CueSource,
    Degradation,
    Rendered,
    Rendition,
    TextContent,
    TranscriptCue,
)


def test_a_rendition_carries_its_locator() -> None:
    rendition = Rendition(locator=TimeSpan(1.0, 2.0), content=TextContent("hello"))
    assert rendition.locator == TimeSpan(1.0, 2.0)
    assert not rendition.degraded


def test_a_transcript_cue_may_have_no_speaker() -> None:
    """Diarization is capability-gated; every cue works without it."""
    cue = TranscriptCue(span=TimeSpan(0.0, 1.0), text="hi", speaker=None, confidence=None)
    assert cue.speaker is None


def test_rendered_requires_barriers_to_lie_within_the_text() -> None:
    locator_map = LocatorMap.build((LocatorSegment(CharSpan(0, 5), TimeSpan(0.0, 1.0)),))
    with pytest.raises(ValueError, match="barrier"):
        Rendered(
            text="hello",
            locator_map=locator_map,
            barriers=(99,),
            degradations=(),
        )


def test_rendered_requires_the_map_to_cover_the_text() -> None:
    locator_map = LocatorMap.build((LocatorSegment(CharSpan(0, 3), TimeSpan(0.0, 1.0)),))
    with pytest.raises(ValueError, match="must cover the text"):
        Rendered(text="hello", locator_map=locator_map, barriers=(), degradations=())


def test_a_budget_of_none_means_unbounded() -> None:
    assert Budget(max_chars=None).permits(10_000_000)
    assert not Budget(max_chars=100).permits(101)


def test_a_degradation_says_what_was_dropped() -> None:
    d = Degradation(what="frame_sampling", detail="reduced to 1 frame per 30s")
    assert "30s" in d.detail


def test_a_cue_is_said_unless_it_says_otherwise() -> None:
    """Every producer that predates this field is a transcriber. A default of
    CAPTIONED would mislabel all of them, which is the exact failure the field
    exists to prevent."""
    cue = TranscriptCue(span=TimeSpan(0.0, 1.0), text="hi", speaker=None, confidence=None)
    assert cue.source is CueSource.SAID


def test_a_caption_can_say_it_was_written() -> None:
    cue = TranscriptCue(
        span=TimeSpan(0.0, 1.0),
        text="[music playing]",
        speaker=None,
        confidence=None,
        source=CueSource.CAPTIONED,
    )
    assert cue.source is CueSource.CAPTIONED


# --- provenance on a single answer ----------------------------------------


def test_a_rendition_carries_no_degradations_by_default() -> None:
    """Every producer that predates the field reports nothing, which is what
    they meant. A default of anything else would put words in their mouth."""
    assert Rendition(locator=PageRef(1), content=TextContent("x")).degradations == ()


def test_a_rendition_can_say_what_is_wrong_with_it_rather_than_only_that_it_is() -> None:
    """`degraded` is a bit; a converted page image needs a sentence.

    A LibreOffice rendering of a PowerPoint is a rendering, not the thing
    itself — fonts substitute — and that is a fact about an answer that is
    otherwise perfectly good. It is not `degraded=True`, which would tell an
    agent to distrust the image; it is provenance, and it needs somewhere to
    go that is not the bit.
    """
    rendition = Rendition(
        locator=PageRef(4),
        content=TextContent("x"),
        degradations=(Degradation(what="converted", detail="fonts may have been substituted"),),
    )
    assert not rendition.degraded
    assert rendition.degradations[0].what == "converted"
