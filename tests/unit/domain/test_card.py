from readeverything.domain.card import Card, Segment
from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
from readeverything.domain.locators import CharSpan


def _ref() -> SourceRef:
    return SourceRef(
        uri="/a.txt",
        mime=MimeType.parse("text/plain"),
        content_hash=ContentHash("deadbeef"),
        size_bytes=12,
    )


def test_a_card_exposes_affordance_names() -> None:
    card = Card(
        ref=_ref(),
        kind=MediaKind.TEXT,
        facts={"lines": 3},
        outline=(Segment(CharSpan(0, 12), "whole file"),),
        excerpt="hello world",
        affordances=(),
    )
    assert card.affordance_names() == ()
    assert card.facts["lines"] == 3
