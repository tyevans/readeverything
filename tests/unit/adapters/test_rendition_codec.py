"""Round-trip fidelity, which is the only property that makes caching safe.

A cache that returns something *almost* right is worse than no cache: the
caller cannot tell. `CharSpan` and `ByteRange` have identical field shapes, so
a codec that resolves the union by shape returns byte offsets labelled as
character offsets, and nothing raises.
"""

import pytest

from readeverything.adapters.rendition_codec import decode_rendition, encode_rendition
from readeverything.domain.errors import DomainError
from readeverything.domain.locators import BBox, ByteRange, CharSpan, Locator, PageRef, TimeSpan
from readeverything.domain.rendition import (
    ImageContent,
    Rendition,
    RenditionContent,
    StructuredContent,
    TextContent,
)

LOCATORS = [
    CharSpan(0, 5),
    ByteRange(0, 5),
    TimeSpan(0.0, 1.5),
    PageRef(3),
    BBox(page=None, x=0.0, y=0.0, w=1.0, h=1.0),
]

CONTENTS = [
    TextContent(text="hello"),
    TextContent(text=""),
    StructuredContent(rows=({"a": 1, "b": None, "c": "x"},)),
    ImageContent(data=bytes(range(256)), mime="image/png"),
]


@pytest.mark.parametrize("locator", LOCATORS)
@pytest.mark.parametrize("content", CONTENTS)
@pytest.mark.parametrize("degraded", [True, False])
def test_every_rendition_shape_round_trips_exactly(
    locator: Locator, content: RenditionContent, degraded: bool
) -> None:
    """Equality is not enough — the concrete TYPE must survive too.

    `CharSpan(0, 5) == ByteRange(0, 5)` is False for these dataclasses, so
    equality does catch this one. The explicit type assertion stays anyway: it
    names the property being protected, so a future change to __eq__ cannot
    quietly remove the protection.
    """
    original = Rendition(locator=locator, content=content, degraded=degraded)
    restored = decode_rendition(encode_rendition(original))
    assert restored == original
    assert type(restored.locator) is type(locator)
    assert type(restored.content) is type(content)


def test_a_byte_range_never_comes_back_as_a_char_span() -> None:
    """The specific corruption this codec exists to prevent, named.

    Measured against `TypeAdapter(Rendition)` before this codec was written:
    ByteRange went in, CharSpan came out, nothing raised.
    """
    restored = decode_rendition(
        encode_rendition(Rendition(locator=ByteRange(0, 5), content=TextContent(text="x")))
    )
    assert type(restored.locator) is ByteRange


def test_arbitrary_image_bytes_survive() -> None:
    """Every byte value, including the ones that are not valid utf-8."""
    data = bytes(range(256))
    restored = decode_rendition(
        encode_rendition(
            Rendition(
                locator=BBox(page=None, x=0.0, y=0.0, w=1.0, h=1.0),
                content=ImageContent(data=data, mime="image/png"),
            )
        )
    )
    assert isinstance(restored.content, ImageContent)
    assert restored.content.data == data


def test_an_unknown_tag_is_refused_rather_than_guessed() -> None:
    """A cache entry written by a newer version must not be decoded by guess.

    Refusing is a miss, which costs a recomputation. Guessing costs a wrong
    answer.
    """
    with pytest.raises(DomainError):
        decode_rendition(b'{"locator":{"__type__":"Nonesuch"},"content":{},"degraded":false}')
