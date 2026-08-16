"""The citation promise for HTML: every block points back at the bytes it came from."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from readeverything.adapters.html_text import html_blocks, html_title

SIMPLE = """<html>
<head><title>A Page</title></head>
<body>
<h1>Chapter One</h1>
<p>It was a dark night.</p>
<p>Then it was not.</p>
</body>
</html>
"""


def test_blocks_are_returned_in_document_order() -> None:
    assert [b.text for b in html_blocks(SIMPLE)] == [
        "Chapter One",
        "It was a dark night.",
        "Then it was not.",
    ]


def test_a_heading_carries_its_level_and_body_text_carries_zero() -> None:
    assert [b.level for b in html_blocks(SIMPLE)] == [1, 0, 0]


@pytest.mark.parametrize("tag", ["h1", "h2", "h3", "h4", "h5", "h6"])
def test_every_heading_level_is_recognised(tag: str) -> None:
    (block,) = html_blocks(f"<body><{tag}>Title</{tag}></body>")
    assert block.level == int(tag[1])


def test_the_span_of_every_block_points_at_the_source_text_it_came_from() -> None:
    """The whole reason this adapter tracks offsets: a citation stays checkable.

    Slicing the ORIGINAL html at a block's span must yield that block's text.
    If this fails, a quote can no longer be verified against the file, which is
    the promise `LocatorSegment.locator` exists to keep.
    """
    for block in html_blocks(SIMPLE):
        source = SIMPLE[block.span.start : block.span.end]
        assert " ".join(source.split()) == block.text


def test_script_and_style_contents_are_not_text() -> None:
    html = "<body><script>var x = 1;</script><style>p{color:red}</style><p>Real.</p></body>"
    assert [b.text for b in html_blocks(html)] == ["Real."]


def test_entities_are_resolved() -> None:
    (block,) = html_blocks("<body><p>Tom &amp; Jerry</p></body>")
    assert block.text == "Tom & Jerry"


def test_whitespace_inside_a_block_is_collapsed() -> None:
    (block,) = html_blocks("<body><p>one\n   two\t\tthree</p></body>")
    assert block.text == "one two three"


def test_a_block_with_nested_markup_reads_as_one_run_of_prose() -> None:
    (block,) = html_blocks("<body><p>A <em>very</em> bold <b>claim</b>.</p></body>")
    assert block.text == "A very bold claim."


def test_text_outside_any_block_tag_is_still_a_block() -> None:
    """Loose text is content. Dropping it would silently lose the page."""
    assert [b.text for b in html_blocks("<body>Bare words.</body>")] == ["Bare words."]


def test_a_block_that_is_only_whitespace_is_dropped() -> None:
    blocks = html_blocks("<body><p>   </p><p>Real.</p></body>")
    assert [b.text for b in blocks] == ["Real."]


def test_the_title_is_read_from_the_head() -> None:
    assert html_title(SIMPLE) == "A Page"


def test_a_page_with_no_title_has_none() -> None:
    assert html_title("<body><p>Hi.</p></body>") is None


def test_a_document_with_no_text_yields_no_blocks() -> None:
    assert html_blocks("<html><head><title>Empty</title></head><body></body></html>") == ()


def test_unclosed_tags_do_not_lose_the_text_after_them() -> None:
    """Real-world HTML is malformed. A strict parser would drop this page."""
    assert [b.text for b in html_blocks("<body><p>One<p>Two</body>")] == ["One", "Two"]


@given(
    st.lists(
        st.text(alphabet=st.characters(blacklist_characters="<>&\r"), min_size=1).filter(
            lambda s: s.strip()
        ),
        min_size=1,
        max_size=8,
    )
)
def test_every_block_span_slices_back_to_its_own_text(paragraphs: list[str]) -> None:
    """The citation property, over arbitrary prose rather than one fixture."""
    html = "<body>" + "".join(f"<p>{p}</p>" for p in paragraphs) + "</body>"
    for block in html_blocks(html):
        assert " ".join(html[block.span.start : block.span.end].split()) == block.text
