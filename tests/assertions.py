"""Narrowing helpers shared by tests.

`Rendition.content` is a union, so `rendition.content.text` does not type-check
even when the affordance under test can only answer with text. Reaching for
`# type: ignore` there would silence the one check that notices when a handler
starts answering with something else -- an image or a table -- which is a real
way for these tests to become vacuous. `text_of` narrows instead: it asserts
the kind and hands back the string, so a handler that changes shape fails on
the assertion rather than on an attribute error thirty lines later.
"""

from __future__ import annotations

from readeverything.domain.rendition import Rendition, TextContent


def text_of(rendition: Rendition) -> str:
    """The rendition's text, or an assertion failure naming what it was instead."""
    content = rendition.content
    assert isinstance(content, TextContent), f"expected TextContent, got {type(content).__name__}"
    return content.text
