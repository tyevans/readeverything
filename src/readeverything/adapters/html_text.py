"""HTML prose, and the offset in the source each run of it came from.

Stripping tags is the easy half. The half that matters is that every block
remembers WHERE it was, because a `LocatorSegment` pairs a span of flattened
text with a locator into the artifact, and an HTML handler that forgot the
second half would hand back citations no one could check against the file.

So this is not a "strip tags" helper: it is an offset-preserving reader. The
text is what you read; `HtmlBlock.span` is what you cite.

Built on stdlib `html.parser` rather than BeautifulSoup or lxml. Two reasons,
and only the second is about dependencies. `HTMLParser.getpos()` reports the
source position of every event, which is exactly the fact this module exists to
preserve — the tree libraries throw it away, so using one would mean giving up
the offsets and then reconstructing them by searching for the text, which is
ambiguous the moment a page repeats a sentence. That it also costs no
dependency is a bonus, not the argument.

No I/O and no environment here, matching `odf.py` and `ooxml.py`: a string in,
data out.
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser

from readeverything.domain.locators import CharSpan

#: Tags whose content is code, not prose. Their text is dropped entirely — a
#: page's stylesheet is not something an agent asked to read.
_SKIPPED = frozenset({"script", "style"})

#: Tags that end the previous run of prose and begin a new one. Inline tags
#: (`em`, `a`, `span`) are deliberately absent: they mark up a sentence rather
#: than separating two, and splitting on them would shred every paragraph that
#: contains a link into three unreadable fragments.
_BLOCK = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "caption",
        "dd",
        "div",
        "dt",
        "figcaption",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "nav",
        "p",
        "pre",
        "section",
        "td",
        "th",
        "tr",
    }
)

#: A body block's level. Zero rather than None so `HtmlBlock.level` is one
#: comparable number, matching `OdfBlock.level`.
_BODY_LEVEL = 0

_HEADINGS = {f"h{level}": level for level in range(1, 7)}


@dataclass(frozen=True, slots=True)
class HtmlBlock:
    """One run of prose. `level` is 0 for body text, 1-6 for a heading.

    `span` indexes the ORIGINAL html string, not `text`. The two differ:
    `text` has had its whitespace collapsed and its entities resolved, so it is
    generally shorter. Slicing the source at `span` and collapsing whitespace
    is what reproduces `text` — that round trip is the citation guarantee, and
    it has a property test.
    """

    level: int
    text: str
    span: CharSpan


class _Reader(HTMLParser):
    """Accumulates prose per block, remembering where each block started."""

    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=True)
        self._source = source
        self._line_starts = _line_starts(source)
        self.blocks: list[HtmlBlock] = []
        self.title: str | None = None
        self._chunks: list[str] = []
        self._start: int | None = None
        self._level = _BODY_LEVEL
        # A depth rather than a flag: `<script>` nested inside a skipped
        # region should not reopen prose when the inner one closes.
        self._skipping = 0
        self._in_title = False

    def _offset(self, position: tuple[int, int]) -> int:
        """`getpos()`'s 1-based (line, column) as an absolute character offset."""
        line, column = position
        if line - 1 >= len(self._line_starts):
            return len(self._source)
        return min(self._line_starts[line - 1] + column, len(self._source))

    def _flush(self, end: int) -> None:
        text = " ".join("".join(self._chunks).split())
        if text and self._start is not None:
            # `CharSpan` rejects an empty span, and a block whose end lands at
            # or before its start is possible on malformed input. The text
            # exists, so the span must too.
            span = CharSpan(self._start, max(end, self._start + 1))
            self.blocks.append(HtmlBlock(level=self._level, text=text, span=span))
        self._chunks = []
        self._start = None
        self._level = _BODY_LEVEL

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        here = self._offset(self.getpos())
        if tag in _SKIPPED:
            self._flush(here)
            self._skipping += 1
            return
        if self._skipping:
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in _BLOCK:
            self._flush(here)
            # Start the span AFTER the tag, so the cited source is the
            # content the reader saw rather than the markup around it.
            self._start = here + len(self.get_starttag_text() or "")
            self._level = _HEADINGS.get(tag, _BODY_LEVEL)

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED:
            self._skipping = max(0, self._skipping - 1)
            return
        if self._skipping:
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in _BLOCK:
            self._flush(self._offset(self.getpos()))

    def handle_data(self, data: str) -> None:
        if self._skipping:
            return
        if self._in_title:
            collapsed = " ".join(data.split())
            if collapsed:
                self.title = collapsed
            return
        if self._start is None:
            if not data.strip():
                return
            # Text outside any block tag is still text. Dropping it would
            # silently lose a page whose author never wrote a `<p>`.
            self._start = self._offset(self.getpos())
        self._chunks.append(data)

    def close(self) -> None:
        super().close()
        self._flush(len(self._source))


def _line_starts(source: str) -> tuple[int, ...]:
    """Absolute offset of the first character of each line."""
    starts = [0]
    for index, character in enumerate(source):
        if character == "\n":
            starts.append(index + 1)
    return tuple(starts)


def _read(source: str) -> _Reader:
    reader = _Reader(source)
    reader.feed(source)
    reader.close()
    return reader


def html_blocks(source: str) -> tuple[HtmlBlock, ...]:
    """Every run of prose in `source`, in document order, with its span."""
    return tuple(_read(source).blocks)


def html_title(source: str) -> str | None:
    """The document's `<title>`, or None when it has none."""
    return _read(source).title
