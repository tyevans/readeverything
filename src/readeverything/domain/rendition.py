"""The results of doing work: one operation's output, and a whole-source flattening.

`Rendition` answers an affordance. `Rendered` is the indexing feed — flat text
plus the locator map plus hard chunk barriers — and is the contract Plan 2's
query interface consumes.

`Rendered` validates that its map covers its text. A map that stops short would
produce a hit that cannot be cited, discovered at citation time rather than
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NewType

from readeverything.domain.locator_map import LocatorMap
from readeverything.domain.locators import Locator, TimeSpan

SpeakerId = NewType("SpeakerId", str)


@dataclass(frozen=True, slots=True)
class TextContent:
    text: str


@dataclass(frozen=True, slots=True)
class ImageContent:
    """Raw image bytes plus their mimetype, ready for a multimodal content block."""

    data: bytes
    mime: str


@dataclass(frozen=True, slots=True)
class StructuredContent:
    rows: tuple[dict[str, str | int | float | None], ...]


type RenditionContent = TextContent | ImageContent | StructuredContent


@dataclass(frozen=True, slots=True)
class Degradation:
    """Something a handler chose not to produce, or could not produce faithfully.

    Reported rather than silent. Silent truncation is invisible in exactly the
    case where the answer is wrong.
    """

    what: str
    detail: str


@dataclass(frozen=True, slots=True)
class Rendition:
    """One affordance's answer, always located."""

    locator: Locator
    content: RenditionContent
    degraded: bool = False
    #: What is true about this answer that the content itself does not say.
    #:
    #: Distinct from `degraded`, and the distinction is the point. `degraded`
    #: is "do not trust this as the thing you asked for". A page image produced
    #: by converting a PowerPoint through LibreOffice is not that: it is a
    #: perfectly usable image, and it is also a *rendering* rather than the
    #: document itself — fonts substitute, layout engines differ. Setting
    #: `degraded` for it would tell an agent to discount a good answer;
    #: saying nothing would let a rendering pass as the original. This is
    #: where that third thing goes.
    #:
    #: Empty by default, because every producer that predates this field
    #: reported nothing and meant it.
    degradations: tuple[Degradation, ...] = ()


class CueSource(Enum):
    """Where a cue's words came from.

    `video.py` already refuses to let a citation attribute speech to a
    picture — `SPEECH_MARKER` exists for exactly that. Captions are a third
    kind of evidence and need the same care, because a caption is authored
    text: frequently condensed for reading speed rather than verbatim, often
    describing sound nobody spoke (`[music playing]` is the first cue in the
    reference file), and sometimes translated. A reader deciding how far to
    trust a quoted line needs to know which of these produced it.
    """

    SAID = "said"
    CAPTIONED = "captioned"


@dataclass(frozen=True, slots=True)
class TranscriptCue:
    """One utterance, with a speaker when diarization is available."""

    span: TimeSpan
    text: str
    speaker: SpeakerId | None
    confidence: float | None
    #: Defaults to SAID because every producer that predates this field is a
    #: transcriber, and only the caption adapter sets CAPTIONED. A default of
    #: CAPTIONED would mislabel every existing cue, which is the failure this
    #: field exists to prevent.
    source: CueSource = CueSource.SAID


@dataclass(frozen=True, slots=True)
class Rendered:
    """A whole source flattened for indexing."""

    text: str
    locator_map: LocatorMap
    barriers: tuple[int, ...]
    degradations: tuple[Degradation, ...]

    def __post_init__(self) -> None:
        if self.locator_map.length != len(self.text):
            raise ValueError(
                f"the locator map must cover the text exactly: map covers "
                f"{self.locator_map.length}, text is {len(self.text)}"
            )
        for barrier in self.barriers:
            if not 0 <= barrier <= len(self.text):
                raise ValueError(
                    f"barrier {barrier} is outside the text of length {len(self.text)}"
                )
        if list(self.barriers) != sorted(set(self.barriers)):
            raise ValueError("barriers must be sorted and unique")


@dataclass(frozen=True, slots=True)
class Budget:
    """How much a caller is willing to spend on a representation.

    Passed *into* `represent`, not enforced around it: a handler degrades on its
    own terms, because only it knows that dropping frame density costs less than
    dropping transcript.
    """

    max_chars: int | None

    def permits(self, chars: int) -> bool:
        return self.max_chars is None or chars <= self.max_chars
