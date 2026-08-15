"""Video.

`TimeSpan` has existed in the domain since the first spec with no producer.
This is its first. Where `pdf.py` maps every character to the page it came
from, this maps every character to the moment it describes — the same property
one dimension over, and the reason the locator vocabulary is shared.

The card costs a probe and decodes no frame, exactly as the PDF card costs a
probe and extracts no text. Duration, resolution and stream layout are the
facts that shape an agent's next move; paying a decode to learn them would
defeat progressive disclosure on the one media type where a decode is most
expensive.

`requires()` is `{FFMPEG}` — unlike `image.py`, which registers without a
vision model and merely drops affordances. Without ffmpeg there is no way to
learn anything at all about a container, so the registry drops this handler
entirely and video files fall to the binary fallback.

The probe and the extractor arrive by injection. Nothing here imports an
adapter.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from readeverything.domain.affordance import Affordance
from readeverything.domain.capability import Capability
from readeverything.domain.card import Card, Segment
from readeverything.domain.errors import UnknownAffordanceError
from readeverything.domain.identity import MediaKind, SourceRef
from readeverything.domain.locator_map import LocatorMap, LocatorSegment
from readeverything.domain.locators import ByteRange, CharSpan, TimeSpan
from readeverything.domain.rendition import (
    Budget,
    Degradation,
    Rendered,
    Rendition,
)
from readeverything.ports.frames import FrameExtractor
from readeverything.ports.source import SourceReader
from readeverything.ports.streams import MediaFacts, StreamProbe
from readeverything.ports.vision import VisionModel

#: How often `represent` samples the timeline, in seconds, unless the caller
#: says otherwise.
DEFAULT_SAMPLE_INTERVAL_S = 5.0

#: The width of one frame when `StreamInfo.frame_rate` is `None` — which the
#: ffprobe adapter returns for a zero denominator in `r_frame_rate`, a real
#: case in malformed and image-sequence containers. A frame is a point in time
#: and `TimeSpan` forbids `start >= end`, so a frame's span is
#: `[t, t + frame_duration)`; without a frame rate there is no honest width to
#: use, and 1/25 s is stated here as an assumption rather than derived. It is
#: only ever a floor on a span that the next sample's timestamp normally ends.
FALLBACK_FRAME_DURATION_S = 1.0 / 25.0

#: Every sampled moment's text ends with this, and that moment's
#: `LocatorSegment` INCLUDES it — the same reason `pdf.PAGE_SEPARATOR` exists.
#: `CharSpan` rejects `start >= end`, so a moment whose model returned nothing
#: would otherwise contribute a zero-width span and break the map.
MOMENT_SEPARATOR = "\n"

_FRAME_PROMPT = "Describe what is visible in this video frame, in one or two sentences."


def _timestamp(seconds: float) -> str:
    """`h:mm:ss.s`, the form a person reads off a player's scrubber."""
    hours, rest = divmod(seconds, 3600.0)
    minutes, secs = divmod(rest, 60.0)
    return f"{int(hours):d}:{int(minutes):02d}:{secs:04.1f}"


def _sample_times(duration_s: float, interval_s: float) -> tuple[float, ...]:
    """Sample timestamps: zero first, then every `interval_s` inside the file.

    Always at least one, because a video with a duration has a first moment.
    """
    times = [0.0]
    t = interval_s
    while t < duration_s:
        times.append(t)
        t += interval_s
    return tuple(times)


class VideoHandler:
    """Reads a video's timeline, and maps every character to the moment it describes."""

    mime_patterns: ClassVar[tuple[str, ...]] = ("kind:video",)
    priority: ClassVar[int] = 0
    handler_id: ClassVar[str] = "video"
    handler_version: ClassVar[int] = 1

    def __init__(
        self,
        *,
        source: SourceReader,
        probe: StreamProbe,
        frames: FrameExtractor,
        vision: VisionModel | None = None,
        sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    ) -> None:
        if sample_interval_s <= 0:
            raise ValueError(f"sample_interval_s must be positive, got {sample_interval_s}")
        self._source = source
        self._probe = probe
        self._frames = frames
        self._vision = vision
        self._interval_s = sample_interval_s

    def requires(self) -> frozenset[Capability]:
        return frozenset({Capability.FFMPEG})

    def affordances(self) -> tuple[Affordance, ...]:
        """None yet — `frame_at` and `describe_frame` are the next task's work."""
        return ()

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition:
        raise UnknownAffordanceError(name, (a.name for a in self.affordances()))

    async def _facts(self, ref: SourceRef) -> tuple[MediaFacts | None, str | None]:
        """The probe's answer and the path it was read from, or `(None, path)`.

        A missing file, a file that is not media and a probe that raises all
        land here as `None`. This handler never raises about its input, so
        every one of them is a degradation rather than an exception.
        """
        try:
            path = await self._source.local_path(ref.uri)
        except Exception:
            return None, None
        try:
            return await self._probe.probe(path), path
        except Exception:
            return None, path

    async def describe(self, ref: SourceRef) -> Card:
        """Duration, resolution and stream layout, from the probe. No frame is decoded."""
        facts, _ = await self._facts(ref)
        if facts is None:
            return Card(
                ref=ref,
                kind=MediaKind.VIDEO,
                facts={"readable": "no", "size_bytes": ref.size_bytes},
                outline=(),
                excerpt=None,
                affordances=self.affordances(),
            )
        video = facts.video_streams[0] if facts.video_streams else None
        audio = facts.audio_streams[0] if facts.audio_streams else None
        card_facts: dict[str, str | int | float] = {
            "readable": "yes",
            "duration_s": facts.duration_s,
            "container": facts.container,
            "video_streams": len(facts.video_streams),
            "audio_streams": len(facts.audio_streams),
            "size_bytes": ref.size_bytes,
        }
        if video is not None:
            card_facts["video_codec"] = video.codec
            card_facts["width"] = video.width if video.width is not None else 0
            card_facts["height"] = video.height if video.height is not None else 0
            if video.frame_rate is not None:
                card_facts["frame_rate"] = video.frame_rate
        if audio is not None:
            card_facts["audio_codec"] = audio.codec
            if audio.sample_rate is not None:
                card_facts["audio_sample_rate"] = audio.sample_rate
            if audio.channels is not None:
                card_facts["audio_channels"] = audio.channels
        return Card(
            ref=ref,
            kind=MediaKind.VIDEO,
            facts=card_facts,
            outline=self._outline(facts),
            excerpt=None,
            affordances=self.affordances(),
        )

    def _outline(self, facts: MediaFacts) -> tuple[Segment, ...]:
        """The sampling grid, so an agent sees the timeline before paying for it."""
        if facts.duration_s <= 0 or not facts.video_streams:
            return ()
        times = _sample_times(facts.duration_s, self._interval_s)
        bounds = self._bounds(times, facts)
        return tuple(
            Segment(TimeSpan(start, end), f"{_timestamp(start)}-{_timestamp(end)}")
            for start, end in bounds
        )

    def _frame_duration(self, facts: MediaFacts) -> float:
        video = facts.video_streams[0] if facts.video_streams else None
        if video is None or video.frame_rate is None or video.frame_rate <= 0:
            return FALLBACK_FRAME_DURATION_S
        return 1.0 / video.frame_rate

    def _bounds(
        self, times: tuple[float, ...], facts: MediaFacts
    ) -> tuple[tuple[float, float], ...]:
        """Each sample owns `[t_i, t_{i+1})`; the last owns `[t_n, duration)`.

        The direct analogue of the PDF handler's page separator: `LocatorMap`
        demands total, gapless coverage, so the stretches BETWEEN sampled
        frames must belong to somebody, and they belong to the sample that
        starts them. A timeline with holes cannot answer "what was on screen
        at 3.1 seconds".

        The last bound is floored at one frame's duration past its start: a
        duration that is an exact multiple of the interval, or a rounded
        duration that lands just below the final sample, would otherwise give
        a zero- or negative-width `TimeSpan`, which the domain rejects.
        """
        frame_duration = self._frame_duration(facts)
        bounds: list[tuple[float, float]] = []
        for index, start in enumerate(times):
            if index + 1 < len(times):
                bounds.append((start, times[index + 1]))
            else:
                bounds.append((start, max(facts.duration_s, start + frame_duration)))
        return tuple(bounds)

    async def represent(self, ref: SourceRef, budget: Budget) -> Rendered:
        facts, path = await self._facts(ref)
        if facts is None or path is None:
            return self._nothing_to_read(
                ref,
                budget,
                summary=f"Unreadable video {ref.uri}, {ref.size_bytes} bytes.",
                what="video unprobeable",
                detail=("the file could not be probed as a video; no timeline could be built"),
            )
        if facts.duration_s <= 0 or not facts.video_streams:
            return self._nothing_to_read(
                ref,
                budget,
                summary=(
                    f"Video {ref.uri} ({facts.container}), "
                    f"{facts.duration_s:g}s, {len(facts.video_streams)} video stream(s)."
                ),
                what="no timeline",
                detail=(
                    "the file reports no video stream or no duration; "
                    "there is no timeline to sample"
                ),
            )
        return await self._timeline(ref, path, facts, budget)

    async def _timeline(
        self, ref: SourceRef, path: str, facts: MediaFacts, budget: Budget
    ) -> Rendered:
        times = _sample_times(facts.duration_s, self._interval_s)
        bounds = self._bounds(times, facts)
        chunks: list[str] = []
        segments: list[LocatorSegment] = []
        barriers: list[int] = []
        missing: list[float] = []
        failed: list[float] = []
        cursor = 0
        for index, (start, end) in enumerate(bounds):
            body, state = await self._moment(path, start)
            if state == "missing":
                missing.append(start)
            elif state == "failed":
                failed.append(start)
            chunk = f"[{_timestamp(start)}] {body}{MOMENT_SEPARATOR}"
            if index:
                # One barrier per moment boundary: a new moment's first
                # character is a hard chunk break, so there are exactly as many
                # barriers as sample count minus one.
                barriers.append(cursor)
            segments.append(
                LocatorSegment(CharSpan(cursor, cursor + len(chunk)), TimeSpan(start, end))
            )
            cursor += len(chunk)
            chunks.append(chunk)
        return self._fit(
            "".join(chunks),
            tuple(segments),
            tuple(barriers),
            budget,
            self._timeline_degradations(missing, failed),
        )

    async def _moment(self, path: str, seconds: float) -> tuple[str, str]:
        """What to say about one sampled moment, and why.

        Without a vision model this still reports the moment — a video is not
        empty because nothing looked at it, which is the scanned-PDF lesson at
        a new site. The line says what was not done rather than being blank.
        """
        try:
            frame = await self._frames.frame_at(path, seconds)
        except Exception:
            frame = None
        if frame is None:
            return "(no frame could be decoded at this moment)", "missing"
        if self._vision is None:
            return (
                f"(frame decoded, {len(frame)} bytes; not described, "
                f"as no vision model is configured)",
                "undescribed",
            )
        try:
            text = await self._vision.describe(frame, "image/png", _FRAME_PROMPT)
        except Exception:
            return "(the vision model failed to describe this frame)", "failed"
        if not text.strip():
            # The port's return type is `str`, so a model that answers with
            # nothing is a legal implementation. An empty description must not
            # reach an index as though it were an observation.
            return "(the vision model returned no description for this frame)", "failed"
        return " ".join(text.split()), "described"

    def _timeline_degradations(
        self, missing: list[float], failed: list[float]
    ) -> tuple[Degradation, ...]:
        """One report per state, not one per moment — a long video is one fact."""
        degradations: list[Degradation] = []
        if self._vision is None:
            degradations.append(
                Degradation(
                    what="vision unavailable: frames were not described",
                    detail=(
                        "no vision model is configured, so no sampled frame was "
                        "described; the timeline reports its moments and their spans only"
                    ),
                )
            )
        if missing:
            degradations.append(
                Degradation(
                    what="frames undecodable",
                    detail=(
                        f"{len(missing)} sampled moment(s) yielded no frame "
                        f"({_listed(missing)}); nothing was described for them"
                    ),
                )
            )
        if failed:
            degradations.append(
                Degradation(
                    what="frame descriptions failed",
                    detail=(
                        f"the vision model produced nothing usable for "
                        f"{len(failed)} sampled moment(s) ({_listed(failed)})"
                    ),
                )
            )
        return tuple(degradations)

    def _nothing_to_read(
        self, ref: SourceRef, budget: Budget, *, summary: str, what: str, detail: str
    ) -> Rendered:
        """A rendition for a file with no moment to point at.

        Located by `ByteRange` rather than `TimeSpan`: no timeline was ever
        observed, and claiming `[0, duration)` would be a claim about a video
        this handler never established exists.
        """
        segments = (
            LocatorSegment(CharSpan(0, len(summary)), ByteRange(0, max(1, ref.size_bytes))),
        )
        return self._fit(summary, segments, (), budget, (Degradation(what=what, detail=detail),))

    def _fit(
        self,
        full: str,
        segments: tuple[LocatorSegment, ...],
        barriers: tuple[int, ...],
        budget: Budget,
        degradations: tuple[Degradation, ...],
    ) -> Rendered:
        """Apply the budget, pruning the map and the barriers along with the text.

        `Rendered` rejects a map that does not cover its text exactly and a
        barrier past the end, so truncation cannot touch the text alone. A
        budget of zero still keeps one character, because `CharSpan(0, 0)`
        raises, and the degradation reports the characters kept rather than the
        budget asked for.
        """
        if budget.max_chars is None or len(full) <= budget.max_chars:
            return Rendered(
                text=full,
                locator_map=LocatorMap.build(segments),
                barriers=barriers,
                degradations=degradations,
            )
        keep = max(1, budget.max_chars)
        text = full[:keep]
        kept = tuple(
            LocatorSegment(CharSpan(s.span.start, min(s.span.end, keep)), s.locator)
            for s in segments
            if s.span.start < keep
        )
        return Rendered(
            text=text,
            locator_map=LocatorMap.build(kept),
            barriers=tuple(barrier for barrier in barriers if barrier < keep),
            degradations=(
                *degradations,
                Degradation(
                    what="text truncated",
                    detail=f"kept {len(text)} of {len(full)} characters",
                ),
            ),
        )


def _listed(times: list[float], limit: int = 10) -> str:
    head = ", ".join(_timestamp(t) for t in times[:limit])
    if len(times) <= limit:
        return head
    return f"{head} and {len(times) - limit} more"
