"""Audio.

`TranscriptCue` has existed in the domain since the first spec with no
producer. This is its first. Where `video.py` maps every character to the
moment it describes, this maps every character to the moment it was *said* —
the same property one medium over, and the reason the locator vocabulary is
shared.

The card costs a probe and decodes nothing, exactly as the video card does:
duration, codec, sample rate and channel count are the facts that shape an
agent's next move, and paying for a decode to learn them would defeat
progressive disclosure.

WHERE THE PDF PRECEDENT STOPS APPLYING. `pdf.py` keeps OCR out of
`represent()` because a scanned PDF still contributes page structure without
it, and `video.py`'s timeline still contributes its moments without a vision
model. An audio file has NO cheaper layer: without transcription it
contributes a duration and a codec, and nothing anyone can ask a question of.
So transcription happens IN `represent()`, gated on the injected
`Transcriber | None` exactly as video's frame descriptions are gated on
`VisionModel | None`. `describe()` stays probe-only and cheap.

`requires()` is `{FFMPEG}` — without it there is no way to extract the track
at all, so the registry drops this handler and audio files fall to the binary
fallback.

The probe, the extractor and the transcriber all arrive by injection. Nothing
here imports an adapter.
"""

from __future__ import annotations

import time
from typing import ClassVar

from pydantic import BaseModel, Field

from readeverything.domain.affordance import Affordance, DetailLevel
from readeverything.domain.capability import Capability
from readeverything.domain.card import Card
from readeverything.domain.errors import UnknownAffordanceError
from readeverything.domain.identity import MediaKind, SourceRef
from readeverything.domain.locator_map import LocatorMap, LocatorSegment
from readeverything.domain.locators import ByteRange, CharSpan, TimeSpan
from readeverything.domain.observation import (
    OperationFinished,
    OperationProgressed,
    OperationStarted,
)
from readeverything.domain.rendition import (
    Budget,
    Degradation,
    Rendered,
    Rendition,
    TextContent,
    TranscriptCue,
)
from readeverything.domain.timeline import clamp_cues_to_duration, tile
from readeverything.ports.audio import AudioExtractor
from readeverything.ports.observation import Observer, emit
from readeverything.ports.source import SourceReader
from readeverything.ports.streams import MediaFacts, StreamProbe
from readeverything.ports.transcription import Transcriber

#: What `represent` calls itself when it narrates.
_OPERATION = "represent"

#: The mimetype handed to the transcriber. `AudioExtractor.extract` is
#: specified to return mono 16kHz WAV bytes whatever the container was, so
#: this describes what is actually passed rather than what was on disk.
EXTRACTED_MIME = "audio/wav"

#: Every cue's text ends with this, and that cue's `LocatorSegment` INCLUDES
#: it — the same reason `video.MOMENT_SEPARATOR` and `pdf.PAGE_SEPARATOR`
#: exist. `CharSpan` rejects `start >= end`, so a cue whose transcriber
#: returned an empty string would otherwise contribute a zero-width span and
#: break the map.
CUE_SEPARATOR = "\n"

#: The narrowest span a cue may own, in seconds. Only ever a floor: cue spans
#: are normally ended by the next cue's start, and this exists because
#: `TimeSpan` forbids `start >= end` while a transcriber is free to emit two
#: cues at the same timestamp, or a final cue at or past a rounded duration.
MIN_CUE_SPAN_S = 0.001

#: `read_span`'s default width, used when the compliance suite invokes every
#: declared affordance with default parameters. A zero-width default would be
#: rejected by `TimeSpan`, so a default has to name some duration; thirty
#: seconds is stated as a convenience, not derived from anything.
DEFAULT_SPAN_S = 30.0


class ReadSpanParams(BaseModel):
    start_s: float = Field(default=0.0, ge=0.0, description="Start of the window, in seconds.")
    end_s: float = Field(
        default=DEFAULT_SPAN_S, gt=0.0, description="End of the window, in seconds."
    )


def _timestamp(seconds: float) -> str:
    """`h:mm:ss.s`, the form a person reads off a player's scrubber."""
    hours, rest = divmod(seconds, 3600.0)
    minutes, secs = divmod(rest, 60.0)
    return f"{int(hours):d}:{int(minutes):02d}:{secs:04.1f}"


def _cue_bounds(
    cues: tuple[TranscriptCue, ...], duration_s: float
) -> tuple[tuple[float, float], ...]:
    """Each cue owns `[start_i, start_{i+1})`; the last owns `[start_n, duration)`.

    A transcript has SILENCE between its cues — people stop talking — and the
    stretches between them have to belong to somebody. The rule itself lives in
    `domain.timeline.tile`, because `VideoHandler` applies exactly the same one
    to its sampled frames and to the two merged; see that module for why the
    first cue starts at 0.0 and why starts are forced strictly increasing.
    """
    return tile(
        (cue.span.start_s for cue in cues), duration_s=duration_s, min_width_s=MIN_CUE_SPAN_S
    )


class AudioHandler:
    """Reads an audio file's speech, and maps every character to the moment it was said."""

    mime_patterns: ClassVar[tuple[str, ...]] = ("kind:audio",)
    priority: ClassVar[int] = 0
    handler_id: ClassVar[str] = "audio"
    handler_version: ClassVar[int] = 1

    def __init__(
        self,
        *,
        source: SourceReader,
        probe: StreamProbe,
        audio: AudioExtractor,
        transcriber: Transcriber | None = None,
        observer: Observer | None = None,
    ) -> None:
        self._source = source
        self._probe = probe
        self._audio = audio
        self._transcriber = transcriber
        self._observer = observer

    def requires(self) -> frozenset[Capability]:
        return frozenset({Capability.FFMPEG})

    def affordances(self) -> tuple[Affordance, ...]:
        """`read_span` only exists when something can listen.

        Gated on the injected transcriber exactly as video gates
        `describe_frame` on its vision model: declaring a tool that can only
        answer "sorry" is the behaviour capability negotiation exists to
        avoid.
        """
        if self._transcriber is None:
            return ()
        return (
            Affordance(
                name="read_span",
                description="Transcribe what was said within a window of the timeline.",
                params=ReadSpanParams,
                requires=frozenset({Capability.FFMPEG, Capability.ASR}),
                level=DetailLevel.SEGMENT,
            ),
        )

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition:
        match name:
            case "read_span" if (transcriber := self._transcriber) is not None:
                if not isinstance(params, ReadSpanParams):
                    raise TypeError(f"expected ReadSpanParams, got {type(params).__name__}")
                return await self._read_span(ref, transcriber, params.start_s, params.end_s)
            case _:
                raise UnknownAffordanceError(name, (a.name for a in self.affordances()))

    def _degraded_span(self, ref: SourceRef, detail: str) -> Rendition:
        """What every unanswerable `read_span` returns.

        Located by `ByteRange` over the whole file rather than by the
        requested `TimeSpan`: a `TimeSpan` would assert a stretch of timeline
        this handler never established the file has. The same reasoning
        `VideoHandler._degraded_frame` applies to a frame it could not decode.
        """
        return Rendition(
            locator=ByteRange(0, max(1, ref.size_bytes)), content=TextContent(detail), degraded=True
        )

    async def _read_span(
        self, ref: SourceRef, transcriber: Transcriber, start_s: float, end_s: float
    ) -> Rendition:
        if start_s >= end_s:
            return self._degraded_span(
                ref,
                f"the requested window {_timestamp(start_s)}-{_timestamp(end_s)} is empty",
            )
        try:
            path = await self._source.local_path(ref.uri)
        except Exception:
            return self._degraded_span(ref, f"{ref.uri} could not be read")
        cues, failure = await self._cues(path, transcriber)
        if failure is not None:
            return self._degraded_span(ref, failure)
        window = tuple(
            cue
            for cue in cues
            if cue.span.start_s < end_s and start_s < cue.span.end_s and cue.text.strip()
        )
        if not window:
            return self._degraded_span(
                ref,
                f"nothing was transcribed between {_timestamp(start_s)} and {_timestamp(end_s)}",
            )
        text = "\n".join(f"[{_timestamp(cue.span.start_s)}] {cue.text.strip()}" for cue in window)
        return Rendition(locator=TimeSpan(start_s, end_s), content=TextContent(text))

    async def _cues(
        self, path: str, transcriber: Transcriber
    ) -> tuple[tuple[TranscriptCue, ...], str | None]:
        """The transcript, or the reason there isn't one.

        The transcriber is passed in rather than read off `self` because the
        only caller has already established it is not `None` — `read_span` is
        not declared without one. A defensive branch here would be a path no
        test could reach honestly.
        """
        audio = await self._audio.extract(path)
        if audio is None:
            return (), "the file has no audio track that could be extracted"
        try:
            return await transcriber.transcribe(audio, EXTRACTED_MIME), None
        except Exception as exc:
            return (), f"transcription failed ({exc})"

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
        """Duration, codec, sample rate and channels, from the probe. Nothing is decoded."""
        facts, _ = await self._facts(ref)
        if facts is None:
            return Card(
                ref=ref,
                kind=MediaKind.AUDIO,
                facts={"readable": "no", "size_bytes": ref.size_bytes},
                outline=(),
                excerpt=None,
                affordances=self.affordances(),
            )
        stream = facts.audio_streams[0] if facts.audio_streams else None
        card_facts: dict[str, str | int | float] = {
            "readable": "yes",
            "duration_s": facts.duration_s,
            "container": facts.container,
            "audio_streams": len(facts.audio_streams),
            "size_bytes": ref.size_bytes,
        }
        if stream is not None:
            card_facts["audio_codec"] = stream.codec
            # Omitted rather than zeroed when the probe could not report them.
            # A sample rate of 0 is a measurement, and claiming one nothing
            # made is the defect this project keeps finding; an absent key
            # admits ignorance.
            if stream.sample_rate is not None:
                card_facts["sample_rate"] = stream.sample_rate
            if stream.channels is not None:
                card_facts["channels"] = stream.channels
        return Card(
            ref=ref,
            kind=MediaKind.AUDIO,
            # No outline. Video's is its sampling grid, known before any
            # decode; an audio file's structure IS its cues, and those cost a
            # transcription. Inventing a uniform grid would present an
            # arbitrary slicing as though it were the file's shape.
            facts=card_facts,
            outline=(),
            excerpt=None,
            affordances=self.affordances(),
        )

    async def represent(self, ref: SourceRef, budget: Budget) -> Rendered:
        """Narrated from end to end, including the paths that transcribe nothing.

        `OperationFinished` is reported in a `finally` and its `elapsed_s` is
        measured rather than estimated, matching `VideoHandler.represent`.
        Transcription happens in ONE call, so there is no per-call progress to
        report during it — `OperationProgressed` fires per cue while the
        timeline is assembled afterward, when the cue count is finally known.
        """
        emit(self._observer, OperationStarted(operation=_OPERATION, ref=ref))
        started = time.perf_counter()
        try:
            return await self._represent(ref, budget)
        finally:
            emit(
                self._observer,
                OperationFinished(
                    operation=_OPERATION, ref=ref, elapsed_s=time.perf_counter() - started
                ),
            )

    async def _represent(self, ref: SourceRef, budget: Budget) -> Rendered:
        facts, path = await self._facts(ref)
        if facts is None or path is None:
            return self._nothing_to_read(
                ref,
                budget,
                summary=f"Unreadable audio {ref.uri}, {ref.size_bytes} bytes.",
                what="audio unprobeable",
                detail="the file could not be probed as audio; no timeline could be built",
            )
        if facts.duration_s <= 0 or not facts.audio_streams:
            return self._nothing_to_read(
                ref,
                budget,
                summary=(
                    f"Audio {ref.uri} ({facts.container}), {facts.duration_s:g}s, "
                    f"{len(facts.audio_streams)} audio stream(s)."
                ),
                what="no audio timeline",
                detail=(
                    "the file reports no audio stream or no duration; "
                    "there is no audio to transcribe"
                ),
            )
        return await self._transcript(ref, path, facts, budget)

    def _summary(self, ref: SourceRef, facts: MediaFacts) -> str:
        codec = facts.audio_streams[0].codec if facts.audio_streams else "unknown"
        return f"Audio {ref.uri} ({facts.container}/{codec}), {facts.duration_s:g}s."

    async def _transcript(
        self, ref: SourceRef, path: str, facts: MediaFacts, budget: Budget
    ) -> Rendered:
        """The transcript, or an honest statement of why there is none.

        Every non-transcript outcome still renders the file's duration and
        codec rather than an empty string: an empty representation would claim
        there was nothing to hear, which is a different — and unverified —
        assertion from "nobody listened".
        """
        if self._transcriber is None:
            return self._nothing_to_read(
                ref,
                budget,
                summary=(
                    f"{self._summary(ref, facts)} Not transcribed: no transcriber is configured."
                ),
                what="transcription unavailable",
                detail=(
                    "no transcriber is configured, so no speech was transcribed; "
                    "the representation reports the file's duration and codec only"
                ),
                span=TimeSpan(0.0, facts.duration_s),
            )
        audio = await self._audio.extract(path)
        if audio is None:
            return self._nothing_to_read(
                ref,
                budget,
                summary=f"{self._summary(ref, facts)} No audio track could be extracted.",
                what="audio track unavailable",
                detail=(
                    "no audio track could be extracted from the file, "
                    "so there was nothing to transcribe"
                ),
                span=TimeSpan(0.0, facts.duration_s),
            )
        try:
            cues = await self._transcriber.transcribe(audio, EXTRACTED_MIME)
        except Exception as exc:
            return self._nothing_to_read(
                ref,
                budget,
                summary=f"{self._summary(ref, facts)} Transcription failed.",
                what="transcription failed",
                detail=f"the transcriber could not transcribe the audio ({exc})",
                span=TimeSpan(0.0, facts.duration_s),
            )
        cues, dropped = clamp_cues_to_duration(cues, facts.duration_s)
        if dropped and not cues:
            return self._nothing_to_read(
                ref,
                budget,
                summary=(
                    f"{self._summary(ref, facts)} Transcribed, and every cue fell "
                    f"outside the file's measured duration."
                ),
                what="transcript outside the file",
                detail=(
                    f"the transcriber returned {dropped} cue(s), all of them starting at or "
                    f"after the probed duration of {facts.duration_s:g}s; none of them could "
                    "be placed on this file's timeline"
                ),
                span=TimeSpan(0.0, facts.duration_s),
            )
        if not cues:
            # Deliberately NOT the same rendering as an absent transcriber.
            # A transcriber that ran and heard nothing says this file is
            # silent or unintelligible; an absent one says nobody listened.
            # Both are honest, and they are different facts.
            return self._nothing_to_read(
                ref,
                budget,
                summary=(f"{self._summary(ref, facts)} Transcribed, and no speech was detected."),
                what="no speech detected",
                detail=(
                    "the transcriber ran over the whole track and returned no cues; "
                    "the audio is silent or contains no intelligible speech"
                ),
                span=TimeSpan(0.0, facts.duration_s),
            )
        degradations = (
            (
                Degradation(
                    what="cues outside the file",
                    detail=(
                        f"{dropped} cue(s) started at or after the probed duration of "
                        f"{facts.duration_s:g}s and were dropped; the transcriber and the "
                        "probe disagree about how long this file is"
                    ),
                ),
            )
            if dropped
            else ()
        )
        return self._fit(*self._flatten(ref, cues, facts.duration_s), budget, degradations)

    def _flatten(
        self, ref: SourceRef, cues: tuple[TranscriptCue, ...], duration_s: float
    ) -> tuple[str, tuple[LocatorSegment, ...]]:
        """The transcript as text and locators, one `OperationProgressed` per cue.

        `total` is `len(cues)` here, not `None` — by the time this runs,
        `transcribe()` has already returned, so the cue count is exactly the
        thing that WAS unknown before it did. There is no earlier point at
        which a partial count could be reported honestly.
        """
        total = len(cues)
        chunks: list[str] = []
        segments: list[LocatorSegment] = []
        cursor = 0
        for done, (cue, (start, end)) in enumerate(
            zip(cues, _cue_bounds(cues, duration_s), strict=True), start=1
        ):
            body = " ".join(cue.text.split())
            speaker = f"{cue.speaker} " if cue.speaker is not None else ""
            chunk = f"[{_timestamp(start)}] {speaker}{body}{CUE_SEPARATOR}"
            segments.append(
                LocatorSegment(CharSpan(cursor, cursor + len(chunk)), TimeSpan(start, end))
            )
            cursor += len(chunk)
            chunks.append(chunk)
            emit(
                self._observer,
                OperationProgressed(operation=_OPERATION, ref=ref, done=done, total=total),
            )
        return "".join(chunks), tuple(segments)

    def _nothing_to_read(
        self,
        ref: SourceRef,
        budget: Budget,
        *,
        summary: str,
        what: str,
        detail: str,
        span: TimeSpan | None = None,
    ) -> Rendered:
        """A rendition for a file with no transcript to point at.

        Located by the file's measured `TimeSpan` when the probe established a
        duration, and by `ByteRange` when it did not — claiming `[0, duration)`
        for a file that was never successfully probed would be a claim about a
        timeline this handler never observed.
        """
        locator = span if span is not None else ByteRange(0, max(1, ref.size_bytes))
        segments = (LocatorSegment(CharSpan(0, len(summary)), locator),)
        return self._fit(summary, segments, budget, (Degradation(what=what, detail=detail),))

    def _fit(
        self,
        full: str,
        segments: tuple[LocatorSegment, ...],
        budget: Budget,
        degradations: tuple[Degradation, ...],
    ) -> Rendered:
        """Apply the budget, pruning the map along with the text.

        The budget is applied to the FLATTENED TEXT, never by transcribing
        less audio: a transcript that stopped early without saying so would be
        indistinguishable from one of a shorter file. `Rendered` rejects a map
        that does not cover its text exactly, so truncation cannot touch the
        text alone. A budget of zero still keeps one character, because
        `CharSpan(0, 0)` raises, and the degradation reports the characters
        KEPT rather than the budget asked for.

        `barriers` is always `()`. Video's scene cuts measure something ffmpeg
        genuinely detects; the audio analogue is a speaker turn, which needs
        diarization. Emitting a barrier at every cue boundary would claim each
        pause is a hard chunk break, which transcription does not establish.
        """
        if budget.max_chars is None or len(full) <= budget.max_chars:
            return Rendered(
                text=full,
                locator_map=LocatorMap.build(segments),
                barriers=(),
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
            barriers=(),
            degradations=(
                *degradations,
                Degradation(
                    what="text truncated",
                    detail=f"kept {len(text)} of {len(full)} characters",
                ),
            ),
        )
