"""Video.

`TimeSpan` has existed in the domain since the first spec with no producer.
This is its first. Where `pdf.py` maps every character to the page it came
from, this maps every character to the moment it describes — the same property
one dimension over, and the reason the locator vocabulary is shared.

A video is two timelines in one file, and with an audio extractor and a
transcriber injected this reads both: sampled frame descriptions and transcript
cues are merged into ONE timestamp-ordered sequence, so a citation resolves the
same way whether it landed on a picture or a sentence. The tiling rule that
makes that sequence total and gapless is `domain.timeline.tile`, shared with
`audio.py` rather than copied — see that module. Without a transcriber nothing
is extracted, nothing is transcribed, and the rendering is exactly what it was
before the transcript existed.

The card costs a probe and decodes no frame, exactly as the PDF card costs a
probe and extracts no text. Duration, resolution and stream layout are the
facts that shape an agent's next move; paying a decode to learn them would
defeat progressive disclosure on the one media type where a decode is most
expensive.

`requires()` is `{FFMPEG}` — unlike `image.py`, which registers without a
vision model and merely drops affordances. Without ffmpeg there is no way to
learn anything at all about a container, so the registry drops this handler
entirely and video files fall to the binary fallback.

The probe, the frame extractor, the audio extractor and the transcriber all
arrive by injection. Nothing here imports an adapter.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import AbstractAsyncContextManager, nullcontext
from typing import ClassVar

from pydantic import BaseModel, Field

from readeverything.domain.affordance import Affordance, DetailLevel
from readeverything.domain.capability import Capability
from readeverything.domain.card import Card, Segment
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
    ImageContent,
    Rendered,
    Rendition,
    TextContent,
    TranscriptCue,
)
from readeverything.domain.timeline import clamp_cues_to_duration, tile
from readeverything.ports.audio import AudioExtractor
from readeverything.ports.frames import FrameExtractor
from readeverything.ports.limits import Limiter
from readeverything.ports.observation import Observer, emit
from readeverything.ports.source import SourceReader
from readeverything.ports.streams import MediaFacts, StreamProbe
from readeverything.ports.transcription import Transcriber
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

#: What marks a transcript line in the merged timeline. Frame descriptions and
#: cues are both `[timestamp] text`, and an agent reading the rendition has to
#: be able to tell what was SEEN from what was SAID — they are different kinds
#: of evidence with different failure modes, and a citation that conflated them
#: would attribute speech to a picture.
SPEECH_MARKER = "(speech)"

#: The mimetype handed to the transcriber. `AudioExtractor.extract` is
#: specified to return mono 16kHz WAV bytes whatever the container was, so this
#: describes what is actually passed rather than what was on disk. The same
#: constant, for the same reason, as `audio.EXTRACTED_MIME`.
EXTRACTED_MIME = "audio/wav"

_FRAME_PROMPT = "Describe what is visible in this video frame, in one or two sentences."

#: What `represent` calls itself when it narrates.
_OPERATION = "represent"

#: What a moment says when fetching it raised instead of returning a state.
#: `_moment` guards both of its calls and returns a state rather than raising,
#: so nothing here should ever be seen; it exists because `asyncio.gather`
#: propagates the first exception, and this handler's contract is that
#: `represent` does not raise. A moment that failed in a way `_moment` did not
#: anticipate is reported as a failed moment, not as an exception out of a read.
_UNEXPECTED_MOMENT = ("(this moment could not be read)", "failed")


class FrameAtParams(BaseModel):
    seconds: float = Field(
        default=0.0, ge=0.0, description="Point in the timeline to extract a frame from."
    )


class DescribeFrameParams(BaseModel):
    seconds: float = Field(
        default=0.0, ge=0.0, description="Point in the timeline to extract a frame from."
    )
    prompt: str = Field(
        default=_FRAME_PROMPT, description="What to ask the vision model about the frame."
    )


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
        audio: AudioExtractor | None = None,
        transcriber: Transcriber | None = None,
        sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
        observer: Observer | None = None,
        limiter: Limiter | None = None,
    ) -> None:
        if sample_interval_s <= 0:
            raise ValueError(f"sample_interval_s must be positive, got {sample_interval_s}")
        self._source = source
        self._probe = probe
        self._frames = frames
        self._vision = vision
        self._audio = audio
        self._transcriber = transcriber
        self._interval_s = sample_interval_s
        self._observer = observer
        self._limiter = limiter

    def _limit(self, capability: Capability) -> AbstractAsyncContextManager[None]:
        """One capability's bound, or nothing at all.

        `None` is unbounded — this handler's behaviour before a limiter existed.
        The handler does not invent a default, because a handler that made its
        own semaphore would bound itself independently of every other handler
        sharing the same endpoint (spec §5.1).

        That is not the same as the library being unbounded by default.
        `build_perception` installs a `SemaphoreLimiter()` when a caller injects
        none, so `None` arrives here only by constructing this handler directly
        or by passing a limiter that deliberately bounds nothing.
        """
        if self._limiter is None:
            return nullcontext()
        return self._limiter.limit(capability)

    def requires(self) -> frozenset[Capability]:
        return frozenset({Capability.FFMPEG})

    def affordances(self) -> tuple[Affordance, ...]:
        affordances: list[Affordance] = [
            Affordance(
                name="frame_at",
                description="Extract one video frame, as a PNG image, at a point in time.",
                params=FrameAtParams,
                requires=frozenset({Capability.FFMPEG}),
                level=DetailLevel.SEGMENT,
            )
        ]
        if self._vision is not None:
            affordances.append(
                Affordance(
                    name="describe_frame",
                    description=(
                        "Extract a video frame and describe it with a vision model in one "
                        "call, saving the round trip through describe_image."
                    ),
                    params=DescribeFrameParams,
                    requires=frozenset({Capability.FFMPEG, Capability.VISION}),
                    level=DetailLevel.DEEP,
                )
            )
        return tuple(affordances)

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition:
        match name:
            case "frame_at":
                if not isinstance(params, FrameAtParams):
                    raise TypeError(f"expected FrameAtParams, got {type(params).__name__}")
                return await self._frame_at(ref, params.seconds)
            case "describe_frame":
                if self._vision is None:
                    raise UnknownAffordanceError(name, (a.name for a in self.affordances()))
                if not isinstance(params, DescribeFrameParams):
                    raise TypeError(f"expected DescribeFrameParams, got {type(params).__name__}")
                return await self._describe_frame(ref, params.seconds, params.prompt)
            case _:
                raise UnknownAffordanceError(name, (a.name for a in self.affordances()))

    def _degraded_frame(self, ref: SourceRef, seconds: float, detail: str) -> Rendition:
        """What every un-decodable-frame request returns.

        Located by `ByteRange` over the whole file rather than a `TimeSpan` at
        `seconds`: a `TimeSpan` would assert a moment the video does not have.
        The same reasoning `PdfHandler._degraded_text` applies to a page number
        past the end.
        """
        return Rendition(
            locator=ByteRange(0, max(1, ref.size_bytes)), content=TextContent(detail), degraded=True
        )

    async def _absent_frame_detail(self, path: str, seconds: float) -> str:
        """Why `frame_at` returned nothing, distinguishing the cause when it can.

        The generic message ("no frame could be decoded at ...") tells an agent
        nothing actionable: it cannot tell whether to retry at a different
        timestamp or give up. A probe of the header (one ffprobe read, paid
        only on this already-degraded path) usually knows the duration, and a
        request past it is the common case worth naming. A negative request is
        named too, though `FrameAtParams` already forbids one via `ge=0.0`.
        Within the duration, or when the probe itself fails or reports no
        duration, this falls back to the generic message rather than guessing:
        claiming "past the end" without knowing where the end is would be the
        same defect wearing different clothes.
        """
        if seconds < 0.0:
            return f"the requested time {_timestamp(seconds)} is negative; there is no such frame"
        try:
            facts = await self._probe.probe(path)
        except Exception:
            facts = None
        if facts is not None and facts.duration_s > 0 and seconds >= facts.duration_s:
            return (
                f"the video is {facts.duration_s:g}s long; "
                f"there is no frame at {_timestamp(seconds)}"
            )
        return f"no frame could be decoded at {_timestamp(seconds)}"

    async def _frame_at(self, ref: SourceRef, seconds: float) -> Rendition:
        try:
            path = await self._source.local_path(ref.uri)
        except Exception:
            return self._degraded_frame(ref, seconds, f"{ref.uri} could not be read")
        try:
            frame = await self._frames.frame_at(path, seconds)
        except Exception:
            frame = None
        if frame is None:
            return self._degraded_frame(
                ref, seconds, await self._absent_frame_detail(path, seconds)
            )
        frame_span = TimeSpan(seconds, seconds + FALLBACK_FRAME_DURATION_S)
        return Rendition(locator=frame_span, content=ImageContent(data=frame, mime="image/png"))

    async def _describe_frame(self, ref: SourceRef, seconds: float, prompt: str) -> Rendition:
        if self._vision is None:
            raise UnknownAffordanceError("describe_frame", (a.name for a in self.affordances()))
        try:
            path = await self._source.local_path(ref.uri)
        except Exception:
            return self._degraded_frame(ref, seconds, f"{ref.uri} could not be read")
        try:
            frame = await self._frames.frame_at(path, seconds)
        except Exception:
            frame = None
        if frame is None:
            return self._degraded_frame(
                ref, seconds, await self._absent_frame_detail(path, seconds)
            )
        try:
            text = await self._vision.describe(frame, "image/png", prompt)
        except Exception:
            return self._degraded_frame(
                ref,
                seconds,
                f"the vision model failed to describe the frame at {_timestamp(seconds)}",
            )
        if not text.strip():
            return self._degraded_frame(
                ref,
                seconds,
                f"the vision model returned no description for the frame at {_timestamp(seconds)}",
            )
        frame_span = TimeSpan(seconds, seconds + FALLBACK_FRAME_DURATION_S)
        return Rendition(locator=frame_span, content=TextContent(" ".join(text.split())))

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
            # Omitted rather than zeroed when the probe could not report them.
            # A width of 0 is a measurement, and claiming one nothing made is the
            # defect this project keeps finding; an absent key admits ignorance,
            # which is what `frame_rate` and `sample_rate` already do.
            if video.width is not None:
                card_facts["width"] = video.width
            if video.height is not None:
                card_facts["height"] = video.height
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
        """Each entry owns `[t_i, t_{i+1})`; the last owns `[t_n, duration)`.

        The direct analogue of the PDF handler's page separator: `LocatorMap`
        demands total, gapless coverage, so the stretches BETWEEN sampled
        frames must belong to somebody, and they belong to the sample that
        starts them. A timeline with holes cannot answer "what was on screen
        at 3.1 seconds".

        The rule lives in `domain.timeline.tile` because `AudioHandler` needs
        exactly the same one for its cues, and `_timeline` needs it a third
        time for the two MERGED — one rule with three copies is a rule that
        drifts. `min_width_s` is one frame's duration here, which is what
        floors the final span when the duration is an exact multiple of the
        interval or rounds to just below the last sample.
        """
        return tile(times, duration_s=facts.duration_s, min_width_s=self._frame_duration(facts))

    async def represent(self, ref: SourceRef, budget: Budget) -> Rendered:
        """Narrated from end to end, including the paths that read nothing.

        `OperationFinished` is reported in a `finally` and its `elapsed_s` is
        measured rather than estimated: a caller watching a read is watching to
        find out whether it is progressing, and a start with no end is the hang
        this narration exists to make visible.

        This never raises about its input — except `asyncio.CancelledError`
        (or any other `BaseException` that is not an `Exception`), which
        propagates. Cancellation is not a claim about the file; it is the
        caller or the runtime asking the work to stop, and folding it into a
        degraded moment would assert something about the video that nothing
        established.
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
        """One timestamp-ordered sequence of what was seen and what was said.

        The sampled moments and the transcript's cues are merged before
        anything is tiled, so a citation resolves to the same timeline whether
        it landed on a picture or on a sentence. WITHOUT A TRANSCRIBER the cue
        list is empty and every step below reduces to what this handler did
        before the transcript existed — deliberately, because every other video
        test in the suite rests on that rendering.
        """
        times = _sample_times(facts.duration_s, self._interval_s)
        cues, transcript_degradations = await self._cues(path, facts)
        entries = _merge(times, cues)
        bounds = self._bounds(tuple(start for start, _ in entries), facts)
        fetched = await self._moments(ref, path, entries)
        chunks: list[str] = []
        segments: list[LocatorSegment] = []
        entry_barriers: list[int] = []
        entry_offsets: list[int] = []
        missing: list[float] = []
        failed: list[float] = []
        cursor = 0
        for index, ((_sampled_at, cue), (start, end)) in enumerate(
            zip(entries, bounds, strict=True)
        ):
            if cue is None:
                # `fetched` is indexed BY ENTRY, so this reads the moment that
                # belongs here whatever order the fetches finished in. The
                # `or` is unreachable — `_moments` holds an outcome at every
                # `cue is None` index by construction — and is what narrows
                # the optional away.
                body, state = fetched[index] or _UNEXPECTED_MOMENT
                if state == "missing":
                    missing.append(start)
                elif state == "failed":
                    failed.append(start)
            else:
                body = _spoken(cue)
            chunk = f"[{_timestamp(start)}] {body}{MOMENT_SEPARATOR}"
            if index:
                # One barrier per entry boundary: a new entry's first character
                # is a hard chunk break. This is the status quo `_barriers`
                # falls back to when scene detection finds nothing to say or
                # fails outright.
                entry_barriers.append(cursor)
            entry_offsets.append(cursor)
            segments.append(
                LocatorSegment(CharSpan(cursor, cursor + len(chunk)), TimeSpan(start, end))
            )
            cursor += len(chunk)
            chunks.append(chunk)
        barriers, scene_degradation = await self._barriers(
            path, tuple(t for t, _ in bounds), entry_offsets, entry_barriers
        )
        return self._fit(
            "".join(chunks),
            tuple(segments),
            barriers,
            budget,
            (
                *self._timeline_degradations(missing, failed),
                *transcript_degradations,
                *scene_degradation,
            ),
        )

    async def _cues(
        self, path: str, facts: MediaFacts
    ) -> tuple[tuple[TranscriptCue, ...], tuple[Degradation, ...]]:
        """The transcript to merge in, and everything that went wrong getting it.

        NOTHING IS ATTEMPTED WITHOUT A TRANSCRIBER, and no degradation is
        reported for its absence. A video with no transcriber configured is not
        a degraded video — it is this handler's whole prior behaviour, and
        `describe()` never promised speech. Contrast `_timeline_degradations`,
        which does report a missing vision model: there the affordance was
        offered and the frames were sampled anyway, so the silence needs
        explaining. Here nothing was ever going to be asked.

        Every other outcome degrades and reports. `AudioExtractor.extract` is
        specified never to raise, but it is an injected implementation and this
        handler's contract is that it never raises about its input, so the call
        is guarded rather than trusted.
        """
        if self._transcriber is None:
            return (), ()
        if self._audio is None:
            return (), (
                Degradation(
                    what="audio track unavailable",
                    detail=(
                        "a transcriber is configured but no audio extractor is, so the "
                        "video's audio track could not be reached; the timeline reports "
                        "its frames only"
                    ),
                ),
            )
        try:
            audio = await self._audio.extract(path)
        except Exception:
            audio = None
        if audio is None:
            return (), (
                Degradation(
                    what="audio track unavailable",
                    detail=(
                        "no audio track could be extracted from the video, so there was "
                        "nothing to transcribe; the timeline reports its frames only"
                    ),
                ),
            )
        try:
            transcribed = await self._transcriber.transcribe(audio, EXTRACTED_MIME)
        except Exception as exc:
            return (), (
                Degradation(
                    what="transcription failed",
                    detail=(
                        f"the transcriber could not transcribe the video's audio ({exc}); "
                        "the timeline reports its frames only"
                    ),
                ),
            )
        cues, dropped = clamp_cues_to_duration(transcribed, facts.duration_s)
        degradations: list[Degradation] = []
        if dropped:
            degradations.append(
                Degradation(
                    what="cues outside the file",
                    detail=(
                        f"{dropped} cue(s) started at or after the probed duration of "
                        f"{facts.duration_s:g}s and were dropped; the transcriber and the "
                        "probe disagree about how long this file is"
                    ),
                )
            )
        if not cues:
            # A transcriber that ran and heard nothing is a different fact from
            # an absent transcriber, and the timeline says so rather than
            # rendering identically to the untranscribed case.
            degradations.append(
                Degradation(
                    what="no speech detected",
                    detail=(
                        "the transcriber ran over the whole track and returned no usable "
                        "cues; the audio is silent or contains no intelligible speech"
                    ),
                )
            )
        return cues, tuple(degradations)

    async def _barriers(
        self,
        path: str,
        starts: tuple[float, ...],
        moment_offsets: list[int],
        moment_barriers: list[int],
    ) -> tuple[tuple[int, ...], tuple[Degradation, ...]]:
        """Hard chunk breaks: at the video's scene cuts when detection finds
        them, at every moment boundary otherwise.

        "No cuts found" and "detection failed" are distinguishable outcomes of
        `scene_cuts` (see `ports/frames.py`): the first degrades to the status
        quo silently, since an unedited video's timeline is exactly right as
        every-moment barriers. The second degrades the same way but reports
        why, so a caller cannot mistake a broken detector for uniform content.
        """
        try:
            cuts = await self._frames.scene_cuts(path)
        except Exception as exc:
            return tuple(moment_barriers), (
                Degradation(
                    what="scene detection failed",
                    detail=(
                        f"scene-cut detection could not run ({exc}); barriers fall back to "
                        "one per sampled moment"
                    ),
                ),
            )
        if not cuts:
            return tuple(moment_barriers), ()
        offsets = {
            moment_offsets[self._bucket(cut, starts)]
            for cut in cuts
            if self._bucket(cut, starts) > 0
        }
        if not offsets:
            return tuple(moment_barriers), ()
        return tuple(sorted(offsets)), ()

    @staticmethod
    def _bucket(seconds: float, starts: tuple[float, ...]) -> int:
        """The index of the last sample whose start is at or before `seconds`."""
        index = 0
        for i, start in enumerate(starts):
            if start <= seconds:
                index = i
            else:
                break
        return index

    async def _moments(
        self,
        ref: SourceRef,
        path: str,
        entries: tuple[tuple[float, TranscriptCue | None], ...],
    ) -> list[tuple[str, str] | None]:
        """Every sampled moment, fetched concurrently, INDEXED BY ENTRY.

        Fetching a moment is slow and independent of every other moment;
        assembling the timeline out of them is fast and strictly ordered. This
        does the first job and hands the second a list it reads positionally,
        so completion order never reaches the assembly and cannot reach the
        text or the locators. `None` marks a cue entry, which fetches nothing.

        The frame is decoded at the moment it was SAMPLED, while the entry is
        labelled and located at its tiled start. The two differ only when a cue
        shares a frame's instant and tiling nudges one of them apart; asking
        ffmpeg for the nudged timestamp instead would decode a frame nobody
        asked about.

        `total` is the number of sampled moments, which IS known before any of
        them runs. `done` counts completions, and it is monotonic because it is
        incremented and read in one step of a single-threaded event loop —
        progress that went 3, 7, 4 would tell a caller nothing.
        """
        sampled = tuple(index for index, (_, cue) in enumerate(entries) if cue is None)
        total = len(sampled)
        done = 0

        async def fetch(seconds: float) -> tuple[str, str]:
            nonlocal done
            try:
                outcome = await self._moment(path, seconds)
            except Exception:
                outcome = _UNEXPECTED_MOMENT
            done += 1
            emit(
                self._observer,
                OperationProgressed(operation=_OPERATION, ref=ref, done=done, total=total),
            )
            return outcome

        # `return_exceptions=True` as well as the guard inside `fetch`: gather
        # propagates the first exception it sees, and this handler must not
        # begin raising from `represent` because it began gathering. The
        # `except Exception:` above deliberately does not catch
        # `CancelledError` — a cancelled fetch's own `CancelledError` must
        # reach the results below undisguised, so the loop that follows can
        # tell it apart from a decode failure and re-raise it instead of
        # turning it into a degraded moment. See `_represent`.
        outcomes = await asyncio.gather(
            *(fetch(entries[index][0]) for index in sampled), return_exceptions=True
        )
        fetched: list[tuple[str, str] | None] = [None] * len(entries)
        for index, outcome in zip(sampled, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                if not isinstance(outcome, Exception):
                    # CancelledError (and any other BaseException that isn't an
                    # Exception) is a request to stop, not a frame we failed to
                    # read. `gather(return_exceptions=True)` hands back a
                    # cancelled child's own CancelledError as a result instead
                    # of propagating it, so a blanket BaseException-to-degraded
                    # mapping would put a claim about the file into the
                    # timeline that nothing established. Let it propagate.
                    raise outcome
                outcome = _UNEXPECTED_MOMENT
            fetched[index] = outcome
        return fetched

    async def _moment(self, path: str, seconds: float) -> tuple[str, str]:
        """What to say about one sampled moment, and why.

        Without a vision model this still reports the moment — a video is not
        empty because nothing looked at it, which is the scanned-PDF lesson at
        a new site. The line says what was not done rather than being blank.

        The two calls are bounded SEPARATELY, each by the capability it
        actually spends: extraction is an ffmpeg subprocess and description is
        a model call over the network. Holding a vision slot for the duration
        of a decode would spend the endpoint's concurrency on ffmpeg, and a
        single lock around both would let the slower of the two set the pace
        for the other.
        """
        try:
            async with self._limit(Capability.FFMPEG):
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
            async with self._limit(Capability.VISION):
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


def _merge(
    times: tuple[float, ...], cues: tuple[TranscriptCue, ...]
) -> tuple[tuple[float, TranscriptCue | None], ...]:
    """The sampled moments and the cues in one timestamp-ordered sequence.

    A frame description and a cue at the same instant are TWO ENTRIES, not a
    conflict: they say different things about that moment, and merging them
    into one line would force a choice about which evidence outranks the other
    that nothing here is in a position to make. `domain.timeline.tile` gives
    the second of the pair a start one frame's width later, which is what keeps
    both spans non-empty.

    The sort is stable and frames sort first at a tie, so a frame keeps the
    exact timestamp it was decoded at and the cue is the one nudged. That
    direction matters: the frame's timestamp is a request this handler made of
    ffmpeg, while the cue's is a measurement someone else reported.
    """
    entries: list[tuple[float, TranscriptCue | None]] = [(t, None) for t in times]
    entries.extend((cue.span.start_s, cue) for cue in cues)
    entries.sort(key=lambda entry: (entry[0], entry[1] is not None))
    return tuple(entries)


def _spoken(cue: TranscriptCue) -> str:
    """One cue as a line of the merged timeline, marked as speech."""
    speaker = f"{cue.speaker} " if cue.speaker is not None else ""
    body = " ".join(cue.text.split())
    if not body:
        # `TranscriptCue.text` is a `str`, so a transcriber emitting an empty
        # one is a legal implementation. It must not reach an index as though
        # something had been heard.
        return f"{SPEECH_MARKER} (the transcriber returned no text for this cue)"
    return f"{SPEECH_MARKER} {speaker}{body}"


def _listed(times: list[float], limit: int = 10) -> str:
    head = ", ".join(_timestamp(t) for t in times[:limit])
    if len(times) <= limit:
        return head
    return f"{head} and {len(times) - limit} more"
