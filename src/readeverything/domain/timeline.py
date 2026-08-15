"""Laying entries out along a timeline, and keeping them inside the file.

Two rules live here because three callers need them and a rule with three
copies is a rule that drifts. `AudioHandler` tiles cue starts, `VideoHandler`
tiles sampled frame moments, and `VideoHandler` tiles the two MERGED when a
transcriber is configured. The third caller is what forced the extraction: it
must agree with the other two exactly, or a video's transcript would sit on a
different timeline from the same file's audio.

They are domain rules rather than handler helpers because they are statements
about `LocatorMap` and `TimeSpan`, not about ffmpeg or whisper. `LocatorMap`
demands total, gapless, zero-start coverage and `TimeSpan` forbids
`start >= end`; `tile` is precisely the function that turns a list of moments
into spans satisfying both. Nothing here decodes, probes or infers anything.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from readeverything.domain.locators import TimeSpan
from readeverything.domain.rendition import TranscriptCue


def tile(
    starts: Iterable[float], *, duration_s: float, min_width_s: float
) -> tuple[tuple[float, float], ...]:
    """Each entry owns `[start_i, start_{i+1})`; the last owns `[start_n, duration)`.

    The stretches BETWEEN entries — the silence between two cues, the seconds
    between two sampled frames — have to belong to somebody, because
    `LocatorMap` must be total and gapless. They belong to the entry that most
    recently spoke: a citation landing in a pause resolves to the utterance or
    the frame before it, which is the reading a person scrubbing a player would
    make anyway.

    THE FIRST ENTRY STARTS AT 0.0 regardless of where its own timestamp lands —
    whisper's first cue is often a second or two in, and a video's first sample
    happens to be at zero already. The opening span includes any lead-in
    because the map is of the FILE, not of what was found in it.

    Starts are forced strictly increasing, and the last span is floored at
    `min_width_s`. Both exist for the same reason: `TimeSpan` forbids
    `start >= end`, while a transcriber may legally emit two cues at the same
    timestamp, a frame sample and a cue may legally coincide, and a rounded
    duration may legally land at or below the final entry's start.

    No explicit "(silence)" or "(nothing sampled)" entry is emitted for a gap.
    That would need a detection pass with thresholds and false positives, and
    would put text in an index describing something nothing measured.
    """
    forced: list[float] = []
    for index, start in enumerate(starts):
        forced.append(0.0 if index == 0 else max(start, forced[-1] + min_width_s))
    bounds: list[tuple[float, float]] = []
    for index, start in enumerate(forced):
        if index + 1 < len(forced):
            bounds.append((start, forced[index + 1]))
        else:
            bounds.append((start, max(duration_s, start + min_width_s)))
    return tuple(bounds)


def clamp_cues_to_duration(
    cues: tuple[TranscriptCue, ...], duration_s: float
) -> tuple[tuple[TranscriptCue, ...], int]:
    """The cues the probed file actually has room for, and how many it did not.

    A transcriber is free to disagree with the probe about how long a file is,
    and when it does, one of them is wrong. A cue that merely OVERHANGS the end
    is truncated: the utterance did happen, and only its tail is in dispute. A
    cue that STARTS at or past the duration is dropped entirely — keeping it
    would put a locator on a moment the file does not contain, which is exactly
    the claim this library exists not to make.

    The count of dropped cues is returned rather than swallowed so a handler
    can report the disagreement as a `Degradation`. Silently discarding them
    would hide the fact that the two measurements differ.
    """
    kept = tuple(cue for cue in cues if cue.span.start_s < duration_s)
    clamped = tuple(
        cue
        if cue.span.end_s <= duration_s
        else replace(cue, span=TimeSpan(cue.span.start_s, duration_s))
        for cue in kept
    )
    return clamped, len(cues) - len(kept)
