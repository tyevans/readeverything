"""Which file took the time, and — for a video — how far into it we are.

`Perception` does not fan out across a directory itself — the caller writes
that loop, over `list()` — so these tests write it too, and check that the
observer sees each file's start and finish and can attribute elapsed time per
`uri` without threading any state of its own.

Per-frame progress needs a handler that has frames to report, so the last
test here wires `VideoHandler` by hand over stubbed ffmpeg ports and an
explicitly-declared `CapabilitySet` — `build_perception` would only reach the
real ffmpeg binary, which not every machine running these tests has.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.detection import PuremagicDetector
from readeverything.adapters.hashing import ContentHasher, StatMemo
from readeverything.adapters.local_source import LocalFileSource
from readeverything.composition import build_perception
from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.domain.observation import (
    OperationFinished,
    OperationProgressed,
    OperationStarted,
)
from readeverything.domain.rendition import Budget
from readeverything.handlers.video import VideoHandler
from readeverything.pipeline.perception import Perception
from readeverything.pipeline.resolution import ResolutionMemo
from readeverything.ports.streams import MediaFacts, StreamInfo
from readeverything.registry.registry import MimeTypeRegistry
from readeverything.testing.fakes import FakeVision, RecordingObserver

pytestmark = pytest.mark.integration

#: Enough of an mp4 header for detection to say `video/mp4`. The bytes after
#: it are never decoded: the stream probe and the frame extractor are stubs,
#: which is the point — this test must run on a machine with no ffmpeg.
_MP4_HEADER = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41"

#: Six seconds at the two-second sampling interval below: t=0, 2, 4.
_DURATION_S = 6.0
_INTERVAL_S = 2.0
_EXPECTED_MOMENTS = 3


class _StubStreams:
    """A stream probe that answers without ffprobe."""

    async def probe(self, path: str) -> MediaFacts:
        return MediaFacts(
            duration_s=_DURATION_S,
            container="mov,mp4",
            streams=(
                StreamInfo(
                    kind="video",
                    codec="h264",
                    width=320,
                    height=240,
                    frame_rate=10.0,
                    sample_rate=None,
                    channels=None,
                ),
            ),
        )


class _StubFrames:
    """Frame bytes whose length varies with the timestamp, without ffmpeg.

    `FakeVision` describes an image by its byte length, so distinct lengths
    keep the moments distinguishable in the assembled timeline.
    """

    async def frame_at(self, path: str, seconds: float) -> bytes | None:
        return b"frame".ljust(64 + int(seconds * 10), b"\0")

    async def scene_cuts(self, path: str, threshold: float = 0.4) -> tuple[float, ...]:
        return ()


def _video_perception(root: Path, observer: RecordingObserver) -> Perception:
    """A `Perception` that genuinely dispatches an mp4 to `VideoHandler`.

    The capability set is declared, not probed: `Capability.FFMPEG` is what
    the registry filters on, and declaring it is what makes the handler
    survive registration on a machine that has no ffmpeg to probe for.
    """
    source = LocalFileSource(root=root)
    vision = FakeVision()
    handler = VideoHandler(
        source=source,
        probe=_StubStreams(),
        frames=_StubFrames(),
        vision=vision,
        sample_interval_s=_INTERVAL_S,
        observer=observer,
    )
    capabilities = CapabilitySet.of(
        {Capability.FFMPEG: "stub-ffmpeg@1", Capability.VISION: vision.model_id}
    )
    return Perception(
        source=source,
        detector=PuremagicDetector(),
        hasher=ContentHasher(source=source, memo=StatMemo()),
        registry=MimeTypeRegistry(handlers=[handler], capabilities=capabilities),
        artifacts=InMemoryArtifactStore(),
        memo=ResolutionMemo(),
    )


async def test_an_observer_sees_a_whole_directory_being_read(media_root: Path) -> None:
    """Every file in a mixed directory reports a start and a finish.

    Which handler serves which file is whatever `build_perception` registers
    on this machine — this test asserts the start/finish envelope around each
    `uri`, not what happens inside any one read. Per-frame progress is
    `test_a_videos_frames_are_reported_as_progress` below.
    """
    recorder = RecordingObserver()
    perception = await build_perception(
        media_root, vision=FakeVision(), observer=recorder, probe_binaries=False
    )
    for uri in await perception.list("."):
        await perception.represent(uri, Budget(max_chars=None))

    finished = [e for e in recorder.events if isinstance(e, OperationFinished)]
    assert {e.ref.uri for e in finished} == set(await perception.list("."))
    assert all(e.elapsed_s >= 0.0 for e in finished)


async def test_observing_changes_nothing(media_root: Path) -> None:
    """Attaching an observer perturbs no rendition anywhere in the directory."""
    recorder = RecordingObserver()
    unobserved = await build_perception(media_root, vision=FakeVision(), probe_binaries=False)
    observed = await build_perception(
        media_root, vision=FakeVision(), observer=recorder, probe_binaries=False
    )
    for uri in await unobserved.list("."):
        before = await unobserved.represent(uri, Budget(max_chars=None))
        after = await observed.represent(uri, Budget(max_chars=None))
        assert before.text == after.text
    assert recorder.events, "the observed arm must actually have been observed"


async def test_a_videos_frames_are_reported_as_progress(tmp_path: Path) -> None:
    """"Each frame complete" is a claim only `OperationProgressed` can settle."""
    (tmp_path / "clip.mp4").write_bytes(_MP4_HEADER + bytes(512))
    recorder = RecordingObserver()
    perception = _video_perception(tmp_path, recorder)

    card = await perception.inspect("clip.mp4")
    assert card.kind == "video", "the video handler must be the one that served this"
    await perception.represent("clip.mp4", Budget(max_chars=None))

    progressed = [e for e in recorder.events if isinstance(e, OperationProgressed)]
    assert [e.done for e in progressed] == list(range(1, _EXPECTED_MOMENTS + 1))
    assert {e.total for e in progressed} == {_EXPECTED_MOMENTS}
    assert {e.ref.uri for e in progressed} == {"clip.mp4"}
    assert isinstance(recorder.events[0], OperationStarted)
    assert isinstance(recorder.events[-1], OperationFinished)
