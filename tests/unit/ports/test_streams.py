from __future__ import annotations

import pytest

from readeverything.ports.streams import MediaFacts, StreamInfo, StreamProbe


def _video_stream() -> StreamInfo:
    return StreamInfo(
        kind="video",
        codec="h264",
        width=320,
        height=240,
        frame_rate=10.0,
        sample_rate=None,
        channels=None,
    )


def _audio_stream() -> StreamInfo:
    return StreamInfo(
        kind="audio",
        codec="aac",
        width=None,
        height=None,
        frame_rate=None,
        sample_rate=44100,
        channels=2,
    )


def test_video_streams_and_audio_streams_split_by_kind() -> None:
    facts = MediaFacts(
        duration_s=5.0, container="mov,mp4,m4a", streams=(_video_stream(), _audio_stream())
    )
    assert facts.video_streams == (_video_stream(),)
    assert facts.audio_streams == (_audio_stream(),)


def test_negative_duration_is_rejected() -> None:
    with pytest.raises(ValueError, match="duration_s"):
        MediaFacts(duration_s=-1.0, container="mov,mp4", streams=())


def test_an_unrelated_object_does_not_satisfy_the_port() -> None:
    assert not isinstance(object(), StreamProbe)
