from readeverything.ports.clips import ClipModel
from readeverything.ports.vision import VisionModel
from readeverything.testing.fakes import FakeClipModel, FakeVision


def test_the_fake_satisfies_the_port() -> None:
    assert isinstance(FakeClipModel(), ClipModel)


def test_an_object_without_watch_does_not_satisfy_the_port() -> None:
    assert not isinstance(object(), ClipModel)


def test_a_vision_model_is_not_a_clip_model() -> None:
    """The reason the two are separate protocols. Our own server described
    stills for months before it could decode a clip, and a handler must be
    able to offer one while truthfully reporting it cannot do the other."""
    assert isinstance(FakeVision(), VisionModel)
    assert not isinstance(FakeVision(), ClipModel)


async def test_the_fake_is_deterministic_and_derived_from_its_input() -> None:
    watcher = FakeClipModel()
    first = await watcher.watch(b"1234", "video/mp4", "what happens")
    second = await watcher.watch(b"1234", "video/mp4", "what happens")
    assert first == second
    assert "4 bytes" in first


def test_the_fake_declares_a_model_id() -> None:
    assert FakeClipModel().model_id == "fake-clip@1"
