import pytest

from readeverything.ports.vision import VisionModel
from readeverything.testing.fakes import FakeVision


def test_the_fake_satisfies_the_port() -> None:
    assert isinstance(FakeVision(), VisionModel)


def test_an_object_without_describe_does_not_satisfy_the_port() -> None:
    assert not isinstance(object(), VisionModel)


async def test_the_fake_is_deterministic_and_derived_from_its_input() -> None:
    """Unit tests must never assert on model text, so the fake must not invent any."""
    vision = FakeVision()
    first = await vision.describe(b"1234", "image/png", "what is this")
    second = await vision.describe(b"1234", "image/png", "what is this")
    assert first == second
    assert "4 bytes" in first
    assert "image/png" in first


def test_the_fake_declares_a_model_id() -> None:
    """The id feeds the capability fingerprint, so a fake needs one too."""
    assert FakeVision().model_id == "fake-vision@1"


def test_a_data_protocol_cannot_be_used_with_issubclass() -> None:
    """Documents a real limitation: VisionModel has a non-method member.

    `runtime_checkable` supports `isinstance` for data protocols but not
    `issubclass`. Anyone adding a ports test that mirrors the method-only
    protocols will hit this, so it is pinned rather than left to surprise them.
    """
    with pytest.raises(TypeError):
        issubclass(FakeVision, VisionModel)  # type: ignore[misc]
