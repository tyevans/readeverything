import pytest
from pydantic import BaseModel

from readeverything.domain.affordance import Affordance, DetailLevel
from readeverything.domain.capability import Capability, CapabilitySet


class FrameParams(BaseModel):
    at_s: float


def _affordance() -> Affordance:
    return Affordance(
        name="get_frame_at",
        description="Return the video frame at a given time.",
        params=FrameParams,
        requires=frozenset({Capability.FFMPEG}),
        level=DetailLevel.SEGMENT,
    )


def test_an_affordance_is_available_when_its_capabilities_are_present() -> None:
    caps = CapabilitySet.of({Capability.FFMPEG: "7.1"})
    assert _affordance().is_available(caps)


def test_an_affordance_is_unavailable_when_a_capability_is_missing() -> None:
    assert not _affordance().is_available(CapabilitySet.empty())


def test_an_affordance_name_must_be_a_valid_tool_identifier() -> None:
    with pytest.raises(ValueError, match="must be a valid identifier"):
        Affordance(
            name="get frame",
            description="x",
            params=FrameParams,
            requires=frozenset(),
            level=DetailLevel.DEEP,
        )


def test_an_affordance_requires_a_description() -> None:
    """The description becomes the tool docstring; a blank one blinds the model."""
    with pytest.raises(ValueError, match="description must not be blank"):
        Affordance(
            name="get_frame_at",
            description="  ",
            params=FrameParams,
            requires=frozenset(),
            level=DetailLevel.DEEP,
        )
