"""ModelProbe: the VISION revision is derived from the injected model, not asserted."""

from __future__ import annotations

from readeverything.adapters.model_probe import ModelProbe
from readeverything.domain.capability import Capability
from readeverything.testing.fakes import FakeVision


async def test_the_vision_revision_is_the_injected_model_s_own_id() -> None:
    """The revision cannot disagree with the model, because it is derived from it.

    Before this, the VISION revision and `VisionModel.model_id` were two
    independent inputs that happened to agree by convention. Once artifacts are
    cached, disagreement means keys that misdescribe the model that produced
    them.
    """
    probe = ModelProbe(vision=FakeVision())
    assert await probe.revision(Capability.VISION) == FakeVision().model_id


async def test_no_vision_model_means_no_vision_capability() -> None:
    assert await ModelProbe(vision=None).revision(Capability.VISION) is None
