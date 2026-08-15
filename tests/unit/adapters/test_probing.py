"""discover(): first probe to answer wins, and only wanted capabilities are probed."""

from __future__ import annotations

from readeverything.adapters.binary_probe import BinaryProbe
from readeverything.adapters.model_probe import ModelProbe
from readeverything.adapters.probing import discover
from readeverything.domain.capability import Capability
from readeverything.testing.fakes import FakeVision


async def test_discovery_reports_only_what_answered() -> None:
    capabilities = await discover(
        probes=[BinaryProbe(executables={}), ModelProbe(vision=FakeVision())],
        capabilities=list(Capability),
    )
    assert capabilities.satisfies({Capability.VISION})
    assert not capabilities.satisfies({Capability.FFMPEG})


async def test_discovery_only_probes_requested_capabilities() -> None:
    """A caller who asks about VISION only should not pay for the other probes."""

    class ExplodingProbe:
        async def revision(self, capability: Capability) -> str | None:
            if capability is not Capability.VISION:
                raise AssertionError(f"should not have been probed: {capability}")
            return None

    capabilities = await discover(
        probes=[ExplodingProbe()],
        capabilities={Capability.VISION},
    )
    assert capabilities.revisions == {}
