"""Running every probe and assembling what answered."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.ports.probe import CapabilityProbe


async def discover(
    *,
    probes: Sequence[CapabilityProbe],
    capabilities: Iterable[Capability] | None = None,
) -> CapabilitySet:
    """Probe for each capability and report only what answered.

    First answer wins, so ordering the probes orders precedence. A capability
    no probe answers for is absent, which is what makes the result an
    observation rather than a hopeful assertion. If `capabilities` is given,
    only those are probed — a caller should not pay for probes it never asked
    about.
    """
    wanted = list(Capability) if capabilities is None else list(capabilities)
    found: dict[Capability, str] = {}
    for capability in wanted:
        for probe in probes:
            revision = await probe.revision(capability)
            if revision is not None:
                found[capability] = revision
                break
    return CapabilitySet.of(found)
