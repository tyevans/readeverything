from readeverything.domain.capability import Capability, CapabilitySet


def test_an_empty_set_satisfies_only_an_empty_requirement() -> None:
    empty = CapabilitySet.empty()
    assert empty.satisfies(frozenset())
    assert not empty.satisfies({Capability.VISION})


def test_satisfies_requires_every_capability() -> None:
    caps = CapabilitySet.of({Capability.VISION: "qwen3.8@rev1"})
    assert caps.satisfies({Capability.VISION})
    assert not caps.satisfies({Capability.VISION, Capability.ASR})


def test_fingerprint_is_stable_across_insertion_order() -> None:
    a = CapabilitySet.of({Capability.VISION: "v1", Capability.ASR: "w1"})
    b = CapabilitySet.of({Capability.ASR: "w1", Capability.VISION: "v1"})
    assert a.fingerprint() == b.fingerprint()


def test_fingerprint_changes_when_a_model_revision_changes() -> None:
    """Swapping the VLM must invalidate cached descriptions."""
    a = CapabilitySet.of({Capability.VISION: "qwen3.8@rev1"})
    b = CapabilitySet.of({Capability.VISION: "qwen3.8@rev2"})
    assert a.fingerprint() != b.fingerprint()


def test_binaries_and_models_are_the_same_kind_of_capability() -> None:
    caps = CapabilitySet.of({Capability.FFMPEG: "7.1", Capability.VISION: "v1"})
    assert caps.satisfies({Capability.FFMPEG, Capability.VISION})
