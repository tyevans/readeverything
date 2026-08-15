import pytest

from readeverything.domain.errors import (
    CapabilityUnavailableError,
    InfrastructureError,
    ReadEverythingError,
    UnknownAffordanceError,
)


def test_every_error_descends_from_the_root() -> None:
    assert issubclass(UnknownAffordanceError, ReadEverythingError)
    assert issubclass(InfrastructureError, ReadEverythingError)


def test_capability_unavailable_names_what_is_missing() -> None:
    with pytest.raises(CapabilityUnavailableError, match="vision"):
        raise CapabilityUnavailableError(missing=frozenset({"vision"}))
