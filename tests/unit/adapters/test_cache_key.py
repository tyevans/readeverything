from pathlib import Path

import pytest

from readeverything.adapters.cache_key import artifact_key
from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.domain.errors import DomainError
from readeverything.domain.identity import ContentHash

CAPS = CapabilitySet.of({Capability.VISION: "qwen3.8@rev1"})


def _key(**overrides: object) -> str:
    kwargs: dict[str, object] = {
        "content_hash": ContentHash("aaa"),
        "handler_id": "video",
        "handler_version": 1,
        "affordance": "describe_frame",
        "params": {"at_s": 1.5},
        "capabilities": CAPS,
    }
    kwargs.update(overrides)
    return artifact_key(**kwargs)  # type: ignore[arg-type]


def test_the_same_derivation_yields_the_same_key() -> None:
    assert _key() == _key()


def test_param_order_does_not_change_the_key() -> None:
    assert _key(params={"a": 1, "b": 2}) == _key(params={"b": 2, "a": 1})


def test_different_content_is_a_different_key() -> None:
    assert _key() != _key(content_hash=ContentHash("bbb"))


def test_a_handler_version_bump_invalidates() -> None:
    """A fixed extraction bug must invalidate exactly what it should."""
    assert _key() != _key(handler_version=2)


def test_different_params_are_a_different_key() -> None:
    assert _key() != _key(params={"at_s": 2.5})


def test_swapping_the_model_invalidates() -> None:
    """Otherwise the cache silently serves a mixture from two models."""
    other = CapabilitySet.of({Capability.VISION: "qwen3.8@rev2"})
    assert _key() != _key(capabilities=other)


def test_a_non_primitive_param_is_refused() -> None:
    """`default=str` would silently collide with the plain string "a"."""
    with pytest.raises(DomainError, match="not JSON-primitive"):
        _key(params={"path": Path("a")})


def test_nested_primitives_are_allowed() -> None:
    """Structure is fine; only unserialisable leaves are refused."""
    assert _key(params={"a": [1, 2, {"b": None}]}) == _key(params={"a": [1, 2, {"b": None}]})


def test_a_non_primitive_nested_deep_is_refused() -> None:
    with pytest.raises(DomainError, match="not JSON-primitive"):
        _key(params={"a": [1, {"b": Path("x")}]})


def test_the_refused_type_is_named_in_the_message() -> None:
    with pytest.raises(DomainError, match="PosixPath"):
        _key(params={"path": Path("a")})
