"""Deriving the artifact cache key.

The key is the whole derivation, not just the file. Every component earns its
place:

- `content_hash`   a moved or renamed file is a hit; an edited one is a miss.
                   This is why there is no staleness protocol: no mutable key.
- `handler_id`     two handlers may produce different things from one file.
- `handler_version` a fixed extraction bug invalidates exactly what it should.
- `affordance`+`params` the operation and its arguments.
- `capability_fingerprint` the model revisions behind the capabilities. This is
                   the one that is easy to forget, and forgetting it means
                   swapping the vision model silently serves a mixture of
                   descriptions produced by two different models.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from readeverything.domain.capability import CapabilitySet
from readeverything.domain.errors import DomainError
from readeverything.domain.identity import ContentHash

_PRIMITIVES = (str, int, float, bool, type(None))


def _reject_non_primitives(value: Any, path: str) -> None:
    """Refuse anything `json.dumps` could not represent losslessly.

    The alternative was `default=str`, which silently coerced. That made
    `{"path": Path("a")}` and `{"path": "a"}` the same cache key — two different
    derivations sharing one artifact, which is the worst failure this component
    has. Refusing is louder and the caller has the information to fix it.
    """
    if isinstance(value, _PRIMITIVES):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                # `json.dumps` stringifies keys, so {1: "v"} and {"1": "v"}
                # would hash the same — the same collision as `default=str`,
                # one level down.
                raise DomainError(
                    f"cache key param {path} has a non-string key: {type(key).__name__}. "
                    f"Affordance params must be JSON-representable so the key is stable."
                )
            _reject_non_primitives(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_non_primitives(item, f"{path}[{index}]")
        return
    raise DomainError(
        f"cache key param {path} is not JSON-primitive: {type(value).__name__}. "
        f"Affordance params must be JSON-representable so the key is stable."
    )


def artifact_key(
    *,
    content_hash: ContentHash,
    handler_id: str,
    handler_version: int,
    affordance: str,
    params: Mapping[str, Any],
    capabilities: CapabilitySet,
) -> str:
    """A stable digest of one derivation."""
    _reject_non_primitives(dict(params), "params")
    payload = json.dumps(
        {
            "content_hash": str(content_hash),
            "handler_id": handler_id,
            "handler_version": handler_version,
            "affordance": affordance,
            "params": params,
            "capabilities": capabilities.fingerprint(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=32).hexdigest()
