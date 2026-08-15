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
from readeverything.domain.identity import ContentHash


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
        default=str,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=32).hexdigest()
