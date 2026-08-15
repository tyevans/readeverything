"""Bounding how much of a capability runs at once.

This library samples a video's frames and calls a vision model on each one; an
unbounded caller can flood a vision endpoint or saturate every core with
ffmpeg processes without ever seeing it happen. `Limiter` is the seam a caller
plugs a bound into, per capability, without this library prescribing a
concurrency framework.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Protocol, runtime_checkable

from readeverything.domain.capability import Capability


@runtime_checkable
class Limiter(Protocol):
    def limit(self, capability: Capability) -> AbstractAsyncContextManager[None]: ...
