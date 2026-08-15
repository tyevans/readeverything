"""Turning exceptions into results, exactly once, at the model boundary.

Ports and handlers raise; the tool pack returns. The split is by audience: a
raised exception is the right signal for a caller that can branch on it, and
the wrong one for a model, which sees a traceback, cannot act on it, and burns
a turn discovering that.

`never_raises` catches `Exception` deliberately broadly: an adapter bug is
exactly as unhelpful to a model as an expected failure, and letting one
through would make the guarantee conditional on our own code being correct.
It does NOT catch `BaseException` subclasses outside `Exception` — notably
`asyncio.CancelledError`, `KeyboardInterrupt` and `SystemExit` — which must
keep propagating. Swallowing `CancelledError` would break task cancellation.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    value: Any
    error: str | None
    error_type: str | None

    def render(self) -> str:
        """A compact string for a model to read."""
        if self.ok:
            return str(self.value)
        return f"ERROR ({self.error_type}): {self.error}"


def never_raises[**P](
    fn: Callable[P, Awaitable[Any]],
) -> Callable[P, Awaitable[ToolResult]]:
    """Wrap an async callable so it returns a `ToolResult` instead of raising."""

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> ToolResult:
        try:
            return ToolResult(ok=True, value=await fn(*args, **kwargs), error=None, error_type=None)
        except Exception as exc:  # broad on purpose: the whole point of this decorator
            return ToolResult(ok=False, value=None, error=str(exc), error_type=type(exc).__name__)

    return wrapper
