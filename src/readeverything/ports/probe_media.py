"""What can be said about a document without reading its content.

`inspect` must stay cheap: Spec 1's progressive-disclosure design rests on a
card costing no real work, and page count is exactly the fact that shapes an
agent's next move. A probe that extracted text in order to count pages would
defeat that, so this type carries no text and the protocol has no way to
return any.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class DocumentFacts:
    """Cheap facts about a paginated document."""

    page_count: int
    page_sizes: tuple[tuple[float, float], ...]
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.page_count < 0:
            raise ValueError(f"page_count must not be negative, got {self.page_count}")
        if len(self.page_sizes) != self.page_count:
            # These two fields describe one document. Disagreement means a card
            # claiming a page count nothing measured.
            raise ValueError(
                f"page_count {self.page_count} disagrees with {len(self.page_sizes)} page sizes"
            )
        for width, height in self.page_sizes:
            if width <= 0 or height <= 0:
                raise ValueError(f"page size must be positive, got {width}x{height}")


@runtime_checkable
class MediaProbe(Protocol):
    """Cheap structural facts about a document, without extracting content."""

    async def probe(self, data: bytes) -> DocumentFacts: ...
