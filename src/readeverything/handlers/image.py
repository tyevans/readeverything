"""Images.

The first handler whose useful work needs a model. Its card deliberately does
not: dimensions, format and mode come from the header alone, so pointing an
agent at a directory of photographs costs no inference. Everything a model must
answer is behind an affordance the agent chooses to invoke.

`requires()` is empty on purpose. A deployment with no vision model can still
list, size and identify images — it just cannot describe them, and the registry
drops those affordances rather than the handler.
"""

from __future__ import annotations

import io
from typing import ClassVar

from PIL import Image, UnidentifiedImageError

from readeverything.domain.affordance import Affordance
from readeverything.domain.capability import Capability
from readeverything.domain.card import Card, Segment
from readeverything.domain.identity import MediaKind, SourceRef
from readeverything.domain.locators import BBox
from readeverything.ports.source import SourceReader
from readeverything.ports.vision import VisionModel

#: The whole frame, in the normalised coordinates every BBox uses.
WHOLE_IMAGE = BBox(page=None, x=0.0, y=0.0, w=1.0, h=1.0)


class ImageHandler:
    """Reads raster images, and describes them when a vision model is present."""

    mime_patterns: ClassVar[tuple[str, ...]] = ("kind:image",)
    priority: ClassVar[int] = 0
    handler_id: ClassVar[str] = "image"
    handler_version: ClassVar[int] = 1

    def __init__(self, *, source: SourceReader, vision: VisionModel | None = None) -> None:
        self._source = source
        self._vision = vision

    def requires(self) -> frozenset[Capability]:
        return frozenset()

    def affordances(self) -> tuple[Affordance, ...]:
        return ()

    async def _open(self, ref: SourceRef) -> Image.Image | None:
        """The decoded image, or None if the bytes are not a readable image."""
        data = await self._source.read_bytes(ref.uri)
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError, ValueError):
            return None
        return image

    async def describe(self, ref: SourceRef) -> Card:
        image = await self._open(ref)
        if image is None:
            return Card(
                ref=ref,
                kind=MediaKind.IMAGE,
                facts={"decodable": False, "size_bytes": ref.size_bytes},
                outline=(),
                excerpt=None,
                affordances=self.affordances(),
            )
        return Card(
            ref=ref,
            kind=MediaKind.IMAGE,
            facts={
                "decodable": True,
                "width": image.width,
                "height": image.height,
                "format": image.format or "unknown",
                "mode": image.mode,
                "size_bytes": ref.size_bytes,
            },
            outline=(Segment(WHOLE_IMAGE, "whole image"),),
            excerpt=None,
            affordances=self.affordances(),
        )
