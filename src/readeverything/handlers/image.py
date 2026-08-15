"""Images.

The first handler whose useful work needs a model. Its card deliberately does
not: dimensions, format and mode come from the header alone, so pointing an
agent at a directory of photographs costs no inference. Everything a model must
answer is behind an affordance the agent chooses to invoke.

The card costs no model call. It does pay a full decode, because `Image.open`
is lazy and truncation only surfaces on first pixel access — that decode is
what makes the never-raise property real.

`requires()` is empty on purpose. A deployment with no vision model can still
list, size and identify images — it just cannot describe them, and the registry
drops those affordances rather than the handler.
"""

from __future__ import annotations

import io
from typing import ClassVar

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from readeverything.domain.affordance import Affordance, DetailLevel
from readeverything.domain.capability import Capability
from readeverything.domain.card import Card, Segment
from readeverything.domain.errors import DomainError, InfrastructureError, UnknownAffordanceError
from readeverything.domain.identity import MediaKind, SourceRef
from readeverything.domain.locator_map import LocatorMap, LocatorSegment
from readeverything.domain.locators import BBox, CharSpan
from readeverything.domain.rendition import (
    Budget,
    Degradation,
    ImageContent,
    Rendered,
    Rendition,
    TextContent,
)
from readeverything.ports.source import SourceReader
from readeverything.ports.vision import VisionModel

#: The whole frame, in the normalised coordinates every BBox uses.
WHOLE_IMAGE = BBox(page=None, x=0.0, y=0.0, w=1.0, h=1.0)

_DESCRIBE_PROMPT = "Describe this image in two or three sentences."
_OCR_PROMPT = (
    "Transcribe all text visible in this image, exactly as written. "
    "If there is no text, reply with: (no text)"
)


class DescribeImageParams(BaseModel):
    prompt: str = Field(
        default=_DESCRIBE_PROMPT, description="What to ask the model about the image."
    )


class OcrParams(BaseModel):
    pass


class CropParams(BaseModel):
    x: float = Field(default=0.0, ge=0.0, le=1.0, description="Left edge, 0-1 of image width.")
    y: float = Field(default=0.0, ge=0.0, le=1.0, description="Top edge, 0-1 of image height.")
    w: float = Field(default=1.0, gt=0.0, le=1.0, description="Width, 0-1 of image width.")
    h: float = Field(default=1.0, gt=0.0, le=1.0, description="Height, 0-1 of image height.")


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
        crop = Affordance(
            name="crop_region",
            description=(
                "Return a rectangular region of the image as PNG bytes. "
                "Coordinates are fractions of the image, 0 to 1."
            ),
            params=CropParams,
            requires=frozenset(),
            level=DetailLevel.SEGMENT,
        )
        if self._vision is None:
            return (crop,)
        return (
            crop,
            Affordance(
                name="describe_image",
                description="Describe what is visible in the image, in prose.",
                params=DescribeImageParams,
                requires=frozenset({Capability.VISION}),
                level=DetailLevel.DEEP,
            ),
            Affordance(
                name="ocr",
                description="Transcribe text visible in the image.",
                params=OcrParams,
                requires=frozenset({Capability.VISION}),
                level=DetailLevel.DEEP,
            ),
        )

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

    async def _require_image(self, ref: SourceRef) -> Image.Image:
        image = await self._open(ref)
        if image is None:
            raise DomainError(f"{ref.uri} is not a readable image")
        return image

    async def _see(self, ref: SourceRef, prompt: str) -> str:
        if self._vision is None:
            raise UnknownAffordanceError("describe_image", (a.name for a in self.affordances()))
        data = await self._source.read_bytes(ref.uri)
        text = await self._vision.describe(data, str(ref.mime), prompt)
        if not text.strip():
            # The port's return type is `str`, so a model that answers with
            # nothing is a legal implementation. An empty description must not
            # reach an index as though it were an observation.
            raise InfrastructureError(f"vision model returned no description for {ref.uri}")
        return text

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition:
        match name:
            case "crop_region":
                if not isinstance(params, CropParams):
                    raise TypeError(f"expected CropParams, got {type(params).__name__}")
                image = await self._require_image(ref)
                box = (
                    int(params.x * image.width),
                    int(params.y * image.height),
                    max(
                        int((params.x + params.w) * image.width),
                        int(params.x * image.width) + 1,
                    ),
                    max(
                        int((params.y + params.h) * image.height),
                        int(params.y * image.height) + 1,
                    ),
                )
                buffer = io.BytesIO()
                image.crop(box).save(buffer, format="PNG")
                return Rendition(
                    locator=BBox(page=None, x=params.x, y=params.y, w=params.w, h=params.h),
                    content=ImageContent(data=buffer.getvalue(), mime="image/png"),
                )
            case "describe_image":
                if not isinstance(params, DescribeImageParams):
                    raise TypeError(f"expected DescribeImageParams, got {type(params).__name__}")
                text = await self._see(ref, params.prompt)
                return Rendition(locator=WHOLE_IMAGE, content=TextContent(text))
            case "ocr":
                if not isinstance(params, OcrParams):
                    raise TypeError(f"expected OcrParams, got {type(params).__name__}")
                text = await self._see(ref, _OCR_PROMPT)
                return Rendition(locator=WHOLE_IMAGE, content=TextContent(text))
            case _:
                raise UnknownAffordanceError(name, (a.name for a in self.affordances()))

    async def represent(self, ref: SourceRef, budget: Budget) -> Rendered:
        image = await self._open(ref)
        degradations: tuple[Degradation, ...] = ()
        described = ""
        if image is None:
            facts = f"Unreadable image {ref.uri}, {ref.size_bytes} bytes."
            # Pillow reads a wide format set. If it cannot open these bytes this
            # library has no standing to call them an image, and asking a model
            # to describe them invites a hallucination indexed as an
            # observation. Say what failed and stop.
            degradations = (
                Degradation(
                    what="image undecodable",
                    detail=f"{ref.uri} could not be decoded as an image; no description attempted",
                ),
            )
            return self._rendered(facts, budget, degradations)
        facts = (
            f"Image {ref.uri}, {image.width}x{image.height} "
            f"{image.format or 'unknown'} ({image.mode}), {ref.size_bytes} bytes."
        )
        if self._vision is None:
            degradations = (
                Degradation(
                    what="vision unavailable",
                    detail="no vision model configured; only metadata was indexed",
                ),
            )
        else:
            try:
                described = await self._see(ref, _DESCRIBE_PROMPT)
            except InfrastructureError as exc:
                # An empty or failed completion is not a description. Saying so
                # is better than indexing silence as an observation.
                degradations = (Degradation(what="vision unavailable", detail=str(exc)),)
        full = f"{facts} {described}".strip() if described else facts
        return self._rendered(full, budget, degradations)

    def _rendered(
        self, full: str, budget: Budget, degradations: tuple[Degradation, ...]
    ) -> Rendered:
        """Apply the budget and report what was actually kept.

        A zero-width rendition is inexpressible: `CharSpan(0, 0)` raises, and
        `Rendered` requires `locator_map.length == len(text)`. So a budget of
        zero still keeps one character, and the degradation must report that one
        character rather than the budget it was asked for — otherwise the
        rendition and its own degradation contradict each other.
        """
        text = full
        if budget.max_chars is not None and len(full) > budget.max_chars:
            text = full[: budget.max_chars] or full[:1] or "?"
            degradations = (
                *degradations,
                Degradation(
                    what="text truncated",
                    detail=f"kept {len(text)} of {len(full)} characters",
                ),
            )
        return Rendered(
            text=text,
            locator_map=LocatorMap.build((LocatorSegment(CharSpan(0, len(text)), WHOLE_IMAGE),)),
            barriers=(),
            degradations=degradations,
        )
