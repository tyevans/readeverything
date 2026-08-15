"""Rectangles, shared by every handler that can hand pixels to a model.

The unit-square check lived in `image.py` alone, which meant a PDF page region
and a video frame region had no boundary validation at all — a caller's
mistake surfaced as a bare `ValueError` from inside `BBox`, mid-crop, rather
than as a rejected parameter at the edge.

Cropping does NOT reduce what a vision call costs. Measured 2026-08-15 against
qwen3.8-27b-mtp, a 720x480 frame and a 72x48 crop of it both cost 1,140 prompt
tokens: the server resizes to a fixed grid, so cost is per image, not per
pixel. A region is worth asking for because it puts the rectangle in the
locator instead of in prose, not because it saves anything.
"""

from __future__ import annotations

import io

from PIL import Image
from pydantic import BaseModel, Field, model_validator

from readeverything.domain.locators import BBox


class RegionParams(BaseModel):
    """A rectangle in normalised coordinates, defaulting to the whole frame."""

    x: float = Field(default=0.0, ge=0.0, le=1.0, description="Left edge, 0-1 of width.")
    y: float = Field(default=0.0, ge=0.0, le=1.0, description="Top edge, 0-1 of height.")
    w: float = Field(default=1.0, gt=0.0, le=1.0, description="Width, 0-1 of width.")
    h: float = Field(default=1.0, gt=0.0, le=1.0, description="Height, 0-1 of height.")

    @model_validator(mode="after")
    def _stay_inside_the_frame(self) -> RegionParams:
        """A crop running off the edge is a parameter error, so reject it here.

        `BBox` catches it too, but only once the crop is already running — the
        caller then sees a bare `ValueError` from deep inside the domain rather
        than a rejection at the boundary where their mistake was. The `BBox`
        check stays as the backstop for every other path that builds one.
        """
        if self.x + self.w > 1.0 or self.y + self.h > 1.0:
            raise ValueError(
                f"crop must be within the unit square, got x={self.x} y={self.y} "
                f"w={self.w} h={self.h}"
            )
        return self

    @property
    def is_whole_frame(self) -> bool:
        return (self.x, self.y, self.w, self.h) == (0.0, 0.0, 1.0, 1.0)


def crop_to_region(image: Image.Image, region: RegionParams) -> bytes:
    """`image` cropped to `region`, as PNG bytes.

    The `max(..., + 1)` guards keep a thin rectangle from rounding to zero
    width or height: PIL would accept the degenerate box and produce an image
    no locator can describe.
    """
    box = (
        int(region.x * image.width),
        int(region.y * image.height),
        max(int((region.x + region.w) * image.width), int(region.x * image.width) + 1),
        max(int((region.y + region.h) * image.height), int(region.y * image.height) + 1),
    )
    buffer = io.BytesIO()
    image.crop(box).save(buffer, format="PNG")
    return buffer.getvalue()


def region_bbox(region: RegionParams, page: int | None = None) -> BBox:
    """The locator for `region`, on `page` when the medium has pages."""
    return BBox(page=page, x=region.x, y=region.y, w=region.w, h=region.h)
