import importlib
import io
import sys

import pytest
from PIL import Image

from readeverything.handlers.regions import RegionParams, crop_to_region, region_bbox


def _image(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


def test_default_region_is_the_whole_image() -> None:
    region = RegionParams()
    data = crop_to_region(_image(100, 50), region)
    assert Image.open(io.BytesIO(data)).size == (100, 50)


def test_region_crops_to_the_requested_fraction() -> None:
    region = RegionParams(x=0.25, y=0.25, w=0.5, h=0.5)
    data = crop_to_region(_image(100, 50), region)
    assert Image.open(io.BytesIO(data)).size == (50, 25)


def test_a_sliver_keeps_at_least_one_pixel() -> None:
    """A rectangle that rounds to zero width is inexpressible as an image."""
    region = RegionParams(x=0.0, y=0.0, w=0.001, h=0.001)
    data = crop_to_region(_image(100, 50), region)
    assert Image.open(io.BytesIO(data)).size == (1, 1)


def test_a_region_running_off_the_edge_is_rejected_at_the_boundary() -> None:
    with pytest.raises(ValueError, match="unit square"):
        RegionParams(x=0.8, y=0.0, w=0.5, h=1.0)


def test_crop_returns_png() -> None:
    data = crop_to_region(_image(10, 10), RegionParams())
    assert Image.open(io.BytesIO(data)).format == "PNG"


def test_region_bbox_carries_the_page_when_given() -> None:
    box = region_bbox(RegionParams(x=0.1, y=0.2, w=0.3, h=0.4), page=7)
    assert (box.page, box.x, box.y, box.w, box.h) == (7, 0.1, 0.2, 0.3, 0.4)


def test_region_bbox_has_no_page_by_default() -> None:
    assert region_bbox(RegionParams()).page is None


def test_the_module_and_region_params_stay_usable_without_pillow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`video.py` will import this module while Pillow may be absent, so the
    import itself and the non-cropping API must not require PIL. Only
    `crop_to_region` — which actually needs pixels — may fail without it."""
    monkeypatch.setitem(sys.modules, "PIL", None)
    for module in [m for m in sys.modules if m.startswith("readeverything.handlers.regions")]:
        monkeypatch.delitem(sys.modules, module, raising=False)
    regions = importlib.import_module("readeverything.handlers.regions")
    region = regions.RegionParams(x=0.1, y=0.2, w=0.3, h=0.4)
    assert region.is_whole_frame is False
    assert regions.region_bbox(region).x == 0.1
