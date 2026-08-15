"""A real directory of mixed files, shared by the integration tier."""

from __future__ import annotations

import io
from pathlib import Path

import pytest


def _png_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    # Small and uniform, not single-pixel: a real vision endpoint can reject a
    # degenerate 1x1 image before the model ever sees it, which looks like an
    # endpoint failure rather than the payload failure it actually is.
    Image.new("RGB", (64, 64), (200, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def media_root(tmp_path: Path) -> Path:
    (tmp_path / "notes.txt").write_text("hello from the integration tier")
    (tmp_path / "data.bin").write_bytes(bytes(range(256)))
    (tmp_path / "photo.png").write_bytes(_png_bytes())
    return tmp_path
