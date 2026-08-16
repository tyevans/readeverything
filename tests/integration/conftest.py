"""A real directory of mixed files, shared by the integration tier."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from tests.fixtures_media import audio_only, ffmpeg_available, video_with_audio
from tests.fixtures_office import (
    docx_bytes,
    odp_bytes,
    ods_bytes,
    odt_bytes,
    pptx_bytes,
    xlsx_bytes,
)
from tests.fixtures_pdf import blank, born_digital, scanned_like


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
    if ffmpeg_available():
        (tmp_path / "clip.mp4").write_bytes(video_with_audio())
        (tmp_path / "clip.wav").write_bytes(audio_only())
    return tmp_path


@pytest.fixture
def documents_root(tmp_path: Path) -> Path:
    (tmp_path / "report.pdf").write_bytes(
        born_digital(["Section one.", "Section two.", "Section three."])
    )
    (tmp_path / "scan.pdf").write_bytes(scanned_like())
    (tmp_path / "blank.pdf").write_bytes(blank())
    return tmp_path


@pytest.fixture
def office_root(tmp_path: Path) -> Path:
    """One of each office family — Spec 9 §1.1's acceptance scenario."""
    (tmp_path / "policy.docx").write_bytes(docx_bytes(comment="Check this number."))
    (tmp_path / "deck.pptx").write_bytes(pptx_bytes(picture_on=(2,)))
    (tmp_path / "book.xlsx").write_bytes(xlsx_bytes(formulas=True, cached=True))
    (tmp_path / "notes.odt").write_bytes(odt_bytes())
    (tmp_path / "slides.odp").write_bytes(odp_bytes())
    (tmp_path / "sheet.ods").write_bytes(ods_bytes())
    return tmp_path
