"""`AudioHandler` reached through `build_perception`, not constructed by hand."""

from __future__ import annotations

from pathlib import Path

import pytest

from readeverything.composition import build_perception
from tests.fixtures_media import ffmpeg_available

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not ffmpeg_available(), reason="requires ffmpeg")
async def test_read_span_is_not_offered_without_a_transcriber(media_root: Path) -> None:
    """Negotiation, not a runtime apology."""
    perception = await build_perception(media_root)
    card = await perception.inspect("clip.wav")
    assert "read_span" not in {a.name for a in card.affordances}
