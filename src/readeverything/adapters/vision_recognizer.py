"""`TextRecognizer` over an existing `VisionModel`.

This adapter renders nothing itself. A caller — the PDF handler, for a scanned
page — renders the page and hands over the bytes; this adapter's only job is
phrasing the OCR prompt and forwarding to the wrapped model, matching how
`ImageHandler`'s own OCR prompt works.
"""

from __future__ import annotations

from readeverything.ports.vision import VisionModel

_OCR_PROMPT = (
    "Transcribe all text visible in this image, exactly as written. "
    "If there is no text, reply with: (no text)"
)


class VisionTextRecognizer:
    """Wraps a `VisionModel` to answer the narrower `TextRecognizer` protocol."""

    def __init__(self, *, vision: VisionModel) -> None:
        self._vision = vision
        #: The wrapped model's id, so OCR artifacts invalidate when it changes.
        self.model_id = vision.model_id

    async def recognize(self, image: bytes, mime: str) -> str:
        return await self._vision.describe(image, mime, _OCR_PROMPT)
