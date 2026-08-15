"""Reading text out of a rendered page.

`TextRecognizer` is deliberately narrower than `VisionModel`: it answers one
question, "what text is in this image", rather than an arbitrary prompt. A PDF
handler asking for OCR should not need to know how to phrase a vision prompt,
and a recognizer that wraps a `VisionModel` should not need to expose the
prompt it uses.

`model_id` is not used for dispatch. It feeds `CapabilitySet.fingerprint()`, so
OCR artifacts invalidate when the model behind them changes, while extracted
text — which depends on no model — stays cached.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TextRecognizer(Protocol):
    #: Provider-qualified and versioned. See `VisionModel.model_id` for why.
    model_id: str

    async def recognize(self, image: bytes, mime: str) -> str:
        """Transcribe the text visible in `image`.

        Returns the model's transcription. Raises `InfrastructureError` if the
        model answered with nothing usable.
        """
        ...
