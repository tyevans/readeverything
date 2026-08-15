from readeverything.adapters.vision_recognizer import VisionTextRecognizer
from readeverything.testing.fakes import FakeVision


async def test_the_recognizer_carries_the_model_id_it_recognises_with() -> None:
    """It feeds the capability fingerprint, so OCR artifacts invalidate when
    the model changes while extracted text — which does not depend on a model
    — stays cached."""
    recognizer = VisionTextRecognizer(vision=FakeVision())
    assert recognizer.model_id == FakeVision().model_id


async def test_recognize_forwards_the_image_to_the_wrapped_vision_model() -> None:
    vision = FakeVision()
    recognizer = VisionTextRecognizer(vision=vision)
    text = await recognizer.recognize(b"\x89PNG", "image/png")
    assert vision.calls == 1
    assert "image/png" in text
