"""Does the real transcriber behave the way the design assumes?

Marked `live` and deselected by default. Additionally, both tests here skip
whenever `READEVERYTHING_LIVE_WHISPER_MODEL_DIR` is unset — which on this
machine is always, since no whisper weights are present and none are
downloaded, implicitly or otherwise. The skip is the honest state: this
machine's real experience is the degraded no-transcriber path, and that path
is fully covered at the unit level (see `tests/handlers/` / `tests/adapters/`
for `_EmptyTranscriber` and `FakeTranscriber` coverage).

To run these locally, point the env var at a directory holding a **converted
CTranslate2** whisper model — not a raw Hugging Face checkout. `faster-whisper`
loads CT2 models; convert one with `ct2-transformers-converter` first, e.g.:

    ct2-transformers-converter --model openai/whisper-base.en \\
        --output_dir /path/to/whisper-base.en-ct2

Then:

    READEVERYTHING_LIVE_WHISPER_MODEL_DIR=/path/to/whisper-base.en-ct2 \\
        uv run pytest tests/live/test_transcription.py -m live -v

These assert on STRUCTURE, never on transcribed text. The audio fixture is a
generated sine tone (see `tests/fixtures_media.py`), which is not speech — a
real model will return zero or nonsense cues against it, and that is fine.
What must hold is that whatever comes back is ordered, non-degenerate, and
locatable in time; word accuracy is a bench concern, not a test concern.
"""

from __future__ import annotations

import pytest

from readeverything.adapters.cache_key import artifact_key
from readeverything.adapters.whisper_transcriber import WhisperTranscriber
from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.domain.identity import ContentHash
from tests.fixtures_media import audio_only, ffmpeg_available

pytestmark = pytest.mark.live


def _require_model_dir(model_dir: str | None) -> str:
    if model_dir is None:
        pytest.skip(
            "no whisper model configured: set READEVERYTHING_LIVE_WHISPER_MODEL_DIR "
            "to a directory holding a converted CTranslate2 whisper model "
            "(faster-whisper needs a CT2 conversion, not a raw Hugging Face "
            "checkout) to run this test"
        )
    return model_dir


async def test_a_real_transcriber_returns_located_cues(
    live_whisper_model_dir: str | None,
) -> None:
    """Structure only, never the words.

    Model quality is a bench concern; what must be true is that cues came
    back are ordered and non-degenerate, and that each carries a `TimeSpan`
    that could locate it in the file.

    The fixture is a sine tone, not speech, so a real transcriber may
    legitimately return zero cues here. This test therefore establishes only
    that the call completes and that whatever comes back is well-formed — it
    does NOT establish that the transcriber can recognise speech at all.
    """
    model_dir = _require_model_dir(live_whisper_model_dir)
    if not ffmpeg_available():
        pytest.skip("ffmpeg not available to generate the audio fixture")

    transcriber = WhisperTranscriber(model_dir=model_dir)
    audio = audio_only(seconds=3)

    cues = await transcriber.transcribe(audio, "audio/wav")

    assert isinstance(cues, tuple)
    previous_end = 0.0
    for cue in cues:
        assert cue.span.start_s >= previous_end
        assert cue.span.end_s > cue.span.start_s
        assert isinstance(cue.text, str)
        previous_end = cue.span.end_s


async def test_swapping_the_model_changes_the_cache_key(
    live_whisper_model_dir: str | None,
) -> None:
    """A transcript is keyed on the ASR model that produced it, so an index
    cannot silently mix two models' readings of the same audio.

    `Transcriber.model_id` feeds the capability fingerprint, which feeds
    `artifact_key`. Two transcribers with different model ids must produce
    different keys for the same audio. This needs no network — mirrors
    `test_vision_endpoint.py::test_swapping_the_model_changes_every_cache_key`.
    """
    _require_model_dir(live_whisper_model_dir)

    def key_for(model_id: str) -> str:
        return artifact_key(
            content_hash=ContentHash("a" * 64),
            handler_id="audio",
            handler_version=1,
            affordance="transcribe",
            params={},
            capabilities=CapabilitySet.of({Capability.ASR: model_id}),
        )

    model_id = f"faster-whisper/{live_whisper_model_dir}@int8"
    assert key_for(model_id) != key_for("faster-whisper/some-other-model@int8")
    assert key_for(model_id) == key_for(model_id)
