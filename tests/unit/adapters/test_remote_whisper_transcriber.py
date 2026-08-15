"""The remote transcriber against a mocked whisper.cpp server.

The bodies here are copied from a real `/inference` response (see the adapter's
module docstring for the captured shape), so the parsing asserted below is
parsing of something that actually came off the wire — including the
zero-width final segment, which is not a hypothetical.
"""

from __future__ import annotations

import json

import httpx
import pytest

from readeverything.adapters.remote_whisper_transcriber import RemoteWhisperTranscriber
from readeverything.domain.errors import InfrastructureError
from readeverything.ports.transcription import Transcriber

AUDIO = b"RIFF\x00\x00\x00\x00WAVEfmt "

#: Trimmed from a real response: two usable segments and one zero-width.
VERBOSE_JSON = {
    "task": "transcribe",
    "language": "english",
    "duration": 20.0,
    "segments": [
        {"id": 0, "text": " Music", "start": 0.0, "end": 16.0, "avg_logprob": -0.44},
        {
            "id": 1,
            "text": " When I say computer, you probably think of something",
            "start": 16.0,
            "end": 20.0,
            "avg_logprob": -0.08,
        },
        {"id": 2, "text": " extremely", "start": 20.0, "end": 20.0, "avg_logprob": -0.03},
    ],
}


def _transcriber(
    handler: object, *, base_url: str = "http://whisper.invalid:8083"
) -> RemoteWhisperTranscriber:
    return RemoteWhisperTranscriber(
        base_url=base_url,
        model_id="whisper.cpp/large-v3@q5_0",
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
    )


def _replying(payload: object, status: int = 200) -> RemoteWhisperTranscriber:
    return _transcriber(lambda request: httpx.Response(status, json=payload))


def test_it_satisfies_the_port() -> None:
    assert isinstance(_replying(VERBOSE_JSON), Transcriber)


async def test_it_returns_located_cues() -> None:
    cues = await _replying(VERBOSE_JSON).transcribe(AUDIO, "audio/wav")

    assert [(c.span.start_s, c.span.end_s) for c in cues] == [(0.0, 16.0), (16.0, 20.0)]
    assert cues[1].text == "When I say computer, you probably think of something"


async def test_it_drops_the_zero_width_segment_the_real_server_emits() -> None:
    """`TimeSpan` rejects a zero-width span and widening one would assert a
    duration nothing observed, so the segment is dropped rather than repaired."""
    cues = await _replying(VERBOSE_JSON).transcribe(AUDIO, "audio/wav")

    assert all(c.span.end_s > c.span.start_s for c in cues)
    assert "extremely" not in " ".join(c.text for c in cues)


async def test_confidence_is_never_populated_from_a_log_probability() -> None:
    """`avg_logprob` is whisper's own diagnostic, not a measured certainty."""
    cues = await _replying(VERBOSE_JSON).transcribe(AUDIO, "audio/wav")

    assert all(c.confidence is None for c in cues)


async def test_it_posts_the_multipart_shape_the_server_documents() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.content
        return httpx.Response(200, json=VERBOSE_JSON)

    await _transcriber(handler).transcribe(AUDIO, "audio/wav")

    assert seen["url"] == "http://whisper.invalid:8083/inference"
    body = bytes(seen["body"])  # type: ignore[arg-type]
    # verbose_json is the only format carrying per-segment start/end.
    assert b'name="response_format"\r\n\r\nverbose_json' in body
    assert b'name="temperature"\r\n\r\n0.0' in body
    assert b'name="file"' in body
    assert AUDIO in body


async def test_a_trailing_slash_on_the_base_url_does_not_double_up() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=VERBOSE_JSON)

    await _transcriber(handler, base_url="http://whisper.invalid:8083/").transcribe(
        AUDIO, "audio/wav"
    )

    assert seen == ["http://whisper.invalid:8083/inference"]


async def test_an_empty_segment_list_is_an_empty_transcript_not_an_error() -> None:
    cues = await _replying({"segments": []}).transcribe(AUDIO, "audio/wav")
    assert cues == ()


async def test_a_non_2xx_answer_raises_infrastructure_error() -> None:
    with pytest.raises(InfrastructureError):
        await _replying(VERBOSE_JSON, status=500).transcribe(AUDIO, "audio/wav")


async def test_a_transport_failure_raises_infrastructure_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    with pytest.raises(InfrastructureError):
        await _transcriber(handler).transcribe(AUDIO, "audio/wav")


async def test_a_body_without_segments_raises_rather_than_reading_as_silence() -> None:
    """A misconfigured `response_format` must not look like a file with
    nothing in it."""
    with pytest.raises(InfrastructureError):
        await _replying({"text": "one blob, no timings"}).transcribe(AUDIO, "audio/wav")


async def test_a_malformed_segment_raises() -> None:
    with pytest.raises(InfrastructureError):
        await _replying({"segments": [{"text": "no timings here"}]}).transcribe(AUDIO, "audio/wav")


async def test_non_json_raises() -> None:
    transcriber = _transcriber(
        lambda request: httpx.Response(200, content=b"<html>not json</html>")
    )
    with pytest.raises(InfrastructureError):
        await transcriber.transcribe(AUDIO, "audio/wav")


def test_the_model_id_is_the_callers_to_state() -> None:
    """The server names no model, so the adapter must not invent one: the id
    feeds the capability fingerprint and therefore every cache key."""
    assert _replying(VERBOSE_JSON).model_id == "whisper.cpp/large-v3@q5_0"
    assert json.dumps(VERBOSE_JSON)  # the fixture is a real JSON body
