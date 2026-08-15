"""`Transcriber` over a remote `whisper.cpp` server.

The sibling of `whisper_transcriber.py`, and its opposite in one respect: that
adapter refuses the network on principle, this one is nothing but network. Both
satisfy the same port, so the composition root chooses which machine does the
work without any handler learning about it.

THE WIRE, verified against a running whisper.cpp server at
`http://192.168.1.14:8083` on 2026-08-15, not assumed from documentation:

    POST /inference
      multipart/form-data:
        file             the audio bytes
        temperature      "0.0"
        response_format  "verbose_json"

    -> 200, application/json:
       {"task": "transcribe", "language": "english", "duration": 20.0,
        "text": " ...", "segments": [{"id": 0, "text": " Music",
        "start": 0.0, "end": 16.0, "words": [...], "avg_logprob": -0.44,
        "no_speech_prob": 1.9e-05}, ...],
        "detected_language": "english", ...}

Note the endpoint is `/inference`, NOT OpenAI's `/v1/audio/transcriptions` —
whisper.cpp's server serves 404 there, and `/v1/models` does not exist either,
which is why `model_id` is a required constructor argument below. Only
`response_format=verbose_json` carries per-segment `start`/`end`; the default
`json` returns a single blob of text, which would leave every cue unlocatable
in time and defeat the point of a `TranscriptCue`.

Two behaviours of the real server that this adapter must absorb:

  * **Zero-width segments are real.** The 20-second sample above ended with
    `{"start": 20.0, "end": 20.0, "text": " extremely"}`. `TimeSpan` rejects a
    zero-width span, and widening it would assert a duration nothing observed,
    so such segments are dropped — the same rule, for the same reason, as
    `whisper_transcriber.py`.
  * **Word `probability` is not confidence.** The response carries per-word
    probabilities and per-segment `avg_logprob`. Neither is a measured
    certainty about the transcript, so `confidence` stays `None`, matching the
    local adapter exactly. A number here would be a fabrication with a decimal
    point on it.

`InfrastructureError` is raised on transport failure, non-2xx, unparseable
JSON, and malformed segments. The port permits this; the HANDLER is what must
never raise, and it is what catches this and degrades.
"""

from __future__ import annotations

from typing import Any

import httpx

from readeverything.domain.errors import InfrastructureError
from readeverything.domain.locators import TimeSpan
from readeverything.domain.rendition import TranscriptCue


class RemoteWhisperTranscriber:
    """Speech-to-text against a whisper.cpp server over HTTP."""

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        timeout_s: float = 300.0,
        temperature: float = 0.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """`model_id` is required and has no default.

        whisper.cpp's server exposes no endpoint that names the weights it
        loaded — `/v1/models` is a 404 — so this adapter cannot discover what
        actually ran. `model_id` feeds `CapabilitySet.fingerprint()` and
        therefore every artifact cache key derived from it: a default here
        would let two different models' transcripts share a key silently.
        Making the caller state it keeps the fingerprint honest by
        construction, since only the caller knows what the server was started
        with.

        `transport` is a seam for tests: an `httpx.MockTransport` lets the
        wire format above be asserted without a server. Production leaves it
        `None` and gets httpx's default.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._temperature = temperature
        self._transport = transport
        self.model_id = model_id

    async def transcribe(self, audio: bytes, mime: str) -> tuple[TranscriptCue, ...]:
        """Transcribe `audio` into ordered, non-overlapping cues.

        Raises `InfrastructureError` if the server could not be reached, did
        not answer 2xx, or answered with something that is not the documented
        shape. `mime` is passed through as the upload's content type; the
        server sniffs the container itself.
        """
        url = f"{self._base_url}/inference"
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_s, transport=self._transport
            ) as client:
                response = await client.post(
                    url,
                    files={"file": ("audio", audio, mime)},
                    data={
                        "temperature": str(self._temperature),
                        "response_format": "verbose_json",
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            raise InfrastructureError(f"whisper.cpp server at {url} failed: {exc}") from exc
        except ValueError as exc:
            raise InfrastructureError(
                f"whisper.cpp server at {url} answered with non-JSON: {exc}"
            ) from exc

        return _cues_from(payload, url)


def _cues_from(payload: Any, url: str) -> tuple[TranscriptCue, ...]:
    """Segments out of a verbose_json body, or `InfrastructureError`.

    A body without `segments` is an error rather than an empty transcript:
    silence and "the server answered in a shape this adapter does not
    understand" must not look the same to a caller, or a configuration mistake
    reads as a file with nothing in it.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        raise InfrastructureError(
            f"whisper.cpp server at {url} answered without a 'segments' list — "
            "response_format=verbose_json is required for located cues"
        )

    cues: list[TranscriptCue] = []
    for segment in payload["segments"]:
        try:
            start = float(segment["start"])
            end = float(segment["end"])
            text = str(segment["text"])
        except (TypeError, KeyError, ValueError) as exc:
            raise InfrastructureError(
                f"whisper.cpp server at {url} returned a malformed segment: {segment!r}"
            ) from exc
        # See the module docstring: the real server emits start == end.
        if start >= end:
            continue
        cues.append(
            TranscriptCue(
                span=TimeSpan(start, end),
                text=text.strip(),
                speaker=None,
                confidence=None,
            )
        )
    return tuple(cues)
