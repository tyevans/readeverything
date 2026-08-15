"""A `ClipModel` over llama.cpp's `input_video` content part.

`input_video` is a llama.cpp extension, not part of OpenAI's schema. The
OpenAI-shaped alternative is `video_url`, which is what vLLM accepts and what
Qwen's own model card documents; the two servers do not agree, and a client
that guesses wrong gets a 400 rather than a graceful degradation. This adapter
speaks llama.cpp's dialect because that is what the endpoint this project
targets speaks.

Verified working against llama.cpp b10438 on 2026-08-15. Before that build the
identical request failed with "Failed to load image or audio file" — the
container decoder llama.cpp shells out to was not reachable from the server
process. Worth knowing, because the failure looks like a malformed request and
is not one: the same bytes that fail on one build succeed on the next with no
client change.

The empty-completion and content-block handling is `vision_langchain`'s,
imported rather than copied. The failure modes are identical and a second copy
would drift apart at the first fix.
"""

from __future__ import annotations

import base64

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from readeverything.adapters.vision_langchain import _flatten
from readeverything.domain.errors import InfrastructureError


class LangChainClipModel:
    """Describes a clip by sending it as an `input_video` content part."""

    def __init__(self, *, chat: BaseChatModel, model_id: str) -> None:
        self._chat = chat
        self.model_id = model_id

    async def watch(self, clip: bytes, mime: str, prompt: str) -> str:
        encoded = base64.b64encode(clip).decode("ascii")
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                # The mime is not sent. llama.cpp sniffs the container itself
                # and takes no type hint here, so passing one would be a field
                # the server ignores and a reader would believe.
                {"type": "input_video", "input_video": {"data": encoded}},
            ]
        )
        try:
            response = await self._chat.ainvoke([message])
        except Exception as exc:
            raise InfrastructureError(f"clip model call failed: {exc}") from exc
        flattened = _flatten(response.content)
        if flattened is None:
            raise InfrastructureError(
                f"clip model {self.model_id} returned an unrecognised content shape: "
                f"{type(response.content).__name__}"
            )
        text = flattened.strip()
        if not text:
            raise InfrastructureError(
                f"clip model {self.model_id} returned an empty completion; "
                f"a reasoning model may have spent its budget before answering"
            )
        return text


def build_openai_clip_model(
    *,
    base_url: str,
    model: str,
    api_key: str = "not-needed",
    timeout_s: float = 300.0,
    max_tokens: int = 1500,
    thinking: bool = False,
) -> LangChainClipModel:
    """Build a clip model against an OpenAI-compatible endpoint.

    Every value is an argument. Nothing here reads the environment — a caller
    running two differently-configured instances in one process must be able
    to, and `test_reads_no_environment` enforces it.

    `timeout_s` and `max_tokens` both default higher than
    `build_openai_vision_model`'s, for the same reason: a clip is many frames
    of prompt. A 30-second clip is roughly 65,000 prompt tokens, which takes
    the server appreciably longer to read than one still, and gives a reasoning
    model appreciably more to think about before it answers.

    `thinking` defaults to False, as it does for vision. Reasoning about a clip
    is budget spent deciding how to describe it rather than describing it, and
    an exhausted budget arrives here as an empty completion — a paid-for call
    that returns nothing.
    """
    from langchain_openai import ChatOpenAI

    chat = ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,  # type: ignore[arg-type]
        timeout=timeout_s,
        max_completion_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )
    return LangChainClipModel(chat=chat, model_id=f"openai/{model}")
