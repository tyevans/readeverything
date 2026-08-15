"""A `VisionModel` over any OpenAI-compatible chat endpoint.

The chat model is injected rather than constructed, so every unit test runs
offline and the one test that touches a network is explicit about it. Use
`build_openai_vision_model` at a composition root to make a real one.

Two failure modes get their own handling because both are common and both are
silent if ignored:

**Empty completions.** Reasoning models split output into a reasoning channel
and a content channel. A model that spends its budget thinking returns empty
content, and returning `""` from here would put an empty string into an index
as though it were an observation about the image. It is a failure, not an
answer.

**Content blocks.** Some providers return a list of typed blocks rather than a
string. Flattening the text blocks is not a nicety — a bare `str()` of the list
would index a Python repr.
"""

from __future__ import annotations

import base64
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from readeverything.domain.errors import InfrastructureError


def _flatten(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return ""


class LangChainVisionModel:
    """Describes images by sending them to a chat model as a data URL."""

    def __init__(self, *, chat: BaseChatModel, model_id: str) -> None:
        self._chat = chat
        self.model_id = model_id

    async def describe(self, data: bytes, mime: str, prompt: str) -> str:
        encoded = base64.b64encode(data).decode("ascii")
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                },
            ]
        )
        try:
            response = await self._chat.ainvoke([message])
        except Exception as exc:
            raise InfrastructureError(f"vision model call failed: {exc}") from exc
        text = _flatten(response.content).strip()
        if not text:
            raise InfrastructureError(
                f"vision model {self.model_id} returned an empty completion; "
                f"a reasoning model may have spent its budget before answering"
            )
        return text


def build_openai_vision_model(
    *,
    base_url: str,
    model: str,
    api_key: str = "not-needed",
    timeout_s: float = 120.0,
    max_tokens: int = 1024,
) -> LangChainVisionModel:
    """Build a vision model against an OpenAI-compatible endpoint.

    Every value is an argument. Nothing here reads the environment — a caller
    running two differently-configured instances in one process must be able to,
    and `test_reads_no_environment` enforces it.

    `model_id` is derived as `openai/{model}` so the capability fingerprint is
    provider-qualified rather than a bare family name.
    """
    from langchain_openai import ChatOpenAI

    chat = ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,  # type: ignore[arg-type]
        timeout=timeout_s,
        max_completion_tokens=max_tokens,
    )
    return LangChainVisionModel(chat=chat, model_id=f"openai/{model}")
