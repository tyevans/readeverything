import base64

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from readeverything.adapters.vision_langchain import (
    LangChainVisionModel,
    build_openai_vision_model,
)
from readeverything.domain.errors import InfrastructureError
from readeverything.ports.vision import VisionModel

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8


class _RecordingChat:
    """A stand-in chat model that records what it was sent."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.sent: list[BaseMessage] = []

    async def ainvoke(self, messages: list[BaseMessage], **kwargs: object) -> AIMessage:
        self.sent = list(messages)
        return AIMessage(content=self._reply)


def _model(reply: str = "a small green square") -> tuple[LangChainVisionModel, _RecordingChat]:
    chat = _RecordingChat(reply)
    return LangChainVisionModel(chat=chat, model_id="test/model@1"), chat  # type: ignore[arg-type]


def test_it_satisfies_the_port() -> None:
    model, _ = _model()
    assert isinstance(model, VisionModel)


async def test_it_returns_the_model_text() -> None:
    model, _ = _model("a small green square")
    assert await model.describe(PNG, "image/png", "what is this") == "a small green square"


async def test_the_image_is_sent_as_a_base64_data_url() -> None:
    """The endpoint takes data URLs, not file paths — the bytes must be inlined."""
    model, chat = _model()
    await model.describe(PNG, "image/png", "what is this")
    content = chat.sent[0].content
    assert isinstance(content, list)
    blocks = [part for part in content if isinstance(part, dict)]
    image_parts = [part for part in blocks if part.get("type") == "image_url"]
    assert len(image_parts) == 1
    url = image_parts[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == PNG


async def test_the_prompt_is_sent_alongside_the_image() -> None:
    model, chat = _model()
    await model.describe(PNG, "image/png", "count the squares")
    content = chat.sent[0].content
    assert isinstance(content, list)
    blocks = [part for part in content if isinstance(part, dict)]
    text_parts = [part for part in blocks if part.get("type") == "text"]
    assert text_parts[0]["text"] == "count the squares"


async def test_an_empty_completion_raises_rather_than_returning_nothing() -> None:
    """A reasoning model that spends its budget thinking returns empty content.

    Returning "" would enter the index as an observation about the image.
    """
    model, _ = _model("")
    with pytest.raises(InfrastructureError, match="empty completion"):
        await model.describe(PNG, "image/png", "what is this")


async def test_a_whitespace_only_completion_also_raises() -> None:
    model, _ = _model("   \n  ")
    with pytest.raises(InfrastructureError, match="empty completion"):
        await model.describe(PNG, "image/png", "what is this")


async def test_a_list_shaped_completion_is_flattened() -> None:
    """Some providers return content blocks rather than a bare string."""
    chat = _RecordingChat("")

    async def _blocks(messages: list[BaseMessage], **kwargs: object) -> AIMessage:
        return AIMessage(content=[{"type": "text", "text": "a green square"}])

    chat.ainvoke = _blocks  # type: ignore[method-assign]
    model = LangChainVisionModel(chat=chat, model_id="test/model@1")  # type: ignore[arg-type]
    assert await model.describe(PNG, "image/png", "what") == "a green square"


async def test_a_transport_failure_becomes_an_infrastructure_error() -> None:
    class _Failing:
        async def ainvoke(self, messages: list[BaseMessage], **kwargs: object) -> AIMessage:
            raise ConnectionError("refused")

    model = LangChainVisionModel(chat=_Failing(), model_id="test/model@1")  # type: ignore[arg-type]
    with pytest.raises(InfrastructureError, match="vision model call failed"):
        await model.describe(PNG, "image/png", "what is this")


async def test_an_unrecognised_content_shape_is_reported_as_such() -> None:
    """Not an 'empty completion' — that diagnosis would send a debugger the wrong way.

    `AIMessage` itself validates `content` as `str | list`, so a genuinely
    unrecognised shape (an int) can only arrive via a chat model that skips
    that validation — exactly the kind of loosely-typed third-party response
    this adapter must tolerate.
    """

    class _Response:
        content = 42

    class _Weird:
        async def ainvoke(self, messages: list[BaseMessage], **kwargs: object) -> _Response:
            return _Response()

    model = LangChainVisionModel(chat=_Weird(), model_id="test/model@1")  # type: ignore[arg-type]
    with pytest.raises(InfrastructureError, match="unrecognised content shape"):
        await model.describe(PNG, "image/png", "what is this")


async def test_a_list_of_only_reasoning_blocks_is_an_empty_completion() -> None:
    """A recognised shape that contains no text IS the reasoning-budget case."""

    class _Reasoning:
        async def ainvoke(self, messages: list[BaseMessage], **kwargs: object) -> AIMessage:
            return AIMessage(content=[{"type": "reasoning", "text": "thinking..."}])

    model = LangChainVisionModel(chat=_Reasoning(), model_id="test/model@1")  # type: ignore[arg-type]
    with pytest.raises(InfrastructureError, match="empty completion"):
        await model.describe(PNG, "image/png", "what is this")


async def test_a_list_of_bare_strings_is_flattened() -> None:
    class _Strings:
        async def ainvoke(self, messages: list[BaseMessage], **kwargs: object) -> AIMessage:
            return AIMessage(content=["a green ", "square"])

    model = LangChainVisionModel(chat=_Strings(), model_id="test/model@1")  # type: ignore[arg-type]
    assert await model.describe(PNG, "image/png", "what") == "a green square"


async def test_the_factory_passes_the_endpoint_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo in base_url or model_id would otherwise only surface live."""
    captured: dict[str, object] = {}

    class _FakeChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("langchain_openai.ChatOpenAI", _FakeChatOpenAI)
    model = build_openai_vision_model(base_url="http://x/v1/", model="m")
    assert captured["base_url"] == "http://x/v1/"
    assert captured["model"] == "m"
    assert model.model_id == "openai/m"


def test_thinking_is_off_unless_a_caller_asks_for_it() -> None:
    """A reasoning model asked to describe a picture spends its budget
    deciding how to describe the picture. Measured against the live server, a
    two-frame call with a 300-token budget produced 300 tokens of reasoning
    and no answer — which this adapter then correctly raises on as an empty
    completion, having paid for the call.

    Describing an image is not a task reasoning improves, so the default
    departs from the model's own.
    """
    model = build_openai_vision_model(base_url="http://localhost:1/v1", model="m")
    assert model._chat.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_a_caller_can_ask_for_the_reasoning_channel_back() -> None:
    model = build_openai_vision_model(base_url="http://localhost:1/v1", model="m", thinking=True)
    assert model._chat.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
