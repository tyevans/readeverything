import base64

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from readeverything.adapters.clip_langchain import (
    LangChainClipModel,
    build_openai_clip_model,
)
from readeverything.domain.errors import InfrastructureError
from readeverything.ports.clips import ClipModel

CLIP = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 16


class _RecordingChat:
    """A stand-in chat model that records what it was sent."""

    def __init__(self, reply: object) -> None:
        self._reply = reply
        self.sent: list[BaseMessage] = []

    async def ainvoke(self, messages: list[BaseMessage], **kwargs: object) -> AIMessage:
        self.sent = list(messages)
        return AIMessage(content=self._reply)  # type: ignore[arg-type]


def _model(
    reply: object = "a rainbow band scrolls left to right",
) -> tuple[LangChainClipModel, _RecordingChat]:
    chat = _RecordingChat(reply)
    return LangChainClipModel(chat=chat, model_id="test/model@1"), chat  # type: ignore[arg-type]


def test_it_satisfies_the_port() -> None:
    model, _ = _model()
    assert isinstance(model, ClipModel)


async def test_it_returns_the_model_text() -> None:
    model, _ = _model("a rainbow band scrolls left to right")
    answer = await model.watch(CLIP, "video/mp4", "what changes?")
    assert answer == "a rainbow band scrolls left to right"


async def test_the_clip_is_sent_as_an_input_video_part() -> None:
    """`input_video` is llama.cpp's content-part type, NOT OpenAI's
    `video_url` — the two servers do not agree, and vLLM takes the other one.
    Verified against the live server on 2026-08-15; before that build the same
    request returned "Failed to load image or audio file" because ffmpeg was
    not reachable from the server process.
    """
    model, chat = _model()
    await model.watch(CLIP, "video/mp4", "what changes?")
    content = chat.sent[0].content
    assert isinstance(content, list)
    parts = [part for part in content if isinstance(part, dict)]
    video = [part for part in parts if part.get("type") == "input_video"]
    assert len(video) == 1
    assert video[0]["input_video"]["data"] == base64.b64encode(CLIP).decode("ascii")


async def test_the_prompt_travels_with_the_clip() -> None:
    model, chat = _model()
    await model.watch(CLIP, "video/mp4", "what changes?")
    text = [
        part
        for part in chat.sent[0].content
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    assert text[0]["text"] == "what changes?"


async def test_an_empty_completion_is_a_failure_not_an_answer() -> None:
    """More likely here than for a still, not less: a clip is many frames of
    prompt, so a reasoning model has more to think about before it starts
    writing. Measured, a two-frame call with a 300-token budget spent all 300
    reasoning and returned empty content."""
    model, _ = _model("")
    with pytest.raises(InfrastructureError, match="empty completion"):
        await model.watch(CLIP, "video/mp4", "?")


async def test_content_blocks_are_flattened_not_repr_ed() -> None:
    model, _ = _model([{"type": "text", "text": "a "}, {"type": "text", "text": "clip"}])
    assert await model.watch(CLIP, "video/mp4", "?") == "a clip"


async def test_an_unrecognised_content_shape_is_its_own_failure() -> None:
    """Kept distinct from an empty completion: collapsing the two would send
    someone debugging an unfamiliar server toward reasoning budgets instead of
    toward the actual problem.

    The response is a bare object rather than an `AIMessage`, because
    `AIMessage` validates its own content and would reject the shape before the
    adapter ever saw it. A server returning something langchain never modelled
    is exactly the case this guard exists for.
    """

    class _Odd:
        content = 42

    class _OddChat:
        async def ainvoke(self, messages: list[BaseMessage], **kwargs: object) -> _Odd:
            return _Odd()

    model = LangChainClipModel(chat=_OddChat(), model_id="test/model@1")  # type: ignore[arg-type]
    with pytest.raises(InfrastructureError, match="unrecognised content shape"):
        await model.watch(CLIP, "video/mp4", "?")


async def test_a_failing_call_becomes_an_infrastructure_error() -> None:
    class _Boom:
        async def ainvoke(self, messages: list[BaseMessage], **kwargs: object) -> AIMessage:
            raise RuntimeError("connection reset")

    model = LangChainClipModel(chat=_Boom(), model_id="test/model@1")  # type: ignore[arg-type]
    with pytest.raises(InfrastructureError, match="clip model call failed"):
        await model.watch(CLIP, "video/mp4", "?")


def test_the_builder_qualifies_the_model_id_by_provider() -> None:
    model = build_openai_clip_model(base_url="http://localhost:1/v1", model="qwen")
    assert model.model_id == "openai/qwen"


def test_the_builder_turns_thinking_off() -> None:
    """Same reason as the vision builder: reasoning about a clip is budget
    spent on deciding how to describe it rather than on describing it, and an
    exhausted budget arrives here as an empty completion."""
    model = build_openai_clip_model(base_url="http://localhost:1/v1", model="qwen")
    assert model._chat.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_the_builder_reads_no_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every value is an argument, so two differently-configured instances can
    run in one process."""
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-be-used")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://elsewhere/v1")
    model = build_openai_clip_model(
        base_url="http://localhost:1/v1", model="qwen", api_key="explicit"
    )
    assert str(model._chat.openai_api_base) == "http://localhost:1/v1"
