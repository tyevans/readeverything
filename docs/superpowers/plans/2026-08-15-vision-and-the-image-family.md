# Vision and the Image Family Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove capability negotiation against a real multimodal model server rather than fakes, by shipping the `VisionModel` port, a live LangChain adapter, and the image handler family — and pay off the two deferrals the spec says get more expensive with every handler.

**Architecture:** Two owed cleanups first (a `ContentHashing` port so `Perception` stops depending on a concrete adapter, and closing an `artifact_key` collision) while they are still cheap. Then the `VisionModel` port, an adapter injected with a `BaseChatModel` so unit tests stay offline, one thin `live`-marked task that proves the endpoint answers, and finally an image handler whose affordances are genuinely VISION-gated — the first handler whose capability filtering is exercised against a real model.

**Tech Stack:** Python 3.13, pydantic 2, Pillow, langchain-openai against an OpenAI-compatible endpoint, pytest + hypothesis.

**Spec:** `docs/superpowers/specs/2026-08-14-readeverything-perception-core-design.md` (amended after Plan 1 — read §5, §7, §13 and §14b)

## Scope

Plan 1 is merged on `main`: 150 tests, five gates green, 36 source modules.

**In scope:** spec §14b's two "owed early" deferrals; the `VisionModel` port from §5; the `image/*` row of §7; live validation of the assumption §14b names as the largest unvalidated one.

**Not in scope** (later plans): video, audio, PDF/Office, HTML, tabular, archive handlers; ASR and diarization; the deepagents backend decorator; the redstring sink and composition root; cache wiring; per-capability concurrency limits.

## Global Constraints

- Python `>=3.13`. PEP 695 syntax; never module-level `TypeVar`.
- `mypy --strict` must pass, with `warn_unused_ignores = true`. Everything typed.
- **The library reads no environment.** Enforced by `tests/unit/test_reads_no_environment.py`, which scans `src/` only — *tests* may read env for the live endpoint, the library may not.
- Value objects are `@dataclass(frozen=True, slots=True)`; caller-facing models are pydantic `BaseModel`.
- Ports are `typing.Protocol` + `@runtime_checkable`.
- Ruff line-length 100, `select = ["E","F","I","N","W","UP","B","C4","SIM","RUF"]` (note `BLE` is NOT selected).
- **Bare `assert` is not permitted in `src/` outside `readeverything/testing/`** — bandit's B101 skip is scoped to that package and `python -O` strips asserts. Raise explicitly instead.
- Each third-party client is confined to one directory, asserted by `tests/unit/test_dependencies_stay_confined.py`. This plan adds two entries; the liveness half of that test fails if an entry names a file that does not import it.
- Coverage gate is `fail_under = 90` and currently sits at 91% — thin headroom, so new code needs tests.
- All commands run through `uv run`. `make check` is the gate.

---

### Task 1: The `ContentHashing` port

Spec §5 and §14b: `Perception` depends on the concrete `ContentHasher`, the one non-hexagonal seam in the core. import-linter cannot catch it because `pipeline` legitimately sits above `adapters`.

**Files:**
- Create: `src/readeverything/ports/hashing.py`
- Modify: `src/readeverything/pipeline/perception.py` (import and the `hasher` annotation)
- Modify: `src/readeverything/__init__.py` (export `ContentHashing`)
- Test: `tests/unit/ports/test_hashing_port.py`

**Interfaces:**
- Consumes: `ContentHash` from `readeverything.domain.identity`.
- Produces: `ContentHashing` Protocol with `async def hash(self, uri: str) -> ContentHash`. Task 8 relies on `Perception` accepting any object satisfying it.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/ports/test_hashing_port.py
from pathlib import Path

from readeverything.adapters.hashing import ContentHasher
from readeverything.adapters.local_source import LocalFileSource
from readeverything.domain.identity import ContentHash
from readeverything.ports.hashing import ContentHashing


class _PrecomputedHasher:
    """A caller supplying hashes from elsewhere — the case the port exists for."""

    def __init__(self, value: str) -> None:
        self._value = value

    async def hash(self, uri: str) -> ContentHash:
        return ContentHash(self._value)


def test_the_bundled_adapter_satisfies_the_port(tmp_path: Path) -> None:
    source = LocalFileSource(root=tmp_path)
    assert isinstance(ContentHasher(source=source), ContentHashing)


def test_an_unrelated_hasher_satisfies_the_port_without_inheriting() -> None:
    """Structural typing is the point: a caller must not have to subclass."""
    assert isinstance(_PrecomputedHasher("abc"), ContentHashing)


def test_an_object_without_hash_does_not_satisfy_the_port() -> None:
    assert not isinstance(object(), ContentHashing)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/ports/test_hashing_port.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'readeverything.ports.hashing'`

- [ ] **Step 3: Write the port**

```python
# src/readeverything/ports/hashing.py
"""Turning a source into a stable identity.

Split out after Plan 1, where `Perception` annotated the concrete
`ContentHasher` and so was the one collaborator in the core that could not be
substituted. import-linter permits it — `pipeline` sits above `adapters` — which
is exactly why it needed a human to notice.

The port exists for callers who already know the hash: a content-addressed
store that hands one over, a manifest, a build system that hashed the file
minutes ago. Re-reading a two-hour video to recompute what the caller already
has is the cost of not having this.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from readeverything.domain.identity import ContentHash


@runtime_checkable
class ContentHashing(Protocol):
    async def hash(self, uri: str) -> ContentHash:
        """The stable identity of the bytes at `uri`."""
        ...
```

- [ ] **Step 4: Annotate `Perception` against the port**

In `src/readeverything/pipeline/perception.py`, replace the `ContentHasher` import with the port and change the constructor annotation. The import line becomes:

```python
from readeverything.ports.hashing import ContentHashing
```

and the parameter:

```python
        hasher: ContentHashing,
```

Delete the now-unused `from readeverything.adapters.hashing import ContentHasher` import. Nothing else changes — `ContentHasher` already satisfies the port structurally.

- [ ] **Step 5: Export it**

Add to `_LAZY` and the `TYPE_CHECKING` block in `src/readeverything/__init__.py`, in the existing redundant-alias style:

```python
    "ContentHashing": "readeverything.ports.hashing",
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/unit/ports -v && uv run mypy && uv run lint-imports`
Expected: all pass. `lint-imports` matters here — `pipeline` must now reach `ports.hashing` rather than `adapters.hashing`.

- [ ] **Step 7: Commit**

```bash
git add src/readeverything/ports/hashing.py src/readeverything/pipeline/perception.py src/readeverything/__init__.py tests/unit/ports/test_hashing_port.py
git commit -m "feat(ports): add ContentHashing so Perception depends on no concrete adapter"
```

---

### Task 2: Close the `artifact_key` collision

Spec §14b: `json.dumps(..., default=str)` makes `{"path": Path("a")}` and `{"path": "a"}` produce the same key. Free to fix now — no caller passes non-primitives and the cache is not wired — and a silent wrong-answer bug once it is.

**Files:**
- Modify: `src/readeverything/adapters/cache_key.py`
- Test: `tests/unit/adapters/test_cache_key.py` (add cases)

**Interfaces:**
- Consumes: `ContentHash`, `CapabilitySet`.
- Produces: `artifact_key(...)` unchanged in signature; now raises `DomainError` on a non-JSON-primitive param value.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/adapters/test_cache_key.py`:

```python
from pathlib import Path

import pytest

from readeverything.domain.errors import DomainError


def test_a_non_primitive_param_is_refused() -> None:
    """`default=str` would silently collide with the plain string "a"."""
    with pytest.raises(DomainError, match="not JSON-primitive"):
        _key(params={"path": Path("a")})


def test_nested_primitives_are_allowed() -> None:
    """Structure is fine; only unserialisable leaves are refused."""
    assert _key(params={"a": [1, 2, {"b": None}]}) == _key(params={"a": [1, 2, {"b": None}]})


def test_a_non_primitive_nested_deep_is_refused() -> None:
    with pytest.raises(DomainError, match="not JSON-primitive"):
        _key(params={"a": [1, {"b": Path("x")}]})


def test_the_refused_type_is_named_in_the_message() -> None:
    with pytest.raises(DomainError, match="PosixPath"):
        _key(params={"path": Path("a")})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/adapters/test_cache_key.py -v`
Expected: FAIL — `default=str` currently coerces `Path` silently, so no exception is raised.

- [ ] **Step 3: Implement the guard**

In `src/readeverything/adapters/cache_key.py`, add above `artifact_key`:

```python
_PRIMITIVES = (str, int, float, bool, type(None))


def _reject_non_primitives(value: Any, path: str) -> None:
    """Refuse anything `json.dumps` could not represent losslessly.

    The alternative was `default=str`, which silently coerced. That made
    `{"path": Path("a")}` and `{"path": "a"}` the same cache key — two different
    derivations sharing one artifact, which is the worst failure this component
    has. Refusing is louder and the caller has the information to fix it.
    """
    if isinstance(value, _PRIMITIVES):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_primitives(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_non_primitives(item, f"{path}[{index}]")
        return
    raise DomainError(
        f"cache key param {path} is not JSON-primitive: {type(value).__name__}. "
        f"Affordance params must be JSON-representable so the key is stable."
    )
```

Then in `artifact_key`, before building the payload:

```python
    _reject_non_primitives(dict(params), "params")
```

and remove `default=str` from the `json.dumps` call.

Add the imports it needs: `from collections.abc import Mapping` is already there; add `from readeverything.domain.errors import DomainError`.

Note `bool` must be checked before `int` conceptually, but `isinstance(True, int)` is `True` and both are permitted, so ordering does not matter here.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/adapters -v && uv run mypy`
Expected: all pass, including the six pre-existing key tests.

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/adapters/cache_key.py tests/unit/adapters/test_cache_key.py
git commit -m "fix(adapters): refuse non-primitive cache-key params instead of coercing them"
```

---

### Task 3: The `VisionModel` port

**Files:**
- Create: `src/readeverything/ports/vision.py`
- Modify: `src/readeverything/testing/fakes.py` (align `FakeVision`, add `FakeVisionRefusing`)
- Modify: `src/readeverything/__init__.py`
- Test: `tests/unit/ports/test_vision_port.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `VisionModel` Protocol with `model_id: str` and `async def describe(self, data: bytes, mime: str, prompt: str) -> str`. Tasks 4, 6 and 7 depend on exactly this shape.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/ports/test_vision_port.py
import pytest

from readeverything.ports.vision import VisionModel
from readeverything.testing.fakes import FakeVision


def test_the_fake_satisfies_the_port() -> None:
    assert isinstance(FakeVision(), VisionModel)


def test_an_object_without_describe_does_not_satisfy_the_port() -> None:
    assert not isinstance(object(), VisionModel)


async def test_the_fake_is_deterministic_and_derived_from_its_input() -> None:
    """Unit tests must never assert on model text, so the fake must not invent any."""
    vision = FakeVision()
    first = await vision.describe(b"1234", "image/png", "what is this")
    second = await vision.describe(b"1234", "image/png", "what is this")
    assert first == second
    assert "4 bytes" in first
    assert "image/png" in first


def test_the_fake_declares_a_model_id() -> None:
    """The id feeds the capability fingerprint, so a fake needs one too."""
    assert FakeVision().model_id == "fake-vision@1"


def test_a_data_protocol_cannot_be_used_with_issubclass() -> None:
    """Documents a real limitation: VisionModel has a non-method member.

    `runtime_checkable` supports `isinstance` for data protocols but not
    `issubclass`. Anyone adding a ports test that mirrors the method-only
    protocols will hit this, so it is pinned rather than left to surprise them.
    """
    with pytest.raises(TypeError):
        issubclass(FakeVision, VisionModel)  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/ports/test_vision_port.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'readeverything.ports.vision'`

- [ ] **Step 3: Write the port**

```python
# src/readeverything/ports/vision.py
"""Turning pixels into words.

The whole multimodal strategy of this library rests on this one method: content
is described into text at the edge, and only the description enters an index or
a knowledge graph. A frame never becomes a claim — what a model asserts about
the frame does, with the frame's locator as provenance.

`model_id` is not used for dispatch. It feeds `CapabilitySet.fingerprint()`, so
that swapping the model changes every artifact cache key derived from it.
Without it the cache would serve a mixture of descriptions produced by two
different models, which is invisible until someone reads two answers side by
side and cannot explain the difference in voice.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VisionModel(Protocol):
    #: Provider-qualified and versioned, e.g. "openai/qwen3.8-27b-mtp@2026-08".
    #: A bare family name makes "re-derive everything the old model touched"
    #: unanswerable.
    model_id: str

    async def describe(self, data: bytes, mime: str, prompt: str) -> str:
        """Answer `prompt` about the image in `data`.

        Returns the model's text. Raises `InfrastructureError` if the model
        answered with nothing usable — see the adapter for why an empty
        completion is a real and common failure rather than a valid answer.
        """
        ...
```

- [ ] **Step 4: Align the fake and add a refusing one**

In `src/readeverything/testing/fakes.py`, `FakeVision` already matches the port. Add an explicit annotation so it is checked, and add a second fake for the failure path:

```python
class FakeVision:
    """Describes an image by its size, deterministically."""

    model_id: str = "fake-vision@1"

    async def describe(self, data: bytes, mime: str, prompt: str) -> str:
        return f"[{mime} image of {len(data)} bytes] {prompt}"


class FakeVisionRefusing:
    """A vision model that answers with nothing.

    Not a hypothetical: reasoning models split their output into a reasoning
    channel and a content channel, and a model that spends its whole budget
    reasoning returns empty content. A handler must degrade rather than emit an
    empty description as if it were an observation.
    """

    model_id: str = "fake-vision-refusing@1"

    async def describe(self, data: bytes, mime: str, prompt: str) -> str:
        from readeverything.domain.errors import InfrastructureError

        raise InfrastructureError("the model returned an empty completion")
```

- [ ] **Step 5: Export it**

Add to `_LAZY` and the `TYPE_CHECKING` block in `src/readeverything/__init__.py`:

```python
    "VisionModel": "readeverything.ports.vision",
    "FakeVisionRefusing": "readeverything.testing.fakes",
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/unit -v && uv run mypy && uv run lint-imports`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/readeverything/ports/vision.py src/readeverything/testing/fakes.py src/readeverything/__init__.py tests/unit/ports/test_vision_port.py
git commit -m "feat(ports): add the VisionModel port"
```

---

### Task 4: The LangChain vision adapter (offline)

The adapter takes a `BaseChatModel` rather than building one, so every test in this task runs offline. Task 5 builds a real one and hits the network exactly once.

**Files:**
- Create: `src/readeverything/adapters/vision_langchain.py`
- Modify: `pyproject.toml` (add the `vision` extra)
- Modify: `tests/unit/test_dependencies_stay_confined.py` (add the confinement entry)
- Test: `tests/unit/adapters/test_vision_langchain.py`

**Interfaces:**
- Consumes: `VisionModel` (Task 3); `InfrastructureError` from `readeverything.domain.errors`.
- Produces: `LangChainVisionModel(chat=..., model_id=...)` satisfying `VisionModel`; `build_openai_vision_model(base_url=..., model=..., api_key=..., timeout_s=...) -> LangChainVisionModel`.

- [ ] **Step 1: Add the extra and the confinement entry**

In `pyproject.toml`, under `[project.optional-dependencies]`:

```toml
vision = ["langchain-openai>=0.2", "langchain-core>=0.3"]
```

In `tests/unit/test_dependencies_stay_confined.py`, add to `CONFINED`:

```python
    "langchain_openai": {"adapters/vision_langchain.py"},
```

Note `langchain_core` is already confined to `agent/tools.py`; this adapter needs it too, so extend that entry to `{"agent/tools.py", "adapters/vision_langchain.py"}`.

Run `uv sync --all-extras`.

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit/adapters/test_vision_langchain.py
import base64

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from readeverything.adapters.vision_langchain import LangChainVisionModel
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
    image_parts = [part for part in content if part.get("type") == "image_url"]
    assert len(image_parts) == 1
    url = image_parts[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == PNG


async def test_the_prompt_is_sent_alongside_the_image() -> None:
    model, chat = _model()
    await model.describe(PNG, "image/png", "count the squares")
    content = chat.sent[0].content
    assert isinstance(content, list)
    text_parts = [part for part in content if part.get("type") == "text"]
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/adapters/test_vision_langchain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'readeverything.adapters.vision_langchain'`

- [ ] **Step 4: Write the adapter**

```python
# src/readeverything/adapters/vision_langchain.py
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
        max_tokens=max_tokens,
    )
    return LangChainVisionModel(chat=chat, model_id=f"openai/{model}")
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/unit/adapters -v && uv run mypy && make check`
Expected: all pass. If `test_the_confinement_table_is_live` fails, it means an entry names a file that does not import its module — fix the table, not the test.

- [ ] **Step 6: Commit**

```bash
git add src/readeverything/adapters/vision_langchain.py pyproject.toml tests/unit/adapters/test_vision_langchain.py tests/unit/test_dependencies_stay_confined.py uv.lock
git commit -m "feat(adapters): add a LangChain vision adapter with an injected chat model"
```

---

### Task 5: Live validation of the endpoint

**This is the task that needs the model server.** Everything before it runs offline; everything after it depends on the assumption this task tests. Spec §14b calls this the largest unvalidated assumption in the design.

**Files:**
- Create: `tests/live/__init__.py`, `tests/live/conftest.py`, `tests/live/test_vision_endpoint.py`
- Modify: `pyproject.toml` (add `tests/live` to testpaths if needed — the `live` marker already exists and is deselected by default)

**Interfaces:**
- Consumes: `build_openai_vision_model` (Task 4); `CapabilitySet`, `Capability`; `artifact_key` (Task 2).
- Produces: nothing the library depends on. This task's deliverable is knowledge.

- [ ] **Step 1: Write the fixture**

```python
# tests/live/conftest.py
"""Configuration for tests that touch the real model server.

Tests may read the environment; the library may not. `test_reads_no_environment`
scans `src/` only, which is the line: configuration reaches the library as
constructor arguments, and it is the caller's business where it came from.
"""

import os

import pytest

from readeverything.adapters.vision_langchain import (
    LangChainVisionModel,
    build_openai_vision_model,
)

DEFAULT_BASE_URL = "http://192.168.1.14/v1/"
DEFAULT_MODEL = "qwen3.8-27b-mtp"


@pytest.fixture(scope="session")
def live_base_url() -> str:
    return os.environ.get("READEVERYTHING_LIVE_BASE_URL", DEFAULT_BASE_URL)


@pytest.fixture(scope="session")
def live_model_name() -> str:
    return os.environ.get("READEVERYTHING_LIVE_MODEL", DEFAULT_MODEL)


@pytest.fixture
def live_vision(live_base_url: str, live_model_name: str) -> LangChainVisionModel:
    return build_openai_vision_model(base_url=live_base_url, model=live_model_name)
```

- [ ] **Step 2: Write the live tests**

```python
# tests/live/test_vision_endpoint.py
"""Does the real endpoint behave the way the design assumes?

Marked `live` and deselected by default. Run with:
    uv run pytest tests/live -m live -v

These assert on STRUCTURE, never on model text. What is being validated is that
the transport works, that the model accepts an inlined image, and that the
identity feeding the cache key is real — not that the model describes anything
well. Description quality is a bench concern, not a test.
"""

import pytest

from readeverything.adapters.vision_langchain import LangChainVisionModel
from readeverything.adapters.cache_key import artifact_key
from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.domain.identity import ContentHash

pytestmark = pytest.mark.live

#: A 1x1 red PNG. Small enough that a failure is the endpoint, not the payload.
RED_PIXEL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c636060f80f000101010018dd8db000"
    "00000049454e44ae426082"
)


async def test_the_endpoint_answers_with_text(live_vision: LangChainVisionModel) -> None:
    """The transport works and the model accepts an inlined image."""
    answer = await live_vision.describe(
        RED_PIXEL_PNG, "image/png", "Describe this image in one short sentence."
    )
    assert answer.strip()


async def test_the_answer_is_not_an_echo_of_the_prompt(
    live_vision: LangChainVisionModel,
) -> None:
    """Guards the failure where a model returns the prompt back verbatim.

    That would pass a bare truthiness check while proving nothing about vision.
    """
    prompt = "Describe this image in one short sentence."
    answer = await live_vision.describe(RED_PIXEL_PNG, "image/png", prompt)
    assert answer.strip() != prompt


async def test_the_model_id_is_provider_qualified(
    live_vision: LangChainVisionModel, live_model_name: str
) -> None:
    """A bare family name makes 're-derive what the old model touched' unanswerable."""
    assert live_vision.model_id == f"openai/{live_model_name}"


def test_swapping_the_model_changes_every_cache_key(live_model_name: str) -> None:
    """The whole reason `model_id` exists.

    Without this, changing the model silently serves a mixture of descriptions
    produced by two different models under one key.
    """
    def key_for(model_id: str) -> str:
        return artifact_key(
            content_hash=ContentHash("a" * 64),
            handler_id="image",
            handler_version=1,
            affordance="describe_image",
            params={"prompt": "what is this"},
            capabilities=CapabilitySet.of({Capability.VISION: model_id}),
        )

    assert key_for(f"openai/{live_model_name}") != key_for("openai/some-other-model")
    assert key_for(f"openai/{live_model_name}") == key_for(f"openai/{live_model_name}")
```

- [ ] **Step 3: Confirm the live tests are deselected by default**

Run: `uv run pytest -q`
Expected: the same count as before this task — `tests/live` collected but deselected by the existing `addopts = "-m 'not integration and not live and not accuracy'"`. If they run, the marker is not applied; fix `pytestmark` rather than the config.

- [ ] **Step 4: Run the live tests against the real server**

**Coordinate before running this — the box may be busy with other inference.**

Run: `uv run pytest tests/live -m live -v`
Expected: 4 passed.

If the endpoint rejects the image-content shape, that is the finding this task exists to produce: report the exact error rather than reshaping the payload until something sticks. The likely variants are `image_url` as a bare string rather than `{"url": ...}`, or the model requiring `detail`. Report which, and stop.

- [ ] **Step 5: Commit**

```bash
git add tests/live pyproject.toml
git commit -m "test(live): validate the vision endpoint and the capability fingerprint"
```

---

### Task 6: The image handler's card

Card first, affordances second — a card must be cheap and must not invoke a model, so it is worth landing and reviewing on its own.

**Files:**
- Create: `src/readeverything/handlers/image.py`
- Modify: `pyproject.toml` (`images` extra), `tests/unit/test_dependencies_stay_confined.py`
- Test: `tests/unit/handlers/test_image_handler.py`

**Interfaces:**
- Consumes: `SourceReader`, all domain types, `VisionModel` (Task 3).
- Produces: `ImageHandler(source=..., vision=None)`. Task 7 adds its affordances.

- [ ] **Step 1: Add the extra and confinement entry**

`pyproject.toml`:

```toml
images = ["pillow>=11.0"]
```

`tests/unit/test_dependencies_stay_confined.py`, in `CONFINED`:

```python
    "PIL": {"handlers/image.py"},
```

Run `uv sync --all-extras`.

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit/handlers/test_image_handler.py
import io

import pytest
from PIL import Image

from readeverything.domain.capability import Capability
from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
from readeverything.handlers.image import ImageHandler
from readeverything.testing.fakes import FakeSource


def _png(width: int = 8, height: int = 4, colour: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


PNG = _png()


def _ref(size: int = len(PNG)) -> SourceRef:
    return SourceRef(
        uri="a.png",
        mime=MimeType.parse("image/png"),
        content_hash=ContentHash("e" * 64),
        size_bytes=size,
    )


def _handler() -> ImageHandler:
    return ImageHandler(source=FakeSource({"a.png": PNG, "somewhere/else": PNG}))


async def test_the_card_reports_dimensions_and_format() -> None:
    card = await _handler().describe(_ref())
    assert card.kind is MediaKind.IMAGE
    assert card.facts["width"] == 8
    assert card.facts["height"] == 4
    assert card.facts["format"] == "PNG"
    assert card.facts["mode"] == "RGB"


async def test_the_card_has_no_excerpt() -> None:
    """An image has no cheap textual excerpt; describing it costs a model call."""
    assert (await _handler().describe(_ref())).excerpt is None


async def test_the_card_outlines_the_whole_image() -> None:
    card = await _handler().describe(_ref())
    assert len(card.outline) == 1
    assert card.outline[0].label == "whole image"


async def test_a_handler_without_vision_requires_nothing() -> None:
    """The handler stays usable for metadata even with no model configured."""
    assert _handler().requires() == frozenset()


async def test_an_undecodable_image_still_produces_a_card() -> None:
    """There is no unsupported-file path; a corrupt image is a thin card."""
    handler = ImageHandler(source=FakeSource({"a.png": b"not an image at all"}))
    card = await handler.describe(_ref(size=19))
    assert card.kind is MediaKind.IMAGE
    assert card.facts["decodable"] is False
    assert card.outline == ()


async def test_a_decodable_image_says_so() -> None:
    card = await _handler().describe(_ref())
    assert card.facts["decodable"] is True
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/handlers/test_image_handler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'readeverything.handlers.image'`

- [ ] **Step 4: Write the handler's card half**

```python
# src/readeverything/handlers/image.py
"""Images.

The first handler whose useful work needs a model. Its card deliberately does
not: dimensions, format and mode come from the header alone, so pointing an
agent at a directory of photographs costs no inference. Everything a model must
answer is behind an affordance the agent chooses to invoke.

`requires()` is empty on purpose. A deployment with no vision model can still
list, size and identify images — it just cannot describe them, and the registry
drops those affordances rather than the handler.
"""

from __future__ import annotations

import io
from typing import ClassVar

from PIL import Image, UnidentifiedImageError

from readeverything.domain.affordance import Affordance
from readeverything.domain.capability import Capability
from readeverything.domain.card import Card, Segment
from readeverything.domain.identity import MediaKind, SourceRef
from readeverything.domain.locators import BBox
from readeverything.ports.source import SourceReader
from readeverything.ports.vision import VisionModel

#: The whole frame, in the normalised coordinates every BBox uses.
WHOLE_IMAGE = BBox(page=None, x=0.0, y=0.0, w=1.0, h=1.0)


class ImageHandler:
    """Reads raster images, and describes them when a vision model is present."""

    mime_patterns: ClassVar[tuple[str, ...]] = ("kind:image",)
    priority: ClassVar[int] = 0
    handler_id: ClassVar[str] = "image"
    handler_version: ClassVar[int] = 1

    def __init__(self, *, source: SourceReader, vision: VisionModel | None = None) -> None:
        self._source = source
        self._vision = vision

    def requires(self) -> frozenset[Capability]:
        return frozenset()

    def affordances(self) -> tuple[Affordance, ...]:
        return ()

    async def _open(self, ref: SourceRef) -> Image.Image | None:
        """The decoded image, or None if the bytes are not a readable image."""
        data = await self._source.read_bytes(ref.uri)
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError, ValueError):
            return None
        return image

    async def describe(self, ref: SourceRef) -> Card:
        image = await self._open(ref)
        if image is None:
            return Card(
                ref=ref,
                kind=MediaKind.IMAGE,
                facts={"decodable": False, "size_bytes": ref.size_bytes},
                outline=(),
                excerpt=None,
                affordances=self.affordances(),
            )
        return Card(
            ref=ref,
            kind=MediaKind.IMAGE,
            facts={
                "decodable": True,
                "width": image.width,
                "height": image.height,
                "format": image.format or "unknown",
                "mode": image.mode,
                "size_bytes": ref.size_bytes,
            },
            outline=(Segment(WHOLE_IMAGE, "whole image"),),
            excerpt=None,
            affordances=self.affordances(),
        )
```

Note `facts` is typed `Mapping[str, str | int | float]` on `Card`; `bool` is a subclass of `int`, so `decodable` is valid without widening the type.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/unit/handlers -v && uv run mypy && make check`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/readeverything/handlers/image.py pyproject.toml tests/unit/handlers/test_image_handler.py tests/unit/test_dependencies_stay_confined.py uv.lock
git commit -m "feat(handlers): add the image handler's card"
```

---

### Task 7: The image handler's affordances

**Files:**
- Modify: `src/readeverything/handlers/image.py`
- Test: `tests/unit/handlers/test_image_handler.py` (extend)

**Interfaces:**
- Consumes: Task 6's `ImageHandler`; `VisionModel`; `FakeVision`, `FakeVisionRefusing` (Task 3).
- Produces: `DescribeImageParams`, `OcrParams`, `CropParams`; three affordances; `represent`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/handlers/test_image_handler.py`:

```python
from readeverything.domain.errors import UnknownAffordanceError
from readeverything.domain.locators import BBox
from readeverything.domain.rendition import Budget, ImageContent, TextContent
from readeverything.handlers.image import (
    CropParams,
    DescribeImageParams,
    ImageHandler,
    OcrParams,
)
from readeverything.testing.fakes import FakeVision, FakeVisionRefusing
from readeverything.testing.handler_compliance import MediaHandlerCompliance


def _seeing() -> ImageHandler:
    return ImageHandler(
        source=FakeSource({"a.png": PNG, "somewhere/else": PNG}), vision=FakeVision()
    )


def test_without_vision_only_crop_is_offered() -> None:
    """Cropping is pure Pillow; describing and OCR are not."""
    names = tuple(a.name for a in _handler().affordances())
    assert names == ("crop_region",)


def test_with_vision_all_three_are_offered() -> None:
    names = tuple(a.name for a in _seeing().affordances())
    assert set(names) == {"crop_region", "describe_image", "ocr"}


def test_the_model_backed_affordances_declare_the_vision_capability() -> None:
    """The registry filters on this; a wrong declaration makes negotiation a lie."""
    by_name = {a.name: a for a in _seeing().affordances()}
    assert by_name["describe_image"].requires == frozenset({Capability.VISION})
    assert by_name["ocr"].requires == frozenset({Capability.VISION})
    assert by_name["crop_region"].requires == frozenset()


async def test_describe_image_returns_text_located_at_the_whole_frame() -> None:
    rendition = await _seeing().invoke(_ref(), "describe_image", DescribeImageParams())
    assert isinstance(rendition.content, TextContent)
    assert rendition.locator == BBox(page=None, x=0.0, y=0.0, w=1.0, h=1.0)


async def test_describe_image_passes_the_prompt_through() -> None:
    rendition = await _seeing().invoke(
        _ref(), "describe_image", DescribeImageParams(prompt="count the squares")
    )
    assert isinstance(rendition.content, TextContent)
    assert "count the squares" in rendition.content.text


async def test_ocr_locates_its_text_at_the_whole_frame() -> None:
    rendition = await _seeing().invoke(_ref(), "ocr", OcrParams())
    assert isinstance(rendition.content, TextContent)
    assert rendition.locator == BBox(page=None, x=0.0, y=0.0, w=1.0, h=1.0)


async def test_crop_region_returns_image_bytes_located_at_the_crop() -> None:
    params = CropParams(x=0.0, y=0.0, w=0.5, h=1.0)
    rendition = await _seeing().invoke(_ref(), "crop_region", params)
    assert isinstance(rendition.content, ImageContent)
    assert rendition.content.mime == "image/png"
    assert rendition.locator == BBox(page=None, x=0.0, y=0.0, w=0.5, h=1.0)


async def test_a_crop_is_actually_cropped() -> None:
    """Verified by decoding the result, not by trusting the locator."""
    params = CropParams(x=0.0, y=0.0, w=0.5, h=1.0)
    rendition = await _seeing().invoke(_ref(), "crop_region", params)
    assert isinstance(rendition.content, ImageContent)
    cropped = Image.open(io.BytesIO(rendition.content.data))
    assert cropped.size == (4, 4)


async def test_a_crop_of_an_undecodable_image_raises_a_domain_error() -> None:
    from readeverything.domain.errors import DomainError

    handler = ImageHandler(source=FakeSource({"a.png": b"nonsense"}), vision=FakeVision())
    with pytest.raises(DomainError, match="not a readable image"):
        await handler.invoke(_ref(size=8), "crop_region", CropParams(x=0.0, y=0.0, w=1.0, h=1.0))


async def test_invoking_a_vision_affordance_without_vision_raises() -> None:
    """Handler-level guard; the registry normally prevents this from being reachable."""
    with pytest.raises(UnknownAffordanceError, match="describe_image"):
        await _handler().invoke(_ref(), "describe_image", DescribeImageParams())


async def test_represent_without_vision_states_the_facts_and_degrades() -> None:
    """A card's worth of truth, plus an honest note that description was unavailable."""
    rendered = await _handler().represent(_ref(), Budget(max_chars=None))
    assert "8x4" in rendered.text
    assert rendered.degradations
    assert "vision" in rendered.degradations[0].what


async def test_represent_with_vision_describes_the_image(
) -> None:
    rendered = await _seeing().represent(_ref(), Budget(max_chars=None))
    assert "image/png" in rendered.text
    assert rendered.degradations == ()
    assert rendered.locator_map.length == len(rendered.text)


async def test_represent_degrades_when_the_model_returns_nothing() -> None:
    """An empty completion must not enter an index as an observation."""
    handler = ImageHandler(
        source=FakeSource({"a.png": PNG}), vision=FakeVisionRefusing()
    )
    rendered = await handler.represent(_ref(), Budget(max_chars=None))
    assert "8x4" in rendered.text
    assert rendered.degradations
    assert "vision" in rendered.degradations[0].what


class TestImageHandlerCompliance(MediaHandlerCompliance):
    @pytest.fixture
    def handler(self) -> ImageHandler:
        return _seeing()

    @pytest.fixture
    def content(self) -> bytes:
        return PNG

    @pytest.fixture
    def ref(self, content: bytes) -> SourceRef:
        return _ref()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/handlers/test_image_handler.py -v`
Expected: FAIL — `ImportError` for `CropParams`/`DescribeImageParams`/`OcrParams`.

- [ ] **Step 3: Implement the affordances**

Add to `src/readeverything/handlers/image.py`:

```python
from pydantic import BaseModel, Field

from readeverything.domain.affordance import DetailLevel
from readeverything.domain.errors import DomainError, InfrastructureError, UnknownAffordanceError
from readeverything.domain.locator_map import LocatorMap, LocatorSegment
from readeverything.domain.locators import CharSpan
from readeverything.domain.rendition import (
    Budget,
    Degradation,
    ImageContent,
    Rendered,
    Rendition,
    TextContent,
)

_DESCRIBE_PROMPT = "Describe this image in two or three sentences."
_OCR_PROMPT = (
    "Transcribe all text visible in this image, exactly as written. "
    "If there is no text, reply with: (no text)"
)


class DescribeImageParams(BaseModel):
    prompt: str = Field(
        default=_DESCRIBE_PROMPT, description="What to ask the model about the image."
    )


class OcrParams(BaseModel):
    pass


class CropParams(BaseModel):
    x: float = Field(ge=0.0, le=1.0, description="Left edge, 0-1 of image width.")
    y: float = Field(ge=0.0, le=1.0, description="Top edge, 0-1 of image height.")
    w: float = Field(gt=0.0, le=1.0, description="Width, 0-1 of image width.")
    h: float = Field(gt=0.0, le=1.0, description="Height, 0-1 of image height.")
```

Replace `affordances()` with:

```python
    def affordances(self) -> tuple[Affordance, ...]:
        crop = Affordance(
            name="crop_region",
            description=(
                "Return a rectangular region of the image as PNG bytes. "
                "Coordinates are fractions of the image, 0 to 1."
            ),
            params=CropParams,
            requires=frozenset(),
            level=DetailLevel.SEGMENT,
        )
        if self._vision is None:
            return (crop,)
        return (
            crop,
            Affordance(
                name="describe_image",
                description="Describe what is visible in the image, in prose.",
                params=DescribeImageParams,
                requires=frozenset({Capability.VISION}),
                level=DetailLevel.DEEP,
            ),
            Affordance(
                name="ocr",
                description="Transcribe text visible in the image.",
                params=OcrParams,
                requires=frozenset({Capability.VISION}),
                level=DetailLevel.DEEP,
            ),
        )
```

Add `invoke` and `represent`:

```python
    async def _require_image(self, ref: SourceRef) -> Image.Image:
        image = await self._open(ref)
        if image is None:
            raise DomainError(f"{ref.uri} is not a readable image")
        return image

    async def _see(self, ref: SourceRef, prompt: str) -> str:
        if self._vision is None:
            raise UnknownAffordanceError("describe_image", (a.name for a in self.affordances()))
        data = await self._source.read_bytes(ref.uri)
        return await self._vision.describe(data, str(ref.mime), prompt)

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition:
        match name:
            case "crop_region":
                if not isinstance(params, CropParams):
                    raise TypeError(f"expected CropParams, got {type(params).__name__}")
                image = await self._require_image(ref)
                box = (
                    int(params.x * image.width),
                    int(params.y * image.height),
                    max(int((params.x + params.w) * image.width), int(params.x * image.width) + 1),
                    max(int((params.y + params.h) * image.height), int(params.y * image.height) + 1),
                )
                buffer = io.BytesIO()
                image.crop(box).save(buffer, format="PNG")
                return Rendition(
                    locator=BBox(page=None, x=params.x, y=params.y, w=params.w, h=params.h),
                    content=ImageContent(data=buffer.getvalue(), mime="image/png"),
                )
            case "describe_image":
                if not isinstance(params, DescribeImageParams):
                    raise TypeError(
                        f"expected DescribeImageParams, got {type(params).__name__}"
                    )
                text = await self._see(ref, params.prompt)
                return Rendition(locator=WHOLE_IMAGE, content=TextContent(text))
            case "ocr":
                if not isinstance(params, OcrParams):
                    raise TypeError(f"expected OcrParams, got {type(params).__name__}")
                text = await self._see(ref, _OCR_PROMPT)
                return Rendition(locator=WHOLE_IMAGE, content=TextContent(text))
            case _:
                raise UnknownAffordanceError(name, (a.name for a in self.affordances()))

    async def represent(self, ref: SourceRef, budget: Budget) -> Rendered:
        image = await self._open(ref)
        if image is None:
            facts = f"Unreadable image {ref.uri}, {ref.size_bytes} bytes."
        else:
            facts = (
                f"Image {ref.uri}, {image.width}x{image.height} "
                f"{image.format or 'unknown'} ({image.mode}), {ref.size_bytes} bytes."
            )
        degradations: tuple[Degradation, ...] = ()
        described = ""
        if self._vision is None:
            degradations = (
                Degradation(
                    what="vision unavailable",
                    detail="no vision model configured; only metadata was indexed",
                ),
            )
        else:
            try:
                described = await self._see(ref, _DESCRIBE_PROMPT)
            except InfrastructureError as exc:
                # An empty or failed completion is not a description. Saying so
                # is better than indexing silence as an observation.
                degradations = (
                    Degradation(what="vision unavailable", detail=str(exc)),
                )
        full = f"{facts} {described}".strip() if described else facts
        text = full
        if budget.max_chars is not None and len(full) > budget.max_chars:
            degradations = (
                *degradations,
                Degradation(
                    what="text truncated",
                    detail=f"kept {budget.max_chars} of {len(full)} characters",
                ),
            )
            text = full[: budget.max_chars]
        if not text:
            text = full[:1] or "?"
        return Rendered(
            text=text,
            locator_map=LocatorMap.build(
                (LocatorSegment(CharSpan(0, len(text)), WHOLE_IMAGE),)
            ),
            barriers=(),
            degradations=degradations,
        )
```

Note the compliance law compares a bounded render against an unbounded one, so `represent` must be deterministic across two calls with the same input. `FakeVision` is deterministic, which is why the compliance fixture uses it.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/handlers -v && uv run mypy && make check`
Expected: all pass, including the six inherited compliance laws.

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/handlers/image.py tests/unit/handlers/test_image_handler.py
git commit -m "feat(handlers): add describe_image, ocr and crop_region"
```

---

### Task 8: Capability negotiation end to end, against a real model

The point of the whole plan: `Perception` over an image, with and without a real vision model, proving the agent's view of a file reflects what the deployment can actually do.

**Files:**
- Create: `tests/live/test_image_negotiation.py`
- Test: `tests/unit/pipeline/test_perception_image.py`
- Modify: `src/readeverything/__init__.py` (export `ImageHandler`)

**Interfaces:**
- Consumes: `Perception`, `MimeTypeRegistry`, `ImageHandler`, `build_openai_vision_model`.
- Produces: nothing new.

- [ ] **Step 1: Write the offline test**

```python
# tests/unit/pipeline/test_perception_image.py
"""Capability negotiation over an image, through the full stack, offline."""

import io
from pathlib import Path

import pytest
from PIL import Image

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.detection import PuremagicDetector
from readeverything.adapters.hashing import ContentHasher
from readeverything.adapters.local_source import LocalFileSource
from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.domain.errors import UnknownAffordanceError
from readeverything.domain.rendition import TextContent
from readeverything.handlers.binary import BinaryHandler
from readeverything.handlers.image import ImageHandler
from readeverything.pipeline.perception import Perception
from readeverything.registry.registry import MimeTypeRegistry
from readeverything.testing.fakes import FakeVision


def _perception(tmp_path: Path, *, seeing: bool) -> Perception:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 4), (0, 128, 0)).save(buffer, format="PNG")
    (tmp_path / "photo.png").write_bytes(buffer.getvalue())
    source = LocalFileSource(root=tmp_path)
    vision = FakeVision() if seeing else None
    capabilities = (
        CapabilitySet.of({Capability.VISION: FakeVision().model_id})
        if seeing
        else CapabilitySet.empty()
    )
    return Perception(
        source=source,
        detector=PuremagicDetector(),
        hasher=ContentHasher(source=source),
        registry=MimeTypeRegistry(
            handlers=(
                ImageHandler(source=source, vision=vision),
                BinaryHandler(source=source),
            ),
            capabilities=capabilities,
        ),
        artifacts=InMemoryArtifactStore(),
    )


async def test_a_png_dispatches_to_the_image_handler(tmp_path: Path) -> None:
    card = await _perception(tmp_path, seeing=False).inspect("photo.png")
    assert card.facts["width"] == 8


async def test_without_vision_the_agent_sees_only_crop(tmp_path: Path) -> None:
    card = await _perception(tmp_path, seeing=False).inspect("photo.png")
    assert card.affordance_names() == ("crop_region",)


async def test_with_vision_the_agent_sees_all_three(tmp_path: Path) -> None:
    card = await _perception(tmp_path, seeing=True).inspect("photo.png")
    assert set(card.affordance_names()) == {"crop_region", "describe_image", "ocr"}


async def test_without_vision_describe_is_not_invocable(tmp_path: Path) -> None:
    perception = _perception(tmp_path, seeing=False)
    with pytest.raises(UnknownAffordanceError):
        await perception.invoke("photo.png", "describe_image", {})


async def test_with_vision_describe_is_invocable(tmp_path: Path) -> None:
    rendition = await _perception(tmp_path, seeing=True).invoke("photo.png", "describe_image", {})
    assert isinstance(rendition.content, TextContent)
    assert rendition.content.text
```

- [ ] **Step 2: Run it, then export `ImageHandler`**

Run: `uv run pytest tests/unit/pipeline -v`
Expected: FAIL on the import of `readeverything.handlers.image` only if Task 6/7 are incomplete; otherwise these should pass once written.

Add to `_LAZY` and the `TYPE_CHECKING` block in `src/readeverything/__init__.py`:

```python
    "ImageHandler": "readeverything.handlers.image",
```

- [ ] **Step 3: Write the live test**

```python
# tests/live/test_image_negotiation.py
"""The same negotiation, against the real model server.

Structure only — never assert on what the model says.
"""

import io
from pathlib import Path

import pytest
from PIL import Image

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.detection import PuremagicDetector
from readeverything.adapters.hashing import ContentHasher
from readeverything.adapters.local_source import LocalFileSource
from readeverything.adapters.vision_langchain import LangChainVisionModel
from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.domain.rendition import TextContent
from readeverything.handlers.binary import BinaryHandler
from readeverything.handlers.image import ImageHandler
from readeverything.pipeline.perception import Perception
from readeverything.registry.registry import MimeTypeRegistry

pytestmark = pytest.mark.live


def _perception(tmp_path: Path, vision: LangChainVisionModel) -> Perception:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (0, 128, 0)).save(buffer, format="PNG")
    (tmp_path / "photo.png").write_bytes(buffer.getvalue())
    source = LocalFileSource(root=tmp_path)
    return Perception(
        source=source,
        detector=PuremagicDetector(),
        hasher=ContentHasher(source=source),
        registry=MimeTypeRegistry(
            handlers=(
                ImageHandler(source=source, vision=vision),
                BinaryHandler(source=source),
            ),
            capabilities=CapabilitySet.of({Capability.VISION: vision.model_id}),
        ),
        artifacts=InMemoryArtifactStore(),
    )


async def test_a_real_model_describes_a_real_image(
    tmp_path: Path, live_vision: LangChainVisionModel
) -> None:
    perception = _perception(tmp_path, live_vision)
    card = await perception.inspect("photo.png")
    assert "describe_image" in card.affordance_names()
    rendition = await perception.invoke("photo.png", "describe_image", {})
    assert isinstance(rendition.content, TextContent)
    assert rendition.content.text.strip()


async def test_represent_against_a_real_model_reports_no_degradation(
    tmp_path: Path, live_vision: LangChainVisionModel
) -> None:
    """If the real model answers, nothing should claim vision was unavailable."""
    from readeverything.domain.rendition import Budget

    perception = _perception(tmp_path, live_vision)
    rendered = await perception.represent("photo.png", Budget(max_chars=None))
    assert rendered.locator_map.length == len(rendered.text)
    assert not any(d.what == "vision unavailable" for d in rendered.degradations)
```

- [ ] **Step 4: Run everything**

Run: `make check`
Expected: all five gates pass; live tests deselected.

**Coordinate before this one — it needs the model server:**

Run: `uv run pytest tests/live -m live -v`
Expected: 6 passed (4 from Task 5, 2 here).

- [ ] **Step 5: Commit**

```bash
git add tests/unit/pipeline/test_perception_image.py tests/live/test_image_negotiation.py src/readeverything/__init__.py
git commit -m "test: prove capability negotiation over images, offline and live"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
| --- | --- |
| §5 `ContentHashing` port | 1 |
| §5 `VisionModel` port | 3 |
| §7 `image/*` row (Pillow, describe/ocr/crop) | 6, 7 |
| §8 cache-key correctness (`default=str` collision) | 2 |
| §13 laws exercised against the new handler | 7 (inherited compliance suite) |
| §13 model quality measured, not asserted | 5, 8 (structure only, never model text) |
| §14b "owed early" deferrals | 1, 2 |
| §14b largest unvalidated assumption | 5, 8 |
| §14 reference model deployment | 5 |

**Deliberate gaps:** `TextRecognizer` (spec §5) is not built — OCR goes through `VisionModel`, and a tesseract adapter has no caller until a no-vision deployment needs one. `exiftool`/`BinaryProbe` are not built: Pillow supplies dimensions, and no handler yet needs binary discovery. Cache wiring stays out of scope, so `artifact_key` remains uncalled by `Perception` — Task 2 fixes it before it has callers, which is the point.

**Type consistency:** `VisionModel.describe(data, mime, prompt) -> str` is spelled identically in the port, both fakes, the adapter and `ImageHandler._see`. `model_id` is a plain attribute everywhere. `WHOLE_IMAGE` is the single `BBox` constant used by `describe_image`, `ocr`, `represent` and the card's outline.

**One risk named:** Task 5 Step 4 may find the endpoint rejects the `image_url` content shape. The instruction is to report the exact error and stop, not to reshape the payload until something sticks — the finding is the deliverable.
