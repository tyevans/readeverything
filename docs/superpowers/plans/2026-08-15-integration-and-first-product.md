# Integration and the First Usable Product — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a well-built set of components into a library a stranger can install and use in under ten lines.

**Architecture:** Build outward, not downward. Wire the artifact cache that already exists into the pipeline that already ignores it; give capability negotiation the discovery half it never had; add a composition root so callers stop hand-assembling ten objects; add an integration test tier so seams are verified rather than assumed; close the residual defects; and ship the deepagents integration the distribution is named for.

**Tech Stack:** Python 3.13, pydantic v2, Pillow, langchain-core, langchain-openai, deepagents 0.7.x, pytest, mypy --strict, ruff, import-linter, bandit, coverage.

**Spec:** `docs/superpowers/specs/2026-08-15-readeverything-integration-and-first-product-design.md`

## Global Constraints

- **The library reads NO environment variables under `src/`.** Every input arrives as a constructor or function argument. Enforced by `tests/unit/test_reads_no_environment.py` — do not weaken that test.
- **Python 3.13, PEP 695 inline type parameters.** `class Foo[T]`, `def f[**P]`, `type X = ...`. A module-level `TypeVar` is a defect.
- **`mypy --strict` with `warn_unused_ignores = true`** over `src` and `tests`. No new `# type: ignore` without a comment naming why.
- **Import-linter layered contract, `exhaustive = true`.** Layers, outermost first: `testing, agent, pipeline, registry, handlers, adapters, ports, domain`. A lower layer never imports a higher one.
- **`readeverything.testing` may import only `ports` and `domain`.** Second contract, `forbidden`.
- **Third-party imports are pinned by an AST test** (`tests/unit/test_dependencies_stay_confined.py`) because import-linter cannot see third-party imports. Any new third-party import in a new module must be added there.
- **`langchain_core` may be imported only in `agent/tools.py` and `adapters/vision_*.py`.** `deepagents` may be imported only in the one module Task 12 creates.
- **Ruff line-length 100**, rules `E,F,I,N,W,UP,B,C4,SIM,RUF`, `E501` ignored. Run `uv run --all-extras ruff format` before committing.
- **The tool pack never raises.** `agent/` returns structured results; exceptions become `ToolResult`.
- **Never assert on model text.** Fakes produce mechanically-derived output; tests assert structure and locators.
- **A law must be able to fail.** A compliance suite that no implementation subclasses is not passing, it is unexercised.
- **Coverage floor rises from 90 to 92** in Task 11, not before.
- Run the full gate set with `make check`. All five stages must be green at every commit.

---

## File Structure

**Created:**

| File | Responsibility |
| --- | --- |
| `src/readeverything/ports/probe.py` | `CapabilityProbe` protocol — one method, returns a revision or `None`. |
| `src/readeverything/adapters/binary_probe.py` | Probes OS executables by running their version flag. |
| `src/readeverything/adapters/model_probe.py` | Derives a capability revision from an injected model object's own identity. |
| `src/readeverything/pipeline/resolution.py` | The stat-keyed resolution memo. Separate from the artifact store because it has a different invalidation rule. |
| `src/readeverything/composition.py` | `build_perception` — the whole product surface. Top layer; imports everything. |
| `src/readeverything/agent/deepagents_backend.py` | `MediaAwareBackend`. The only module that may import `deepagents`. |
| `tests/integration/` | The third test tier. Real components, fake models only. |
| `README.md` | Currently 0 bytes. Becomes the thing that makes the acceptance sentence true. |

**Modified:**

| File | Change |
| --- | --- |
| `src/readeverything/pipeline/perception.py` | Consult the resolution memo and the artifact store. Stop doing everything twice. |
| `src/readeverything/handlers/binary.py` | Announce synthesized text; stop inventing a byte. |
| `src/readeverything/handlers/text.py` | Announce synthesized text; clamp `end` symmetrically. |
| `src/readeverything/agent/tools.py` | Name the real route for image content instead of a tool that does not exist. |
| `src/readeverything/__init__.py` | Export the vision builders, the probes, and `build_perception`. |
| `pyproject.toml` | `deepagents` extra; coverage floor; import-linter layer for the new module. |
| `docs/superpowers/specs/2026-08-14-readeverything-perception-core-design.md` | Correct the stale §14b. |

**Layering note.** `composition.py` sits at the top of the layer order — above `agent` — because it imports handlers, adapters, registry, pipeline, and agent. Task 8 adds it to the import-linter contract as the new outermost layer. `testing` stays confined to ports and domain and is unaffected.

---

## Task 1: Announce synthesized text

Two handlers emit text *about* a file rather than *from* it, and both fabricate a locator to satisfy `Rendered`'s length invariant. The spec (§8.1) rules that the type-level fix waits for Spec 2; what lands now is making the fiction visible through the `Degradation` channel that already exists.

**Files:**
- Modify: `src/readeverything/handlers/binary.py` (`represent`)
- Modify: `src/readeverything/handlers/text.py` (`represent`)
- Test: `tests/unit/handlers/test_binary_handler.py`, `tests/unit/handlers/test_text_handler.py`

**Interfaces:**
- Consumes: `Degradation(what: str, detail: str)` from `readeverything.domain.rendition`.
- Produces: the constant `SYNTHESIZED = "synthesized description"` used as `Degradation.what` by both handlers. Later tasks and tests match on this exact string.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/handlers/test_binary_handler.py`:

```python
async def test_a_binary_representation_announces_that_it_is_synthesized() -> None:
    """The text describes the file; it was not extracted from it.

    Nothing in the bytes says "No textual content could be extracted" — the
    handler wrote that. A consumer indexing this must be able to tell it apart
    from text that actually came out of the file, because attributing it to the
    file's content is a false claim about the file.
    """
    rendered = await _handler().represent(_ref(), Budget(max_chars=None))
    assert any(d.what == "synthesized description" for d in rendered.degradations)


async def test_an_empty_binary_file_does_not_claim_a_byte_that_is_not_there() -> None:
    """`ByteRange(0, max(1, size))` invents byte 0 for a 0-byte file.

    The `max(1, ...)` exists because ByteRange rejects start >= end, so the
    fabrication is structural rather than careless — which is exactly why it
    needs announcing rather than silently patching.
    """
    handler = BinaryHandler(source=FakeSource({"empty.bin": b""}))
    ref = _ref(uri="empty.bin", size_bytes=0)
    rendered = await handler.represent(ref, Budget(max_chars=None))
    assert any(d.what == "synthesized description" for d in rendered.degradations)
```

In `tests/unit/handlers/test_text_handler.py`:

```python
async def test_the_empty_file_placeholder_announces_that_it_is_synthesized() -> None:
    """`[empty text file: x]` is 24 characters mapped over a file of zero.

    The placeholder is correct behaviour and must stay. What must not stay is
    an indexer being unable to tell that those characters are not in the file.
    """
    handler = TextHandler(source=FakeSource({"empty.txt": b""}))
    rendered = await handler.represent(_ref(uri="empty.txt", size_bytes=0), Budget(max_chars=None))
    assert rendered.text.startswith("[empty text file:")
    assert any(d.what == "synthesized description" for d in rendered.degradations)


async def test_extracted_text_is_not_announced_as_synthesized() -> None:
    """The marker must distinguish. A marker on everything distinguishes nothing."""
    handler = TextHandler(source=FakeSource({"real.txt": b"actual file content"}))
    rendered = await handler.represent(_ref(uri="real.txt", size_bytes=19), Budget(max_chars=None))
    assert not any(d.what == "synthesized description" for d in rendered.degradations)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run --all-extras pytest tests/unit/handlers/test_binary_handler.py tests/unit/handlers/test_text_handler.py -v -k synthesized
```

Expected: FAIL — no degradation with that `what` is produced.

- [ ] **Step 3: Implement in `binary.py`**

Add the module-level constant near the other constants:

```python
#: `Degradation.what` for text the handler wrote about a file rather than
#: extracted from it. A consumer that indexes renditions must be able to tell
#: the difference, because attributing synthesized text to file content is a
#: false claim about the file.
SYNTHESIZED = "synthesized description"
```

In `represent`, seed the degradations tuple instead of starting empty:

```python
    async def represent(self, ref: SourceRef, budget: Budget) -> Rendered:
        full = (
            f"Binary file {ref.uri} of type {ref.mime}, {ref.size_bytes} bytes. "
            f"No textual content could be extracted."
        )
        text = full
        # Every byte of this rendition was written here, not read from the
        # file. The locator below points into the file because `Rendered`
        # requires a map whose length matches the text; this degradation is
        # what stops that from reading as a claim about the file's contents.
        degradations: tuple[Degradation, ...] = (
            Degradation(
                what=SYNTHESIZED,
                detail="no content was extracted; this text describes the file",
            ),
        )
        if budget.max_chars is not None and len(full) > budget.max_chars:
```

Leave the truncation block exactly as it is — it appends to `degradations` and already reports `len(text)` correctly.

- [ ] **Step 4: Implement in `text.py`**

Add the same constant to `text.py` (each handler owns its own; do not import across handler modules):

```python
#: `Degradation.what` for text the handler wrote about a file rather than
#: extracted from it. See `binary.SYNTHESIZED` — the same string, deliberately,
#: so a consumer matches one value across every handler.
SYNTHESIZED = "synthesized description"
```

Change the empty branch of `represent`:

```python
        if not full:
            # Only a genuinely empty source earns this. A truncated one is not
            # empty, and saying so would index a false claim about the file.
            text = f"[empty text file: {ref.uri}]"
            degradations = (
                Degradation(
                    what=SYNTHESIZED,
                    detail="the file is empty; this text describes it",
                ),
            )
```

- [ ] **Step 5: Run the full unit suite**

```bash
uv run --all-extras pytest tests/unit -q
```

Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
uv run --all-extras ruff format src tests
git add src/readeverything/handlers/binary.py src/readeverything/handlers/text.py tests/unit/handlers/
git commit -m "fix(handlers): announce text that describes a file rather than comes from it"
```

---

## Task 2: Name the real route for image content

**Files:**
- Modify: `src/readeverything/agent/tools.py` (`_render_rendition`)
- Test: `tests/unit/agent/test_tools.py`

**Interfaces:**
- Consumes: `Rendition`, `ImageContent` from `readeverything.domain.rendition`.
- Produces: `_render_rendition(rendition: Rendition, affordances: tuple[str, ...] = ()) -> str` — the added second parameter is the affordance names available for that file. `build_tools` passes them; existing callers passing one argument keep working.

- [ ] **Step 1: Write the failing test**

```python
def test_image_content_names_an_affordance_that_exists() -> None:
    """The pack has three tools and none of them is a vision tool.

    The old text told the model to "pass to a vision tool to read it". There is
    no vision tool. Instructing a model toward a tool that does not exist is
    the same defect shape as a degradation describing a cause nothing checked —
    text asserting something nothing established — and it reaches the model.
    """
    rendition = Rendition(
        locator=BBox(page=None, x=0.0, y=0.0, w=1.0, h=1.0),
        content=ImageContent(data=b"\x89PNG", mime="image/png"),
    )
    rendered = _render_rendition(rendition, ("describe_image", "ocr"))
    assert "vision tool" not in rendered
    assert "describe_image" in rendered


def test_image_content_says_so_plainly_when_nothing_can_read_it() -> None:
    """With no vision capability the honest answer is that it cannot be read
    here — not a pointer at an affordance the registry filtered out."""
    rendition = Rendition(
        locator=BBox(page=None, x=0.0, y=0.0, w=1.0, h=1.0),
        content=ImageContent(data=b"\x89PNG", mime="image/png"),
    )
    rendered = _render_rendition(rendition, ())
    assert "vision tool" not in rendered
    assert "cannot be read" in rendered
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run --all-extras pytest tests/unit/agent/test_tools.py -v -k image_content
```

Expected: FAIL — `_render_rendition` takes one argument.

- [ ] **Step 3: Implement**

Replace the `ImageContent` arm and the signature:

```python
#: Affordances that turn image bytes into something a text model can read.
#: Named here so the rendering and the handler cannot drift apart silently.
_IMAGE_READING_AFFORDANCES = ("describe_image", "ocr")


def _render_rendition(rendition: Rendition, affordances: tuple[str, ...] = ()) -> str:
    match rendition.content:
        case TextContent(text=text):
            body = text
        case StructuredContent(rows=rows):
            body = json.dumps(list(rows), indent=2)
        case ImageContent(data=data, mime=mime):
            usable = [a for a in _IMAGE_READING_AFFORDANCES if a in affordances]
            if usable:
                route = " or ".join(usable)
                body = (
                    f"[{mime} image, {len(data)} bytes — "
                    f"call invoke_affordance with {route} to read it]"
                )
            else:
                # No vision capability is registered here. Saying so is true and
                # actionable; pointing at an affordance the registry filtered
                # out would send the model after a tool call that cannot work.
                body = (
                    f"[{mime} image, {len(data)} bytes — "
                    f"cannot be read here: no vision capability is configured]"
                )
        case _:
            body = f"[unrenderable content: {type(rendition.content).__name__}]"
    marker = " (degraded)" if rendition.degraded else ""
    return f"located at {rendition.locator!r}{marker}:\n{body}"
```

- [ ] **Step 4: Pass the affordances at the call site**

In `build_tools`, change `invoke_affordance`:

```python
    @never_raises
    async def invoke_affordance(
        uri: str, affordance: str, params: Mapping[str, Any] | None = None
    ) -> str:
        card = await perception.inspect(uri)
        rendition = await perception.invoke(uri, affordance, params or {})
        return _render_rendition(rendition, tuple(a.name for a in card.affordances))
```

- [ ] **Step 5: Run the tests**

```bash
uv run --all-extras pytest tests/unit/agent -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
uv run --all-extras ruff format src tests
git add src/readeverything/agent/tools.py tests/unit/agent/test_tools.py
git commit -m "fix(agent): name a route that exists for image content"
```

**Note for the reviewer and for Task 5:** this adds a second `perception.inspect` call inside `invoke_affordance`, which is deliberate and temporary — Task 5's resolution memo makes it nearly free. If Task 5 is dropped, this call becomes a real cost and the reviewer should say so.

---

## Task 3: Clamp `end` symmetrically in `TextHandler.invoke`

**Files:**
- Modify: `src/readeverything/handlers/text.py` (`invoke`)
- Test: `tests/unit/handlers/test_text_handler.py`

**Interfaces:**
- Consumes: `ReadRangeParams` with `start: int`, `end: int` (already defined in `text.py`).
- Produces: no signature change.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize(
    ("content", "start", "end"),
    [(b"x", 1, 5), (b"x", 0, 1), (b"hello", 4, 99), (b"hello", 10, 20)],
)
async def test_read_range_returns_a_span_matching_the_text_it_returns(
    content: bytes, start: int, end: int
) -> None:
    """The clamp forced `start` into range and left `end` alone.

    Whenever `start >= len(text) - 1` the caller's `end` was discarded and
    exactly one character came back regardless of what was asked for. The
    rendition's own locator is the check: it must describe the text beside it.
    """
    handler = TextHandler(source=FakeSource({"f.txt": content}))
    ref = _ref(uri="f.txt", size_bytes=len(content))
    rendition = await handler.invoke(ref, "read_range", ReadRangeParams(start=start, end=end))
    assert isinstance(rendition.content, TextContent)
    span = rendition.locator
    assert isinstance(span, CharSpan)
    assert span.end - span.start == len(rendition.content.text)
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run --all-extras pytest tests/unit/handlers/test_text_handler.py -v -k span_matching
```

Expected: FAIL on at least one parameter set — the returned span and the returned text disagree.

- [ ] **Step 3: Implement**

Replace the clamping block in `invoke`:

```python
        # Clamp both ends against the text, not just `start`. Clamping `start`
        # alone silently discarded the caller's `end` whenever `start` was at
        # or past the last character, always returning exactly one character —
        # a rendition whose locator did not describe the text beside it.
        length = len(text)
        start = max(0, min(params.start, length - 1))
        end = max(start + 1, min(params.end, length))
        body = text[start:end]
```

and build the rendition from `CharSpan(start, end)`.

- [ ] **Step 4: Run the tests**

```bash
uv run --all-extras pytest tests/unit/handlers -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uv run --all-extras ruff format src tests
git add src/readeverything/handlers/text.py tests/unit/handlers/test_text_handler.py
git commit -m "fix(handlers): clamp both ends of a read_range against the text"
```

---

## Task 4: Force the error paths that have never run

Five branches exist and no test has ever entered them. Each gets a test that forces it.

**Files:**
- Test: `tests/unit/adapters/test_detection.py`, `tests/unit/adapters/test_local_source.py`, `tests/unit/adapters/test_hashing.py`
- Modify: none expected. If a test reveals a defect, fix it and say so in the report.

**Interfaces:**
- Consumes: `PuremagicDetector`, `LocalFileSource`, `StatMemo` — all unchanged.
- Produces: nothing new.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/adapters/test_detection.py
async def test_a_detector_whose_library_raises_still_returns_a_mimetype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """puremagic raising must degrade to the fallback, not propagate.

    Detection sits in front of every read. A library that raises on a malformed
    header would take down every call for that file, so the except exists — it
    had simply never run.
    """
    import puremagic

    def _boom(*args: object, **kwargs: object) -> object:
        raise ValueError("malformed header")

    monkeypatch.setattr(puremagic, "magic_string", _boom)
    mime = await PuremagicDetector().detect("f.bin", b"\x00\x01")
    assert str(mime) == "application/octet-stream"


# tests/unit/adapters/test_local_source.py
async def test_size_of_an_unreadable_path_raises_source_unreadable(tmp_path: Path) -> None:
    source = LocalFileSource(root=tmp_path)
    with pytest.raises(SourceUnreadableError):
        await source.size("does-not-exist.txt")


async def test_walking_a_missing_directory_raises_source_unreadable(tmp_path: Path) -> None:
    source = LocalFileSource(root=tmp_path)
    with pytest.raises(SourceUnreadableError):
        await source.walk("no-such-dir")


async def test_streaming_a_missing_file_raises_and_leaves_no_handle_open(
    tmp_path: Path,
) -> None:
    """The `finally: close()` has never run under an error.

    A leaked handle per failed stream is the kind of defect that only shows up
    under load, which is the worst time to find it.
    """
    source = LocalFileSource(root=tmp_path)
    with pytest.raises(SourceUnreadableError):
        async for _ in source.stream("missing.bin", chunk_size=8):
            pass


# tests/unit/adapters/test_hashing.py
def test_a_memo_for_a_vanished_file_misses_rather_than_raising(tmp_path: Path) -> None:
    """A file deleted between put and get must be a miss.

    The memo is an optimisation: a miss costs a rehash, never a wrong answer.
    Raising here would turn a stale optimisation into a failed read.
    """
    path = tmp_path / "gone.txt"
    path.write_bytes(b"data")
    memo = StatMemo()
    memo.put(path, ContentHash("abc"))
    path.unlink()
    assert memo.get(path) is None
```

- [ ] **Step 2: Run to see which fail**

```bash
uv run --all-extras pytest tests/unit/adapters -v -k "unreadable or vanished or library_raises or no_handle_open"
```

Expected: these describe branches believed to exist. Any that fails has found a real defect — fix the source, and note it in the report. Any that passes was simply untested; keep it.

- [ ] **Step 3: Confirm the branches are now covered**

```bash
uv run --all-extras pytest tests/unit --cov=readeverything --cov-report=term-missing 2>&1 | grep -E "detection|local_source|hashing"
```

Expected: the previously-listed missing lines in `detection.py`, `local_source.py`, and `hashing.py` are gone.

- [ ] **Step 4: Commit**

```bash
uv run --all-extras ruff format tests
git add tests/unit/adapters/
git commit -m "test(adapters): force the error branches that had never run"
```

---

## Task 5: The resolution memo

`Perception._ref` does four I/O operations per call and `_resolve` calls it from `inspect`, `invoke`, and `represent`. One `inspect` plus one `invoke` on the same path hashes the whole file twice.

**Files:**
- Create: `src/readeverything/pipeline/resolution.py`
- Modify: `src/readeverything/pipeline/perception.py`
- Test: `tests/unit/pipeline/test_resolution.py`

**Interfaces:**
- Consumes: `SourceRef`, `FileSource.local_path(uri) -> str`.
- Produces:
  - `class ResolutionMemo` with `get(uri: str, stat_key: tuple[int, int, int, int] | None) -> SourceRef | None` and `put(uri: str, stat_key: tuple[int, int, int, int] | None, ref: SourceRef) -> None`.
  - `async def stat_key(source: SourceStat, uri: str) -> tuple[int, int, int, int] | None` — `(st_dev, st_ino, st_size, st_mtime_ns)`, or `None` when the path cannot be stat'd.
  - `Perception.__init__` gains `memo: ResolutionMemo | None = None`. Default `None` means no memoization, so every existing test keeps its current behaviour.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/pipeline/test_resolution.py
async def test_a_second_resolve_of_one_path_does_not_rehash() -> None:
    """The point of the memo, stated as a call count rather than a duration.

    Timing assertions are flaky; counting the calls through a real adapter is
    not. Hashing is the expensive operation and it scales with file size.
    """
    counting = CountingHasher(inner=ContentHasher(source=source))
    perception = _perception(hasher=counting, memo=ResolutionMemo())
    await perception.inspect("a.txt")
    await perception.inspect("a.txt")
    assert counting.calls == 1


async def test_a_rewritten_file_is_resolved_again() -> None:
    """A stale ref is the worst thing this cache could produce.

    The memo is keyed on (dev, inode, size, mtime_ns), so a rewrite invalidates
    even when the size is unchanged.
    """
    path = tmp_path / "a.txt"
    path.write_bytes(b"first")
    perception = _perception(memo=ResolutionMemo())
    before = await perception.inspect("a.txt")
    os.utime(path, ns=(0, 0))
    path.write_bytes(b"secnd")  # same length, different content
    after = await perception.inspect("a.txt")
    assert before.ref.content_hash != after.ref.content_hash


async def test_a_hit_and_a_miss_are_indistinguishable() -> None:
    """The property that makes the memo safe to have at all."""
    cold = await _perception(memo=None).inspect("a.txt")
    warm_perception = _perception(memo=ResolutionMemo())
    await warm_perception.inspect("a.txt")
    warm = await warm_perception.inspect("a.txt")
    assert cold.ref == warm.ref


async def test_a_source_that_cannot_be_stat_is_never_memoized() -> None:
    """Without a stat there is no invalidation rule, so there is no caching.

    A non-local source (an object store) has no inode. Memoizing it on the uri
    alone would serve a stale ref forever after the object changed.
    """
    perception = _perception(source=FakeSource({"a.txt": b"x"}), memo=ResolutionMemo())
    first = await perception.inspect("a.txt")
    second = await perception.inspect("a.txt")
    assert first.ref == second.ref  # correct, just not cached
```

Write `CountingHasher` in the test file:

```python
class CountingHasher:
    """Counts `hash` calls through a real hasher rather than replacing it."""

    def __init__(self, *, inner: ContentHashing) -> None:
        self._inner = inner
        self.calls = 0

    async def hash(self, uri: str) -> ContentHash:
        self.calls += 1
        return await self._inner.hash(uri)
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run --all-extras pytest tests/unit/pipeline/test_resolution.py -v
```

Expected: FAIL — `readeverything.pipeline.resolution` does not exist.

- [ ] **Step 3: Implement `resolution.py`**

```python
"""Remembering what a path resolved to, for as long as the path stands still.

This is deliberately NOT the artifact store, and the difference is the whole
reason it is a separate file. The artifact store is content-addressed: its key
contains the content hash, so an entry can never go stale and it never needs
invalidating. This memo is keyed on a *path*, which is mutable, so it carries an
invalidation rule — `(dev, inode, size, mtime_ns)`, the same rule `StatMemo`
already uses for hashes.

Conflating the two would give the content-addressed store a staleness protocol
it does not need and must not acquire.

A source that cannot be stat'd is never memoized. Without a stat there is no
invalidation rule, and caching on the uri alone would serve a stale ref forever
after a non-local object changed.
"""

from __future__ import annotations

from pathlib import Path

from readeverything.domain.identity import SourceRef
from readeverything.ports.source import SourceStat

type StatKey = tuple[int, int, int, int]


async def stat_key(source: SourceStat, uri: str) -> StatKey | None:
    """`(dev, inode, size, mtime_ns)` for `uri`, or None if it cannot be stat'd."""
    try:
        path = Path(await source.local_path(uri))
        stat = path.stat()
    except (OSError, NotImplementedError):
        return None
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


class ResolutionMemo:
    """Maps a uri to the `SourceRef` it produced, while its stat is unchanged."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[StatKey, SourceRef]] = {}

    def get(self, uri: str, key: StatKey | None) -> SourceRef | None:
        if key is None:
            return None
        entry = self._entries.get(uri)
        if entry is None or entry[0] != key:
            return None
        return entry[1]

    def put(self, uri: str, key: StatKey | None, ref: SourceRef) -> None:
        if key is None:
            return
        self._entries[uri] = (key, ref)
```

- [ ] **Step 4: Wire it into `Perception`**

```python
    def __init__(
        self,
        *,
        source: FileSource,
        detector: MimeDetector,
        hasher: ContentHashing,
        registry: MimeTypeRegistry,
        artifacts: ArtifactStore,
        memo: ResolutionMemo | None = None,
    ) -> None:
        self._source = source
        self._detector = detector
        self._hasher = hasher
        self._registry = registry
        self._artifacts = artifacts
        self._memo = memo

    async def _ref(self, uri: str) -> SourceRef:
        key = None if self._memo is None else await stat_key(self._source, uri)
        if self._memo is not None:
            cached = self._memo.get(uri, key)
            if cached is not None:
                return cached
        head = await self._source.read_range(uri, 0, _HEAD_BYTES)
        ref = SourceRef(
            uri=uri,
            mime=await self._detector.detect(uri, head),
            content_hash=await self._hasher.hash(uri),
            size_bytes=await self._source.size(uri),
        )
        if self._memo is not None:
            self._memo.put(uri, key, ref)
        return ref
```

- [ ] **Step 5: Run the tests**

```bash
uv run --all-extras pytest tests/unit -q
```

Expected: PASS, including all pre-existing pipeline tests unchanged (they pass no `memo`, so they get the old behaviour).

- [ ] **Step 6: Commit**

```bash
uv run --all-extras ruff format src tests
git add src/readeverything/pipeline/ tests/unit/pipeline/test_resolution.py
git commit -m "feat(pipeline): remember what a path resolved to while its stat holds"
```

---

## Task 6: Consult the artifact store

`Perception` has held an `ArtifactStore` since Plan 1 and has never read it.

**Files:**
- Modify: `src/readeverything/pipeline/perception.py` (`invoke`)
- Test: `tests/unit/pipeline/test_perception_caching.py`

**Interfaces:**
- Consumes: `artifact_key(*, content_hash, handler_id, handler_version, affordance, params, capabilities) -> str`; `ArtifactStore.get(key) -> bytes | None`, `.put(key, value) -> None`; `MediaHandler.handler_id`/`handler_version` ClassVars.
- Produces: `MimeTypeRegistry.capabilities` property returning the `CapabilitySet` — needed for the key and currently private. Add it in this task.

- [ ] **Step 1: Write the failing tests**

```python
async def test_a_repeated_invoke_returns_an_identical_rendition() -> None:
    """A hit and a miss must be indistinguishable in their result.

    This is the property that makes caching safe to have at all, and it is what
    the tests assert — not a speedup, which would be a timing assertion.
    """
    perception = _perception(artifacts=InMemoryArtifactStore())
    first = await perception.invoke("a.txt", "read_range", {"start": 0, "end": 5})
    second = await perception.invoke("a.txt", "read_range", {"start": 0, "end": 5})
    assert first == second


async def test_a_second_invoke_does_not_re-enter_the_handler() -> None:
    handler = CountingTextHandler(source=source)
    perception = _perception(handlers=[handler], artifacts=InMemoryArtifactStore())
    await perception.invoke("a.txt", "read_range", {"start": 0, "end": 5})
    await perception.invoke("a.txt", "read_range", {"start": 0, "end": 5})
    assert handler.invocations == 1


async def test_different_params_do_not_share_an_artifact() -> None:
    perception = _perception(artifacts=InMemoryArtifactStore())
    a = await perception.invoke("a.txt", "read_range", {"start": 0, "end": 5})
    b = await perception.invoke("a.txt", "read_range", {"start": 5, "end": 10})
    assert a != b


async def test_a_changed_capability_fingerprint_does_not_serve_the_old_artifact() -> None:
    """Swapping the model must miss, or the index becomes a mixture.

    This is the component of the key that is easiest to forget and the one
    whose absence is silent.
    """
    store = InMemoryArtifactStore()
    first = await _perception(
        artifacts=store, capabilities=CapabilitySet.of({Capability.VISION: "model-a"})
    ).invoke("a.txt", "read_range", {"start": 0, "end": 5})
    second = await _perception(
        artifacts=store, capabilities=CapabilitySet.of({Capability.VISION: "model-b"})
    ).invoke("a.txt", "read_range", {"start": 0, "end": 5})
    assert first == second  # same input, same answer
    assert len(store.keys()) == 2  # but stored under two keys


async def test_a_handler_without_a_version_is_not_cached() -> None:
    """Cache participation is the handler's decision, not the pipeline's."""
    handler = CountingTextHandler(source=source)
    handler.handler_version = 0  # 0 means "do not cache me"
    perception = _perception(handlers=[handler], artifacts=InMemoryArtifactStore())
    await perception.invoke("a.txt", "read_range", {"start": 0, "end": 5})
    await perception.invoke("a.txt", "read_range", {"start": 0, "end": 5})
    assert handler.invocations == 2
```

`InMemoryArtifactStore` needs a `keys()` accessor for the fourth test; add it if absent.

- [ ] **Step 2: Run to verify they fail**

```bash
uv run --all-extras pytest tests/unit/pipeline/test_perception_caching.py -v
```

Expected: FAIL — nothing is cached.

- [ ] **Step 3: Add the `capabilities` property to the registry**

```python
    @property
    def capabilities(self) -> CapabilitySet:
        """What this deployment can do. Part of every artifact cache key."""
        return self._capabilities
```

- [ ] **Step 4: Implement caching in `invoke`**

Renditions are serialized as JSON via pydantic so the store stays a plain `bytes` interface.

```python
    async def invoke(self, uri: str, name: str, params: Mapping[str, Any]) -> Rendition:
        """Invoke a named affordance. Raises if it is not available here."""
        ref, handler = await self._resolve(uri)
        affordance = self._affordance(handler, name)
        validated = affordance.params.model_validate(dict(params))

        # `handler_version` of 0 means the handler is opting out: its output is
        # cheap or nondeterministic enough that an artifact would cost more than
        # it saves. The decision belongs to the handler, which knows what it
        # does, not to the pipeline, which does not.
        if handler.handler_version == 0:
            return await handler.invoke(ref, name, validated)

        key = artifact_key(
            content_hash=ref.content_hash,
            handler_id=handler.handler_id,
            handler_version=handler.handler_version,
            affordance=name,
            params=validated.model_dump(mode="json"),
            capabilities=self._registry.capabilities,
        )
        cached = await self._artifacts.get(key)
        if cached is not None:
            return Rendition.model_validate_json(cached)
        rendition = await handler.invoke(ref, name, validated)
        await self._artifacts.put(key, rendition.model_dump_json().encode("utf-8"))
        return rendition
```

If `Rendition` is a frozen dataclass rather than a pydantic model, serialize with `pydantic.TypeAdapter(Rendition)` instead — check before writing, and use whichever the type actually is. Report which you found.

- [ ] **Step 5: Run the tests**

```bash
uv run --all-extras pytest tests/unit -q && uv run --all-extras mypy
```

Expected: PASS and clean.

- [ ] **Step 6: Commit**

```bash
uv run --all-extras ruff format src tests
git add src/readeverything/pipeline/perception.py src/readeverything/registry/registry.py tests/unit/pipeline/test_perception_caching.py
git commit -m "feat(pipeline): consult the artifact store that has been injected since plan 1"
```

---

## Task 7: Capability discovery

**Files:**
- Create: `src/readeverything/ports/probe.py`, `src/readeverything/adapters/binary_probe.py`, `src/readeverything/adapters/model_probe.py`
- Test: `tests/unit/adapters/test_binary_probe.py`, `tests/unit/adapters/test_model_probe.py`
- Modify: `tests/unit/test_dependencies_stay_confined.py` (new module uses `subprocess` — add it)

**Interfaces:**
- Produces:
  - `class CapabilityProbe(Protocol)` with `async def revision(self, capability: Capability) -> str | None`.
  - `class BinaryProbe` — `__init__(self, *, timeout_s: float = 5.0)`; probes `FFMPEG`, `EXIFTOOL`, `LIBREOFFICE`, `TESSERACT`.
  - `class ModelProbe` — `__init__(self, *, vision: VisionModel | None = None)`; returns `vision.model_id` for `Capability.VISION`.
  - `async def discover(*, probes: Sequence[CapabilityProbe], capabilities: Iterable[Capability] | None = None) -> CapabilitySet`, in a third small module `src/readeverything/adapters/probing.py`. It belongs beside neither probe: it orders them, so it depends on both and neither depends on it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/adapters/test_binary_probe.py
async def test_a_missing_binary_is_unavailable() -> None:
    probe = BinaryProbe(executables={Capability.FFMPEG: "definitely-not-a-real-binary-xyz"})
    assert await probe.revision(Capability.FFMPEG) is None


async def test_a_present_binary_reports_a_revision() -> None:
    """`echo` stands in for ffmpeg: present, runs, prints something.

    What is under test is the probe's contract — locate, run, capture — not
    ffmpeg's version string, which would make this test require ffmpeg.
    """
    probe = BinaryProbe(executables={Capability.FFMPEG: "echo"}, version_flag="7.1")
    revision = await probe.revision(Capability.FFMPEG)
    assert revision is not None and "7.1" in revision


async def test_a_binary_that_hangs_is_unavailable_rather_than_hanging(tmp_path: Path) -> None:
    """A hung `--version` must not hang composition.

    Composition happens at startup. A probe without a timeout turns one broken
    binary into an application that never finishes starting.
    """
    script = tmp_path / "hang"
    script.write_text("#!/bin/sh\nsleep 30\n")
    script.chmod(0o755)
    probe = BinaryProbe(executables={Capability.FFMPEG: str(script)}, timeout_s=0.2)
    assert await probe.revision(Capability.FFMPEG) is None


async def test_a_probe_never_raises() -> None:
    """Under uncertainty the library offers less, never more."""
    probe = BinaryProbe(executables={})
    assert await probe.revision(Capability.TESSERACT) is None


# tests/unit/adapters/test_model_probe.py
async def test_the_vision_revision_is_the_injected_model_s_own_id() -> None:
    """The revision cannot disagree with the model, because it is derived from it.

    Before this, the VISION revision and `VisionModel.model_id` were two
    independent inputs that happened to agree by convention. Once artifacts are
    cached, disagreement means keys that misdescribe the model that produced
    them.
    """
    probe = ModelProbe(vision=FakeVision())
    assert await probe.revision(Capability.VISION) == FakeVision().model_id


async def test_no_vision_model_means_no_vision_capability() -> None:
    assert await ModelProbe(vision=None).revision(Capability.VISION) is None


# discovery
async def test_discovery_reports_only_what_answered() -> None:
    capabilities = await discover(
        probes=[BinaryProbe(executables={}), ModelProbe(vision=FakeVision())],
        capabilities=list(Capability),
    )
    assert capabilities.satisfies({Capability.VISION})
    assert not capabilities.satisfies({Capability.FFMPEG})
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run --all-extras pytest tests/unit/adapters/test_binary_probe.py tests/unit/adapters/test_model_probe.py -v
```

Expected: FAIL — the modules do not exist.

- [ ] **Step 3: Write `ports/probe.py`**

```python
"""Asking the machine what it can actually do.

Capability negotiation was sound from Plan 1 and its inputs were not: a caller
hand-asserted that ffmpeg existed, and a wrong assertion registered affordances
that could not run — the exact failure negotiation exists to prevent. A probe
replaces an assertion with an observation.

A probe never raises and never guesses. `None` means unavailable, so under
uncertainty the library offers less rather than more.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from readeverything.domain.capability import Capability


@runtime_checkable
class CapabilityProbe(Protocol):
    async def revision(self, capability: Capability) -> str | None:
        """A revision string if this capability is genuinely available, else None."""
        ...
```

- [ ] **Step 4: Write `adapters/binary_probe.py`**

```python
"""Probing OS executables by running them.

Availability means *it ran and reported a version*, not that a file exists at a
path. A binary that is present and broken is not a capability, and finding that
out at composition time is much cheaper than finding it out inside a handler
three layers down.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Mapping

from readeverything.domain.capability import Capability

#: The executable each capability is provided by, and the flag that makes it
#: print a version and exit. Both are fixed here rather than caller-supplied:
#: this module runs subprocesses, and the argument vector must never contain
#: anything a caller can influence.
DEFAULT_EXECUTABLES: Mapping[Capability, str] = {
    Capability.FFMPEG: "ffmpeg",
    Capability.EXIFTOOL: "exiftool",
    Capability.LIBREOFFICE: "libreoffice",
    Capability.TESSERACT: "tesseract",
}


class BinaryProbe:
    """Reports a capability available when its executable runs and answers."""

    def __init__(
        self,
        *,
        executables: Mapping[Capability, str] | None = None,
        version_flag: str = "-version",
        timeout_s: float = 5.0,
    ) -> None:
        self._executables = dict(DEFAULT_EXECUTABLES if executables is None else executables)
        self._version_flag = version_flag
        self._timeout_s = timeout_s

    async def revision(self, capability: Capability) -> str | None:
        name = self._executables.get(capability)
        if name is None:
            return None
        located = shutil.which(name)
        if located is None:
            return None
        try:
            process = await asyncio.create_subprocess_exec(
                located,
                self._version_flag,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError:
            return None
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=self._timeout_s)
        except TimeoutError:
            # A hung `--version` must not hang composition, which happens at
            # startup. Kill it and report unavailable.
            process.kill()
            await process.wait()
            return None
        if process.returncode != 0:
            return None
        first_line = stdout.decode("utf-8", errors="replace").strip().splitlines()
        return first_line[0].strip() if first_line else None
```

**Security note for the reviewer:** `create_subprocess_exec` takes an argument vector, never a shell string, and both the executable name and the flag come from module constants or explicit constructor arguments — never from a uri, a file, or model output. Bandit must stay clean.

- [ ] **Step 5: Write `adapters/model_probe.py`**

```python
"""Deriving a capability's revision from the model object itself.

This closes a seam Plan 2's review flagged: the VISION revision in a
`CapabilitySet` and the `model_id` of the injected `VisionModel` were
independent inputs with nothing requiring them to agree. Deriving one from the
other makes disagreement impossible rather than merely discouraged — which
matters once artifacts are cached, because a key that misdescribes the model
that produced it serves a mixture of two models' output as though it were one.
"""

from __future__ import annotations

from readeverything.domain.capability import Capability
from readeverything.ports.vision import VisionModel


class ModelProbe:
    """Reports model-backed capabilities from the models actually injected."""

    def __init__(self, *, vision: VisionModel | None = None) -> None:
        self._vision = vision

    async def revision(self, capability: Capability) -> str | None:
        if capability is Capability.VISION and self._vision is not None:
            return self._vision.model_id
        return None
```

- [ ] **Step 6: Write `adapters/probing.py`**

```python
"""Running every probe and assembling what answered."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.ports.probe import CapabilityProbe


async def discover(
    *,
    probes: Sequence[CapabilityProbe],
    capabilities: Iterable[Capability] | None = None,
) -> CapabilitySet:
    """Probe for each capability and report only what answered.

    First answer wins, so ordering the probes orders precedence. A capability
    no probe answers for is absent, which is what makes the result an
    observation rather than a hopeful assertion.
    """
    wanted = list(Capability) if capabilities is None else list(capabilities)
    found: dict[Capability, str] = {}
    for capability in wanted:
        for probe in probes:
            revision = await probe.revision(capability)
            if revision is not None:
                found[capability] = revision
                break
    return CapabilitySet.of(found)
```

- [ ] **Step 7: Add `subprocess`-family imports to the dependency-confinement test**

`binary_probe.py` imports `asyncio` and `shutil`. Add the module to whatever allowlist `tests/unit/test_dependencies_stay_confined.py` maintains, with a comment naming why.

- [ ] **Step 8: Run everything**

```bash
uv run --all-extras pytest tests/unit -q && uv run --all-extras mypy && uv run --all-extras bandit -c pyproject.toml -r src -q
```

Expected: PASS, clean, no bandit findings.

- [ ] **Step 9: Commit**

```bash
uv run --all-extras ruff format src tests
git add src/readeverything/ports/probe.py src/readeverything/adapters/binary_probe.py src/readeverything/adapters/model_probe.py src/readeverything/adapters/probing.py tests/unit/adapters/ tests/unit/test_dependencies_stay_confined.py
git commit -m "feat(adapters): give capability negotiation the discovery half it never had"
```

---

## Task 8: The composition root

**Files:**
- Create: `src/readeverything/composition.py`
- Modify: `pyproject.toml` (import-linter layers), `src/readeverything/__init__.py`
- Test: `tests/unit/test_composition.py`

**Interfaces:**
- Produces:

```python
async def build_perception(
    root: Path | str,
    *,
    vision: VisionModel | None = None,
    capabilities: CapabilitySet | None = None,
    artifacts: ArtifactStore | None = None,
    probe_binaries: bool = True,
) -> Perception
```

- [ ] **Step 1: Write the failing tests**

```python
async def test_a_root_is_the_only_thing_a_caller_must_supply(tmp_path: Path) -> None:
    """The acceptance sentence, as a test.

    Before this, a caller assembled ten objects in dependency order and had to
    know which handler classes existed to do it.
    """
    (tmp_path / "a.txt").write_text("hello")
    perception = await build_perception(tmp_path)
    card = await perception.inspect("a.txt")
    assert card.ref.uri == "a.txt"


async def test_a_base_install_without_pillow_still_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Handlers whose dependencies are absent are omitted, not fatal.

    The front door advertises `ImageHandler` on a base install and importing it
    raises ModuleNotFoundError: PIL. A composition root that propagated that
    would make the base install unusable rather than merely narrower.
    """
    monkeypatch.setitem(sys.modules, "PIL", None)
    (tmp_path / "a.txt").write_text("hello")
    perception = await build_perception(tmp_path)
    assert await perception.inspect("a.txt") is not None


async def test_explicit_capabilities_are_used_verbatim_and_nothing_is_probed(
    tmp_path: Path,
) -> None:
    """Tests must be able to declare any capability set without touching the machine."""
    declared = CapabilitySet.of({Capability.FFMPEG: "declared-not-probed"})
    perception = await build_perception(tmp_path, capabilities=declared, probe_binaries=False)
    assert perception.registry.capabilities == declared


async def test_a_vision_model_registers_the_affordances_that_need_it(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(_png_bytes())
    with_vision = await build_perception(tmp_path, vision=FakeVision(), probe_binaries=False)
    names = {a.name for a in (await with_vision.inspect("a.png")).affordances}
    assert {"describe_image", "ocr"} <= names


async def test_no_vision_model_means_those_affordances_are_not_offered(tmp_path: Path) -> None:
    """Negotiation working, not degradation. The agent never sees a tool it cannot use."""
    (tmp_path / "a.png").write_bytes(_png_bytes())
    without = await build_perception(tmp_path, probe_binaries=False)
    names = {a.name for a in (await without.inspect("a.png")).affordances}
    assert "describe_image" not in names
    assert "crop_region" in names


def test_the_composition_root_reads_no_environment() -> None:
    """The constraint that has held since Spec 1 §3, checked at the new top layer."""
    source = Path("src/readeverything/composition.py").read_text()
    assert "os.environ" not in source
    assert "getenv" not in source
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run --all-extras pytest tests/unit/test_composition.py -v
```

Expected: FAIL — the module does not exist.

- [ ] **Step 3: Write `composition.py`**

```python
"""One function between a directory and a working `Perception`.

Everything here is a convenience over the public constructors and nothing more.
If this module can do something a caller assembling the pieces by hand cannot,
that is a bug in the constructors, not a feature of this file — which is why it
takes the same arguments they do and holds no state of its own.

It reads no environment variables. Every input is an argument, so two
differently-configured instances can run in one process.
"""

from __future__ import annotations

from pathlib import Path

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.binary_probe import BinaryProbe
from readeverything.adapters.detection import PuremagicDetector
from readeverything.adapters.hashing import ContentHasher, StatMemo
from readeverything.adapters.local_source import LocalFileSource
from readeverything.adapters.model_probe import ModelProbe
from readeverything.adapters.probing import discover
from readeverything.domain.capability import CapabilitySet
from readeverything.handlers.binary import BinaryHandler
from readeverything.handlers.text import TextHandler
from readeverything.pipeline.perception import Perception
from readeverything.pipeline.resolution import ResolutionMemo
from readeverything.ports.artifacts import ArtifactStore
from readeverything.ports.handler import MediaHandler
from readeverything.ports.source import SourceReader
from readeverything.ports.vision import VisionModel
from readeverything.registry.registry import MimeTypeRegistry


def _optional_image_handler(
    source: SourceReader, vision: VisionModel | None
) -> list[MediaHandler]:
    """`ImageHandler` when Pillow is importable, nothing when it is not.

    Pillow lives behind the `images` extra. A base install must yield a working
    `Perception` that handles text and binary — narrower, not broken — so the
    import failure is a registration decision here rather than an exception
    reaching the caller.
    """
    try:
        from readeverything.handlers.image import ImageHandler
    except ImportError:
        return []
    return [ImageHandler(source=source, vision=vision)]


async def build_perception(
    root: Path | str,
    *,
    vision: VisionModel | None = None,
    capabilities: CapabilitySet | None = None,
    artifacts: ArtifactStore | None = None,
    probe_binaries: bool = True,
) -> Perception:
    """A `Perception` over `root`, with everything else defaulted.

    `capabilities` given explicitly is used verbatim and nothing is probed —
    a test must be able to declare any capability set without depending on what
    happens to be installed on the machine running it.
    """
    source = LocalFileSource(root=root)
    if capabilities is None:
        probes = [ModelProbe(vision=vision)]
        if probe_binaries:
            probes.append(BinaryProbe())
        capabilities = await discover(probes=probes)
    handlers: list[MediaHandler] = [
        TextHandler(source=source),
        *_optional_image_handler(source, vision),
        # The fallback claims "*", so it must be last: the registry breaks a
        # rank tie by registration order, and a fallback registered first would
        # shadow nothing but would rank ahead of an equally-specific match.
        BinaryHandler(source=source),
    ]
    return Perception(
        source=source,
        detector=PuremagicDetector(),
        hasher=ContentHasher(source=source, memo=StatMemo()),
        registry=MimeTypeRegistry(handlers=handlers, capabilities=capabilities),
        artifacts=InMemoryArtifactStore() if artifacts is None else artifacts,
        memo=ResolutionMemo(),
    )
```

- [ ] **Step 4: Expose the registry on `Perception`**

The composition test asserts `perception.registry.capabilities`. Add:

```python
    @property
    def registry(self) -> MimeTypeRegistry:
        """The registry this perception dispatches through."""
        return self._registry
```

- [ ] **Step 5: Add the new outermost layer to import-linter**

In `pyproject.toml`, the layered contract becomes:

```toml
layers = [
    "composition",
    "testing",
    "agent",
    "pipeline",
    "registry",
    "handlers",
    "adapters",
    "ports",
    "domain",
]
```

- [ ] **Step 6: Export from the front door**

Add to `_LAZY`, keeping it alphabetically sorted:

```python
    "BinaryProbe": "readeverything.adapters.binary_probe",
    "CapabilityProbe": "readeverything.ports.probe",
    "LangChainVisionModel": "readeverything.adapters.vision_langchain",
    "ModelProbe": "readeverything.adapters.model_probe",
    "ResolutionMemo": "readeverything.pipeline.resolution",
    "build_openai_vision_model": "readeverything.adapters.vision_langchain",
    "build_perception": "readeverything.composition",
    "discover": "readeverything.adapters.probing",
```

Add matching `TYPE_CHECKING` imports in the same style as the existing block.

- [ ] **Step 7: Add a test that the front door and the registry cannot drift**

```python
def test_every_bundled_handler_is_exported_from_the_front_door() -> None:
    """Adding a handler must mean adding it in exactly one place.

    A handler registered in the composition root but missing from the front
    door — or the reverse — is a surface that lies about itself.
    """
    import readeverything

    registered = {"TextHandler", "BinaryHandler", "ImageHandler"}
    assert registered <= set(readeverything.__all__)
```

- [ ] **Step 8: Run everything**

```bash
make check
```

Expected: all five gates green.

- [ ] **Step 9: Commit**

```bash
uv run --all-extras ruff format src tests
git add src/readeverything/composition.py src/readeverything/__init__.py src/readeverything/pipeline/perception.py pyproject.toml tests/unit/test_composition.py
git commit -m "feat: one function between a directory and a working perception"
```

---

## Task 9: A clear error when Pillow is missing

`ImageHandler` resolves lazily from the front door and then fails on `PIL`, naming a package the user never asked for instead of the extra they need.

**Files:**
- Modify: `src/readeverything/handlers/image.py` (the `PIL` import)
- Test: `tests/unit/handlers/test_image_handler.py`

**Interfaces:** no signature change.

- [ ] **Step 1: Write the failing test**

```python
def test_a_missing_pillow_names_the_extra_not_the_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ModuleNotFoundError: No module named 'PIL'` is a true statement that
    does not help. The user installed `deepagents-read-everything`; the thing
    they can act on is the name of the extra."""
    monkeypatch.setitem(sys.modules, "PIL", None)
    for module in [m for m in sys.modules if m.startswith("readeverything.handlers.image")]:
        monkeypatch.delitem(sys.modules, module, raising=False)
    with pytest.raises(ImportError, match=r"images"):
        importlib.import_module("readeverything.handlers.image")
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run --all-extras pytest tests/unit/handlers/test_image_handler.py -v -k missing_pillow
```

Expected: FAIL — the raised message names `PIL`, not `images`.

- [ ] **Step 3: Implement**

Replace the bare import in `image.py`:

```python
try:
    from PIL import Image, UnidentifiedImageError
except ImportError as exc:  # pragma: no cover - exercised via a patched sys.modules
    raise ImportError(
        "readeverything's image support needs Pillow, which ships in the "
        "'images' extra: pip install 'deepagents-read-everything[images]'. "
        "The composition root omits image handling when Pillow is absent, so "
        "reaching this means the handler was imported directly."
    ) from exc
```

- [ ] **Step 4: Run the tests**

```bash
uv run --all-extras pytest tests/unit/handlers -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uv run --all-extras ruff format src tests
git add src/readeverything/handlers/image.py tests/unit/handlers/test_image_handler.py
git commit -m "fix(handlers): name the extra rather than the package Pillow lives in"
```

---

## Task 10: The integration tier

**Files:**
- Create: `tests/integration/__init__.py`, `tests/integration/conftest.py`, `tests/integration/test_artifact_store_compliance.py`, `tests/integration/test_agent_path.py`, `tests/integration/test_capability_negotiation.py`, `tests/integration/test_caching.py`
- Modify: `pyproject.toml` (`testpaths`, and confirm `integration` is not deselected)

**Interfaces:**
- Consumes: `build_perception`, `build_tools`, `ArtifactStoreCompliance`, `InMemoryArtifactStore`, `FilesystemArtifactStore`, `FakeVision`.
- Produces: `tests/integration/conftest.py` supplying a `media_root` fixture — a real `tmp_path` containing `notes.txt`, `data.bin`, and `photo.png`.

The tier's rule, stated in `tests/integration/__init__.py` as a module docstring so it cannot be lost:

> An integration test constructs real components through `build_perception` and
> asserts on behaviour crossing at least two module boundaries. It may use a
> fake **model**, because model output is nondeterministic and this project has
> never asserted on model text. It may not use a fake **source, detector,
> hasher, store, or registry** — those are the seams under test.

- [ ] **Step 1: Give the shipped law an implementation to fail against**

```python
# tests/integration/test_artifact_store_compliance.py
"""The `ArtifactStore` law, run against both real stores.

The suite has shipped in the wheel since Plan 1 and no implementation had ever
subclassed it. An unexercised law is not a law that passes — it is one that has
never been given the chance to fail, which is the thing this project says it
does not do.
"""

import pytest

from readeverything.adapters.artifact_store import FilesystemArtifactStore, InMemoryArtifactStore
from readeverything.testing.artifact_compliance import ArtifactStoreCompliance


class TestInMemoryArtifactStoreCompliance(ArtifactStoreCompliance):
    @pytest.fixture
    def store(self) -> InMemoryArtifactStore:
        return InMemoryArtifactStore()


class TestFilesystemArtifactStoreCompliance(ArtifactStoreCompliance):
    @pytest.fixture
    def store(self, tmp_path) -> FilesystemArtifactStore:
        return FilesystemArtifactStore(root=tmp_path)
```

- [ ] **Step 2: Run it and expect it to reveal something**

```bash
uv run --all-extras pytest tests/integration/test_artifact_store_compliance.py -v
```

Expected: the four laws run against both stores for the first time. If either fails — particularly `test_entries_are_immutable`, which asserts a second `put` does not replace — that is a real defect in a store that was never checked. Fix the store, not the law, and report it.

- [ ] **Step 3: The full agent path**

```python
# tests/integration/test_agent_path.py
async def test_an_agent_can_see_a_directory_of_mixed_files(media_root) -> None:
    """The acceptance sentence, end to end, through the tools an agent holds.

    Every component here is real except the model. This is the first test in
    the project where the tool pack meets image handling at all.
    """
    perception = await build_perception(media_root, vision=FakeVision(), probe_binaries=False)
    tools = {tool.name: tool for tool in build_tools(perception)}
    assert set(tools) == {"inspect_path", "list_paths", "invoke_affordance"}

    listing = await tools["list_paths"].ainvoke({"uri": "."})
    assert "notes.txt" in listing and "photo.png" in listing

    card = json.loads(await tools["inspect_path"].ainvoke({"uri": "photo.png"}))
    assert card["kind"] == "image"
    assert "describe_image" in {a["name"] for a in card["affordances"]}

    described = await tools["invoke_affordance"].ainvoke(
        {"uri": "photo.png", "affordance": "describe_image", "params": {}}
    )
    assert "located at" in described


async def test_a_tool_call_against_a_missing_file_returns_rather_than_raises(
    media_root,
) -> None:
    """The tool pack never raises. An agent gets a result it can read and retry."""
    perception = await build_perception(media_root, probe_binaries=False)
    tools = {tool.name: tool for tool in build_tools(perception)}
    result = await tools["inspect_path"].ainvoke({"uri": "nope.txt"})
    assert "ERROR" in result
```

- [ ] **Step 4: Capability negotiation, end to end**

```python
# tests/integration/test_capability_negotiation.py
async def test_the_same_directory_offers_less_without_a_vision_model(media_root) -> None:
    """One directory, two deployments, and the difference is visible to the agent."""
    with_vision = await build_perception(media_root, vision=FakeVision(), probe_binaries=False)
    without = await build_perception(media_root, probe_binaries=False)

    rich = {a.name for a in (await with_vision.inspect("photo.png")).affordances}
    plain = {a.name for a in (await without.inspect("photo.png")).affordances}

    assert plain < rich
    assert "crop_region" in plain  # needs no model, so it survives
    assert {"describe_image", "ocr"} & plain == set()


async def test_nothing_unavailable_is_ever_offered(media_root) -> None:
    """The design goal, asserted directly: an agent never sees a tool it cannot use."""
    perception = await build_perception(media_root, probe_binaries=False)
    capabilities = perception.registry.capabilities
    for uri in await perception.list("."):
        for affordance in (await perception.inspect(uri)).affordances:
            assert affordance.is_available(capabilities)
```

- [ ] **Step 5: Caching, end to end**

```python
# tests/integration/test_caching.py
class CountingHasher:
    """Wraps a real hasher and counts calls through it.

    This is the one place the tier's rule bends, and the bend is narrow on
    purpose: the hasher is real and does real work, and this only observes how
    often it is asked. Counting is not faking — a fake would answer instead of
    the adapter, and then the test would be about the fake.
    """

    def __init__(self, *, inner: ContentHashing) -> None:
        self._inner = inner
        self.calls = 0

    async def hash(self, uri: str) -> ContentHash:
        self.calls += 1
        return await self._inner.hash(uri)


async def test_a_second_look_at_a_file_hashes_it_once(media_root) -> None:
    """One inspect plus one invoke used to hash the whole file twice.

    Asserted as a call count rather than a duration: timing assertions are
    flaky, and the claim being made is about repeated work, not about speed.
    """
    source = LocalFileSource(root=media_root)
    hasher = CountingHasher(inner=ContentHasher(source=source, memo=StatMemo()))
    perception = Perception(
        source=source,
        detector=PuremagicDetector(),
        hasher=hasher,
        registry=MimeTypeRegistry(
            handlers=[TextHandler(source=source), BinaryHandler(source=source)],
            capabilities=CapabilitySet.empty(),
        ),
        artifacts=InMemoryArtifactStore(),
        memo=ResolutionMemo(),
    )
    await perception.inspect("notes.txt")
    await perception.invoke("notes.txt", "read_range", {"start": 0, "end": 4})
    assert hasher.calls == 1


async def test_a_rewritten_file_is_seen_again(media_root) -> None:
    """The failure mode that would make caching worse than no caching."""
    perception = await build_perception(media_root, probe_binaries=False)
    before = await perception.inspect("notes.txt")
    (media_root / "notes.txt").write_text("completely different content here")
    after = await perception.inspect("notes.txt")
    assert before.ref.content_hash != after.ref.content_hash
```

If `test_a_second_look_at_a_file_hashes_it_once` reports 2, the resolution memo
is not being consulted on the `invoke` path — check that `_resolve` goes through
`_ref` in both callers rather than that the memo is broken.

- [ ] **Step 6: Make the tier run by default**

Confirm `addopts = "-m 'not integration and not live and not accuracy'"` would deselect this tier, and change it. The `integration` marker's meaning changes from "requires OS binaries" to "real components, no network", so update the marker description too:

```toml
markers = [
    "integration: real components wired together; no network, no model server",
    "live: requires a running model server",
    "accuracy: quality benchmark, not a pass/fail test",
    "slow: takes more than a second",
]
addopts = "-m 'not live and not accuracy'"
```

A tier deselected by default is a tier that rots — which is exactly what happened to this marker already.

- [ ] **Step 7: Run everything**

```bash
make check
```

Expected: all five gates green, with the integration tier included in the default run.

- [ ] **Step 8: Commit**

```bash
uv run --all-extras ruff format tests
git add tests/integration/ pyproject.toml
git commit -m "test: an integration tier, and a law that can finally fail"
```

---

## Task 11: The README, and the coverage floor

**Files:**
- Modify: `README.md` (currently 0 bytes), `pyproject.toml` (`fail_under`)
- Modify: `docs/superpowers/specs/2026-08-14-readeverything-perception-core-design.md` (§14b)
- Test: `tests/integration/test_readme_example.py`

**Interfaces:**
- Consumes: `build_perception`, `build_tools`.

- [ ] **Step 1: Write the failing test**

The README's example must be executable, not aspirational.

```python
# tests/integration/test_readme_example.py
async def test_the_readme_example_runs(tmp_path) -> None:
    """A README example that does not run is a bug report with good formatting.

    This test is the reason the example in the README can be trusted, and it is
    why the example must stay small enough to assert on.
    """
    (tmp_path / "notes.txt").write_text("the quick brown fox")
    perception = await build_perception(tmp_path)
    card = await perception.inspect("notes.txt")
    tools = build_tools(perception)
    assert card.kind == "text"
    assert len(tools) == 3
```

- [ ] **Step 2: Write the README**

It must contain: what the library does in two sentences; the copyable base-install example matching the test above; the same example extended with vision via `build_openai_vision_model`; the `create_deep_agent(tools=build_tools(perception))` composition; a statement that the library reads the filesystem and never the environment; and an honest support table.

The support table must list only what exists — text, images, and a binary fallback — and must not describe audio, video, PDF, or office documents as though they are supported. A README that promises unbuilt handlers is the same defect as a degradation that describes a cause nothing checked.

- [ ] **Step 3: Correct the stale §14b in Spec 1**

Two items are listed as owed that Plan 2 closed. Mark them closed with a note, in the same strikethrough style §14b already uses for the `BinaryHandler` entry:

- `Perception.hasher` is already annotated against the `ContentHashing` port.
- `artifact_key`'s `default=str` collision now raises `DomainError`.

Add a line recording that cache wiring, the composition root, and capability discovery were closed by Plan 3, so the deferral list stops misreporting itself.

- [ ] **Step 4: Raise the coverage floor**

```bash
uv run --all-extras pytest --cov=readeverything --cov-report=term | tail -3
```

If total coverage is at or above 92.5%, set `fail_under = 92` in `pyproject.toml`. If it is below, leave it at 90 and record the measured number and the reason in the commit message. Do not add tests solely to reach a number — the spec's §11 risk row anticipates this and pre-authorises leaving the floor alone.

- [ ] **Step 5: Run everything**

```bash
make check
```

- [ ] **Step 6: Commit**

```bash
uv run --all-extras ruff format tests
git add README.md pyproject.toml docs/superpowers/specs/ tests/integration/test_readme_example.py
git commit -m "docs: a README whose example is a passing test"
```

---

## Task 12: `MediaAwareBackend` — ordered last, droppable

The spec (§7.3) rules this in against a recommendation to skip it, and (§10) explicitly excludes it from acceptance. If it does not land cleanly, drop it, record why, and finish the plan.

**Files:**
- Create: `src/readeverything/agent/deepagents_backend.py`
- Modify: `pyproject.toml` (`deepagents` extra, dependency-confinement allowlist)
- Test: `tests/integration/test_deepagents_backend.py`

**Interfaces:**
- Consumes: `deepagents.backends.protocol.BackendProtocol` with `ls`, `read(file_path, offset=0, limit=2000)`, `grep`, `glob`, `write`, plus async mirrors `als`/`aread`/`agrep`/`aglob`/`awrite`. Verified against `deepagents==0.7.6`.
- Produces: `class MediaAwareBackend(BackendProtocol)` — `__init__(self, *, perception: Perception, inner: BackendProtocol)`.

- [ ] **Step 1: Confirm the protocol before writing against it**

```bash
uv run --no-project --with deepagents python -c "
import inspect, deepagents.backends.protocol as p
print(inspect.signature(p.BackendProtocol.read))
print([m for m in dir(p.BackendProtocol) if not m.startswith('_')])
print(inspect.getsource(p.ReadResult))
"
```

If the surface differs from the interfaces block above, stop and report — the spec's §7 ruling was made against 0.7.6 and a changed surface may change the ruling.

- [ ] **Step 2: Write the failing test**

```python
async def test_reading_an_image_returns_a_description_rather_than_base64(
    media_root,
) -> None:
    """The whole point, in one assertion.

    deepagents' own FilesystemBackend base64-encodes a photograph to fit
    `ReadResult.content: str`. That IS the mangling — this library exists to put
    something meaningful in that string instead.
    """
    perception = await build_perception(media_root, vision=FakeVision(), probe_binaries=False)
    backend = MediaAwareBackend(perception=perception, inner=FilesystemBackend(root_dir=media_root))
    result = await backend.aread("photo.png")
    assert "base64" not in result.content
    assert len(result.content) < 2000


async def test_a_text_file_is_delegated_untouched(media_root) -> None:
    """Only the read path for media changes. Everything else is the wrapped backend."""
    perception = await build_perception(media_root, probe_binaries=False)
    inner = FilesystemBackend(root_dir=media_root)
    backend = MediaAwareBackend(perception=perception, inner=inner)
    assert (await backend.aread("notes.txt")).content == (await inner.aread("notes.txt")).content


async def test_writes_go_to_the_wrapped_backend(media_root) -> None:
    """This library does not write. Composition means it does not have to refuse."""
    perception = await build_perception(media_root, probe_binaries=False)
    backend = MediaAwareBackend(perception=perception, inner=FilesystemBackend(root_dir=media_root))
    await backend.awrite("new.txt", "written")
    assert (media_root / "new.txt").read_text() == "written"
```

- [ ] **Step 3: Add the extra**

```toml
deepagents = ["deepagents>=0.7,<0.8", "langchain-core>=0.3"]
```

Pinned below 0.8 because their surface moves fast, and an unpinned dependency on a fast-moving 0.x is how a library acquires breakage it did not choose.

- [ ] **Step 4: Implement**

```python
"""A deepagents backend whose `read` understands media.

deepagents' `BackendProtocol.read` must return `str`. Its built-in
`FilesystemBackend` satisfies that for a photograph by base64-encoding it,
which is a true encoding of the bytes and tells a model nothing. This library
exists to put something meaningful in that string instead — so the protocol
asking for text is the reason this fits, not an obstacle to it.

Only the read path is overridden. `ls`, `glob`, `grep` and `write` delegate to
a wrapped backend, which is how a read-only library composes with a protocol
that includes writing: it does not have to refuse a method it simply does not
implement.

This is the only module in the package that may import `deepagents`.
"""

from __future__ import annotations

from deepagents.backends.protocol import BackendProtocol, ReadResult

from readeverything.domain.rendition import Budget
from readeverything.pipeline.perception import Perception

#: Card kinds deepagents' own backend already handles well. Everything else is
#: what this backend exists for.
_ALREADY_TEXT = frozenset({"text"})


class MediaAwareBackend(BackendProtocol):
    """Wraps a backend and answers `read` for media with a representation."""

    def __init__(self, *, perception: Perception, inner: BackendProtocol) -> None:
        self._perception = perception
        self._inner = inner

    def ls(self, path: str):  # type: ignore[no-untyped-def]  # protocol's own return types
        return self._inner.ls(path)

    def glob(self, pattern: str, path=None):  # type: ignore[no-untyped-def]
        return self._inner.glob(pattern, path)

    def grep(self, pattern: str, path=None, glob=None, *, max_count=None):  # type: ignore[no-untyped-def]
        return self._inner.grep(pattern, path, glob, max_count=max_count)

    def write(self, file_path: str, content: str):  # type: ignore[no-untyped-def]
        return self._inner.write(file_path, content)

    async def als(self, path: str):  # type: ignore[no-untyped-def]
        return await self._inner.als(path)

    async def aglob(self, pattern: str, path=None):  # type: ignore[no-untyped-def]
        return await self._inner.aglob(pattern, path)

    async def agrep(self, pattern: str, path=None, glob=None, *, max_count=None):  # type: ignore[no-untyped-def]
        return await self._inner.agrep(pattern, path, glob, max_count=max_count)

    async def awrite(self, file_path: str, content: str):  # type: ignore[no-untyped-def]
        return await self._inner.awrite(file_path, content)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """A representation for media, the wrapped backend's answer otherwise.

        Implemented natively rather than inheriting the protocol's
        `asyncio.to_thread` default: `Perception` is already async, and
        thread-wrapping an async pipeline would be a defect rather than a
        convenience.
        """
        try:
            card = await self._perception.inspect(file_path)
        except Exception:
            # A file this library cannot resolve is still a file the wrapped
            # backend may be able to read. Failing over is the whole reason
            # this composes rather than replaces.
            return await self._inner.aread(file_path, offset, limit)
        if str(card.kind) in _ALREADY_TEXT:
            return await self._inner.aread(file_path, offset, limit)
        # `limit` is deepagents' paging contract. Honouring it as a Budget keeps
        # a large representation from blowing the caller's context, which is the
        # same thing the parameter means for a text file.
        rendered = await self._perception.represent(file_path, Budget(max_chars=limit))
        return ReadResult(content=rendered.text)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """The sync entry point.

        `asyncio.run` here is safe only because a sync `read` cannot be called
        from inside a running loop — deepagents calls `aread` in async contexts.
        """
        import asyncio

        return asyncio.run(self.aread(file_path, offset, limit))
```

Check `ReadResult`'s real field names against Step 1's output before writing —
if it takes more than `content`, fill the rest from the wrapped backend's answer
rather than inventing values.

- [ ] **Step 5: Confine the import**

Add `deepagents` to the dependency-confinement test's allowlist for this module only, and extend the import-linter forbidden contract so no other module may import it.

- [ ] **Step 6: Run everything**

```bash
make check
```

- [ ] **Step 7: Commit**

```bash
uv run --all-extras ruff format src tests
git add src/readeverything/agent/deepagents_backend.py pyproject.toml tests/integration/test_deepagents_backend.py tests/unit/test_dependencies_stay_confined.py
git commit -m "feat(agent): give a deep agent's own file tools eyes"
```

---

## Plan Self-Review

**Spec coverage.** §3 cache → Tasks 5, 6. §4 capability discovery → Task 7. §5 composition root and surface → Tasks 8, 9, 11. §6 integration tier → Task 10. §7 deepagents → Task 11 (layer one, in the README and its test) and Task 12 (layer two). §8.1 → Task 1. §8.2 → Task 2. §8.3 → Task 3. §8.4 → Task 4. §8.5 → Task 11. §10 acceptance criteria 1–8 → Tasks 8, 10, 11.

**Ordering.** Defect fixes come first (Tasks 1–4) because they are small, independent, and touch files later tasks modify — landing them first means later diffs are not tangled with them. Task 2 knowingly adds a redundant `inspect` that Task 5 makes cheap; this is flagged in the task itself so a reviewer does not have to rediscover it.

**Known gaps a reviewer should hold me to.**
- Task 6 assumes `Rendition` is JSON-serializable via pydantic. The task says to check and report which it is rather than assuming — if it is a plain frozen dataclass with a `Locator` union, serialization may need a `TypeAdapter` and that is a bigger change than the step implies.
- Task 10's first caching test is deliberately left as a sketch with an explicit note, because it must wrap a real hasher and the wrapping seam depends on Task 5's final shape.
- Task 12 depends on a third-party API verified today; Step 1 re-verifies before any code is written.
