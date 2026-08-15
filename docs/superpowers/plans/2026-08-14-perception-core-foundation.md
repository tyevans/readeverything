# readeverything Perception Core — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the perception core of `readeverything` — domain model, ports, mimetype registry with capability negotiation, content-addressed artifact cache, two reference handlers, and a framework-agnostic tool pack — so an agent can inspect any path and receive a card with capability-filtered affordances.

**Architecture:** Hexagonal, layered `domain → ports → adapters → registry/handlers → pipeline → agent`, enforced by import-linter with `exhaustive = true` plus an AST test pinning each third-party client to one directory. Handlers *declare* capabilities and affordances; the registry *filters*; the tool pack *materializes*. Nothing reads the environment.

**Tech Stack:** Python 3.13, pydantic 2, hatchling, uv, mypy strict, ruff, import-linter, pytest + hypothesis, LangChain (tool pack only).

**Spec:** `docs/superpowers/specs/2026-08-14-readeverything-perception-core-design.md`
**Prerequisite RFC (separate repo, not this plan):** `docs/rfcs/0001-source-spans-carry-caller-provenance.md`

## Scope

This plan covers spec sections §3, §4, §5, §6, §8, §10 (tool pack half), §12, §13, §14, and the text/binary rows of §7.

**Not in this plan** — each needs the contract this plan proves, and each ships independently:
- Plan 2: format handlers (video, audio, image, PDF/Office, HTML, tabular, archive) — spec §7
- Plan 3: deepagents backend decorator, redstring sink, composition root — spec §10 (second half), §11
- Plan R: the redstring RFC, authored in the redstring repo

This plan has **no dependency on the redstring RFC** and can proceed fully in parallel with it. `redstring` is not even a dependency until Plan 3.

## Global Constraints

- Python `>=3.13`. PEP 695 inline type parameters (`class Foo[T]`), never module-level `TypeVar`.
- `mypy --strict` must pass. Everything is typed; `py.typed` ships.
- **The library reads no environment.** No `os.environ`, no `os.getenv`, no `dotenv`. All configuration is constructor arguments. Enforced by test in Task 16.
- Ports are `typing.Protocol` + `@runtime_checkable`, decomposed into narrow capability slices.
- Value objects are `@dataclass(frozen=True, slots=True)`. Caller-facing models are pydantic `BaseModel`.
- Third-party imports are confined to one directory each (spec §3 table), enforced by AST test in Task 16.
- Ruff line-length 100. Import name is `readeverything`; distribution name is `deepagents-read-everything`.
- All commands run through `uv run`.
- Async throughout for anything touching I/O. Pure domain code is sync.

---

### Task 1: Project scaffolding and quality gates

**Files:**
- Create: `pyproject.toml` (replaces the stub), `Makefile`, `src/readeverything/__init__.py`, `src/readeverything/py.typed`, `tests/unit/test_smoke.py`
- Delete: `main.py`

**Interfaces:**
- Consumes: nothing
- Produces: the `readeverything` package root; `make check` as the single gate

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "deepagents-read-everything"
version = "0.1.0"
description = "Give an agent eyes into a filesystem: mimetype-dispatched media representations with locator-carrying provenance."
readme = "README.md"
license = "MIT"
authors = [{ name = "Ty Evans", email = "tyler@poorlythoughtout.com" }]
requires-python = ">=3.13"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3 :: Only",
    "Typing :: Typed",
]
dependencies = [
    "pydantic>=2.12,<3",
    "puremagic>=1.28",
    "charset-normalizer>=3.4",
]

[project.optional-dependencies]
langchain = ["langchain-core>=0.3"]
dev = [
    "mypy>=1.13",
    "ruff==0.16.2",
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-cov>=6.0",
    "hypothesis>=6.115",
    "import-linter>=2.1",
    "bandit>=1.8",
    "pip-audit>=2.7",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/readeverything"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM", "RUF"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.13"
strict = true
warn_return_any = true
warn_unused_ignores = true
files = ["src", "tests"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "integration: requires OS binaries",
    "live: requires a running model server",
    "accuracy: quality benchmark, not a pass/fail test",
    "slow: takes more than a second",
]
addopts = "-m 'not integration and not live and not accuracy'"

[tool.importlinter]
root_packages = ["readeverything"]
include_external_packages = false

[[tool.importlinter.contracts]]
name = "Layered architecture"
type = "layers"
containers = ["readeverything"]
exhaustive = true
layers = [
    "testing",
    "agent",
    "pipeline",
    "registry",
    "handlers",
    "adapters",
    "ports",
    "domain",
]

[[tool.importlinter.contracts]]
name = "The testing toolkit sees only ports and domain"
type = "forbidden"
source_modules = ["readeverything.testing"]
forbidden_modules = [
    "readeverything.agent",
    "readeverything.pipeline",
    "readeverything.registry",
    "readeverything.handlers",
    "readeverything.adapters",
]
```

- [ ] **Step 2: Create the package skeleton**

Create empty `__init__.py` files so import-linter's `exhaustive` contract has every layer to see, and an empty `py.typed`:

```bash
mkdir -p src/readeverything/{domain,ports,adapters,handlers,registry,pipeline,agent,testing}
for d in "" /domain /ports /adapters /handlers /registry /pipeline /agent /testing; do
  touch "src/readeverything${d}/__init__.py"
done
touch src/readeverything/py.typed
mkdir -p tests/unit
rm -f main.py
```

- [ ] **Step 3: Write `Makefile`**

```makefile
UV := uv run --all-extras

.PHONY: check lint types arch sec test fmt

check: lint types arch sec test

lint:
	$(UV) ruff check .
	$(UV) ruff format --check .

fmt:
	$(UV) ruff format .
	$(UV) ruff check --fix .

types:
	$(UV) mypy

arch:
	$(UV) lint-imports

sec:
	$(UV) bandit -q -c pyproject.toml -r src
	$(UV) pip-audit

test:
	$(UV) pytest
```

- [ ] **Step 4: Write the smoke test**

```python
# tests/unit/test_smoke.py
def test_package_imports() -> None:
    import readeverything

    assert readeverything.__name__ == "readeverything"
```

- [ ] **Step 5: Run the full gate**

Run: `uv sync --all-extras && make check`
Expected: all five stages PASS. If `lint-imports` complains about an unplaced package, add it to the `layers` list — that is the contract working.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: scaffold readeverything package with quality gates"
```

---

### Task 2: Domain — source identity

**Files:**
- Create: `src/readeverything/domain/identity.py`
- Test: `tests/unit/domain/test_identity.py`

**Interfaces:**
- Consumes: nothing
- Produces: `ContentHash`, `MimeType`, `MediaKind`, `SourceRef`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/domain/test_identity.py
import pytest

from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef


def test_mimetype_parses_type_and_subtype() -> None:
    mime = MimeType.parse("video/mp4")
    assert mime.type == "video"
    assert mime.subtype == "mp4"
    assert str(mime) == "video/mp4"


def test_mimetype_extracts_structured_suffix() -> None:
    assert MimeType.parse("application/epub+zip").suffix == "zip"
    assert MimeType.parse("video/mp4").suffix is None


def test_mimetype_lowercases_and_drops_parameters() -> None:
    mime = MimeType.parse("TEXT/Plain; charset=utf-8")
    assert str(mime) == "text/plain"


def test_mimetype_rejects_a_string_without_a_slash() -> None:
    with pytest.raises(ValueError, match="not a mimetype"):
        MimeType.parse("video")


def test_media_kind_is_derived_from_the_type() -> None:
    assert MediaKind.for_mime(MimeType.parse("video/mp4")) is MediaKind.VIDEO
    assert MediaKind.for_mime(MimeType.parse("audio/flac")) is MediaKind.AUDIO
    assert MediaKind.for_mime(MimeType.parse("image/png")) is MediaKind.IMAGE
    assert MediaKind.for_mime(MimeType.parse("text/plain")) is MediaKind.TEXT
    assert MediaKind.for_mime(MimeType.parse("application/pdf")) is MediaKind.BINARY


def test_source_ref_rejects_a_negative_size() -> None:
    with pytest.raises(ValueError, match="size_bytes"):
        SourceRef(
            uri="/a.txt",
            mime=MimeType.parse("text/plain"),
            content_hash=ContentHash("abc"),
            size_bytes=-1,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/domain/test_identity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'readeverything.domain.identity'`

- [ ] **Step 3: Write the implementation**

```python
# src/readeverything/domain/identity.py
"""What a source is, and how it is named.

`SourceRef` is the only handle a handler ever gets. It carries no filesystem
path semantics on purpose: `uri` is opaque to the domain, so an archive member
addressed as `/a.zip!inner.txt` and an object-store key are the same kind of
thing. Bytes are reached through the `FileSource` port, never from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType, Self

#: A blake2b hex digest of a source's bytes. See `adapters.hashing`.
ContentHash = NewType("ContentHash", str)


@dataclass(frozen=True, slots=True)
class MimeType:
    """A parsed mimetype, without parameters.

    Parameters are dropped rather than kept: `text/plain; charset=utf-8` and
    `text/plain` must dispatch to the same handler, and keeping the parameter
    would make the registry's exact-match step depend on whether the detector
    happened to report an encoding. Encoding is a handler's concern and is
    re-derived from content.
    """

    type: str
    subtype: str
    suffix: str | None = None

    @classmethod
    def parse(cls, raw: str) -> Self:
        value = raw.split(";", 1)[0].strip().lower()
        if "/" not in value:
            raise ValueError(f"not a mimetype: {raw!r}")
        type_, subtype = value.split("/", 1)
        if not type_ or not subtype:
            raise ValueError(f"not a mimetype: {raw!r}")
        suffix = subtype.rsplit("+", 1)[1] if "+" in subtype else None
        return cls(type=type_, subtype=subtype, suffix=suffix)

    def __str__(self) -> str:
        return f"{self.type}/{self.subtype}"


class MediaKind(StrEnum):
    """The coarse family a mimetype belongs to.

    This is the registry's fourth dispatch step and nothing else. It is
    deliberately coarser than the handler families: `application/pdf` is
    `BINARY` here and still reaches a PDF handler, because that match happens
    at the exact-mimetype step long before this one.
    """

    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"
    BINARY = "binary"

    @classmethod
    def for_mime(cls, mime: MimeType) -> MediaKind:
        match mime.type:
            case "video":
                return cls.VIDEO
            case "audio":
                return cls.AUDIO
            case "image":
                return cls.IMAGE
            case "text":
                return cls.TEXT
            case _:
                return cls.BINARY


@dataclass(frozen=True, slots=True)
class SourceRef:
    """A specific sequence of bytes, and what is known about it cheaply."""

    uri: str
    mime: MimeType
    content_hash: ContentHash
    size_bytes: int

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise ValueError(f"size_bytes must not be negative, got {self.size_bytes}")
        if not self.uri:
            raise ValueError("uri must not be empty")
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/domain/test_identity.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/domain/identity.py tests/unit/domain/test_identity.py
git commit -m "feat(domain): add MimeType, MediaKind and SourceRef"
```

---

### Task 3: Domain — locators

**Files:**
- Create: `src/readeverything/domain/locators.py`
- Test: `tests/unit/domain/test_locators.py`

**Interfaces:**
- Consumes: nothing
- Produces: `TimeSpan`, `PageRef`, `BBox`, `CharSpan`, `ByteRange`, `Locator` type alias

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/domain/test_locators.py
import pytest

from readeverything.domain.locators import BBox, ByteRange, CharSpan, PageRef, TimeSpan


def test_char_span_is_half_open_and_reports_length() -> None:
    assert CharSpan(0, 5).length == 5


def test_char_span_rejects_an_inverted_range() -> None:
    with pytest.raises(ValueError, match="start must be less than end"):
        CharSpan(5, 5)


def test_char_span_overlap_is_exclusive_at_the_boundary() -> None:
    assert CharSpan(0, 5).overlaps(CharSpan(4, 9))
    assert not CharSpan(0, 5).overlaps(CharSpan(5, 9))


def test_time_span_rejects_a_negative_start() -> None:
    with pytest.raises(ValueError, match="start_s must not be negative"):
        TimeSpan(-1.0, 2.0)


def test_page_ref_is_one_indexed() -> None:
    assert PageRef(1).page == 1
    with pytest.raises(ValueError, match="page must be at least 1"):
        PageRef(0)


def test_bbox_requires_normalised_coordinates() -> None:
    BBox(page=1, x=0.0, y=0.0, w=1.0, h=1.0)
    with pytest.raises(ValueError, match="must be within the unit square"):
        BBox(page=1, x=0.5, y=0.0, w=0.8, h=0.1)


def test_byte_range_rejects_an_inverted_range() -> None:
    with pytest.raises(ValueError, match="start must be less than end"):
        ByteRange(10, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/domain/test_locators.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/readeverything/domain/locators.py
"""Where in a source something is.

One vocabulary, shared by cards, affordance results, the locator map, chunk
barriers and citations. Every locator is pure data: speaker attribution is
*not* here, because a speaker is a property of an utterance rather than of a
position, and putting it on `TimeSpan` would mean every other locator carried a
field that is always `None`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CharSpan:
    """A half-open range of characters, `[start, end)`."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start must not be negative, got {self.start}")
        if self.start >= self.end:
            raise ValueError(f"start must be less than end, got {self.start} >= {self.end}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: CharSpan) -> bool:
        """True when the two ranges share at least one character.

        Exclusive at the boundary, because the ranges are half-open: `[0, 5)`
        and `[5, 9)` are adjacent, not overlapping. Getting this wrong would
        attach every chunk's provenance to its neighbour as well.
        """
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True, slots=True)
class ByteRange:
    """A half-open range of bytes, `[start, end)`."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start must not be negative, got {self.start}")
        if self.start >= self.end:
            raise ValueError(f"start must be less than end, got {self.start} >= {self.end}")


@dataclass(frozen=True, slots=True)
class TimeSpan:
    """A range of wall-clock time within a media stream, in seconds."""

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.start_s < 0:
            raise ValueError(f"start_s must not be negative, got {self.start_s}")
        if self.start_s >= self.end_s:
            raise ValueError(f"start_s must be less than end_s, got {self.start_s} >= {self.end_s}")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class PageRef:
    """A page of a paginated document, 1-indexed as a reader would count."""

    page: int

    def __post_init__(self) -> None:
        if self.page < 1:
            raise ValueError(f"page must be at least 1, got {self.page}")


@dataclass(frozen=True, slots=True)
class BBox:
    """A rectangle on a page, in normalised coordinates.

    Normalised rather than pixel coordinates so a locator survives the page
    being rendered at a different DPI — which it will be, since the card path
    and the VLM path render at different sizes.
    """

    page: int | None
    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"w and h must be positive, got {self.w}x{self.h}")
        if not (0.0 <= self.x and 0.0 <= self.y and self.x + self.w <= 1.0 and self.y + self.h <= 1.0):
            raise ValueError(
                f"must be within the unit square, got "
                f"x={self.x} y={self.y} w={self.w} h={self.h}"
            )


type Locator = TimeSpan | PageRef | BBox | CharSpan | ByteRange
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/domain/test_locators.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/domain/locators.py tests/unit/domain/test_locators.py
git commit -m "feat(domain): add the Locator vocabulary"
```

---

### Task 4: Domain — LocatorMap

This is where citation correctness lives. It gets property tests.

**Files:**
- Create: `src/readeverything/domain/locator_map.py`
- Test: `tests/unit/domain/test_locator_map.py`

**Interfaces:**
- Consumes: `CharSpan`, `Locator` from Task 3
- Produces: `LocatorSegment`, `LocatorMap` with `LocatorMap.build(segments)`, `.resolve(offset) -> Locator`, `.resolve_span(CharSpan) -> tuple[Locator, ...]`, `.length`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/domain/test_locator_map.py
import pytest
from hypothesis import given
from hypothesis import strategies as st

from readeverything.domain.locator_map import LocatorMap, LocatorSegment
from readeverything.domain.locators import CharSpan, TimeSpan


def _map() -> LocatorMap:
    return LocatorMap.build(
        (
            LocatorSegment(CharSpan(0, 10), TimeSpan(0.0, 1.0)),
            LocatorSegment(CharSpan(10, 25), TimeSpan(1.0, 2.5)),
            LocatorSegment(CharSpan(25, 30), TimeSpan(2.5, 3.0)),
        )
    )


def test_length_is_the_end_of_the_last_segment() -> None:
    assert _map().length == 30


def test_resolve_returns_the_containing_segments_locator() -> None:
    m = _map()
    assert m.resolve(0) == TimeSpan(0.0, 1.0)
    assert m.resolve(9) == TimeSpan(0.0, 1.0)
    assert m.resolve(10) == TimeSpan(1.0, 2.5)
    assert m.resolve(29) == TimeSpan(2.5, 3.0)


def test_resolve_rejects_an_offset_outside_the_map() -> None:
    with pytest.raises(ValueError, match="outside the map"):
        _map().resolve(30)
    with pytest.raises(ValueError, match="outside the map"):
        _map().resolve(-1)


def test_resolve_span_returns_every_overlapping_locator_in_order() -> None:
    assert _map().resolve_span(CharSpan(8, 26)) == (
        TimeSpan(0.0, 1.0),
        TimeSpan(1.0, 2.5),
        TimeSpan(2.5, 3.0),
    )


def test_resolve_span_of_a_single_segment_returns_one_locator() -> None:
    assert _map().resolve_span(CharSpan(11, 20)) == (TimeSpan(1.0, 2.5),)


def test_build_rejects_a_gap() -> None:
    with pytest.raises(ValueError, match="gapless"):
        LocatorMap.build(
            (
                LocatorSegment(CharSpan(0, 10), TimeSpan(0.0, 1.0)),
                LocatorSegment(CharSpan(12, 20), TimeSpan(1.0, 2.0)),
            )
        )


def test_build_rejects_segments_that_do_not_start_at_zero() -> None:
    with pytest.raises(ValueError, match="must start at 0"):
        LocatorMap.build((LocatorSegment(CharSpan(3, 10), TimeSpan(0.0, 1.0)),))


def test_build_rejects_unsorted_segments() -> None:
    with pytest.raises(ValueError, match="gapless"):
        LocatorMap.build(
            (
                LocatorSegment(CharSpan(10, 20), TimeSpan(1.0, 2.0)),
                LocatorSegment(CharSpan(0, 10), TimeSpan(0.0, 1.0)),
            )
        )


def test_empty_map_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one segment"):
        LocatorMap.build(())


@st.composite
def maps(draw: st.DrawFn) -> LocatorMap:
    lengths = draw(st.lists(st.integers(min_value=1, max_value=50), min_size=1, max_size=20))
    segments: list[LocatorSegment] = []
    cursor = 0
    for i, length in enumerate(lengths):
        segments.append(
            LocatorSegment(CharSpan(cursor, cursor + length), TimeSpan(float(i), float(i) + 1.0))
        )
        cursor += length
    return LocatorMap.build(tuple(segments))


@given(maps())
def test_resolution_is_total(m: LocatorMap) -> None:
    """Every offset in the map resolves. A hole here is an uncitable passage."""
    for offset in range(m.length):
        m.resolve(offset)


@given(maps())
def test_resolution_is_monotonic(m: LocatorMap) -> None:
    """Resolution never goes backwards as the offset advances."""
    seen: list[TimeSpan] = []
    for offset in range(m.length):
        locator = m.resolve(offset)
        assert isinstance(locator, TimeSpan)
        if not seen or seen[-1] != locator:
            seen.append(locator)
    assert seen == sorted(seen, key=lambda t: t.start_s)


@given(maps())
def test_resolve_span_agrees_with_pointwise_resolution(m: LocatorMap) -> None:
    """The span API is a compression of the pointwise one, not a second scheme."""
    span = CharSpan(0, m.length)
    pointwise: list[TimeSpan] = []
    for offset in range(m.length):
        locator = m.resolve(offset)
        assert isinstance(locator, TimeSpan)
        if not pointwise or pointwise[-1] != locator:
            pointwise.append(locator)
    assert m.resolve_span(span) == tuple(pointwise)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/domain/test_locator_map.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/readeverything/domain/locator_map.py
"""Character offsets to locators, and back again.

The structure on which citation correctness rests. A retrieval hit knows only
that it came from characters 4100-4380 of some flattened text; this is what
turns that into "00:42:15 to 00:42:31".

The map is required to be **gapless and starting at zero** rather than merely
sorted. A sparse map would let `resolve` fail for an offset that a chunker
happily produced, and it would fail at citation time — after the answer was
already computed, and far from whatever produced the hole. A handler that has
nothing to say about a region must say so with a segment, not with a gap.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Self

from readeverything.domain.locators import CharSpan, Locator


@dataclass(frozen=True, slots=True)
class LocatorSegment:
    """One contiguous run of text that shares a single locator."""

    span: CharSpan
    locator: Locator


@dataclass(frozen=True, slots=True)
class LocatorMap:
    """A total, monotonic mapping from character offset to locator.

    Construct with `build`, which validates and precomputes the bisection
    index. The constructor is not private, but a directly-constructed instance
    whose `starts` disagrees with its `segments` is rejected too.
    """

    segments: tuple[LocatorSegment, ...]
    starts: tuple[int, ...] = field(compare=False, repr=False)

    @classmethod
    def build(cls, segments: tuple[LocatorSegment, ...]) -> Self:
        return cls(segments=segments, starts=tuple(s.span.start for s in segments))

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("a locator map needs at least one segment")
        # Sortedness is checked before the start-at-0 rule, and the order is
        # load-bearing. Both `[(3,10)]` and `[(10,20), (0,10)]` have a first
        # segment that does not start at 0, so a start-at-0 check placed first
        # would report "must start at 0" for the unsorted case too — and the
        # two failures want different messages, because they are different
        # mistakes. Only a sortedness check can tell them apart.
        for i in range(len(self.segments) - 1):
            if self.segments[i].span.start >= self.segments[i + 1].span.start:
                raise ValueError(
                    f"segments must be sorted and gapless: expected a segment starting at "
                    f"{self.segments[i].span.end}, got one starting at "
                    f"{self.segments[i + 1].span.start}"
                )
        if self.segments[0].span.start != 0:
            raise ValueError(f"segments must start at 0, got {self.segments[0].span.start}")
        cursor = 0
        for segment in self.segments:
            if segment.span.start != cursor:
                raise ValueError(
                    f"segments must be sorted and gapless: expected a segment starting at "
                    f"{cursor}, got one starting at {segment.span.start}"
                )
            cursor = segment.span.end
        if self.starts != tuple(s.span.start for s in self.segments):
            raise ValueError("starts does not match segments; use LocatorMap.build")

    @property
    def length(self) -> int:
        """Total characters covered."""
        return self.segments[-1].span.end

    def resolve(self, offset: int) -> Locator:
        """The locator for a single character offset."""
        if not 0 <= offset < self.length:
            raise ValueError(f"offset {offset} is outside the map of length {self.length}")
        index = bisect_right(self.starts, offset) - 1
        return self.segments[index].locator

    def resolve_span(self, span: CharSpan) -> tuple[Locator, ...]:
        """Every locator overlapping `span`, in document order.

        Returns a tuple, not a single locator, and callers must not assume
        length 1: a chunk spanning 00:42:15-00:43:02 genuinely covers several
        transcript cues, and the honest citation is the union of them.
        """
        if span.end > self.length:
            raise ValueError(f"span {span} is outside the map of length {self.length}")
        first = bisect_right(self.starts, span.start) - 1
        return tuple(
            segment.locator
            for segment in self.segments[first:]
            if segment.span.overlaps(span)
        )
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/domain/test_locator_map.py -v`
Expected: 12 passed (9 example-based, 3 property-based)

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/domain/locator_map.py tests/unit/domain/test_locator_map.py
git commit -m "feat(domain): add LocatorMap with totality and monotonicity properties"
```

---

### Task 5: Domain — capabilities and errors

**Files:**
- Create: `src/readeverything/domain/capability.py`, `src/readeverything/domain/errors.py`
- Test: `tests/unit/domain/test_capability.py`, `tests/unit/domain/test_errors.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Capability`, `CapabilitySet` with `.satisfies(frozenset[Capability]) -> bool` and `.fingerprint() -> str`; `ReadEverythingError` and its subclasses

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/domain/test_capability.py
from readeverything.domain.capability import Capability, CapabilitySet


def test_an_empty_set_satisfies_only_an_empty_requirement() -> None:
    empty = CapabilitySet.empty()
    assert empty.satisfies(frozenset())
    assert not empty.satisfies({Capability.VISION})


def test_satisfies_requires_every_capability() -> None:
    caps = CapabilitySet.of({Capability.VISION: "qwen3.8@rev1"})
    assert caps.satisfies({Capability.VISION})
    assert not caps.satisfies({Capability.VISION, Capability.ASR})


def test_fingerprint_is_stable_across_insertion_order() -> None:
    a = CapabilitySet.of({Capability.VISION: "v1", Capability.ASR: "w1"})
    b = CapabilitySet.of({Capability.ASR: "w1", Capability.VISION: "v1"})
    assert a.fingerprint() == b.fingerprint()


def test_fingerprint_changes_when_a_model_revision_changes() -> None:
    """Swapping the VLM must invalidate cached descriptions."""
    a = CapabilitySet.of({Capability.VISION: "qwen3.8@rev1"})
    b = CapabilitySet.of({Capability.VISION: "qwen3.8@rev2"})
    assert a.fingerprint() != b.fingerprint()


def test_binaries_and_models_are_the_same_kind_of_capability() -> None:
    caps = CapabilitySet.of({Capability.FFMPEG: "7.1", Capability.VISION: "v1"})
    assert caps.satisfies({Capability.FFMPEG, Capability.VISION})
```

```python
# tests/unit/domain/test_errors.py
import pytest

from readeverything.domain.errors import (
    CapabilityUnavailableError,
    InfrastructureError,
    ReadEverythingError,
    UnknownAffordanceError,
)


def test_every_error_descends_from_the_root() -> None:
    assert issubclass(UnknownAffordanceError, ReadEverythingError)
    assert issubclass(InfrastructureError, ReadEverythingError)


def test_capability_unavailable_names_what_is_missing() -> None:
    with pytest.raises(CapabilityUnavailableError, match="vision"):
        raise CapabilityUnavailableError(missing=frozenset({"vision"}))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/domain/test_capability.py tests/unit/domain/test_errors.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementations**

```python
# src/readeverything/domain/capability.py
"""What this deployment can do.

Model capabilities and OS binaries are the same kind of thing. A missing
`ffmpeg` must degrade exactly like a missing vision model, because from a
handler's point of view both are "I cannot produce that". One mechanism means
there is one place degradation is decided, and no special cases.

Each capability carries a **revision string** — a model id plus revision, or a
binary version. It is not used for matching; it exists so the artifact cache
key changes when the thing behind a capability changes. Without it, swapping
the vision model silently serves a mixture of descriptions from two models.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Self


class Capability(StrEnum):
    VISION = "vision"
    ASR = "asr"
    DIARIZATION = "diarization"
    TEXT_LLM = "text_llm"
    FFMPEG = "ffmpeg"
    EXIFTOOL = "exiftool"
    LIBREOFFICE = "libreoffice"
    TESSERACT = "tesseract"


@dataclass(frozen=True, slots=True)
class CapabilitySet:
    """The capabilities available, each with the revision behind it."""

    revisions: Mapping[Capability, str]

    @classmethod
    def empty(cls) -> Self:
        return cls(revisions={})

    @classmethod
    def of(cls, revisions: Mapping[Capability, str]) -> Self:
        return cls(revisions=dict(revisions))

    def satisfies(self, required: frozenset[Capability] | set[Capability]) -> bool:
        return all(capability in self.revisions for capability in required)

    def fingerprint(self) -> str:
        """A stable digest of what is installed, for the artifact cache key."""
        digest = hashlib.blake2b(digest_size=16)
        for capability in sorted(self.revisions, key=str):
            digest.update(str(capability).encode("utf-8"))
            digest.update(b"\x00")
            digest.update(self.revisions[capability].encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()
```

```python
# src/readeverything/domain/errors.py
"""The exception taxonomy.

Two families under one root, following eventsource-py's split: a domain error
means the request did not make sense, an infrastructure error means the world
did not cooperate. The distinction is what lets a caller retry one and not the
other.

Note that the *tool pack* never raises any of these — it converts them to
structured results, because a traceback reaching a model is a wasted turn. See
`readeverything/agent/results.py`.
"""

from __future__ import annotations

from collections.abc import Iterable


class ReadEverythingError(Exception):
    """Root of every error this library raises."""


class DomainError(ReadEverythingError):
    """The request did not make sense."""


class InfrastructureError(ReadEverythingError):
    """The world did not cooperate."""


class UnknownAffordanceError(DomainError):
    """An affordance was invoked that the handler does not declare."""

    def __init__(self, name: str, available: Iterable[str]) -> None:
        offered = ", ".join(sorted(available)) or "none"
        super().__init__(f"unknown affordance {name!r}; this handler offers: {offered}")
        self.name = name


class CapabilityUnavailableError(DomainError):
    """Something required a capability this deployment does not have.

    Reaching this from the registry path is a bug: the registry filters
    unsatisfied handlers and affordances out before anything can invoke them.
    It exists for direct handler use, where no filtering happened.
    """

    def __init__(self, missing: frozenset[str] | set[str]) -> None:
        super().__init__(f"missing capabilities: {', '.join(sorted(missing))}")
        self.missing = frozenset(missing)


class SourceUnreadableError(InfrastructureError):
    """A source could not be read."""


class ProbeFailedError(InfrastructureError):
    """An external probe or tool failed."""
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/domain -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/domain/capability.py src/readeverything/domain/errors.py tests/unit/domain/test_capability.py tests/unit/domain/test_errors.py
git commit -m "feat(domain): add CapabilitySet and the error taxonomy"
```

---

### Task 6: Domain — affordances, cards, renditions

**Files:**
- Create: `src/readeverything/domain/affordance.py`, `src/readeverything/domain/card.py`, `src/readeverything/domain/rendition.py`
- Test: `tests/unit/domain/test_affordance.py`, `tests/unit/domain/test_card.py`, `tests/unit/domain/test_rendition.py`

**Interfaces:**
- Consumes: Tasks 2-5
- Produces: `DetailLevel`, `Affordance`, `Segment`, `Card`, `TextContent`, `ImageContent`, `StructuredContent`, `Rendition`, `Degradation`, `Rendered`, `Budget`, `SpeakerId`, `TranscriptCue`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/domain/test_affordance.py
import pytest
from pydantic import BaseModel

from readeverything.domain.affordance import Affordance, DetailLevel
from readeverything.domain.capability import Capability, CapabilitySet


class FrameParams(BaseModel):
    at_s: float


def _affordance() -> Affordance:
    return Affordance(
        name="get_frame_at",
        description="Return the video frame at a given time.",
        params=FrameParams,
        requires=frozenset({Capability.FFMPEG}),
        level=DetailLevel.SEGMENT,
    )


def test_an_affordance_is_available_when_its_capabilities_are_present() -> None:
    caps = CapabilitySet.of({Capability.FFMPEG: "7.1"})
    assert _affordance().is_available(caps)


def test_an_affordance_is_unavailable_when_a_capability_is_missing() -> None:
    assert not _affordance().is_available(CapabilitySet.empty())


def test_an_affordance_name_must_be_a_valid_tool_identifier() -> None:
    with pytest.raises(ValueError, match="must be a valid identifier"):
        Affordance(
            name="get frame",
            description="x",
            params=FrameParams,
            requires=frozenset(),
            level=DetailLevel.DEEP,
        )


def test_an_affordance_requires_a_description() -> None:
    """The description becomes the tool docstring; a blank one blinds the model."""
    with pytest.raises(ValueError, match="description must not be blank"):
        Affordance(
            name="get_frame_at",
            description="  ",
            params=FrameParams,
            requires=frozenset(),
            level=DetailLevel.DEEP,
        )
```

```python
# tests/unit/domain/test_card.py
from readeverything.domain.card import Card, Segment
from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
from readeverything.domain.locators import CharSpan


def _ref() -> SourceRef:
    return SourceRef(
        uri="/a.txt",
        mime=MimeType.parse("text/plain"),
        content_hash=ContentHash("deadbeef"),
        size_bytes=12,
    )


def test_a_card_exposes_affordance_names() -> None:
    card = Card(
        ref=_ref(),
        kind=MediaKind.TEXT,
        facts={"lines": 3},
        outline=(Segment(CharSpan(0, 12), "whole file"),),
        excerpt="hello world",
        affordances=(),
    )
    assert card.affordance_names() == ()
    assert card.facts["lines"] == 3
```

```python
# tests/unit/domain/test_rendition.py
import pytest

from readeverything.domain.locators import CharSpan, TimeSpan
from readeverything.domain.rendition import (
    Budget,
    Degradation,
    Rendered,
    Rendition,
    TextContent,
    TranscriptCue,
)
from readeverything.domain.locator_map import LocatorMap, LocatorSegment


def test_a_rendition_carries_its_locator() -> None:
    rendition = Rendition(locator=TimeSpan(1.0, 2.0), content=TextContent("hello"))
    assert rendition.locator == TimeSpan(1.0, 2.0)
    assert not rendition.degraded


def test_a_transcript_cue_may_have_no_speaker() -> None:
    """Diarization is capability-gated; every cue works without it."""
    cue = TranscriptCue(span=TimeSpan(0.0, 1.0), text="hi", speaker=None, confidence=None)
    assert cue.speaker is None


def test_rendered_requires_barriers_to_lie_within_the_text() -> None:
    locator_map = LocatorMap.build((LocatorSegment(CharSpan(0, 5), TimeSpan(0.0, 1.0)),))
    with pytest.raises(ValueError, match="barrier"):
        Rendered(
            text="hello",
            locator_map=locator_map,
            barriers=(99,),
            degradations=(),
        )


def test_rendered_requires_the_map_to_cover_the_text() -> None:
    locator_map = LocatorMap.build((LocatorSegment(CharSpan(0, 3), TimeSpan(0.0, 1.0)),))
    with pytest.raises(ValueError, match="must cover the text"):
        Rendered(text="hello", locator_map=locator_map, barriers=(), degradations=())


def test_a_budget_of_none_means_unbounded() -> None:
    assert Budget(max_chars=None).permits(10_000_000)
    assert not Budget(max_chars=100).permits(101)


def test_a_degradation_says_what_was_dropped() -> None:
    d = Degradation(what="frame_sampling", detail="reduced to 1 frame per 30s")
    assert "30s" in d.detail
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/domain -v`
Expected: FAIL with `ModuleNotFoundError` for the three new modules

- [ ] **Step 3: Write the implementations**

```python
# src/readeverything/domain/affordance.py
"""What a handler says it can do, before it does any of it.

An `Affordance` is a *declaration*, not a bound callable. That is the whole
design: the registry decides what to expose without executing anything, and the
tool pack materialises the survivors into tools. If an affordance were a
callable, deciding availability would mean holding a live handler, and
capability negotiation would have to happen at call time — which is exactly the
"tool exists but returns sorry" behaviour this avoids.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from readeverything.domain.capability import Capability, CapabilitySet


class DetailLevel(StrEnum):
    """How much work invoking this affordance is likely to be."""

    CARD = "card"
    SEGMENT = "segment"
    DEEP = "deep"


@dataclass(frozen=True, slots=True)
class Affordance:
    """One operation a handler offers over a source."""

    name: str
    description: str
    params: type[BaseModel]
    requires: frozenset[Capability]
    level: DetailLevel

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"name must be a valid identifier, got {self.name!r}")
        if not self.description.strip():
            raise ValueError("description must not be blank")

    def is_available(self, capabilities: CapabilitySet) -> bool:
        return capabilities.satisfies(self.requires)
```

```python
# src/readeverything/domain/card.py
"""The cheap representation returned on first contact.

Producing a card must not invoke a model and must not process the whole file.
Its cost is bounded by a probe. Everything expensive is behind an affordance,
so the agent chooses what to spend rather than paying for a two-hour video
because it looked at a directory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from readeverything.domain.affordance import Affordance
from readeverything.domain.identity import MediaKind, SourceRef
from readeverything.domain.locators import Locator


@dataclass(frozen=True, slots=True)
class Segment:
    """One labelled region of a source: a scene, a chapter, a page, a cue group."""

    locator: Locator
    label: str


@dataclass(frozen=True, slots=True)
class Card:
    """What a source is, cheaply, and what can be done with it."""

    ref: SourceRef
    kind: MediaKind
    facts: Mapping[str, str | int | float]
    outline: tuple[Segment, ...]
    excerpt: str | None
    affordances: tuple[Affordance, ...]

    def affordance_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.affordances)
```

```python
# src/readeverything/domain/rendition.py
"""The results of doing work: one operation's output, and a whole-source flattening.

`Rendition` answers an affordance. `Rendered` is the indexing feed — flat text
plus the locator map plus hard chunk barriers — and is the contract Plan 2's
query interface consumes.

`Rendered` validates that its map covers its text. A map that stops short would
produce a hit that cannot be cited, discovered at citation time rather than
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

from readeverything.domain.locator_map import LocatorMap
from readeverything.domain.locators import Locator, TimeSpan

SpeakerId = NewType("SpeakerId", str)


@dataclass(frozen=True, slots=True)
class TextContent:
    text: str


@dataclass(frozen=True, slots=True)
class ImageContent:
    """Raw image bytes plus their mimetype, ready for a multimodal content block."""

    data: bytes
    mime: str


@dataclass(frozen=True, slots=True)
class StructuredContent:
    rows: tuple[dict[str, str | int | float | None], ...]


type RenditionContent = TextContent | ImageContent | StructuredContent


@dataclass(frozen=True, slots=True)
class Rendition:
    """One affordance's answer, always located."""

    locator: Locator
    content: RenditionContent
    degraded: bool = False


@dataclass(frozen=True, slots=True)
class TranscriptCue:
    """One utterance, with a speaker when diarization is available."""

    span: TimeSpan
    text: str
    speaker: SpeakerId | None
    confidence: float | None


@dataclass(frozen=True, slots=True)
class Degradation:
    """Something a handler chose not to produce, and why.

    Reported rather than silent. Silent truncation is invisible in exactly the
    case where the answer is wrong.
    """

    what: str
    detail: str


@dataclass(frozen=True, slots=True)
class Rendered:
    """A whole source flattened for indexing."""

    text: str
    locator_map: LocatorMap
    barriers: tuple[int, ...]
    degradations: tuple[Degradation, ...]

    def __post_init__(self) -> None:
        if self.locator_map.length != len(self.text):
            raise ValueError(
                f"the locator map must cover the text exactly: map covers "
                f"{self.locator_map.length}, text is {len(self.text)}"
            )
        for barrier in self.barriers:
            if not 0 <= barrier <= len(self.text):
                raise ValueError(
                    f"barrier {barrier} is outside the text of length {len(self.text)}"
                )
        if list(self.barriers) != sorted(set(self.barriers)):
            raise ValueError("barriers must be sorted and unique")


@dataclass(frozen=True, slots=True)
class Budget:
    """How much a caller is willing to spend on a representation.

    Passed *into* `represent`, not enforced around it: a handler degrades on its
    own terms, because only it knows that dropping frame density costs less than
    dropping transcript.
    """

    max_chars: int | None

    def permits(self, chars: int) -> bool:
        return self.max_chars is None or chars <= self.max_chars
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/domain -v && uv run mypy`
Expected: all pass, mypy clean

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/domain tests/unit/domain
git commit -m "feat(domain): add Affordance, Card, Rendition and Rendered"
```

---

### Task 7: Ports

**Files:**
- Create: `src/readeverything/ports/source.py`, `src/readeverything/ports/detection.py`, `src/readeverything/ports/artifacts.py`, `src/readeverything/ports/handler.py`
- Test: `tests/unit/ports/test_protocols_are_runtime_checkable.py`

**Interfaces:**
- Consumes: Tasks 2-6
- Produces: `SourceStat`, `SourceReader`, `SourceLister`, `FileSource`, `MimeDetector`, `ArtifactStore`, `MediaHandler`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/ports/test_protocols_are_runtime_checkable.py
from readeverything.ports.artifacts import ArtifactStore
from readeverything.ports.detection import MimeDetector
from readeverything.ports.handler import MediaHandler
from readeverything.ports.source import FileSource, SourceLister, SourceReader, SourceStat

PORTS = [ArtifactStore, MimeDetector, MediaHandler, FileSource, SourceLister, SourceReader, SourceStat]


def test_every_port_is_runtime_checkable() -> None:
    """Structural typing is the point: an adapter must not have to inherit."""
    for port in PORTS:
        assert hasattr(port, "_is_runtime_protocol"), f"{port.__name__} is not runtime_checkable"


def test_file_source_composes_the_narrow_slices() -> None:
    """Collaborators annotate the slimmest slice they use."""
    assert issubclass(FileSource, SourceStat)
    assert issubclass(FileSource, SourceReader)
    assert issubclass(FileSource, SourceLister)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/ports -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementations**

```python
# src/readeverything/ports/source.py
"""Reaching bytes, without knowing where they live.

Split into three capability slices because collaborators genuinely need
different ones: a handler reads and never lists, the pipeline stats and never
reads, a directory walk lists and never reads. Annotating the slimmest slice is
what keeps an adapter honest about what it is being asked for.

`uri` is opaque. A local path, an object-store key and an archive member
`"/a.zip!inner.txt"` are all just strings here; only the adapter interprets
them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class SourceStat(Protocol):
    async def exists(self, uri: str) -> bool: ...

    async def size(self, uri: str) -> int: ...


@runtime_checkable
class SourceReader(Protocol):
    async def read_bytes(self, uri: str) -> bytes:
        """The whole source. Prefer `stream` for anything that may be large."""
        ...

    def stream(self, uri: str, *, chunk_size: int = 1 << 20) -> AsyncIterator[bytes]: ...

    async def read_range(self, uri: str, start: int, end: int) -> bytes:
        """Bytes in `[start, end)`."""
        ...

    async def local_path(self, uri: str) -> str:
        """A real filesystem path for this source.

        External tools take paths, not streams, so a non-local adapter must
        materialise a temporary file. This is the one place that cost is
        acknowledged rather than hidden behind a stream nobody can use.
        """
        ...


@runtime_checkable
class SourceLister(Protocol):
    async def walk(self, uri: str) -> Sequence[str]:
        """Every source under `uri`, recursively. Directories are not returned."""
        ...


@runtime_checkable
class FileSource(SourceStat, SourceReader, SourceLister, Protocol):
    """Everything a fully-featured source adapter provides."""
```

```python
# src/readeverything/ports/detection.py
"""Deciding what a source is.

Content is the authority and the filename is a tiebreak, never the reverse. An
extension is a claim by whoever named the file; the bytes are a fact.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from readeverything.domain.identity import MimeType


@runtime_checkable
class MimeDetector(Protocol):
    async def detect(self, uri: str, head: bytes) -> MimeType:
        """The mimetype of a source, given its first bytes and its uri."""
        ...
```

```python
# src/readeverything/ports/artifacts.py
"""Storing what was expensive to derive.

Entries are immutable and content-addressed on the whole derivation, so there
is no invalidation protocol and no staleness: a different input, handler
version, parameter set or model revision is simply a different key.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ArtifactStore(Protocol):
    async def get(self, key: str) -> bytes | None:
        """The stored artifact, or None on a miss."""
        ...

    async def put(self, key: str, value: bytes) -> None:
        """Store an artifact. Storing an existing key is a no-op, not an error."""
        ...
```

```python
# src/readeverything/ports/handler.py
"""What every media handler must be able to say and do.

Handlers are stateless and receive every capability by constructor injection.
A handler never touches a filesystem, never shells out directly and never reads
the environment — it asks an injected port. That is what makes them unit
testable with fakes, and what keeps ffmpeg confined to one adapter module.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel

from readeverything.domain.affordance import Affordance
from readeverything.domain.capability import Capability
from readeverything.domain.card import Card
from readeverything.domain.identity import SourceRef
from readeverything.domain.rendition import Budget, Rendered, Rendition


@runtime_checkable
class MediaHandler(Protocol):
    #: Mimetype patterns this handler claims. See `registry.patterns`.
    mime_patterns: ClassVar[tuple[str, ...]]
    #: Higher wins a tie. Bundled handlers use 0; a caller shadows with 1+.
    priority: ClassVar[int]
    #: Stable identity, part of the artifact cache key.
    handler_id: ClassVar[str]
    #: Bumped when this handler's output changes for the same input.
    handler_version: ClassVar[int]

    def requires(self) -> frozenset[Capability]:
        """Capabilities without which this handler cannot function at all."""
        ...

    def affordances(self) -> tuple[Affordance, ...]:
        """Everything this handler can do, before capability filtering."""
        ...

    async def describe(self, ref: SourceRef) -> Card: ...

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition: ...

    async def represent(self, ref: SourceRef, budget: Budget) -> Rendered: ...
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/ports -v && uv run mypy && uv run lint-imports`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/ports tests/unit/ports
git commit -m "feat(ports): add source, detection, artifact and handler protocols"
```

---

### Task 8: Registry — dispatch and capability filtering

**Files:**
- Create: `src/readeverything/registry/patterns.py`, `src/readeverything/registry/registry.py`
- Test: `tests/unit/registry/test_patterns.py`, `tests/unit/registry/test_registry.py`

**Interfaces:**
- Consumes: `MimeType`, `MediaKind`, `Capability`, `CapabilitySet`, `MediaHandler`, `Affordance`
- Produces: `match_pattern(pattern, mime) -> MatchRank | None`, `MimeTypeRegistry(handlers, capabilities)` with `.resolve(mime) -> MediaHandler`, `.available_affordances(handler) -> tuple[Affordance, ...]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/registry/test_patterns.py
from readeverything.domain.identity import MimeType
from readeverything.registry.patterns import MatchRank, match_pattern


def test_an_exact_pattern_ranks_highest() -> None:
    assert match_pattern("video/mp4", MimeType.parse("video/mp4")) is MatchRank.EXACT


def test_a_suffix_pattern_matches_a_structured_subtype() -> None:
    assert match_pattern("+zip", MimeType.parse("application/epub+zip")) is MatchRank.SUFFIX
    assert match_pattern("+zip", MimeType.parse("application/pdf")) is None


def test_a_type_wildcard_matches_the_family() -> None:
    assert match_pattern("video/*", MimeType.parse("video/webm")) is MatchRank.TYPE
    assert match_pattern("video/*", MimeType.parse("audio/mp3")) is None


def test_a_kind_pattern_matches_the_media_kind() -> None:
    assert match_pattern("kind:text", MimeType.parse("text/markdown")) is MatchRank.KIND
    assert match_pattern("kind:binary", MimeType.parse("application/pdf")) is MatchRank.KIND


def test_the_star_pattern_always_matches_and_ranks_lowest() -> None:
    assert match_pattern("*", MimeType.parse("application/x-anything")) is MatchRank.FALLBACK


def test_a_non_matching_exact_pattern_returns_none() -> None:
    assert match_pattern("video/mp4", MimeType.parse("video/webm")) is None


def test_ranks_are_ordered_most_specific_first() -> None:
    assert (
        MatchRank.EXACT
        < MatchRank.SUFFIX
        < MatchRank.TYPE
        < MatchRank.KIND
        < MatchRank.FALLBACK
    )
```

```python
# tests/unit/registry/test_registry.py
import pytest
from pydantic import BaseModel

from readeverything.domain.affordance import Affordance, DetailLevel
from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.domain.card import Card
from readeverything.domain.identity import MediaKind, MimeType, SourceRef
from readeverything.domain.rendition import Budget, Rendered, Rendition
from readeverything.registry.registry import MimeTypeRegistry, NoHandlerError


class _Params(BaseModel):
    pass


class _Stub:
    """A handler whose behaviour is declared entirely by construction."""

    mime_patterns: tuple[str, ...] = ()
    priority: int = 0
    handler_id: str = "stub"
    handler_version: int = 1

    def __init__(
        self,
        *,
        patterns: tuple[str, ...],
        handler_id: str,
        requires: frozenset[Capability] = frozenset(),
        affordance_requires: frozenset[Capability] = frozenset(),
        priority: int = 0,
    ) -> None:
        self.mime_patterns = patterns
        self.handler_id = handler_id
        self.priority = priority
        self._requires = requires
        self._affordance_requires = affordance_requires

    def requires(self) -> frozenset[Capability]:
        return self._requires

    def affordances(self) -> tuple[Affordance, ...]:
        return (
            Affordance(
                name="free",
                description="Needs nothing.",
                params=_Params,
                requires=frozenset(),
                level=DetailLevel.CARD,
            ),
            Affordance(
                name="costly",
                description="Needs a capability.",
                params=_Params,
                requires=self._affordance_requires,
                level=DetailLevel.DEEP,
            ),
        )

    async def describe(self, ref: SourceRef) -> Card:
        raise NotImplementedError

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition:
        raise NotImplementedError

    async def represent(self, ref: SourceRef, budget: Budget) -> Rendered:
        raise NotImplementedError


def test_resolve_prefers_an_exact_match_over_a_wildcard() -> None:
    exact = _Stub(patterns=("video/mp4",), handler_id="exact")
    wild = _Stub(patterns=("video/*",), handler_id="wild")
    registry = MimeTypeRegistry(handlers=(wild, exact), capabilities=CapabilitySet.empty())
    assert registry.resolve(MimeType.parse("video/mp4")).handler_id == "exact"


def test_resolve_falls_back_through_the_ranks() -> None:
    wild = _Stub(patterns=("video/*",), handler_id="wild")
    star = _Stub(patterns=("*",), handler_id="star")
    registry = MimeTypeRegistry(handlers=(wild, star), capabilities=CapabilitySet.empty())
    assert registry.resolve(MimeType.parse("video/webm")).handler_id == "wild"
    assert registry.resolve(MimeType.parse("audio/mp3")).handler_id == "star"


def test_priority_breaks_a_tie_at_the_same_rank() -> None:
    """A caller shadows a bundled handler without forking it."""
    bundled = _Stub(patterns=("video/mp4",), handler_id="bundled", priority=0)
    custom = _Stub(patterns=("video/mp4",), handler_id="custom", priority=1)
    registry = MimeTypeRegistry(handlers=(bundled, custom), capabilities=CapabilitySet.empty())
    assert registry.resolve(MimeType.parse("video/mp4")).handler_id == "custom"


def test_a_handler_whose_capabilities_are_missing_is_dropped_entirely() -> None:
    needs_ffmpeg = _Stub(
        patterns=("video/mp4",), handler_id="video", requires=frozenset({Capability.FFMPEG})
    )
    star = _Stub(patterns=("*",), handler_id="star")
    registry = MimeTypeRegistry(
        handlers=(needs_ffmpeg, star), capabilities=CapabilitySet.empty()
    )
    assert registry.resolve(MimeType.parse("video/mp4")).handler_id == "star"


def test_a_handler_is_kept_when_its_capabilities_are_present() -> None:
    needs_ffmpeg = _Stub(
        patterns=("video/mp4",), handler_id="video", requires=frozenset({Capability.FFMPEG})
    )
    registry = MimeTypeRegistry(
        handlers=(needs_ffmpeg,), capabilities=CapabilitySet.of({Capability.FFMPEG: "7.1"})
    )
    assert registry.resolve(MimeType.parse("video/mp4")).handler_id == "video"


def test_unsatisfied_affordances_are_filtered_from_a_surviving_handler() -> None:
    """Video still works with no ASR; read_transcript simply does not exist."""
    handler = _Stub(
        patterns=("video/mp4",),
        handler_id="video",
        affordance_requires=frozenset({Capability.ASR}),
    )
    registry = MimeTypeRegistry(handlers=(handler,), capabilities=CapabilitySet.empty())
    names = tuple(a.name for a in registry.available_affordances(handler))
    assert names == ("free",)


def test_all_affordances_survive_when_capabilities_are_present() -> None:
    handler = _Stub(
        patterns=("video/mp4",),
        handler_id="video",
        affordance_requires=frozenset({Capability.ASR}),
    )
    registry = MimeTypeRegistry(
        handlers=(handler,), capabilities=CapabilitySet.of({Capability.ASR: "whisper@1"})
    )
    names = tuple(a.name for a in registry.available_affordances(handler))
    assert names == ("free", "costly")


def test_resolving_with_no_handler_at_all_raises() -> None:
    registry = MimeTypeRegistry(handlers=(), capabilities=CapabilitySet.empty())
    with pytest.raises(NoHandlerError, match="no handler"):
        registry.resolve(MimeType.parse("video/mp4"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/registry -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementations**

```python
# src/readeverything/registry/patterns.py
"""Matching a mimetype against a handler's claims.

Five ranks, most specific first. The ranks are an `IntEnum` so that "more
specific" is expressible as `<`, which is what makes the registry's selection a
plain `min` rather than a chain of conditionals that has to be read to be
believed.
"""

from __future__ import annotations

from enum import IntEnum

from readeverything.domain.identity import MediaKind, MimeType


class MatchRank(IntEnum):
    """How specifically a pattern matched. Lower is more specific."""

    EXACT = 0
    SUFFIX = 1
    TYPE = 2
    KIND = 3
    FALLBACK = 4


def match_pattern(pattern: str, mime: MimeType) -> MatchRank | None:
    """The rank at which `pattern` matches `mime`, or None if it does not.

    Pattern forms:
      - `"video/mp4"`  exact mimetype
      - `"+zip"`       structured suffix
      - `"video/*"`    type wildcard
      - `"kind:text"`  media kind
      - `"*"`          always matches
    """
    if pattern == "*":
        return MatchRank.FALLBACK
    if pattern.startswith("kind:"):
        wanted = pattern.removeprefix("kind:")
        return MatchRank.KIND if MediaKind.for_mime(mime).value == wanted else None
    if pattern.startswith("+"):
        return MatchRank.SUFFIX if mime.suffix == pattern.removeprefix("+") else None
    if pattern.endswith("/*"):
        return MatchRank.TYPE if mime.type == pattern.removesuffix("/*") else None
    return MatchRank.EXACT if str(mime) == pattern else None
```

```python
# src/readeverything/registry/registry.py
"""Choosing a handler, and deciding what it may offer.

Two-stage capability filtering, and the order matters:

1. A handler whose `requires()` is unsatisfied is dropped *before* dispatch, so
   it cannot win a match it could not then serve.
2. A surviving handler's individual affordances are filtered by their own
   requirements.

The consequence is the design goal: with no ASR configured, video still works —
metadata, outline and frames are there, and `read_transcript` does not exist.
The agent never sees a tool it cannot use.
"""

from __future__ import annotations

from collections.abc import Sequence

from readeverything.domain.affordance import Affordance
from readeverything.domain.capability import CapabilitySet
from readeverything.domain.errors import DomainError
from readeverything.domain.identity import MimeType
from readeverything.ports.handler import MediaHandler
from readeverything.registry.patterns import MatchRank, match_pattern


class NoHandlerError(DomainError):
    """Nothing claimed this mimetype.

    Reachable only when no fallback handler is registered. A composition that
    includes `BinaryHandler` cannot produce this, which is why there is no
    "unsupported file" path in normal use.
    """

    def __init__(self, mime: MimeType) -> None:
        super().__init__(f"no handler for {mime}; register a fallback handler with pattern '*'")


class MimeTypeRegistry:
    """Dispatches a mimetype to the most specific handler that can serve it."""

    def __init__(
        self,
        *,
        handlers: Sequence[MediaHandler],
        capabilities: CapabilitySet,
    ) -> None:
        self._capabilities = capabilities
        self._handlers = tuple(h for h in handlers if capabilities.satisfies(h.requires()))

    @property
    def handlers(self) -> tuple[MediaHandler, ...]:
        """The handlers that survived capability filtering."""
        return self._handlers

    def resolve(self, mime: MimeType) -> MediaHandler:
        """The handler for `mime`: most specific rank, then highest priority."""
        best: tuple[MatchRank, int, int] | None = None
        chosen: MediaHandler | None = None
        for index, handler in enumerate(self._handlers):
            ranks = [
                rank
                for pattern in handler.mime_patterns
                if (rank := match_pattern(pattern, mime)) is not None
            ]
            if not ranks:
                continue
            # Negated priority so that a plain `<` comparison means "better":
            # lower rank wins, then higher priority, then earlier registration.
            candidate = (min(ranks), -handler.priority, index)
            if best is None or candidate < best:
                best = candidate
                chosen = handler
        if chosen is None:
            raise NoHandlerError(mime)
        return chosen

    def available_affordances(self, handler: MediaHandler) -> tuple[Affordance, ...]:
        """The affordances of `handler` this deployment can actually serve."""
        return tuple(a for a in handler.affordances() if a.is_available(self._capabilities))
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/registry -v && uv run mypy`
Expected: 15 passed (7 pattern tests + 8 registry tests), mypy clean

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/registry tests/unit/registry
git commit -m "feat(registry): add mimetype dispatch with two-stage capability filtering"
```

---

### Task 9: Adapters — content hashing and the local filesystem

**Files:**
- Create: `src/readeverything/adapters/hashing.py`, `src/readeverything/adapters/local_source.py`
- Test: `tests/unit/adapters/test_hashing.py`, `tests/unit/adapters/test_local_source.py`

**Interfaces:**
- Consumes: `ContentHash`, `SourceReader`
- Produces: `ContentHasher(source, memo=None)` with `.hash(uri) -> ContentHash`; `LocalFileSource(root)` implementing `FileSource`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/adapters/test_hashing.py
from pathlib import Path

from readeverything.adapters.hashing import ContentHasher, StatMemo
from readeverything.adapters.local_source import LocalFileSource


async def test_identical_bytes_hash_identically(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"hello")
    (tmp_path / "b.txt").write_bytes(b"hello")
    hasher = ContentHasher(source=LocalFileSource(root=tmp_path))
    assert await hasher.hash("a.txt") == await hasher.hash("b.txt")


async def test_different_bytes_hash_differently(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"hello")
    (tmp_path / "b.txt").write_bytes(b"world")
    hasher = ContentHasher(source=LocalFileSource(root=tmp_path))
    assert await hasher.hash("a.txt") != await hasher.hash("b.txt")


async def test_the_memo_is_an_optimisation_only(tmp_path: Path) -> None:
    """A cold memo must produce the same answer as a warm one."""
    (tmp_path / "a.txt").write_bytes(b"hello")
    source = LocalFileSource(root=tmp_path)
    warm = ContentHasher(source=source, memo=StatMemo())
    first = await warm.hash("a.txt")
    second = await warm.hash("a.txt")
    cold = await ContentHasher(source=source).hash("a.txt")
    assert first == second == cold


async def test_editing_a_file_changes_its_hash_through_the_memo(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_bytes(b"hello")
    hasher = ContentHasher(source=LocalFileSource(root=tmp_path), memo=StatMemo())
    before = await hasher.hash("a.txt")
    path.write_bytes(b"hello there")
    assert await hasher.hash("a.txt") != before
```

```python
# tests/unit/adapters/test_local_source.py
from pathlib import Path

import pytest

from readeverything.adapters.local_source import LocalFileSource
from readeverything.domain.errors import SourceUnreadableError


async def test_reads_bytes(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"hello")
    assert await LocalFileSource(root=tmp_path).read_bytes("a.txt") == b"hello"


async def test_reads_a_range(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"hello world")
    assert await LocalFileSource(root=tmp_path).read_range("a.txt", 6, 11) == b"world"


async def test_streams_in_chunks(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"abcdef")
    chunks = [c async for c in LocalFileSource(root=tmp_path).stream("a.txt", chunk_size=2)]
    assert chunks == [b"ab", b"cd", b"ef"]


async def test_walk_returns_files_relative_to_the_root(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_bytes(b"x")
    (tmp_path / "b.txt").write_bytes(b"y")
    assert sorted(await LocalFileSource(root=tmp_path).walk(".")) == ["b.txt", "sub/a.txt"]


async def test_escaping_the_root_is_refused(tmp_path: Path) -> None:
    """A traversal must fail loudly, not read an unintended file."""
    with pytest.raises(SourceUnreadableError, match="outside the root"):
        await LocalFileSource(root=tmp_path).read_bytes("../etc/passwd")


async def test_a_missing_file_raises_the_domain_error(tmp_path: Path) -> None:
    with pytest.raises(SourceUnreadableError):
        await LocalFileSource(root=tmp_path).read_bytes("nope.txt")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/adapters -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementations**

```python
# src/readeverything/adapters/local_source.py
"""A `FileSource` over a local directory, and nothing outside it.

The root is a boundary, not a convenience. Every uri is resolved and checked
against it, so `"../../etc/passwd"` fails loudly rather than reading an
unintended file. This is the only sandboxing the library performs, and it is
here rather than in the domain because "what is a path" is adapter knowledge.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from readeverything.domain.errors import SourceUnreadableError


class LocalFileSource:
    """Reads files beneath a single root directory."""

    def __init__(self, *, root: Path | str) -> None:
        self._root = Path(root).resolve()

    def _resolve(self, uri: str) -> Path:
        candidate = (self._root / uri).resolve()
        if candidate != self._root and self._root not in candidate.parents:
            raise SourceUnreadableError(f"{uri!r} resolves outside the root {self._root}")
        return candidate

    async def exists(self, uri: str) -> bool:
        return await asyncio.to_thread(self._resolve(uri).is_file)

    async def size(self, uri: str) -> int:
        path = self._resolve(uri)
        try:
            return await asyncio.to_thread(lambda: path.stat().st_size)
        except OSError as exc:
            raise SourceUnreadableError(f"cannot stat {uri!r}: {exc}") from exc

    async def read_bytes(self, uri: str) -> bytes:
        path = self._resolve(uri)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except OSError as exc:
            raise SourceUnreadableError(f"cannot read {uri!r}: {exc}") from exc

    async def read_range(self, uri: str, start: int, end: int) -> bytes:
        path = self._resolve(uri)

        def _read() -> bytes:
            with path.open("rb") as handle:
                handle.seek(start)
                return handle.read(max(0, end - start))

        try:
            return await asyncio.to_thread(_read)
        except OSError as exc:
            raise SourceUnreadableError(f"cannot read {uri!r}: {exc}") from exc

    async def stream(self, uri: str, *, chunk_size: int = 1 << 20) -> AsyncIterator[bytes]:
        path = self._resolve(uri)
        handle = await asyncio.to_thread(path.open, "rb")
        try:
            while True:
                chunk = await asyncio.to_thread(handle.read, chunk_size)
                if not chunk:
                    return
                yield chunk
        except OSError as exc:
            raise SourceUnreadableError(f"cannot read {uri!r}: {exc}") from exc
        finally:
            await asyncio.to_thread(handle.close)

    async def local_path(self, uri: str) -> str:
        return str(self._resolve(uri))

    async def walk(self, uri: str) -> Sequence[str]:
        base = self._resolve(uri)

        def _walk() -> list[str]:
            return sorted(
                str(p.relative_to(self._root)) for p in base.rglob("*") if p.is_file()
            )

        try:
            return await asyncio.to_thread(_walk)
        except OSError as exc:
            raise SourceUnreadableError(f"cannot walk {uri!r}: {exc}") from exc
```

```python
# src/readeverything/adapters/hashing.py
"""Content hashing, with a stat memo in front of it.

`blake2b` from the standard library rather than blake3: fast enough streamed,
and it costs no dependency in a package whose base install is deliberately
light.

The memo maps `(device, inode, size, mtime_ns)` to a hash. It is an
**optimisation only** — a miss costs a rehash, never a wrong answer — which is
why an edited file with an unchanged size still invalidates: `mtime_ns` is part
of the key.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from readeverything.domain.identity import ContentHash
from readeverything.ports.source import SourceReader

_CHUNK = 1 << 20


class StatMemo:
    """Remembers hashes by identity-and-mtime. Not required for correctness."""

    def __init__(self) -> None:
        self._entries: dict[tuple[int, int, int, int], ContentHash] = {}

    @staticmethod
    def _key(path: Path) -> tuple[int, int, int, int] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def get(self, path: Path) -> ContentHash | None:
        key = self._key(path)
        return None if key is None else self._entries.get(key)

    def put(self, path: Path, value: ContentHash) -> None:
        key = self._key(path)
        if key is not None:
            self._entries[key] = value


class ContentHasher:
    """Hashes a source's bytes, streamed."""

    def __init__(self, *, source: SourceReader, memo: StatMemo | None = None) -> None:
        self._source = source
        self._memo = memo

    async def hash(self, uri: str) -> ContentHash:
        path: Path | None = None
        if self._memo is not None:
            path = Path(await self._source.local_path(uri))
            cached = await asyncio.to_thread(self._memo.get, path)
            if cached is not None:
                return cached
        digest = hashlib.blake2b(digest_size=32)
        async for chunk in self._source.stream(uri, chunk_size=_CHUNK):
            digest.update(chunk)
        value = ContentHash(digest.hexdigest())
        if self._memo is not None and path is not None:
            await asyncio.to_thread(self._memo.put, path, value)
        return value
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/adapters -v && uv run mypy`
Expected: 10 passed, mypy clean

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/adapters tests/unit/adapters
git commit -m "feat(adapters): add LocalFileSource and streamed content hashing"
```

---

### Task 10: Adapters — mimetype detection

**Files:**
- Create: `src/readeverything/adapters/detection.py`
- Test: `tests/unit/adapters/test_detection.py`

**Interfaces:**
- Consumes: `MimeType`, `MimeDetector`
- Produces: `PuremagicDetector()` implementing `MimeDetector`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/adapters/test_detection.py
from readeverything.adapters.detection import PuremagicDetector
from readeverything.domain.identity import MimeType

PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24


async def test_content_beats_a_lying_extension() -> None:
    """An extension is a claim; the bytes are a fact."""
    detected = await PuremagicDetector().detect("photo.txt", PNG_HEADER)
    assert detected.type == "image"


async def test_the_filename_is_used_when_content_is_inconclusive() -> None:
    detected = await PuremagicDetector().detect("notes.md", b"# heading\n")
    assert detected == MimeType.parse("text/markdown")


async def test_utf8_text_without_an_extension_is_plain_text() -> None:
    detected = await PuremagicDetector().detect("notes", b"just some words\n")
    assert detected == MimeType.parse("text/plain")


async def test_undecodable_bytes_without_a_signature_are_octet_stream() -> None:
    detected = await PuremagicDetector().detect("blob", b"\x00\x01\x02\xff\xfe")
    assert detected == MimeType.parse("application/octet-stream")


async def test_empty_content_is_octet_stream() -> None:
    assert await PuremagicDetector().detect("empty", b"") == MimeType.parse(
        "application/octet-stream"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/adapters/test_detection.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/readeverything/adapters/detection.py
"""Detecting a mimetype from bytes, with the filename as a tiebreak.

Order is deliberate and is the whole point of the module:

1. A magic signature in the content. Authoritative when present.
2. The filename's extension. A claim, consulted only when the bytes are silent.
3. Decodable as UTF-8 with no control characters, therefore `text/plain`.
4. `application/octet-stream`, which the binary fallback handler always accepts.

Getting 1 and 2 the wrong way round is the classic defect: a PNG named
`photo.txt` would dispatch to a text handler and produce mojibake that looks
like a corrupt file rather than a misidentified one.
"""

from __future__ import annotations

import mimetypes

import puremagic

from readeverything.domain.identity import MimeType

_OCTET_STREAM = MimeType.parse("application/octet-stream")

#: Bytes that never appear in text a handler could usefully read. Tab, newline
#: and carriage return are excluded because they obviously do.
_CONTROL = frozenset(range(0, 9)) | frozenset(range(14, 32))


class PuremagicDetector:
    """Content-first mimetype detection."""

    async def detect(self, uri: str, head: bytes) -> MimeType:
        if not head:
            return _OCTET_STREAM

        for match in puremagic.magic_string(head):
            if match.mime_type:
                try:
                    return MimeType.parse(match.mime_type)
                except ValueError:
                    continue

        guessed, _ = mimetypes.guess_type(uri)
        if guessed:
            return MimeType.parse(guessed)

        try:
            head.decode("utf-8")
        except UnicodeDecodeError:
            return _OCTET_STREAM
        if any(byte in _CONTROL for byte in head):
            return _OCTET_STREAM
        return MimeType.parse("text/plain")
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/adapters/test_detection.py -v`
Expected: 5 passed. If `test_the_filename_is_used_when_content_is_inconclusive` fails because `mimetypes` does not know `.md` on this platform, add `mimetypes.add_type("text/markdown", ".md")` at module import — do not weaken the assertion.

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/adapters/detection.py tests/unit/adapters/test_detection.py
git commit -m "feat(adapters): add content-first mimetype detection"
```

---

### Task 11: Artifact cache

**Files:**
- Create: `src/readeverything/adapters/cache_key.py`, `src/readeverything/adapters/artifact_store.py`
- Test: `tests/unit/adapters/test_cache_key.py`, `tests/unit/adapters/test_artifact_store.py`

**Interfaces:**
- Consumes: `ContentHash`, `CapabilitySet`, `ArtifactStore`
- Produces: `artifact_key(...) -> str`; `InMemoryArtifactStore()`, `FilesystemArtifactStore(root)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/adapters/test_cache_key.py
from readeverything.adapters.cache_key import artifact_key
from readeverything.domain.capability import Capability, CapabilitySet
from readeverything.domain.identity import ContentHash

CAPS = CapabilitySet.of({Capability.VISION: "qwen3.8@rev1"})


def _key(**overrides: object) -> str:
    kwargs: dict[str, object] = {
        "content_hash": ContentHash("aaa"),
        "handler_id": "video",
        "handler_version": 1,
        "affordance": "describe_frame",
        "params": {"at_s": 1.5},
        "capabilities": CAPS,
    }
    kwargs.update(overrides)
    return artifact_key(**kwargs)  # type: ignore[arg-type]


def test_the_same_derivation_yields_the_same_key() -> None:
    assert _key() == _key()


def test_param_order_does_not_change_the_key() -> None:
    assert _key(params={"a": 1, "b": 2}) == _key(params={"b": 2, "a": 1})


def test_different_content_is_a_different_key() -> None:
    assert _key() != _key(content_hash=ContentHash("bbb"))


def test_a_handler_version_bump_invalidates() -> None:
    """A fixed extraction bug must invalidate exactly what it should."""
    assert _key() != _key(handler_version=2)


def test_different_params_are_a_different_key() -> None:
    assert _key() != _key(params={"at_s": 2.5})


def test_swapping_the_model_invalidates() -> None:
    """Otherwise the cache silently serves a mixture from two models."""
    other = CapabilitySet.of({Capability.VISION: "qwen3.8@rev2"})
    assert _key() != _key(capabilities=other)
```

```python
# tests/unit/adapters/test_artifact_store.py
from pathlib import Path

import pytest

from readeverything.adapters.artifact_store import (
    FilesystemArtifactStore,
    InMemoryArtifactStore,
)


@pytest.fixture(params=["memory", "filesystem"])
def store(request: pytest.FixtureRequest, tmp_path: Path):  # type: ignore[no-untyped-def]
    if request.param == "memory":
        return InMemoryArtifactStore()
    return FilesystemArtifactStore(root=tmp_path)


async def test_a_miss_returns_none(store) -> None:  # type: ignore[no-untyped-def]
    assert await store.get("absent") is None


async def test_a_stored_artifact_round_trips(store) -> None:  # type: ignore[no-untyped-def]
    await store.put("k", b"value")
    assert await store.get("k") == b"value"


async def test_putting_an_existing_key_is_a_noop_not_an_error(store) -> None:  # type: ignore[no-untyped-def]
    """Entries are immutable; a concurrent re-derivation must not explode."""
    await store.put("k", b"first")
    await store.put("k", b"first")
    assert await store.get("k") == b"first"


async def test_binary_content_survives(store) -> None:  # type: ignore[no-untyped-def]
    await store.put("k", bytes(range(256)))
    assert await store.get("k") == bytes(range(256))


async def test_concurrent_writers_do_not_share_a_temp_file(tmp_path: Path) -> None:
    """Two writers of the same key must not collide on one .partial file.

    The rglob assertion is the load-bearing half: it proves no temp file was
    orphaned by a writer that lost the race.
    """
    fs = FilesystemArtifactStore(root=tmp_path)
    await asyncio.gather(*(fs.put("k", b"value") for _ in range(8)))
    assert await fs.get("k") == b"value"
    assert not list(tmp_path.rglob("*.partial"))


async def test_a_key_with_path_characters_is_stored_safely(tmp_path: Path) -> None:
    """A key must never be able to escape the store's root."""
    fs = FilesystemArtifactStore(root=tmp_path)
    await fs.put("../escape", b"x")
    assert await fs.get("../escape") == b"x"
    assert not (tmp_path.parent / "escape").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/adapters/test_cache_key.py tests/unit/adapters/test_artifact_store.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementations**

```python
# src/readeverything/adapters/cache_key.py
"""Deriving the artifact cache key.

The key is the whole derivation, not just the file. Every component earns its
place:

- `content_hash`   a moved or renamed file is a hit; an edited one is a miss.
                   This is why there is no staleness protocol: no mutable key.
- `handler_id`     two handlers may produce different things from one file.
- `handler_version` a fixed extraction bug invalidates exactly what it should.
- `affordance`+`params` the operation and its arguments.
- `capability_fingerprint` the model revisions behind the capabilities. This is
                   the one that is easy to forget, and forgetting it means
                   swapping the vision model silently serves a mixture of
                   descriptions produced by two different models.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from readeverything.domain.capability import CapabilitySet
from readeverything.domain.identity import ContentHash


def artifact_key(
    *,
    content_hash: ContentHash,
    handler_id: str,
    handler_version: int,
    affordance: str,
    params: Mapping[str, Any],
    capabilities: CapabilitySet,
) -> str:
    """A stable digest of one derivation."""
    payload = json.dumps(
        {
            "content_hash": str(content_hash),
            "handler_id": handler_id,
            "handler_version": handler_version,
            "affordance": affordance,
            "params": params,
            "capabilities": capabilities.fingerprint(),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=32).hexdigest()
```

```python
# src/readeverything/adapters/artifact_store.py
"""Where derived artifacts live.

Two adapters, one contract. Entries are immutable, so `put` on an existing key
is a no-op rather than an error — two workers deriving the same artifact
concurrently is normal, not a conflict.

`FilesystemArtifactStore` hashes the key before using it as a filename. Keys
from `artifact_key` are already hex digests, but the store must not depend on
that: a caller passing a raw key with `../` in it would otherwise write outside
the root.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from uuid import uuid4


class InMemoryArtifactStore:
    """A process-lifetime store. Useful in tests and for short-lived agents."""

    def __init__(self) -> None:
        self._entries: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes | None:
        return self._entries.get(key)

    async def put(self, key: str, value: bytes) -> None:
        self._entries.setdefault(key, value)


class FilesystemArtifactStore:
    """A store backed by a directory, sharded two levels to keep dirs small."""

    def __init__(self, *, root: Path | str) -> None:
        self._root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        safe = hashlib.blake2b(key.encode("utf-8"), digest_size=32).hexdigest()
        return self._root / safe[:2] / safe[2:4] / safe

    async def get(self, key: str) -> bytes | None:
        path = self._path(key)

        def _read() -> bytes | None:
            try:
                return path.read_bytes()
            except FileNotFoundError:
                return None

        return await asyncio.to_thread(_read)

    async def put(self, key: str, value: bytes) -> None:
        path = self._path(key)

        def _write() -> None:
            if path.exists():
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename so a crash cannot leave a truncated artifact
            # that later reads as a valid cache hit. The temp name is unique
            # per call, not per key: two workers deriving the same artifact
            # concurrently is the normal case, and a shared temp file would
            # let one writer's bytes be replaced by the other's mid-write.
            temporary = path.with_suffix(f".{uuid4().hex}.partial")
            temporary.write_bytes(value)
            temporary.replace(path)

        await asyncio.to_thread(_write)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/adapters -v && uv run mypy`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/adapters tests/unit/adapters
git commit -m "feat(adapters): add derivation-scoped cache keys and two artifact stores"
```

---

### Task 12: Testing toolkit — fakes and the handler compliance suite

Ships in the wheel so third-party handler authors inherit the contract.

**Files:**
- Create: `src/readeverything/testing/fakes.py`, `src/readeverything/testing/handler_compliance.py`, `src/readeverything/testing/artifact_compliance.py`
- Test: `tests/unit/testing/test_compliance_suite_catches_a_bad_handler.py`

**Interfaces:**
- Consumes: all domain types, `MediaHandler`, `ArtifactStore`
- Produces: `FakeSource`, `FakeVision`, `FakeTranscriber`, `FakeDiarizer`, `MediaHandlerCompliance`, `ArtifactStoreCompliance`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/testing/test_compliance_suite_catches_a_bad_handler.py
"""The compliance suite must fail an adapter that breaks a law.

A suite that passes everything is worse than no suite: it certifies nothing
while looking like it certifies the contract. So the suite is itself tested,
against a handler deliberately built to violate one law.
"""

import pytest
from pydantic import BaseModel

from readeverything.domain.affordance import Affordance, DetailLevel
from readeverything.domain.capability import Capability
from readeverything.domain.card import Card
from readeverything.domain.identity import MediaKind, SourceRef
from readeverything.domain.rendition import Budget, Rendered, Rendition, TextContent
from readeverything.domain.locators import CharSpan
from readeverything.testing.handler_compliance import MediaHandlerCompliance


class _Params(BaseModel):
    pass


class _LyingHandler:
    """Declares an affordance it does not implement — the drift law's target."""

    mime_patterns = ("text/plain",)
    priority = 0
    handler_id = "lying"
    handler_version = 1

    def requires(self) -> frozenset[Capability]:
        return frozenset()

    def affordances(self) -> tuple[Affordance, ...]:
        return (
            Affordance(
                name="declared_but_absent",
                description="Declared and never implemented.",
                params=_Params,
                requires=frozenset(),
                level=DetailLevel.DEEP,
            ),
        )

    async def describe(self, ref: SourceRef) -> Card:
        return Card(
            ref=ref, kind=MediaKind.TEXT, facts={}, outline=(), excerpt=None, affordances=()
        )

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition:
        raise NotImplementedError(name)

    async def represent(self, ref: SourceRef, budget: Budget) -> Rendered:
        raise NotImplementedError


class TestTheSuiteCatchesDrift(MediaHandlerCompliance):
    @pytest.fixture
    def handler(self) -> _LyingHandler:
        return _LyingHandler()

    @pytest.fixture
    def content(self) -> bytes:
        return b"hello"

    async def test_declared_affordances_are_invocable(self, handler, content, ref) -> None:  # type: ignore[no-untyped-def]
        """Override: assert the inherited law FAILS for this deliberately broken handler."""
        with pytest.raises(NotImplementedError):
            await super().test_declared_affordances_are_invocable(handler, content, ref)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/testing -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementations**

```python
# src/readeverything/testing/fakes.py
"""Deterministic stand-ins for everything expensive or nondeterministic.

Unit tests never assert on model text. These fakes produce output derived
mechanically from their input, so a test can assert on *structure and
locators* — the things that must be right — without depending on what a model
happened to say. Model quality is a bench concern, not a test concern.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence

from readeverything.domain.locators import TimeSpan
from readeverything.domain.rendition import SpeakerId, TranscriptCue


class FakeSource:
    """An in-memory `FileSource` over a dict of uri to bytes."""

    def __init__(self, files: Mapping[str, bytes]) -> None:
        self._files = dict(files)

    def _get(self, uri: str) -> bytes:
        if uri not in self._files:
            raise KeyError(uri)
        return self._files[uri]

    async def exists(self, uri: str) -> bool:
        return uri in self._files

    async def size(self, uri: str) -> int:
        return len(self._get(uri))

    async def read_bytes(self, uri: str) -> bytes:
        return self._get(uri)

    async def read_range(self, uri: str, start: int, end: int) -> bytes:
        return self._get(uri)[start:end]

    async def stream(self, uri: str, *, chunk_size: int = 1 << 20) -> AsyncIterator[bytes]:
        data = self._get(uri)
        for offset in range(0, len(data), chunk_size):
            yield data[offset : offset + chunk_size]

    async def local_path(self, uri: str) -> str:
        raise NotImplementedError("FakeSource has no local path; use a tmp_path fixture")

    async def walk(self, uri: str) -> Sequence[str]:
        return sorted(self._files)


class FakeVision:
    """Describes an image by its size, deterministically."""

    model_id = "fake-vision@1"

    async def describe(self, data: bytes, mime: str, prompt: str) -> str:
        return f"[{mime} image of {len(data)} bytes] {prompt}"


class FakeTranscriber:
    """One cue per second, text derived from the index."""

    model_id = "fake-asr@1"

    def __init__(self, *, cues: int = 3) -> None:
        self._cues = cues

    async def transcribe(self, path: str) -> tuple[TranscriptCue, ...]:
        return tuple(
            TranscriptCue(
                span=TimeSpan(float(i), float(i) + 1.0),
                text=f"cue {i}",
                speaker=None,
                confidence=1.0,
            )
            for i in range(self._cues)
        )


class FakeDiarizer:
    """Alternates two speakers, so speaker-turn barriers are exercised."""

    model_id = "fake-diarizer@1"

    async def diarize(self, path: str) -> tuple[tuple[TimeSpan, SpeakerId], ...]:
        return (
            (TimeSpan(0.0, 1.0), SpeakerId("SPEAKER_00")),
            (TimeSpan(1.0, 2.0), SpeakerId("SPEAKER_01")),
            (TimeSpan(2.0, 3.0), SpeakerId("SPEAKER_00")),
        )
```

```python
# src/readeverything/testing/handler_compliance.py
"""The laws every `MediaHandler` must obey.

Subclass, supply a `handler` and a `content` fixture, and inherit the contract.
These are the same bodies the bundled handlers are tested against.
"""

from __future__ import annotations

import pytest

from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
from readeverything.domain.rendition import Budget


class MediaHandlerCompliance:
    """Laws a handler must satisfy to be usable by the registry."""

    @pytest.fixture
    def handler(self) -> object:
        raise NotImplementedError("supply a `handler` fixture")

    @pytest.fixture
    def content(self) -> bytes:
        raise NotImplementedError("supply a `content` fixture")

    @pytest.fixture
    def ref(self, content: bytes) -> SourceRef:
        return SourceRef(
            uri="compliance-subject",
            mime=MimeType.parse("application/octet-stream"),
            content_hash=ContentHash("0" * 64),
            size_bytes=len(content),
        )

    async def test_describe_depends_only_on_content(self, handler, content, ref) -> None:  # type: ignore[no-untyped-def]
        """Same bytes at a different uri produce an identical card body.

        A card that varies with the path would make the artifact cache — which
        is keyed on content — serve one path's card for another's.
        """
        moved = SourceRef(
            uri="somewhere/else",
            mime=ref.mime,
            content_hash=ref.content_hash,
            size_bytes=ref.size_bytes,
        )
        first = await handler.describe(ref)
        second = await handler.describe(moved)
        assert first.kind == second.kind
        assert dict(first.facts) == dict(second.facts)
        assert first.outline == second.outline
        assert first.excerpt == second.excerpt

    async def test_the_card_kind_is_a_media_kind(self, handler, ref) -> None:  # type: ignore[no-untyped-def]
        card = await handler.describe(ref)
        assert isinstance(card.kind, MediaKind)

    async def test_declared_affordances_are_invocable(self, handler, content, ref) -> None:  # type: ignore[no-untyped-def]
        """Every declared affordance can be invoked with default parameters.

        Drift between what a handler declares and what it implements would make
        capability negotiation a lie: the registry would expose a tool that
        cannot run.
        """
        for affordance in handler.affordances():
            params = affordance.params()
            await handler.invoke(ref, affordance.name, params)

    async def test_an_undeclared_affordance_raises(self, handler, ref) -> None:  # type: ignore[no-untyped-def]
        from readeverything.domain.errors import UnknownAffordanceError

        with pytest.raises(UnknownAffordanceError):
            await handler.invoke(ref, "definitely_not_an_affordance", None)  # type: ignore[arg-type]

    async def test_represent_produces_a_map_covering_its_text(self, handler, ref) -> None:  # type: ignore[no-untyped-def]
        """`Rendered` validates this itself; this proves the handler builds one."""
        rendered = await handler.represent(ref, Budget(max_chars=None))
        assert rendered.locator_map.length == len(rendered.text)

    async def test_represent_respects_a_budget_or_reports_degradation(self, handler, ref) -> None:  # type: ignore[no-untyped-def]
        """Truncation must be announced. Silent truncation is invisible in
        exactly the case where the answer is wrong."""
        budget = Budget(max_chars=10)
        rendered = await handler.represent(ref, budget)
        assert budget.permits(len(rendered.text)) or rendered.degradations
```

```python
# src/readeverything/testing/artifact_compliance.py
"""The laws every `ArtifactStore` must obey."""

from __future__ import annotations

import pytest


class ArtifactStoreCompliance:
    """Subclass and supply a `store` fixture."""

    @pytest.fixture
    def store(self) -> object:
        raise NotImplementedError("supply a `store` fixture")

    async def test_a_miss_returns_none(self, store) -> None:  # type: ignore[no-untyped-def]
        assert await store.get("absent-key") is None

    async def test_a_stored_artifact_round_trips(self, store) -> None:  # type: ignore[no-untyped-def]
        await store.put("k", b"value")
        assert await store.get("k") == b"value"

    async def test_entries_are_immutable(self, store) -> None:  # type: ignore[no-untyped-def]
        """A second put must not error and must not replace."""
        await store.put("k", b"first")
        await store.put("k", b"second")
        assert await store.get("k") == b"first"

    async def test_arbitrary_bytes_survive(self, store) -> None:  # type: ignore[no-untyped-def]
        await store.put("k", bytes(range(256)))
        assert await store.get("k") == bytes(range(256))
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/testing -v && uv run lint-imports`
Expected: pass, and the `forbidden` contract confirms `testing` reaches only `ports` and `domain`

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/testing tests/unit/testing
git commit -m "feat(testing): ship handler and artifact-store compliance suites"
```

---

### Task 13: Reference handlers — text and binary fallback

**Files:**
- Create: `src/readeverything/handlers/text.py`, `src/readeverything/handlers/binary.py`
- Test: `tests/unit/handlers/test_text_handler.py`, `tests/unit/handlers/test_binary_handler.py`

**Interfaces:**
- Consumes: all domain types, `SourceReader`
- Produces: `TextHandler(source)`, `BinaryHandler(source)`, and their `ReadRangeParams` / `HexdumpParams`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/handlers/test_text_handler.py
import pytest

from readeverything.domain.errors import UnknownAffordanceError
from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
from readeverything.domain.rendition import Budget, TextContent
from readeverything.handlers.text import ReadRangeParams, TextHandler
from readeverything.testing.fakes import FakeSource
from readeverything.testing.handler_compliance import MediaHandlerCompliance

CONTENT = b"alpha\nbeta\ngamma\n"


def _ref() -> SourceRef:
    return SourceRef(
        uri="a.txt",
        mime=MimeType.parse("text/plain"),
        content_hash=ContentHash("a" * 64),
        size_bytes=len(CONTENT),
    )


def _handler() -> TextHandler:
    return TextHandler(source=FakeSource({"a.txt": CONTENT, "somewhere/else": CONTENT}))


async def test_the_card_reports_line_and_character_counts() -> None:
    card = await _handler().describe(_ref())
    assert card.kind is MediaKind.TEXT
    assert card.facts["lines"] == 3
    assert card.facts["characters"] == len(CONTENT.decode())
    assert card.facts["encoding"] == "utf-8"


async def test_the_card_excerpt_is_bounded() -> None:
    long = ("x" * 5000).encode()
    handler = TextHandler(source=FakeSource({"a.txt": long}))
    ref = SourceRef(
        uri="a.txt",
        mime=MimeType.parse("text/plain"),
        content_hash=ContentHash("b" * 64),
        size_bytes=len(long),
    )
    card = await handler.describe(ref)
    assert card.excerpt is not None
    assert len(card.excerpt) <= 1000


async def test_read_range_returns_the_requested_characters_and_its_locator() -> None:
    rendition = await _handler().invoke("a.txt" and _ref(), "read_range", ReadRangeParams(start=6, end=10))
    assert isinstance(rendition.content, TextContent)
    assert rendition.content.text == "beta"
    assert rendition.locator.start == 6
    assert rendition.locator.end == 10


async def test_read_range_clamps_to_the_end_of_the_text() -> None:
    rendition = await _handler().invoke(_ref(), "read_range", ReadRangeParams(start=12, end=9999))
    assert isinstance(rendition.content, TextContent)
    assert rendition.content.text == "amma\n"


async def test_read_range_on_an_empty_file_raises_a_domain_error() -> None:
    """An empty file has no character range; say so rather than invent one."""
    handler = TextHandler(source=FakeSource({"empty.txt": b""}))
    ref = SourceRef(
        uri="empty.txt",
        mime=MimeType.parse("text/plain"),
        content_hash=ContentHash("d" * 64),
        size_bytes=0,
    )
    with pytest.raises(DomainError, match="empty"):
        await handler.invoke(ref, "read_range", ReadRangeParams(start=0, end=5))


async def test_an_unknown_affordance_raises() -> None:
    with pytest.raises(UnknownAffordanceError, match="read_range"):
        await _handler().invoke(_ref(), "nope", ReadRangeParams(start=0, end=1))


async def test_represent_maps_the_whole_text_with_line_barriers() -> None:
    rendered = await _handler().represent(_ref(), Budget(max_chars=None))
    assert rendered.text == CONTENT.decode()
    assert rendered.locator_map.length == len(rendered.text)
    assert rendered.barriers == ()


async def test_represent_truncates_and_says_so() -> None:
    rendered = await _handler().represent(_ref(), Budget(max_chars=5))
    assert len(rendered.text) == 5
    assert rendered.degradations
    assert "truncated" in rendered.degradations[0].what


class TestTextHandlerCompliance(MediaHandlerCompliance):
    @pytest.fixture
    def handler(self) -> TextHandler:
        return _handler()

    @pytest.fixture
    def content(self) -> bytes:
        return CONTENT

    @pytest.fixture
    def ref(self, content: bytes) -> SourceRef:
        return _ref()
```

```python
# tests/unit/handlers/test_binary_handler.py
import pytest

from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
from readeverything.domain.rendition import Budget, TextContent
from readeverything.handlers.binary import BinaryHandler, HexdumpParams
from readeverything.testing.fakes import FakeSource
from readeverything.testing.handler_compliance import MediaHandlerCompliance

CONTENT = bytes(range(64))


def _ref() -> SourceRef:
    return SourceRef(
        uri="blob.bin",
        mime=MimeType.parse("application/octet-stream"),
        content_hash=ContentHash("c" * 64),
        size_bytes=len(CONTENT),
    )


def _handler() -> BinaryHandler:
    return BinaryHandler(source=FakeSource({"blob.bin": CONTENT, "somewhere/else": CONTENT}))


async def test_the_fallback_always_produces_a_card() -> None:
    """There is no unsupported-file error path; the worst case is a thin card."""
    card = await _handler().describe(_ref())
    assert card.kind is MediaKind.BINARY
    assert card.facts["size_bytes"] == 64
    assert card.excerpt is not None


async def test_hexdump_returns_the_requested_window() -> None:
    rendition = await _handler().invoke(_ref(), "hexdump", HexdumpParams(start=0, length=4))
    assert isinstance(rendition.content, TextContent)
    assert rendition.content.text.startswith("00000000")
    assert "00 01 02 03" in rendition.content.text


async def test_represent_describes_rather_than_dumping() -> None:
    """Feeding a hexdump to an extractor produces noise, not claims."""
    rendered = await _handler().represent(_ref(), Budget(max_chars=None))
    assert "application/octet-stream" in rendered.text
    assert rendered.locator_map.length == len(rendered.text)


class TestBinaryHandlerCompliance(MediaHandlerCompliance):
    @pytest.fixture
    def handler(self) -> BinaryHandler:
        return _handler()

    @pytest.fixture
    def content(self) -> bytes:
        return CONTENT

    @pytest.fixture
    def ref(self, content: bytes) -> SourceRef:
        return _ref()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/handlers -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementations**

```python
# src/readeverything/handlers/text.py
"""Text and source code.

The simplest possible handler that is still a real one: it needs no
capabilities, so it exercises the registry's satisfied path, and it produces a
genuine `LocatorMap` over `CharSpan`s, so it exercises the citation path
end to end without any model or binary being involved.
"""

from __future__ import annotations

from typing import ClassVar

from charset_normalizer import from_bytes
from pydantic import BaseModel, Field

from readeverything.domain.affordance import Affordance, DetailLevel
from readeverything.domain.capability import Capability
from readeverything.domain.card import Card, Segment
from readeverything.domain.errors import DomainError, UnknownAffordanceError
from readeverything.domain.identity import MediaKind, SourceRef
from readeverything.domain.locator_map import LocatorMap, LocatorSegment
from readeverything.domain.locators import CharSpan
from readeverything.domain.rendition import (
    Budget,
    Degradation,
    Rendered,
    Rendition,
    TextContent,
)
from readeverything.ports.source import SourceReader

_EXCERPT_CHARS = 1000


class ReadRangeParams(BaseModel):
    start: int = Field(default=0, ge=0)
    end: int = Field(default=_EXCERPT_CHARS, ge=1)


class TextHandler:
    """Reads decodable text."""

    mime_patterns: ClassVar[tuple[str, ...]] = ("kind:text", "application/json", "application/xml")
    priority: ClassVar[int] = 0
    handler_id: ClassVar[str] = "text"
    handler_version: ClassVar[int] = 1

    def __init__(self, *, source: SourceReader) -> None:
        self._source = source

    def requires(self) -> frozenset[Capability]:
        return frozenset()

    def affordances(self) -> tuple[Affordance, ...]:
        return (
            Affordance(
                name="read_range",
                description=(
                    "Read a character range of a text file. "
                    "Offsets are characters, not bytes, and the end is clamped to the file."
                ),
                params=ReadRangeParams,
                requires=frozenset(),
                level=DetailLevel.SEGMENT,
            ),
        )

    async def _text(self, ref: SourceRef) -> tuple[str, str]:
        """The decoded text and the encoding it was decoded with."""
        data = await self._source.read_bytes(ref.uri)
        if not data:
            return "", "utf-8"
        try:
            return data.decode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            best = from_bytes(data).best()
            if best is None:
                return data.decode("utf-8", errors="replace"), "utf-8/replace"
            return str(best), best.encoding

    async def describe(self, ref: SourceRef) -> Card:
        text, encoding = await self._text(ref)
        lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        outline = (
            (Segment(CharSpan(0, len(text)), "whole file"),) if text else ()
        )
        return Card(
            ref=ref,
            kind=MediaKind.TEXT,
            facts={"lines": lines, "characters": len(text), "encoding": encoding},
            outline=outline,
            excerpt=text[:_EXCERPT_CHARS] if text else None,
            affordances=self.affordances(),
        )

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition:
        if name != "read_range":
            raise UnknownAffordanceError(name, (a.name for a in self.affordances()))
        if not isinstance(params, ReadRangeParams):
            # Not an `assert`: bandit's B101 skip is scoped to the testing
            # package, and `python -O` would strip the check entirely.
            raise TypeError(f"expected ReadRangeParams, got {type(params).__name__}")
        text, _ = await self._text(ref)
        if not text:
            # Every Rendition must carry a locator and there is no honest one
            # for a zero-length file: CharSpan(0, 0) raises by construction.
            raise DomainError(f"{ref.uri} is empty; there is no character range to read")
        start = min(params.start, len(text) - 1)
        end = min(params.end, len(text))
        if start >= end:
            start, end = 0, 1
        return Rendition(locator=CharSpan(start, end), content=TextContent(text[start:end]))

    async def represent(self, ref: SourceRef, budget: Budget) -> Rendered:
        text, _ = await self._text(ref)
        degradations: tuple[Degradation, ...] = ()
        if budget.max_chars is not None and len(text) > budget.max_chars:
            degradations = (
                Degradation(
                    what="text truncated",
                    detail=f"kept {budget.max_chars} of {len(text)} characters",
                ),
            )
            text = text[: budget.max_chars]
        if not text:
            text = f"[empty text file: {ref.uri}]"
        return Rendered(
            text=text,
            locator_map=LocatorMap.build(
                (LocatorSegment(CharSpan(0, len(text)), CharSpan(0, len(text))),)
            ),
            barriers=(),
            degradations=degradations,
        )
```

```python
# src/readeverything/handlers/binary.py
"""The fallback that always succeeds.

Its existence is what removes the "unsupported file" error path: the registry's
last dispatch step always finds this, so the worst outcome of pointing an agent
at an unknown file is a thin, honest card rather than an exception.

`represent` deliberately does **not** emit a hexdump. Feeding hex to an
extractor produces noise that looks like claims. It emits one sentence
describing what the file is, which is the true and useful thing to index.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from readeverything.domain.affordance import Affordance, DetailLevel
from readeverything.domain.capability import Capability
from readeverything.domain.card import Card
from readeverything.domain.errors import UnknownAffordanceError
from readeverything.domain.identity import MediaKind, SourceRef
from readeverything.domain.locator_map import LocatorMap, LocatorSegment
from readeverything.domain.locators import ByteRange, CharSpan
from readeverything.domain.rendition import Budget, Rendered, Rendition, TextContent
from readeverything.ports.source import SourceReader

_EXCERPT_BYTES = 64
_BYTES_PER_LINE = 16


class HexdumpParams(BaseModel):
    start: int = Field(default=0, ge=0)
    length: int = Field(default=_EXCERPT_BYTES, ge=1, le=4096)


def _hexdump(data: bytes, offset: int) -> str:
    lines: list[str] = []
    for index in range(0, len(data), _BYTES_PER_LINE):
        row = data[index : index + _BYTES_PER_LINE]
        hex_part = " ".join(f"{byte:02x}" for byte in row)
        text_part = "".join(chr(b) if 32 <= b < 127 else "." for b in row)
        lines.append(f"{offset + index:08x}  {hex_part:<47}  |{text_part}|")
    return "\n".join(lines)


class BinaryHandler:
    """Describes anything, by describing as little as is honest."""

    mime_patterns: ClassVar[tuple[str, ...]] = ("*",)
    priority: ClassVar[int] = 0
    handler_id: ClassVar[str] = "binary"
    handler_version: ClassVar[int] = 1

    def __init__(self, *, source: SourceReader) -> None:
        self._source = source

    def requires(self) -> frozenset[Capability]:
        return frozenset()

    def affordances(self) -> tuple[Affordance, ...]:
        return (
            Affordance(
                name="hexdump",
                description=(
                    "Dump a window of raw bytes as hex and printable ASCII. "
                    "Use only to identify an unknown format; it is not readable content."
                ),
                params=HexdumpParams,
                requires=frozenset(),
                level=DetailLevel.SEGMENT,
            ),
        )

    async def describe(self, ref: SourceRef) -> Card:
        head = await self._source.read_range(ref.uri, 0, _EXCERPT_BYTES)
        return Card(
            ref=ref,
            kind=MediaKind.BINARY,
            facts={"size_bytes": ref.size_bytes, "mime": str(ref.mime)},
            outline=(),
            excerpt=_hexdump(head, 0) if head else None,
            affordances=self.affordances(),
        )

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition:
        if name != "hexdump":
            raise UnknownAffordanceError(name, (a.name for a in self.affordances()))
        if not isinstance(params, HexdumpParams):
            raise TypeError(f"expected HexdumpParams, got {type(params).__name__}")
        end = params.start + params.length
        data = await self._source.read_range(ref.uri, params.start, end)
        actual_end = params.start + len(data)
        if actual_end <= params.start:
            actual_end = params.start + 1
        return Rendition(
            locator=ByteRange(params.start, actual_end),
            content=TextContent(_hexdump(data, params.start)),
        )

    async def represent(self, ref: SourceRef, budget: Budget) -> Rendered:
        text = (
            f"Binary file {ref.uri} of type {ref.mime}, {ref.size_bytes} bytes. "
            f"No textual content could be extracted."
        )
        if budget.max_chars is not None:
            text = text[: budget.max_chars] or text[:1]
        return Rendered(
            text=text,
            locator_map=LocatorMap.build(
                (LocatorSegment(CharSpan(0, len(text)), ByteRange(0, max(1, ref.size_bytes))),)
            ),
            barriers=(),
            degradations=(),
        )
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/handlers -v && uv run mypy`
Expected: all pass, including the inherited compliance suites for both handlers

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/handlers tests/unit/handlers
git commit -m "feat(handlers): add text and binary-fallback reference handlers"
```

---

### Task 14: Pipeline — the Perception facade

**Files:**
- Create: `src/readeverything/pipeline/perception.py`
- Test: `tests/unit/pipeline/test_perception.py`

**Interfaces:**
- Consumes: `MimeTypeRegistry`, `MimeDetector`, `ContentHasher`, `FileSource`, `ArtifactStore`
- Produces: `Perception(...)` with `.inspect(uri) -> Card`, `.invoke(uri, name, params) -> Rendition`, `.represent(uri, budget) -> Rendered`, `.list(uri) -> Sequence[str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/pipeline/test_perception.py
from pathlib import Path

import pytest

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.detection import PuremagicDetector
from readeverything.adapters.hashing import ContentHasher
from readeverything.adapters.local_source import LocalFileSource
from readeverything.domain.capability import CapabilitySet
from readeverything.domain.identity import MediaKind
from readeverything.domain.rendition import Budget, TextContent
from readeverything.handlers.binary import BinaryHandler
from readeverything.handlers.text import TextHandler
from readeverything.pipeline.perception import Perception
from readeverything.registry.registry import MimeTypeRegistry


@pytest.fixture
def perception(tmp_path: Path) -> Perception:
    (tmp_path / "notes.txt").write_bytes(b"alpha\nbeta\n")
    (tmp_path / "blob.bin").write_bytes(bytes(range(64)))
    source = LocalFileSource(root=tmp_path)
    return Perception(
        source=source,
        detector=PuremagicDetector(),
        hasher=ContentHasher(source=source),
        registry=MimeTypeRegistry(
            handlers=(TextHandler(source=source), BinaryHandler(source=source)),
            capabilities=CapabilitySet.empty(),
        ),
        artifacts=InMemoryArtifactStore(),
    )


async def test_inspect_dispatches_text_to_the_text_handler(perception: Perception) -> None:
    card = await perception.inspect("notes.txt")
    assert card.kind is MediaKind.TEXT
    assert card.facts["lines"] == 2


async def test_inspect_falls_back_for_unknown_binary(perception: Perception) -> None:
    card = await perception.inspect("blob.bin")
    assert card.kind is MediaKind.BINARY


async def test_the_card_only_offers_available_affordances(perception: Perception) -> None:
    card = await perception.inspect("notes.txt")
    assert card.affordance_names() == ("read_range",)


async def test_invoke_routes_to_the_resolved_handler(perception: Perception) -> None:
    rendition = await perception.invoke("notes.txt", "read_range", {"start": 0, "end": 5})
    assert isinstance(rendition.content, TextContent)
    assert rendition.content.text == "alpha"


async def test_invoke_validates_params_against_the_declared_schema(
    perception: Perception,
) -> None:
    with pytest.raises(ValueError):
        await perception.invoke("notes.txt", "read_range", {"start": -5, "end": 1})


async def test_invoke_refuses_an_affordance_the_resolved_handler_does_not_declare(
    perception: Perception,
) -> None:
    """hexdump belongs to BinaryHandler; a text file never reaches it."""
    with pytest.raises(UnknownAffordanceError):
        await perception.invoke("notes.txt", "hexdump", {})


class _GatedParams(BaseModel):
    pass


class _GatedHandler(TextHandler):
    """A text handler with one extra affordance that requires VISION.

    It overrides `affordances()` but NOT `requires()`, so the handler itself
    survives capability filtering and only the one affordance is dropped. That
    distinction is the point: gating the handler would pin handler-level
    filtering, which is a different property.
    """

    handler_id: ClassVar[str] = "gated"

    def affordances(self) -> tuple[Affordance, ...]:
        return (
            *super().affordances(),
            Affordance(
                name="describe_layout",
                description="Describe the visual layout of the text.",
                params=_GatedParams,
                requires=frozenset({Capability.VISION}),
                level=DetailLevel.DEEP,
            ),
        )


def _perception_with(capabilities: CapabilitySet, tmp_path: Path) -> Perception:
    (tmp_path / "notes.txt").write_bytes(b"alpha\nbeta\n")
    source = LocalFileSource(root=tmp_path)
    return Perception(
        source=source,
        detector=PuremagicDetector(),
        hasher=ContentHasher(source=source),
        registry=MimeTypeRegistry(
            handlers=(_GatedHandler(source=source), BinaryHandler(source=source)),
            capabilities=capabilities,
        ),
        artifacts=InMemoryArtifactStore(),
    )


async def test_a_capability_gated_affordance_is_hidden_without_the_capability(
    tmp_path: Path,
) -> None:
    """The agent never sees a tool this deployment cannot serve."""
    perception = _perception_with(CapabilitySet.empty(), tmp_path)
    card = await perception.inspect("notes.txt")
    assert "describe_layout" not in card.affordance_names()
    with pytest.raises(UnknownAffordanceError):
        await perception.invoke("notes.txt", "describe_layout", {})


async def test_a_capability_gated_affordance_appears_when_the_capability_is_present(
    tmp_path: Path,
) -> None:
    """...and does see it when the deployment can serve it.

    This half is what proves the filtering is selective rather than blanket:
    the hidden-case test alone would pass against a Perception that hid
    everything.
    """
    perception = _perception_with(
        CapabilitySet.of({Capability.VISION: "fake-vision@1"}), tmp_path
    )
    card = await perception.inspect("notes.txt")
    assert "describe_layout" in card.affordance_names()


async def test_represent_returns_a_covering_map(perception: Perception) -> None:
    rendered = await perception.represent("notes.txt", Budget(max_chars=None))
    assert rendered.locator_map.length == len(rendered.text)


async def test_list_walks_the_tree(perception: Perception) -> None:
    assert sorted(await perception.list(".")) == ["blob.bin", "notes.txt"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/pipeline -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/readeverything/pipeline/perception.py
"""Detect, dispatch, describe — the one object callers hold.

This is where a uri becomes a `SourceRef` and a `SourceRef` meets its handler.
It is deliberately thin: every decision it makes has been made somewhere
testable already, and it adds only the sequencing.

Params arrive as plain dicts from a tool call and are validated here against
the affordance's declared schema, so a handler's `invoke` may assume a
well-formed model. Validating in the handler instead would repeat the schema in
every handler and let a malformed call reach adapter code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from readeverything.adapters.hashing import ContentHasher
from readeverything.domain.affordance import Affordance
from readeverything.domain.card import Card
from readeverything.domain.errors import UnknownAffordanceError
from readeverything.domain.identity import SourceRef
from readeverything.domain.rendition import Budget, Rendered, Rendition
from readeverything.ports.artifacts import ArtifactStore
from readeverything.ports.detection import MimeDetector
from readeverything.ports.handler import MediaHandler
from readeverything.ports.source import FileSource
from readeverything.registry.registry import MimeTypeRegistry

_HEAD_BYTES = 4096


class Perception:
    """Everything an agent needs to see a filesystem."""

    def __init__(
        self,
        *,
        source: FileSource,
        detector: MimeDetector,
        hasher: ContentHasher,
        registry: MimeTypeRegistry,
        artifacts: ArtifactStore,
    ) -> None:
        self._source = source
        self._detector = detector
        self._hasher = hasher
        self._registry = registry
        self._artifacts = artifacts

    async def _ref(self, uri: str) -> SourceRef:
        head = await self._source.read_range(uri, 0, _HEAD_BYTES)
        return SourceRef(
            uri=uri,
            mime=await self._detector.detect(uri, head),
            content_hash=await self._hasher.hash(uri),
            size_bytes=await self._source.size(uri),
        )

    async def _resolve(self, uri: str) -> tuple[SourceRef, MediaHandler]:
        ref = await self._ref(uri)
        return ref, self._registry.resolve(ref.mime)

    def _affordance(self, handler: MediaHandler, name: str) -> Affordance:
        available = self._registry.available_affordances(handler)
        for affordance in available:
            if affordance.name == name:
                return affordance
        raise UnknownAffordanceError(name, (a.name for a in available))

    async def inspect(self, uri: str) -> Card:
        """The cheap card for `uri`, with affordances filtered to what works here."""
        ref, handler = await self._resolve(uri)
        card = await handler.describe(ref)
        return Card(
            ref=card.ref,
            kind=card.kind,
            facts=card.facts,
            outline=card.outline,
            excerpt=card.excerpt,
            affordances=self._registry.available_affordances(handler),
        )

    async def invoke(self, uri: str, name: str, params: Mapping[str, Any]) -> Rendition:
        """Invoke a named affordance. Raises if it is not available here."""
        ref, handler = await self._resolve(uri)
        affordance = self._affordance(handler, name)
        return await handler.invoke(ref, name, affordance.params.model_validate(dict(params)))

    async def represent(self, uri: str, budget: Budget) -> Rendered:
        """Flatten `uri` for indexing: text plus locator map plus barriers."""
        ref, handler = await self._resolve(uri)
        return await handler.represent(ref, budget)

    async def list(self, uri: str) -> Sequence[str]:
        """Every source under `uri`."""
        return await self._source.walk(uri)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/pipeline -v && uv run mypy`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/pipeline tests/unit/pipeline
git commit -m "feat(pipeline): add the Perception facade"
```

---

### Task 15: Agent — the tool pack

**Files:**
- Create: `src/readeverything/agent/results.py`, `src/readeverything/agent/tools.py`
- Test: `tests/unit/agent/test_results.py`, `tests/unit/agent/test_tools.py`

**Interfaces:**
- Consumes: `Perception`, `Card`, `Rendition`
- Produces: `ToolResult`, `never_raises` decorator, `build_tools(perception) -> list[BaseTool]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/agent/test_results.py
from readeverything.agent.results import ToolResult, never_raises
from readeverything.domain.errors import SourceUnreadableError


async def test_a_success_carries_its_value() -> None:
    @never_raises
    async def fine() -> str:
        return "ok"

    result = await fine()
    assert result.ok
    assert result.value == "ok"
    assert result.error is None


async def test_a_domain_error_becomes_a_structured_failure() -> None:
    """A traceback reaching a model is a wasted and unrecoverable turn."""

    @never_raises
    async def bad() -> str:
        raise SourceUnreadableError("no such file: /nope")

    result = await bad()
    assert not result.ok
    assert result.error is not None
    assert "no such file" in result.error
    assert result.error_type == "SourceUnreadableError"


async def test_an_unexpected_error_is_also_caught() -> None:
    """Not just our exceptions: an adapter bug must not reach the model either."""

    @never_raises
    async def worse() -> str:
        raise ZeroDivisionError("oops")

    result = await worse()
    assert not result.ok
    assert result.error_type == "ZeroDivisionError"


def test_a_result_renders_compactly_for_a_model() -> None:
    assert "ok" in ToolResult(ok=True, value="ok", error=None, error_type=None).render()
    rendered = ToolResult(ok=False, value=None, error="boom", error_type="X").render()
    assert "ERROR" in rendered and "boom" in rendered
```

```python
# tests/unit/agent/test_tools.py
from pathlib import Path

import pytest

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.detection import PuremagicDetector
from readeverything.adapters.hashing import ContentHasher
from readeverything.adapters.local_source import LocalFileSource
from readeverything.agent.tools import build_tools
from readeverything.domain.capability import CapabilitySet
from readeverything.handlers.binary import BinaryHandler
from readeverything.handlers.text import TextHandler
from readeverything.pipeline.perception import Perception
from readeverything.registry.registry import MimeTypeRegistry


@pytest.fixture
def perception(tmp_path: Path) -> Perception:
    (tmp_path / "notes.txt").write_bytes(b"alpha\nbeta\n")
    source = LocalFileSource(root=tmp_path)
    return Perception(
        source=source,
        detector=PuremagicDetector(),
        hasher=ContentHasher(source=source),
        registry=MimeTypeRegistry(
            handlers=(TextHandler(source=source), BinaryHandler(source=source)),
            capabilities=CapabilitySet.empty(),
        ),
        artifacts=InMemoryArtifactStore(),
    )


def test_the_pack_offers_the_three_core_tools(perception: Perception) -> None:
    names = {tool.name for tool in build_tools(perception)}
    assert {"inspect_path", "list_paths", "invoke_affordance"} <= names


def test_every_tool_has_a_description(perception: Perception) -> None:
    """The description is the model's only guidance; a blank one blinds it."""
    for tool in build_tools(perception):
        assert tool.description.strip()


async def test_inspect_path_returns_a_rendered_card(perception: Perception) -> None:
    tool = next(t for t in build_tools(perception) if t.name == "inspect_path")
    output = await tool.ainvoke({"uri": "notes.txt"})
    assert "text/plain" in output
    assert "read_range" in output


async def test_a_missing_file_returns_an_error_string_not_an_exception(
    perception: Perception,
) -> None:
    tool = next(t for t in build_tools(perception) if t.name == "inspect_path")
    output = await tool.ainvoke({"uri": "absent.txt"})
    assert "ERROR" in output


async def test_invoke_affordance_round_trips(perception: Perception) -> None:
    tool = next(t for t in build_tools(perception) if t.name == "invoke_affordance")
    output = await tool.ainvoke(
        {"uri": "notes.txt", "affordance": "read_range", "params": {"start": 0, "end": 5}}
    )
    assert "alpha" in output


async def test_invoking_an_unavailable_affordance_returns_an_error(
    perception: Perception,
) -> None:
    tool = next(t for t in build_tools(perception) if t.name == "invoke_affordance")
    output = await tool.ainvoke({"uri": "notes.txt", "affordance": "hexdump", "params": {}})
    assert "ERROR" in output
    assert "read_range" in output  # the error names what IS available
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/agent -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementations**

```python
# src/readeverything/agent/results.py
"""Turning exceptions into results, exactly once, at the model boundary.

Ports and handlers raise; the tool pack returns. The split is by audience: a
raised exception is the right signal for a caller that can branch on it, and
the wrong one for a model, which sees a traceback, cannot act on it, and burns
a turn discovering that.

`never_raises` catches `BaseException` subclasses via `Exception` deliberately
broadly: an adapter bug is exactly as unhelpful to a model as an expected
failure, and letting one through would make the guarantee conditional on our
own code being correct.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    value: Any
    error: str | None
    error_type: str | None

    def render(self) -> str:
        """A compact string for a model to read."""
        if self.ok:
            return str(self.value)
        return f"ERROR ({self.error_type}): {self.error}"


def never_raises[**P](
    fn: Callable[P, Awaitable[Any]],
) -> Callable[P, Awaitable[ToolResult]]:
    """Wrap an async callable so it returns a `ToolResult` instead of raising."""

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> ToolResult:
        try:
            return ToolResult(ok=True, value=await fn(*args, **kwargs), error=None, error_type=None)
        # Catching broadly is the whole point of this decorator: an adapter bug
        # is exactly as unhelpful to a model as an expected failure. Note this
        # deliberately does not catch BaseException, so CancelledError still
        # propagates and task cancellation keeps working.
        except Exception as exc:
            return ToolResult(
                ok=False, value=None, error=str(exc), error_type=type(exc).__name__
            )

    return wrapper
```

```python
# src/readeverything/agent/tools.py
"""The framework-agnostic tool pack.

Three tools rather than one per affordance. Affordances are per-mimetype and
therefore per-file, so a tool per affordance would mean a tool list that
changes with whatever the agent last looked at — which no agent framework
supports and no model handles well. Instead `inspect_path` *tells* the model
which affordances this file has, and `invoke_affordance` runs one by name. The
card is the discovery mechanism.

This module and `adapters/langchain_*.py` are the only places `langchain` may
be imported.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from readeverything.agent.results import never_raises
from readeverything.domain.card import Card
from readeverything.domain.rendition import (
    ImageContent,
    Rendition,
    StructuredContent,
    TextContent,
)
from readeverything.pipeline.perception import Perception


class InspectParams(BaseModel):
    uri: str = Field(description="Path to inspect, relative to the configured root.")


class ListParams(BaseModel):
    uri: str = Field(default=".", description="Directory to list, relative to the root.")


class InvokeParams(BaseModel):
    uri: str = Field(description="Path the affordance applies to.")
    affordance: str = Field(description="Affordance name, from the card's `affordances`.")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Arguments, matching that affordance's schema."
    )


def _render_card(card: Card) -> str:
    return json.dumps(
        {
            "uri": card.ref.uri,
            "mime": str(card.ref.mime),
            "kind": str(card.kind),
            "size_bytes": card.ref.size_bytes,
            "facts": dict(card.facts),
            "outline": [
                {"label": segment.label, "locator": repr(segment.locator)}
                for segment in card.outline
            ],
            "excerpt": card.excerpt,
            "affordances": [
                {
                    "name": affordance.name,
                    "description": affordance.description,
                    "params": affordance.params.model_json_schema(),
                }
                for affordance in card.affordances
            ],
        },
        indent=2,
    )


def _render_rendition(rendition: Rendition) -> str:
    match rendition.content:
        case TextContent(text=text):
            body = text
        case StructuredContent(rows=rows):
            body = json.dumps(list(rows), indent=2)
        case ImageContent(data=data, mime=mime):
            body = f"[{mime} image, {len(data)} bytes — pass to a vision tool to read it]"
    marker = " (degraded)" if rendition.degraded else ""
    return f"located at {rendition.locator!r}{marker}:\n{body}"


def build_tools(perception: Perception) -> list[BaseTool]:
    """The tool pack over one `Perception`."""

    @never_raises
    async def inspect_path(uri: str) -> str:
        return _render_card(await perception.inspect(uri))

    @never_raises
    async def list_paths(uri: str = ".") -> str:
        return "\n".join(await perception.list(uri))

    @never_raises
    async def invoke_affordance(
        uri: str, affordance: str, params: Mapping[str, Any] | None = None
    ) -> str:
        return _render_rendition(await perception.invoke(uri, affordance, params or {}))

    async def _inspect(uri: str) -> str:
        return (await inspect_path(uri)).render()

    async def _list(uri: str = ".") -> str:
        return (await list_paths(uri)).render()

    async def _invoke(
        uri: str, affordance: str, params: dict[str, Any] | None = None
    ) -> str:
        return (await invoke_affordance(uri, affordance, params)).render()

    return [
        StructuredTool.from_function(
            coroutine=_inspect,
            name="inspect_path",
            description=(
                "Inspect any file and get a compact description of it: its type, size, "
                "key metadata, an outline, a short excerpt, and the list of affordances "
                "available for going deeper. Always call this before invoke_affordance. "
                "Cheap: it never runs a model over the whole file."
            ),
            args_schema=InspectParams,
        ),
        StructuredTool.from_function(
            coroutine=_list,
            name="list_paths",
            description="List every file under a directory, recursively.",
            args_schema=ListParams,
        ),
        StructuredTool.from_function(
            coroutine=_invoke,
            name="invoke_affordance",
            description=(
                "Run one of the affordances named in a file's card — for example reading "
                "a character range, a page, a transcript segment, or a video frame. "
                "The affordance name and its parameter schema come from inspect_path. "
                "Results are always accompanied by a locator saying where in the file "
                "they came from."
            ),
            args_schema=InvokeParams,
        ),
    ]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/agent -v && uv run mypy`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/readeverything/agent tests/unit/agent
git commit -m "feat(agent): add the framework-agnostic tool pack"
```

---

### Task 16: Enforcement tests and the public surface

**Files:**
- Create: `src/readeverything/__init__.py` (rewrite), `tests/unit/test_reads_no_environment.py`, `tests/unit/test_dependencies_stay_confined.py`, `tests/unit/test_public_surface.py`
- Modify: `pyproject.toml` (add coverage config)

**Interfaces:**
- Consumes: everything
- Produces: the lazy public API

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_reads_no_environment.py
"""The library reads no environment.

Configuration is constructor arguments, so a caller can run two differently
configured instances in one process and so tests cannot be affected by the
machine they run on. A single `os.getenv` would silently break both.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "readeverything"

FORBIDDEN_ATTRS = {("os", "environ"), ("os", "getenv")}
FORBIDDEN_MODULES = {"dotenv"}


def _offences(tree: ast.AST) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if (node.value.id, node.attr) in FORBIDDEN_ATTRS:
                found.append(f"{node.value.id}.{node.attr}")
        if isinstance(node, ast.Import):
            found.extend(a.name for a in node.names if a.name.split(".")[0] in FORBIDDEN_MODULES)
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in FORBIDDEN_MODULES:
                found.append(node.module)
    return found


def test_no_module_reads_the_environment() -> None:
    offenders: dict[str, list[str]] = {}
    for path in SRC.rglob("*.py"):
        found = _offences(ast.parse(path.read_text()))
        if found:
            offenders[str(path.relative_to(SRC))] = found
    assert not offenders, f"environment reads found: {offenders}"
```

```python
# tests/unit/test_dependencies_stay_confined.py
"""Each third-party client lives in exactly one place.

import-linter cannot see third-party imports, so it cannot enforce this — and
this is the rule that actually stops a langchain or ffmpeg leak into the
domain. The table is the spec's confinement table, and it must fail when a
module drifts out of its home.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "readeverything"

#: top-level third-party module -> the only files that may import it
CONFINED: dict[str, set[str]] = {
    "langchain_core": {"agent/tools.py"},
    "puremagic": {"adapters/detection.py"},
    "charset_normalizer": {"handlers/text.py"},
    "subprocess": set(),
}


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_each_dependency_is_confined_to_its_declared_home() -> None:
    violations: list[str] = []
    for path in SRC.rglob("*.py"):
        relative = str(path.relative_to(SRC))
        for root in _imported_roots(ast.parse(path.read_text())):
            if root in CONFINED and relative not in CONFINED[root]:
                violations.append(f"{relative} imports {root}")
    assert not violations, f"confinement violated: {violations}"


def test_the_confinement_table_is_live() -> None:
    """An entry naming a file that no longer imports it is stale and must fail."""
    stale: list[str] = []
    for root, homes in CONFINED.items():
        for home in homes:
            path = SRC / home
            if not path.exists() or root not in _imported_roots(ast.parse(path.read_text())):
                stale.append(f"{home} no longer imports {root}")
    assert not stale, f"stale confinement entries: {stale}"
```

```python
# tests/unit/test_public_surface.py
"""The public API is closed and loads no driver."""

import sys
import subprocess

import readeverything


def test_everything_in_all_is_reachable() -> None:
    for name in readeverything.__all__:
        assert getattr(readeverything, name) is not None


def test_all_is_sorted_and_unique() -> None:
    assert readeverything.__all__ == sorted(set(readeverything.__all__))


def test_importing_the_package_loads_no_optional_driver() -> None:
    """`import readeverything` must not pull in langchain."""
    code = "import readeverything, sys; print('langchain_core' in sys.modules)"
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert output.stdout.strip() == "False"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_public_surface.py -v`
Expected: FAIL — `readeverything.__all__` does not exist yet

- [ ] **Step 3: Write the lazy public API**

```python
# src/readeverything/__init__.py
"""readeverything — give an agent eyes into a filesystem.

Imports are lazy (PEP 562). `import readeverything` loads no adapter and no
optional driver, so the base install stays light and a caller who wants only
the domain types never pays for langchain.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from readeverything.adapters.artifact_store import (
        FilesystemArtifactStore,
        InMemoryArtifactStore,
    )
    from readeverything.adapters.detection import PuremagicDetector
    from readeverything.adapters.hashing import ContentHasher, StatMemo
    from readeverything.adapters.local_source import LocalFileSource
    from readeverything.domain.affordance import Affordance, DetailLevel
    from readeverything.domain.capability import Capability, CapabilitySet
    from readeverything.domain.card import Card, Segment
    from readeverything.domain.errors import (
        CapabilityUnavailableError,
        DomainError,
        InfrastructureError,
        ReadEverythingError,
        SourceUnreadableError,
        UnknownAffordanceError,
    )
    from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
    from readeverything.domain.locator_map import LocatorMap, LocatorSegment
    from readeverything.domain.locators import BBox, ByteRange, CharSpan, PageRef, TimeSpan
    from readeverything.domain.rendition import (
        Budget,
        Degradation,
        ImageContent,
        Rendered,
        Rendition,
        SpeakerId,
        StructuredContent,
        TextContent,
        TranscriptCue,
    )
    from readeverything.handlers.binary import BinaryHandler
    from readeverything.handlers.text import TextHandler
    from readeverything.pipeline.perception import Perception
    from readeverything.registry.registry import MimeTypeRegistry, NoHandlerError

_LAZY: dict[str, str] = {
    "Affordance": "readeverything.domain.affordance",
    "BBox": "readeverything.domain.locators",
    "BinaryHandler": "readeverything.handlers.binary",
    "Budget": "readeverything.domain.rendition",
    "ByteRange": "readeverything.domain.locators",
    "Capability": "readeverything.domain.capability",
    "CapabilitySet": "readeverything.domain.capability",
    "CapabilityUnavailableError": "readeverything.domain.errors",
    "Card": "readeverything.domain.card",
    "CharSpan": "readeverything.domain.locators",
    "ContentHash": "readeverything.domain.identity",
    "ContentHasher": "readeverything.adapters.hashing",
    "Degradation": "readeverything.domain.rendition",
    "DetailLevel": "readeverything.domain.affordance",
    "DomainError": "readeverything.domain.errors",
    "FilesystemArtifactStore": "readeverything.adapters.artifact_store",
    "ImageContent": "readeverything.domain.rendition",
    "InMemoryArtifactStore": "readeverything.adapters.artifact_store",
    "InfrastructureError": "readeverything.domain.errors",
    "LocalFileSource": "readeverything.adapters.local_source",
    "LocatorMap": "readeverything.domain.locator_map",
    "LocatorSegment": "readeverything.domain.locator_map",
    "MediaKind": "readeverything.domain.identity",
    "MimeType": "readeverything.domain.identity",
    "MimeTypeRegistry": "readeverything.registry.registry",
    "NoHandlerError": "readeverything.registry.registry",
    "PageRef": "readeverything.domain.locators",
    "Perception": "readeverything.pipeline.perception",
    "PuremagicDetector": "readeverything.adapters.detection",
    "ReadEverythingError": "readeverything.domain.errors",
    "Rendered": "readeverything.domain.rendition",
    "Rendition": "readeverything.domain.rendition",
    "Segment": "readeverything.domain.card",
    "SourceRef": "readeverything.domain.identity",
    "SourceUnreadableError": "readeverything.domain.errors",
    "SpeakerId": "readeverything.domain.rendition",
    "StatMemo": "readeverything.adapters.hashing",
    "StructuredContent": "readeverything.domain.rendition",
    "TextContent": "readeverything.domain.rendition",
    "TextHandler": "readeverything.handlers.text",
    "TimeSpan": "readeverything.domain.locators",
    "TranscriptCue": "readeverything.domain.rendition",
    "UnknownAffordanceError": "readeverything.domain.errors",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_LAZY[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return __all__
```

- [ ] **Step 4: Add coverage config to `pyproject.toml`**

```toml
[tool.coverage.run]
source = ["src/readeverything"]
branch = true

[tool.coverage.report]
fail_under = 90
show_missing = true
```

- [ ] **Step 5: Run the full gate**

Run: `make check`
Expected: lint, types, arch, sec, test all PASS.
If `test_dependencies_stay_confined` fails, the fix is to move the import, not to widen the table.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add lazy public API and architectural enforcement tests"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Covered by |
| --- | --- |
| §3 layering, confinement, no-env | Tasks 1, 16 |
| §4.1 locators | Task 3 |
| §4.2 source identity | Task 2 |
| §4.3 affordance | Task 6 |
| §4.4 card | Task 6 |
| §4.5 transcripts and speakers | Task 6 (`TranscriptCue`); consumers in Plan 2 |
| §4.6 renditions and rendered | Task 6 |
| §4.7 LocatorMap | Task 4 |
| §4.8 capability | Task 5 |
| §5 ports | Task 7 (`MediaProbe`, `FrameExtractor`, `AudioExtractor`, `Transcriber`, `Diarizer`, `VisionModel`, `TextModel`, `BinaryProbe` deferred to Plan 2 with their handlers — a port with no implementer and no consumer would be untested surface) |
| §6 registry and dispatch | Task 8 |
| §7 text, binary rows | Task 13; other rows are Plan 2 |
| §8 caching | Task 11 |
| §10 tool pack | Task 15; deepagents decorator is Plan 3 |
| §11 composition root | Plan 3 |
| §12 errors, budgets | Tasks 5, 6, 13, 15 |
| §13 testing | Tasks 12, 16; generated media fixtures arrive with the handlers that need them in Plan 2 |
| §14 packaging | Tasks 1, 16 |

**Gap accepted deliberately:** §12's per-capability concurrency semaphore has no task here. Nothing in this plan performs concurrent expensive work — the text and binary handlers are pure I/O — so a limiter would be untested scaffolding. It belongs with the first handler that calls a model, in Plan 2.

**Type consistency:** `handler_id`/`handler_version` are spelled identically in `ports/handler.py`, `cache_key.py`, and both handlers. `Affordance.params` is a `type[BaseModel]` everywhere and is instantiated via `model_validate` only in `Perception.invoke`. `LocatorMap.build` is the only construction path used in handlers and tests.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-14-perception-core-foundation.md`.
