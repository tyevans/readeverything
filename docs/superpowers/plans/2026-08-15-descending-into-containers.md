# Descending Into Containers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make an archive member a first-class source, so that `docs.zip!nested.tar.gz!report.pdf` is inspected, paged, OCR'd and cited exactly like a loose file on disk — without one line of any handler changing.

**Architecture:** A `!`-separated uri grammar in `domain/container_uri.py`; an `ArchiveOpener` port in `ports/containers.py` with stdlib zip and tar adapters; and `NestedSource`, a `FileSource` **decorator** in `adapters/nested_source.py` that delegates single-segment uris verbatim to the wrapped source and resolves multi-segment uris left to right. Because `walk` returns members inline and the pipeline dispatches on detected mimetype, every existing handler descends for free.

**Tech Stack:** Python 3.12+, stdlib `zipfile`/`tarfile` only (**no new third-party dependency**), pydantic v2, pytest + pytest-asyncio + hypothesis, `uv run --all-extras` for everything.

**Spec:** `docs/superpowers/specs/2026-08-15-readeverything-descending-into-containers-design.md`

## Global Constraints

- **No new third-party dependency.** `zipfile` and `tarfile` are stdlib. Do not add anything to `pyproject.toml`'s dependency lists.
- **Never read an environment variable.** Every input is an explicit constructor or function argument. `tests/unit/test_reads_no_environment.py` enforces this.
- **Layered import contract** (`pyproject.toml` `[tool.importlinter]`, `exhaustive = true`): `domain` imports nothing internal; `ports` imports only `domain`; `adapters` imports only `ports` + `domain`; `handlers` sits below `registry` and imports only `adapters`/`ports`/`domain`; `composition` is top. `readeverything.testing` may see only `ports` and `domain`.
- **mypy is `strict`** and the coverage gate is **92%**. Test the error paths, not just the happy ones.
- **Handlers never raise about their input.** A handler returns a degraded `Rendition`. The *source layer* is the opposite: `NestedSource` raises `SourceUnreadableError` subclasses, exactly as `LocalFileSource` does.
- **Limits raise, never truncate.** Every `ContainerLimits` breach raises `ContainerLimitExceededError`. Handing a handler half a PDF would make it report on a fragment as though it were whole.
- **Docstrings explain *why*, not *what*.** Match the density and voice of `handlers/pdf.py` and `adapters/local_source.py`: every non-obvious decision carries the reason it was made, and every guard carries the failure it prevents.
- **Run only the tests this plan creates or names.** Do not run the full suite and do not run `make check`.
  `uv run --all-extras pytest <named files> -x -q`
  Lint with `uv run --all-extras ruff check <files>` and type-check with `uv run --all-extras mypy <files>`.
- **Another agent is editing `composition.py` and `adapters/detection.py` concurrently (Spec 9).** Every edit to those two files must be strictly *additive* — new keyword arguments with defaults, new list entries — so the merge is trivial. Do not reformat, reorder, or refactor them.
- **Do not create `handlers/office_*.py`.** That is Spec 9's lane.
- **Do not modify `pipeline/resolution.py`.** Spec §6 is explicit: a member cannot be stat'd, so `stat_key` already returns `None` and members are already never memoized. Its existing rule was written for this case.

## File Structure

| File | Responsibility |
| --- | --- |
| `src/readeverything/domain/container_uri.py` | The `!` grammar: `split_uri`, `join_uri`, `container_of`. Pure functions, no I/O. |
| `src/readeverything/domain/errors.py` (modify) | Adds `ContainerLimitExceededError(SourceUnreadableError)`. |
| `src/readeverything/ports/containers.py` | `ArchiveEntry`, `ArchiveOpener`, `ContainerLimits`, and the mimetype constants naming what is and is not a folder. |
| `src/readeverything/adapters/zip_archive.py` | `ZipArchiveOpener` over stdlib `zipfile`. |
| `src/readeverything/adapters/tar_archive.py` | `TarArchiveOpener` over stdlib `tarfile`, including the solid-container temp cache. |
| `src/readeverything/adapters/nested_source.py` | `CompositeOpener` and `NestedSource` — resolution, limits enforcement, guards, and inline `walk`. |
| `src/readeverything/handlers/archive.py` | `ArchiveHandler` — the container's own card and paged `list_entries`. |
| `src/readeverything/composition.py` (modify) | Two new keyword arguments; wires `NestedSource` and `ArchiveHandler`. |
| `src/readeverything/__init__.py` (modify) | Lazy exports for the new public names. |
| `README.md` (modify) | One section on descending into containers. |

---

### Task 1: The uri grammar

The `!` grammar is normative and frozen (spec §2) because Spec 9's fixtures reference it. It lives in `domain` because both an adapter and a test fixture parse these strings, and `domain` is the only layer both may import.

The escape is the part most likely to rot: a literal `!` in a member name is written `!!`, so splitting is a **scan, not `str.split("!")`**. This is rare enough in practice that getting it wrong would be invisible for a year, which is exactly why it is specified and property-tested now.

**Files:**
- Create: `src/readeverything/domain/container_uri.py`
- Modify: `src/readeverything/domain/errors.py`
- Test: `tests/unit/domain/test_container_uri.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `def split_uri(uri: str) -> tuple[str, ...]`
  - `def join_uri(segments: Sequence[str]) -> str`
  - `def container_of(uri: str) -> str | None`
  - `SEPARATOR: str = "!"`
  - `class ContainerLimitExceededError(SourceUnreadableError)` in `domain/errors.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/domain/test_container_uri.py`:

```python
"""The `!` grammar, including the escape that would otherwise rot unnoticed."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from readeverything.domain.container_uri import container_of, join_uri, split_uri


def test_a_plain_path_is_one_segment() -> None:
    """Nothing changes for a uri with no `!`. This is the compatibility story."""
    assert split_uri("docs/report.pdf") == ("docs/report.pdf",)


def test_splits_on_the_separator() -> None:
    assert split_uri("docs.zip!nested.tar.gz!notes.txt") == (
        "docs.zip",
        "nested.tar.gz",
        "notes.txt",
    )


def test_a_doubled_separator_is_a_literal_one() -> None:
    assert split_uri("a.zip!od!!d.txt") == ("a.zip", "od!d.txt")


def test_join_escapes_a_literal_separator() -> None:
    assert join_uri(("a.zip", "od!d.txt")) == "a.zip!od!!d.txt"


def test_join_of_one_segment_is_that_segment() -> None:
    assert join_uri(("docs/report.pdf",)) == "docs/report.pdf"


def test_container_of_a_plain_path_is_none() -> None:
    assert container_of("docs/report.pdf") is None


def test_container_of_a_member_is_everything_to_its_left() -> None:
    assert container_of("docs.zip!nested.tar.gz!notes.txt") == "docs.zip!nested.tar.gz"


def test_container_of_preserves_the_escape() -> None:
    assert container_of("a.zip!od!!d.txt") == "a.zip"


def test_an_empty_segment_is_refused() -> None:
    with pytest.raises(ValueError, match="empty segment"):
        split_uri("a.zip!")


def test_an_empty_uri_is_refused() -> None:
    with pytest.raises(ValueError, match="empty segment"):
        split_uri("")


def test_joining_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one segment"):
        join_uri(())


def test_joining_an_empty_segment_is_refused() -> None:
    with pytest.raises(ValueError, match="empty segment"):
        join_uri(("a.zip", ""))


@given(st.lists(st.text(min_size=1).filter(lambda s: s), min_size=1, max_size=4))
def test_round_trips_through_join_and_split(segments: list[str]) -> None:
    """The escape is the whole reason this is a property test.

    Hypothesis generates `!` freely, which is the only way this catches a
    naive `str.split("!")` regression years from now.
    """
    assert split_uri(join_uri(segments)) == tuple(segments)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --all-extras pytest tests/unit/domain/test_container_uri.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'readeverything.domain.container_uri'`

- [ ] **Step 3: Write the implementation**

Create `src/readeverything/domain/container_uri.py`:

```python
"""Addressing something inside something else.

`ports/source.py` and `domain/identity.py` have both described this grammar in
prose since Spec 1 -- an archive member addressed as `"/a.zip!inner.txt"` -- 
without anything implementing it. This is that implementation, and it is
normative: Spec 9's fixtures reference these strings.

`!` was chosen because Java's `jar:` URLs have used it for two decades, it is
already written into two docstrings in this repository, and it is legal in
POSIX filenames but vanishingly rare. Rare is not impossible, which is why the
escape below exists rather than a claim that collision cannot happen.

Pure functions and no I/O, because this sits in `domain`: an adapter parses
these strings and so do test fixtures, and `domain` is the only layer both may
import.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Between a container and a member. A literal one inside a segment is doubled.
SEPARATOR = "!"


def split_uri(uri: str) -> tuple[str, ...]:
    """`"a!b!c"` -> `("a", "b", "c")`, honouring the `!!` escape.

    A SCAN, not `str.split(SEPARATOR)`. The difference only shows up on a
    member whose name contains a literal `!`, which is rare enough that a
    `str.split` regression here would be invisible for a year -- so the
    round-trip property test in `tests/unit/domain/test_container_uri.py` is
    the thing actually holding this correct, not review.
    """
    segments: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(uri):
        character = uri[index]
        if character == SEPARATOR:
            if uri[index + 1 : index + 2] == SEPARATOR:
                current.append(SEPARATOR)
                index += 2
                continue
            segments.append("".join(current))
            current = []
            index += 1
            continue
        current.append(character)
        index += 1
    segments.append("".join(current))
    if any(not segment for segment in segments):
        # An empty segment names nothing. Allowing it would let `"a.zip!"`
        # resolve to the archive itself by a second spelling, and two spellings
        # for one source means two provenance stories for one citation.
        raise ValueError(f"empty segment in container uri {uri!r}")
    return tuple(segments)


def join_uri(segments: Sequence[str]) -> str:
    """The inverse of `split_uri`, escaping any literal separator."""
    if not segments:
        raise ValueError("a container uri needs at least one segment")
    if any(not segment for segment in segments):
        raise ValueError(f"empty segment in {list(segments)!r}")
    return SEPARATOR.join(segment.replace(SEPARATOR, SEPARATOR * 2) for segment in segments)


def container_of(uri: str) -> str | None:
    """The uri of what holds `uri`, or None when nothing does.

    Re-joins rather than slicing the original string, so the escape survives:
    slicing at the last raw `!` would cut `"a.zip!od!!d.txt"` in the wrong
    place.
    """
    segments = split_uri(uri)
    if len(segments) == 1:
        return None
    return join_uri(segments[:-1])
```

Append to `src/readeverything/domain/errors.py`, after `SourceUnreadableError`:

```python
class ContainerLimitExceededError(SourceUnreadableError):
    """Descending into a container would have cost more than its limit allows.

    A subclass of `SourceUnreadableError` rather than a sibling, so every
    `except SourceUnreadableError` already written keeps working -- a caller
    that handled an unreadable file handles an unreasonable one the same way.

    Raised rather than returning truncated bytes. Truncation would hand a
    handler half a PDF, which it would then report on as though it were whole.
    """
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --all-extras pytest tests/unit/domain/test_container_uri.py -x -q`
Expected: PASS

- [ ] **Step 5: Lint and type-check**

Run:
```bash
uv run --all-extras ruff check src/readeverything/domain/container_uri.py src/readeverything/domain/errors.py tests/unit/domain/test_container_uri.py
uv run --all-extras ruff format src/readeverything/domain/container_uri.py src/readeverything/domain/errors.py tests/unit/domain/test_container_uri.py
uv run --all-extras mypy src/readeverything/domain/container_uri.py src/readeverything/domain/errors.py
```
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/readeverything/domain/container_uri.py src/readeverything/domain/errors.py tests/unit/domain/test_container_uri.py
git commit -m "feat(domain): the container uri grammar, escape included"
```

---

### Task 2: The `ArchiveOpener` port and `ContainerLimits`

Handlers do not shell out and adapters own format knowledge, so container formats sit behind a port (spec §4). `ContainerLimits` lives here too because it is part of the contract every opener's caller enforces, and the mimetype constants live here because both an adapter (`nested_source`) and a handler (`archive`) need them, and `ports` is the lowest layer both may import.

**Note on `open_member`:** the spec writes `async def open_member(...) -> AsyncIterator[bytes]`. This plan declares it `def open_member(...) -> AsyncIterator[bytes]`, matching `SourceReader.stream` in `ports/source.py` exactly, so an implementation can be a plain `async def` generator. This is a typing shape, not a behavior change.

**Files:**
- Create: `src/readeverything/ports/containers.py`
- Test: `tests/unit/ports/test_containers.py`

**Interfaces:**
- Consumes: `readeverything.domain.identity.MimeType`.
- Produces:
  - `@dataclass(frozen=True, slots=True) class ArchiveEntry` with fields `path: str`, `size_bytes: int`, `compressed_bytes: int`, `is_dir: bool`, `is_symlink: bool`, `modified_epoch_s: float | None`, `byte_offset: int | None`
  - `@runtime_checkable class ArchiveOpener(Protocol)` with `claims(self, mime: MimeType) -> bool`, `async def entries(self, path: str) -> Sequence[ArchiveEntry]`, `def open_member(self, path: str, member: str) -> AsyncIterator[bytes]`
  - `@dataclass(frozen=True, slots=True) class ContainerLimits` with `max_depth: int = 3`, `max_member_bytes: int = 1 << 30`, `max_total_bytes: int = 4 << 30`, `max_members: int = 10_000`, `max_expansion_ratio: float = 200.0`, `max_materialised_bytes: int = 8 << 30`, `walk_members: bool = True`
  - `ARCHIVE_MIMES: frozenset[str]`
  - `NOT_A_FOLDER_MIMES: frozenset[str]`
  - `NOT_A_FOLDER_SUFFIXES: frozenset[str]`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/ports/test_containers.py`:

```python
"""The container port's own invariants."""

import pytest

from readeverything.ports.containers import (
    ARCHIVE_MIMES,
    NOT_A_FOLDER_MIMES,
    NOT_A_FOLDER_SUFFIXES,
    ArchiveEntry,
    ContainerLimits,
)


def test_default_limits_match_the_spec() -> None:
    limits = ContainerLimits()
    assert limits.max_depth == 3
    assert limits.max_member_bytes == 1 << 30
    assert limits.max_total_bytes == 4 << 30
    assert limits.max_members == 10_000
    assert limits.max_expansion_ratio == 200.0
    assert limits.max_materialised_bytes == 8 << 30
    assert limits.walk_members is True


def test_limits_are_frozen() -> None:
    """Configuration a caller passed must not drift under them mid-walk."""
    limits = ContainerLimits()
    with pytest.raises(AttributeError):
        limits.max_depth = 9  # type: ignore[misc]


def test_an_entry_rejects_a_negative_size() -> None:
    with pytest.raises(ValueError, match="size_bytes"):
        ArchiveEntry(
            path="a.txt",
            size_bytes=-1,
            compressed_bytes=0,
            is_dir=False,
            is_symlink=False,
            modified_epoch_s=None,
            byte_offset=None,
        )


def test_an_entry_rejects_an_empty_path() -> None:
    with pytest.raises(ValueError, match="path"):
        ArchiveEntry(
            path="",
            size_bytes=0,
            compressed_bytes=0,
            is_dir=False,
            is_symlink=False,
            modified_epoch_s=None,
            byte_offset=None,
        )


def test_zip_based_documents_are_not_folders() -> None:
    """A .docx is a zip and is emphatically not a directory of XML parts."""
    docx = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert docx in NOT_A_FOLDER_MIMES
    assert "application/epub+zip" in NOT_A_FOLDER_MIMES
    assert ".docx" in NOT_A_FOLDER_SUFFIXES
    assert ".jar" in NOT_A_FOLDER_SUFFIXES


def test_plain_archives_are_claimed() -> None:
    assert "application/zip" in ARCHIVE_MIMES
    assert "application/x-tar" in ARCHIVE_MIMES
    assert "application/gzip" in ARCHIVE_MIMES


def test_the_two_sets_do_not_overlap() -> None:
    """An overlap would make "descend into this" ambiguous by construction."""
    assert not (ARCHIVE_MIMES & NOT_A_FOLDER_MIMES)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --all-extras pytest tests/unit/ports/test_containers.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'readeverything.ports.containers'`

- [ ] **Step 3: Write the implementation**

Create `src/readeverything/ports/containers.py`:

```python
"""Opening a container, and what it may cost.

A container format is adapter knowledge, exactly like a video codec: handlers
do not shell out and nothing above this layer learns what a central directory
is. Two adapters ship against this port, both stdlib -- which is why this
entire feature adds no dependency -- and a caller who wants `.7z` or `.rar`
supplies their own opener without this repository growing a dependency on
either.

`ContainerLimits` sits here rather than in `domain` because it describes what
an opener's CALLER will enforce while consuming this port, not a rule about
what a source is. Every field is an explicit constructor argument, per the
library's standing rule that nothing configures itself from the environment.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from readeverything.domain.identity import MimeType


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    """One member, as its container's own directory describes it.

    `compressed_bytes` is carried alongside `size_bytes` because the
    expansion-ratio guard needs both, and only the container knows the former.
    `byte_offset` is `None` for a solid container, which is the single fact
    that tells a caller whether a ranged read is genuinely ranged -- see the
    `NestedSource` docstring on seekable versus solid.
    """

    path: str
    size_bytes: int
    compressed_bytes: int
    is_dir: bool
    is_symlink: bool
    modified_epoch_s: float | None
    byte_offset: int | None

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("an archive entry's path must not be empty")
        if self.size_bytes < 0:
            raise ValueError(f"size_bytes must not be negative, got {self.size_bytes}")
        if self.compressed_bytes < 0:
            raise ValueError(f"compressed_bytes must not be negative, got {self.compressed_bytes}")


@runtime_checkable
class ArchiveOpener(Protocol):
    """Reads a container's directory, and one member's bytes at a time."""

    def claims(self, mime: MimeType) -> bool:
        """Whether this opener understands `mime`."""
        ...

    async def entries(self, path: str) -> Sequence[ArchiveEntry]:
        """Every member the container declares.

        A probe, not a decompression: reading a zip central directory or
        walking tar headers is a seek and a small read. Anything that costs
        more than that belongs behind `open_member`.
        """
        ...

    def open_member(self, path: str, member: str) -> AsyncIterator[bytes]:
        """The member's decompressed bytes, chunked.

        Declared non-`async` returning an `AsyncIterator`, matching
        `SourceReader.stream`, so an implementation is a plain async
        generator. Chunked rather than whole because the caller's expansion
        guard has to fire MID-STREAM: a zip bomb lies in its header, so
        checking a declared size after the fact is checking the bomb's own
        paperwork.
        """
        ...


@dataclass(frozen=True, slots=True)
class ContainerLimits:
    """What descending into a container is allowed to cost.

    Conservative by default, because the failure this bounds is a zip bomb
    filling a disk. `max_expansion_ratio` is the one that matters and the rest
    are belt: it is checked DURING decompression against bytes written so far,
    never afterwards against a declared size.

    `walk_members` exists because §3.1's inline listing is not free -- walking
    a directory now reads every archive's central directory, which on ten
    thousand zips is ten thousand extra opens. A caller who wants the old
    behavior turns it off rather than losing the feature entirely.
    """

    max_depth: int = 3
    max_member_bytes: int = 1 << 30
    max_total_bytes: int = 4 << 30
    max_members: int = 10_000
    max_expansion_ratio: float = 200.0
    max_materialised_bytes: int = 8 << 30
    walk_members: bool = True


#: Containers that ARE folders: descending into one yields sources a caller
#: wanted. Kept as strings rather than `MimeType` so a membership test costs a
#: `str(mime)` and no parsing.
ARCHIVE_MIMES: frozenset[str] = frozenset(
    {
        "application/zip",
        "application/x-zip-compressed",
        "application/x-tar",
        "application/x-gtar",
        "application/gzip",
        "application/x-gzip",
        "application/x-bzip2",
        "application/x-xz",
    }
)

#: Containers that are NOT folders, and the whole reason §3.1 needed a rule.
#:
#: A `.docx`, `.pptx`, `.xlsx`, `.odt`, `.epub` and `.jar` are all zip files.
#: Descending into one would list `report.docx!word/document.xml` as a source,
#: which is worse than useless: it buries the document itself under a dozen XML
#: parts. Once Spec 9's handlers claim these mimetypes they stop being folders
#: for the general reason -- no handler claims a plain zip, and these are
#: claimed -- and this explicit set is what keeps the behavior correct in the
#: interim rather than briefly wrong.
NOT_A_FOLDER_MIMES: frozenset[str] = frozenset(
    {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.oasis.opendocument.text",
        "application/vnd.oasis.opendocument.presentation",
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/epub+zip",
        "application/java-archive",
        "application/vnd.android.package-archive",
    }
)

#: The same rule, spelled as filenames.
#:
#: Not redundant with `NOT_A_FOLDER_MIMES`: detection is content-first, and the
#: bytes of a `.docx` ARE a zip -- puremagic can and does report
#: `application/zip` for one, which would sail straight past the mimetype set
#: and bury the document. Until Spec 9 §3 teaches the detector to report OOXML
#: and ODF as their own types, the filename is the only signal that survives,
#: and this is one of the narrow places where consulting it is correct rather
#: than lazy.
NOT_A_FOLDER_SUFFIXES: frozenset[str] = frozenset(
    {
        ".docx",
        ".docm",
        ".pptx",
        ".pptm",
        ".xlsx",
        ".xlsm",
        ".odt",
        ".odp",
        ".ods",
        ".epub",
        ".jar",
        ".war",
        ".apk",
        ".whl",
    }
)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --all-extras pytest tests/unit/ports/test_containers.py -x -q`
Expected: PASS

- [ ] **Step 5: Lint and type-check**

Run:
```bash
uv run --all-extras ruff check src/readeverything/ports/containers.py tests/unit/ports/test_containers.py
uv run --all-extras ruff format src/readeverything/ports/containers.py tests/unit/ports/test_containers.py
uv run --all-extras mypy src/readeverything/ports/containers.py
```
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/readeverything/ports/containers.py tests/unit/ports/test_containers.py
git commit -m "feat(ports): the ArchiveOpener port, entry record and container limits"
```

---

### Task 3: `ZipArchiveOpener`

stdlib `zipfile`, and the seekable half of §3.2. A zip's central directory gives every member an offset, so a ranged read is a genuine ranged read and nothing is materialised.

**Files:**
- Create: `src/readeverything/adapters/zip_archive.py`
- Test: `tests/unit/adapters/test_zip_archive.py`

**Interfaces:**
- Consumes: `ArchiveEntry`, `ArchiveOpener` from `ports.containers`; `MimeType` from `domain.identity`; `SourceUnreadableError` from `domain.errors`.
- Produces: `class ZipArchiveOpener` with `__init__(self) -> None`, `claims`, `entries`, `open_member` as declared by `ArchiveOpener`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/adapters/test_zip_archive.py`:

```python
"""The zip opener, against archives built in a tmpdir rather than committed."""

import zipfile
from pathlib import Path

import pytest

from readeverything.adapters.zip_archive import ZipArchiveOpener
from readeverything.domain.errors import SourceUnreadableError
from readeverything.domain.identity import MimeType


def _zip(tmp_path: Path, members: dict[str, bytes]) -> str:
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return str(path)


def test_claims_zip_and_not_tar() -> None:
    opener = ZipArchiveOpener()
    assert opener.claims(MimeType.parse("application/zip"))
    assert not opener.claims(MimeType.parse("application/x-tar"))


async def test_lists_entries_with_sizes(tmp_path: Path) -> None:
    path = _zip(tmp_path, {"a.txt": b"hello", "b.txt": b"x" * 100})
    entries = {e.path: e for e in await ZipArchiveOpener().entries(path)}
    assert entries["a.txt"].size_bytes == 5
    assert entries["b.txt"].size_bytes == 100
    assert entries["b.txt"].compressed_bytes < 100


async def test_entries_carry_a_byte_offset(tmp_path: Path) -> None:
    """A zip is seekable, so every member has a real place in the file."""
    path = _zip(tmp_path, {"a.txt": b"hello"})
    (entry,) = [e for e in await ZipArchiveOpener().entries(path) if not e.is_dir]
    assert entry.byte_offset is not None
    assert entry.byte_offset >= 0


async def test_directory_entries_are_marked(tmp_path: Path) -> None:
    path = tmp_path / "d.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sub/", b"")
        archive.writestr("sub/a.txt", b"hi")
    entries = {e.path: e for e in await ZipArchiveOpener().entries(str(path))}
    assert entries["sub/"].is_dir
    assert not entries["sub/a.txt"].is_dir


async def test_reads_a_member(tmp_path: Path) -> None:
    path = _zip(tmp_path, {"a.txt": b"hello world"})
    chunks = [c async for c in ZipArchiveOpener().open_member(path, "a.txt")]
    assert b"".join(chunks) == b"hello world"


async def test_a_missing_member_raises(tmp_path: Path) -> None:
    path = _zip(tmp_path, {"a.txt": b"hi"})
    with pytest.raises(SourceUnreadableError, match="nope.txt"):
        [c async for c in ZipArchiveOpener().open_member(path, "nope.txt")]


async def test_a_corrupt_archive_raises_rather_than_returning_nothing(tmp_path: Path) -> None:
    """Silence would look like an empty archive, which is a false claim."""
    path = tmp_path / "broken.zip"
    path.write_bytes(b"PK\x03\x04 and then garbage")
    with pytest.raises(SourceUnreadableError):
        await ZipArchiveOpener().entries(str(path))


async def test_a_corrupt_member_fails_on_read_without_blinding_its_neighbours(
    tmp_path: Path,
) -> None:
    """Spec §1.1: one bad member must not cost the agent the other entries."""
    path = _zip(tmp_path, {"good.txt": b"fine", "bad.txt": b"x" * 200})
    raw = bytearray(path if isinstance(path, bytes) else Path(path).read_bytes())
    # Corrupt the deflate stream of the second member without touching the
    # central directory, so listing still succeeds and only the read fails.
    marker = raw.rindex(b"bad.txt")
    raw[marker - 40 : marker - 30] = b"\x00" * 10
    Path(path).write_bytes(bytes(raw))
    opener = ZipArchiveOpener()
    listed = [e.path for e in await opener.entries(str(path))]
    assert "good.txt" in listed and "bad.txt" in listed
    with pytest.raises(SourceUnreadableError):
        [c async for c in opener.open_member(str(path), "bad.txt")]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --all-extras pytest tests/unit/adapters/test_zip_archive.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'readeverything.adapters.zip_archive'`

- [ ] **Step 3: Write the implementation**

Create `src/readeverything/adapters/zip_archive.py`:

```python
"""Zip, through stdlib `zipfile`.

The seekable half of the story: a zip's central directory gives every member an
offset and its own compressed size, so listing costs one seek and a small read,
and a ranged read of a member is a genuine ranged read. Nothing here
materialises anything.

Every `zipfile` failure is converted to `SourceUnreadableError`. A corrupt
archive that returned an empty entry list instead would be indistinguishable
from an empty archive, and an agent would report "this release contains
nothing" about a file it failed to open.
"""

from __future__ import annotations

import asyncio
import zipfile
from collections.abc import AsyncIterator, Sequence

from readeverything.domain.errors import SourceUnreadableError
from readeverything.domain.identity import MimeType
from readeverything.ports.containers import ArchiveEntry

#: What this opener answers `claims` for. `.zip` reaches detection under
#: several spellings depending on which of puremagic or `mimetypes` answered.
_MIMES = frozenset({"application/zip", "application/x-zip-compressed"})

#: Zip stores mode bits in the top 16 of `external_attr`, and S_IFLNK is 0xA000.
_S_IFMT = 0xF000
_S_IFLNK = 0xA000

_CHUNK = 1 << 20


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    """Whether the entry is a symlink, per its stored unix mode.

    Only files written on a unix host carry mode bits at all; `create_system`
    of 3 is the flag for that, and reading `external_attr` without checking it
    would misread a DOS-written entry's attribute byte as a file type.
    """
    if info.create_system != 3:
        return False
    return (info.external_attr >> 16) & _S_IFMT == _S_IFLNK


class ZipArchiveOpener:
    """Reads zip containers. Stateless, so one instance serves every archive."""

    def claims(self, mime: MimeType) -> bool:
        return str(mime) in _MIMES

    async def entries(self, path: str) -> Sequence[ArchiveEntry]:
        def _read() -> list[ArchiveEntry]:
            with zipfile.ZipFile(path) as archive:
                return [
                    ArchiveEntry(
                        path=info.filename,
                        size_bytes=info.file_size,
                        compressed_bytes=info.compress_size,
                        is_dir=info.is_dir(),
                        is_symlink=_is_symlink(info),
                        # `date_time` has no timezone and no sub-second part;
                        # it is reported as a fact, never used for freshness.
                        modified_epoch_s=None,
                        byte_offset=info.header_offset,
                    )
                    for info in archive.infolist()
                    if info.filename
                ]

        try:
            return await asyncio.to_thread(_read)
        except (OSError, zipfile.BadZipFile, ValueError) as exc:
            raise SourceUnreadableError(f"cannot read zip {path!r}: {exc}") from exc

    async def open_member(self, path: str, member: str) -> AsyncIterator[bytes]:
        try:
            archive = await asyncio.to_thread(zipfile.ZipFile, path)
        except (OSError, zipfile.BadZipFile) as exc:
            raise SourceUnreadableError(f"cannot read zip {path!r}: {exc}") from exc
        try:
            handle = await asyncio.to_thread(archive.open, member)
        except (KeyError, OSError, zipfile.BadZipFile) as exc:
            await asyncio.to_thread(archive.close)
            raise SourceUnreadableError(f"cannot read {member!r} from {path!r}: {exc}") from exc
        try:
            while True:
                try:
                    chunk = await asyncio.to_thread(handle.read, _CHUNK)
                except (OSError, zipfile.BadZipFile, EOFError) as exc:
                    # A member whose deflate stream is damaged. This fires on
                    # READ, after `entries` already succeeded, which is exactly
                    # the §1.1 requirement: one bad member must not cost the
                    # agent its neighbours.
                    raise SourceUnreadableError(
                        f"cannot read {member!r} from {path!r}: {exc}"
                    ) from exc
                if not chunk:
                    return
                yield chunk
        finally:
            await asyncio.to_thread(handle.close)
            await asyncio.to_thread(archive.close)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --all-extras pytest tests/unit/adapters/test_zip_archive.py -x -q`
Expected: PASS

If `test_a_corrupt_member_fails_on_read_without_blinding_its_neighbours` does not raise, the byte offset chosen for corruption landed outside the deflate stream. Adjust the slice so it lands inside the *first* member's compressed data (between the end of the local header for `good.txt` and the start of `bad.txt`'s local header) and assert on `good.txt` instead. Do not weaken the assertion to "either raises or returns wrong bytes" — the law is that it raises.

- [ ] **Step 5: Lint and type-check**

Run:
```bash
uv run --all-extras ruff check src/readeverything/adapters/zip_archive.py tests/unit/adapters/test_zip_archive.py
uv run --all-extras ruff format src/readeverything/adapters/zip_archive.py tests/unit/adapters/test_zip_archive.py
uv run --all-extras mypy src/readeverything/adapters/zip_archive.py
```
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/readeverything/adapters/zip_archive.py tests/unit/adapters/test_zip_archive.py
git commit -m "feat(adapters): ZipArchiveOpener over stdlib zipfile"
```

---

### Task 4: `TarArchiveOpener`

stdlib `tarfile`, covering `.tar`, `.tar.gz`, `.tgz`, `.tar.bz2` and `.tar.xz` — and the solid half of §3.2. A `.tar.gz` wraps compression around the *whole* archive, so member *n* cannot be reached without decompressing 0..*n*-1: reading three members naively is three full decompressions. This opener decompresses once into a temp file and reuses it.

**Files:**
- Create: `src/readeverything/adapters/tar_archive.py`
- Test: `tests/unit/adapters/test_tar_archive.py`

**Interfaces:**
- Consumes: `ArchiveEntry` from `ports.containers`; `MimeType`; `ContainerLimitExceededError`, `SourceUnreadableError`.
- Produces: `class TarArchiveOpener` with `__init__(self, *, max_materialised_bytes: int = 8 << 30) -> None`, `claims`, `entries`, `open_member`, and `async def aclose(self) -> None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/adapters/test_tar_archive.py`:

```python
"""The tar opener, including the solid-container materialisation."""

import io
import tarfile
from pathlib import Path

import pytest

from readeverything.adapters.tar_archive import TarArchiveOpener
from readeverything.domain.errors import ContainerLimitExceededError, SourceUnreadableError
from readeverything.domain.identity import MimeType


def _tar(tmp_path: Path, members: dict[str, bytes], *, mode: str = "w", name: str = "a.tar") -> str:
    path = tmp_path / name
    with tarfile.open(path, mode) as archive:  # type: ignore[call-overload]
        for member_name, data in members.items():
            info = tarfile.TarInfo(member_name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return str(path)


def test_claims_tar_family_and_not_zip() -> None:
    opener = TarArchiveOpener()
    assert opener.claims(MimeType.parse("application/x-tar"))
    assert opener.claims(MimeType.parse("application/gzip"))
    assert opener.claims(MimeType.parse("application/x-xz"))
    assert not opener.claims(MimeType.parse("application/zip"))


async def test_lists_entries(tmp_path: Path) -> None:
    path = _tar(tmp_path, {"a.txt": b"hello", "b.txt": b"xy"})
    entries = {e.path: e for e in await TarArchiveOpener().entries(path)}
    assert entries["a.txt"].size_bytes == 5
    assert entries["b.txt"].size_bytes == 2


async def test_an_uncompressed_tar_reports_byte_offsets(tmp_path: Path) -> None:
    path = _tar(tmp_path, {"a.txt": b"hello"})
    (entry,) = await TarArchiveOpener().entries(path)
    assert entry.byte_offset is not None


async def test_a_solid_tar_reports_no_byte_offsets(tmp_path: Path) -> None:
    """Offsets into a gzip stream are not offsets a caller can seek to."""
    path = _tar(tmp_path, {"a.txt": b"hello"}, mode="w:gz", name="a.tar.gz")
    (entry,) = await TarArchiveOpener().entries(path)
    assert entry.byte_offset is None


async def test_reads_a_member_from_a_solid_archive(tmp_path: Path) -> None:
    path = _tar(tmp_path, {"a.txt": b"hello world"}, mode="w:gz", name="a.tar.gz")
    opener = TarArchiveOpener()
    try:
        chunks = [c async for c in opener.open_member(path, "a.txt")]
        assert b"".join(chunks) == b"hello world"
    finally:
        await opener.aclose()


async def test_a_solid_archive_is_decompressed_once(tmp_path: Path) -> None:
    """Three member reads must not mean three full decompressions."""
    path = _tar(
        tmp_path, {"a.txt": b"a", "b.txt": b"b", "c.txt": b"c"}, mode="w:gz", name="a.tar.gz"
    )
    opener = TarArchiveOpener()
    try:
        for member in ("a.txt", "b.txt", "c.txt"):
            assert [c async for c in opener.open_member(path, member)] == [member[0].encode()]
        assert len(opener.materialised) == 1
    finally:
        await opener.aclose()


async def test_aclose_removes_the_materialised_copy(tmp_path: Path) -> None:
    path = _tar(tmp_path, {"a.txt": b"hi"}, mode="w:gz", name="a.tar.gz")
    opener = TarArchiveOpener()
    [c async for c in opener.open_member(path, "a.txt")]
    (temp,) = opener.materialised.values()
    assert Path(temp).exists()
    await opener.aclose()
    assert not Path(temp).exists()


async def test_the_cache_evicts_least_recently_used_rather_than_failing(
    tmp_path: Path,
) -> None:
    """Spec §3.2: a directory of large tarballs should be SLOW, not unreadable.

    Failing at the bound would make it unreadable, so the bound evicts.
    """
    first = _tar(tmp_path, {"a.txt": b"x" * 4096}, mode="w:gz", name="one.tar.gz")
    second = _tar(tmp_path, {"b.txt": b"y" * 4096}, mode="w:gz", name="two.tar.gz")
    # Room for one decompressed tarball, not two.
    opener = TarArchiveOpener(max_materialised_bytes=20_000)
    try:
        assert [c async for c in opener.open_member(first, "a.txt")] == [b"x" * 4096]
        assert [c async for c in opener.open_member(second, "b.txt")] == [b"y" * 4096]
        assert list(opener.materialised) == [second]
        # And the evicted one is still READABLE, just paid for again.
        assert [c async for c in opener.open_member(first, "a.txt")] == [b"x" * 4096]
        assert list(opener.materialised) == [first]
    finally:
        await opener.aclose()


async def test_one_archive_larger_than_the_whole_bound_is_refused(tmp_path: Path) -> None:
    """Eviction cannot help here: there is nothing left to evict."""
    path = _tar(tmp_path, {"a.txt": b"x" * 4096}, mode="w:gz", name="a.tar.gz")
    opener = TarArchiveOpener(max_materialised_bytes=16)
    try:
        with pytest.raises(ContainerLimitExceededError, match="materialis"):
            [c async for c in opener.open_member(path, "a.txt")]
    finally:
        await opener.aclose()


async def test_a_missing_member_raises(tmp_path: Path) -> None:
    path = _tar(tmp_path, {"a.txt": b"hi"})
    with pytest.raises(SourceUnreadableError, match="nope.txt"):
        [c async for c in TarArchiveOpener().open_member(path, "nope.txt")]


async def test_a_symlink_member_is_reported_and_not_opened(tmp_path: Path) -> None:
    """A tarball can carry a link to /etc/passwd. Following it is the hole."""
    path = tmp_path / "l.tar"
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("passwd")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)
    (entry,) = await TarArchiveOpener().entries(str(path))
    assert entry.is_symlink
    with pytest.raises(SourceUnreadableError, match="symlink"):
        [c async for c in TarArchiveOpener().open_member(str(path), "passwd")]


async def test_a_corrupt_archive_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.tar.gz"
    path.write_bytes(b"\x1f\x8b and then garbage")
    with pytest.raises(SourceUnreadableError):
        await TarArchiveOpener().entries(str(path))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --all-extras pytest tests/unit/adapters/test_tar_archive.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'readeverything.adapters.tar_archive'`

- [ ] **Step 3: Write the implementation**

Create `src/readeverything/adapters/tar_archive.py`:

```python
"""Tar, through stdlib `tarfile`, and the solid-container problem.

`.tar` is seekable: a header chain gives every member an offset, and a ranged
read is a real one. `.tar.gz`, `.tar.bz2` and `.tar.xz` are SOLID -- the
compression wraps the whole archive, so member n cannot be reached without
decompressing 0..n-1, and reading three members naively costs three full
decompressions of the same file.

So a solid archive is decompressed ONCE into a temp file and every subsequent
read of any member goes through that copy, for the lifetime of this opener.
That is a cache with a bound (`max_materialised_bytes`), and at the bound it
EVICTS LEAST-RECENTLY-USED rather than failing: failing would make a directory
of large tarballs unreadable, where evicting only makes it slow. The single
case eviction cannot rescue -- one archive larger than the entire bound -- does
raise, because there is nothing left to evict.

The temp files live in a `TemporaryDirectory` this instance owns, so they are
removed by `aclose()` and, failing that, by the directory's own finalizer at
interpreter exit. Nothing here is left behind on a crash path.
"""

from __future__ import annotations

import asyncio
import tarfile
import tempfile
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path

from readeverything.domain.errors import ContainerLimitExceededError, SourceUnreadableError
from readeverything.domain.identity import MimeType
from readeverything.ports.containers import ArchiveEntry

#: Every spelling of the tar family this opener answers `claims` for.
#: `application/gzip` is included because a bare `.gz` of a tar is the common
#: shape and detection reports the OUTER compression, which is the honest
#: answer about the bytes -- so the opener, not the detector, is what discovers
#: a tar inside.
_MIMES = frozenset(
    {
        "application/x-tar",
        "application/x-gtar",
        "application/gzip",
        "application/x-gzip",
        "application/x-bzip2",
        "application/x-xz",
    }
)

_CHUNK = 1 << 20


def _is_solid(path: str) -> bool:
    """Whether reaching member n requires decompressing 0..n-1.

    Decided by reading the file's own magic rather than its extension: a
    `.tgz` and a `.tar.gz` are the same thing under two names, and a `.tar`
    someone gzipped without renaming is the case an extension check gets
    wrong in the expensive direction.
    """
    with open(path, "rb") as handle:
        head = handle.read(6)
    return head[:2] == b"\x1f\x8b" or head[:3] == b"BZh" or head[:6] == b"\xfd7zXZ\x00"


class TarArchiveOpener:
    """Reads the tar family, materialising a solid archive at most once."""

    def __init__(self, *, max_materialised_bytes: int = 8 << 30) -> None:
        self._max_materialised_bytes = max_materialised_bytes
        self._temp = tempfile.TemporaryDirectory(prefix="readeverything-tar-")
        #: Insertion-ordered, and re-inserted on use, so the FIRST key is
        #: always the least recently used one. A plain dict is an LRU list
        #: here because Python guarantees that order.
        self._materialised: OrderedDict[str, str] = OrderedDict()
        self._sizes: dict[str, int] = {}
        self._held = 0
        self._counter = 0
        self._lock = asyncio.Lock()

    @property
    def materialised(self) -> Mapping[str, str]:
        """Solid archives decompressed so far, keyed by their original path.

        Public because the "decompressed once" promise is otherwise
        unobservable, and an unobservable promise is one that quietly stops
        being true.
        """
        return self._materialised

    async def aclose(self) -> None:
        """Remove every decompressed copy this opener made."""
        await asyncio.to_thread(self._temp.cleanup)
        self._materialised.clear()
        self._sizes.clear()
        self._held = 0

    def claims(self, mime: MimeType) -> bool:
        return str(mime) in _MIMES

    async def _seekable_path(self, path: str) -> str:
        """`path` itself when it is a plain tar, else a decompressed copy."""
        try:
            solid = await asyncio.to_thread(_is_solid, path)
        except OSError as exc:
            raise SourceUnreadableError(f"cannot read tar {path!r}: {exc}") from exc
        if not solid:
            return path
        async with self._lock:
            # Re-checked inside the lock: two concurrent reads of the same
            # tarball must decompress it once between them, not once each.
            existing = self._materialised.get(path)
            if existing is not None:
                self._materialised.move_to_end(path)
                return existing
            self._counter += 1
            target = str(Path(self._temp.name) / f"{self._counter}.tar")
            written = await asyncio.to_thread(self._decompress, path, target)
            self._materialised[path] = target
            self._sizes[path] = written
            self._held += written
            await self._evict_to_fit(keep=path)
            return target

    async def _evict_to_fit(self, *, keep: str) -> None:
        """Drop least-recently-used copies until the cache is back inside its bound.

        `keep` is never evicted: it is the copy the caller is about to read,
        and evicting it would mean decompressing the same file twice in one
        call. If it alone is over the bound, `_decompress` already raised.
        """
        while self._held > self._max_materialised_bytes and len(self._materialised) > 1:
            oldest = next(iter(self._materialised))
            if oldest == keep:
                oldest = next(key for key in self._materialised if key != keep)
            target = self._materialised.pop(oldest)
            self._held -= self._sizes.pop(oldest, 0)
            await asyncio.to_thread(Path(target).unlink, True)

    def _decompress(self, path: str, target: str) -> int:
        """Stream the archive out, checking the bound as it goes. Returns bytes written.

        Checked DURING the write rather than against a declared size, for the
        same reason the expansion guard is: a compressed file's header is the
        bomb's own paperwork. This is the ONE place the bound raises: a single
        archive bigger than the whole cache cannot be made to fit by evicting
        anything, because there would be nothing left.
        """
        try:
            with tarfile.open(path, "r:*") as archive:
                stream = archive.fileobj
                if stream is None:  # pragma: no cover - tarfile always sets this
                    raise SourceUnreadableError(f"cannot read tar {path!r}: no stream")
                stream.seek(0)
                written = 0
                with open(target, "wb") as out:
                    while True:
                        chunk = stream.read(_CHUNK)
                        if not chunk:
                            return written
                        written += len(chunk)
                        if written > self._max_materialised_bytes:
                            raise ContainerLimitExceededError(
                                f"{path!r} exceeds max_materialised_bytes "
                                f"({self._max_materialised_bytes}) when decompressed"
                            )
                        out.write(chunk)
        except (OSError, tarfile.TarError, EOFError) as exc:
            raise SourceUnreadableError(f"cannot read tar {path!r}: {exc}") from exc

    async def entries(self, path: str) -> Sequence[ArchiveEntry]:
        solid = await asyncio.to_thread(self._safe_is_solid, path)

        def _read() -> list[ArchiveEntry]:
            with tarfile.open(path, "r:*") as archive:
                return [
                    ArchiveEntry(
                        path=info.name,
                        size_bytes=info.size,
                        # tar does not compress per member, so a member's
                        # compressed size IS its size. The expansion guard
                        # upstream therefore never fires on a plain tar, which
                        # is correct: a plain tar cannot be a bomb.
                        compressed_bytes=info.size,
                        is_dir=info.isdir(),
                        is_symlink=info.issym() or info.islnk(),
                        modified_epoch_s=float(info.mtime),
                        # An offset into a gzip stream is not an offset anyone
                        # can seek to, so a solid archive reports none. This is
                        # the single fact that tells a caller which shape it has.
                        byte_offset=None if solid else info.offset_data,
                    )
                    for info in archive.getmembers()
                    if info.name
                ]

        try:
            return await asyncio.to_thread(_read)
        except (OSError, tarfile.TarError, EOFError, ValueError) as exc:
            raise SourceUnreadableError(f"cannot read tar {path!r}: {exc}") from exc

    def _safe_is_solid(self, path: str) -> bool:
        try:
            return _is_solid(path)
        except OSError as exc:
            raise SourceUnreadableError(f"cannot read tar {path!r}: {exc}") from exc

    async def open_member(self, path: str, member: str) -> AsyncIterator[bytes]:
        readable = await self._seekable_path(path)
        try:
            archive = await asyncio.to_thread(tarfile.open, readable, "r:*")
        except (OSError, tarfile.TarError, EOFError) as exc:
            raise SourceUnreadableError(f"cannot read tar {path!r}: {exc}") from exc
        try:
            try:
                info = await asyncio.to_thread(archive.getmember, member)
            except KeyError as exc:
                raise SourceUnreadableError(f"no member {member!r} in {path!r}") from exc
            if info.issym() or info.islnk():
                # A tarball can carry a link to /etc/passwd, and materialising
                # it would follow that link straight out of the root. Refusing
                # is the only defensible default for a library whose whole
                # sandboxing story is "nothing outside the root". The link is
                # still REPORTED by `entries`, with its target, as a fact.
                raise SourceUnreadableError(
                    f"{member!r} in {path!r} is a symlink to {info.linkname!r}; "
                    "links inside containers are reported but never followed"
                )
            handle = await asyncio.to_thread(archive.extractfile, info)
            if handle is None:
                raise SourceUnreadableError(f"{member!r} in {path!r} has no readable content")
            try:
                while True:
                    try:
                        chunk = await asyncio.to_thread(handle.read, _CHUNK)
                    except (OSError, tarfile.TarError, EOFError) as exc:
                        raise SourceUnreadableError(
                            f"cannot read {member!r} from {path!r}: {exc}"
                        ) from exc
                    if not chunk:
                        return
                    yield chunk
            finally:
                await asyncio.to_thread(handle.close)
        finally:
            await asyncio.to_thread(archive.close)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --all-extras pytest tests/unit/adapters/test_tar_archive.py -x -q`
Expected: PASS

If `_decompress`'s `stream.seek(0)` approach does not yield a plain tar (it depends on `tarfile` exposing the decompressing file object), replace the body with an explicit `gzip.open`/`bz2.open`/`lzma.open` chosen by the magic bytes `_is_solid` already read, keeping the same chunked bound check. Do not remove the bound check.

- [ ] **Step 5: Lint and type-check**

Run:
```bash
uv run --all-extras ruff check src/readeverything/adapters/tar_archive.py tests/unit/adapters/test_tar_archive.py
uv run --all-extras ruff format src/readeverything/adapters/tar_archive.py tests/unit/adapters/test_tar_archive.py
uv run --all-extras mypy src/readeverything/adapters/tar_archive.py
```
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/readeverything/adapters/tar_archive.py tests/unit/adapters/test_tar_archive.py
git commit -m "feat(adapters): TarArchiveOpener, with the solid archive materialised once"
```

---

### Task 5: `NestedSource` — resolution, limits and guards

The decorator itself, minus `walk` (Task 6). Every method splits the uri; **if there is exactly one segment it delegates verbatim to `inner` and returns**. That single line is the whole compatibility story: a perception over loose files behaves identically whether or not this is installed, and every existing `LocalFileSource` test still exercises the real path.

**Files:**
- Create: `src/readeverything/adapters/nested_source.py`
- Test: `tests/unit/adapters/test_nested_source.py`

**Interfaces:**
- Consumes: `split_uri`, `join_uri` from `domain.container_uri`; `ArchiveEntry`, `ArchiveOpener`, `ContainerLimits` from `ports.containers`; `FileSource` and `MimeDetector` from `ports`.
- Produces:
  - `class CompositeOpener` with `__init__(self, *, openers: Sequence[ArchiveOpener]) -> None`, `claims`, `entries`, `open_member`, `async def aclose(self) -> None`, and `def opener_for(self, mime: MimeType) -> ArchiveOpener | None`
  - `class NestedSource` with `__init__(self, inner: FileSource, *, limits: ContainerLimits, archives: ArchiveOpener, detector: MimeDetector) -> None` and the full `FileSource` surface plus `async def aclose(self) -> None`

**Note on `detector`:** the spec's §3 signature shows `inner`, `limits` and `archives`. `walk` must decide descent by *mimetype* (§3.1) and `ArchiveOpener.claims` takes a `MimeType`, so a `MimeDetector` is the only way to obtain one without duplicating `adapters/detection.py`'s content-first rule inside this file. `MimeDetector` is a port, so the layering is unchanged. Nothing else about the signature moves.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/adapters/test_nested_source.py`:

```python
"""`NestedSource`: delegation, resolution, limits and the two guards."""

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from readeverything.adapters.detection import PuremagicDetector
from readeverything.adapters.local_source import LocalFileSource
from readeverything.adapters.nested_source import CompositeOpener, NestedSource
from readeverything.adapters.tar_archive import TarArchiveOpener
from readeverything.adapters.zip_archive import ZipArchiveOpener
from readeverything.domain.errors import ContainerLimitExceededError, SourceUnreadableError
from readeverything.ports.containers import ContainerLimits


def _openers() -> CompositeOpener:
    return CompositeOpener(openers=[ZipArchiveOpener(), TarArchiveOpener()])


def _nested(root: Path, limits: ContainerLimits | None = None) -> NestedSource:
    return NestedSource(
        LocalFileSource(root=root),
        limits=ContainerLimits() if limits is None else limits,
        archives=_openers(),
        detector=PuremagicDetector(),
    )


def _zip(root: Path, name: str, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(root / name, "w", zipfile.ZIP_DEFLATED) as archive:
        for member, data in members.items():
            archive.writestr(member, data)


def _targz(root: Path, name: str, members: dict[str, bytes]) -> None:
    with tarfile.open(root / name, "w:gz") as archive:
        for member, data in members.items():
            info = tarfile.TarInfo(member)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


async def test_a_single_segment_uri_delegates_verbatim(tmp_path: Path) -> None:
    """The compatibility story: loose files behave as if this were not installed."""
    (tmp_path / "a.txt").write_bytes(b"hello")
    source = _nested(tmp_path)
    assert await source.read_bytes("a.txt") == b"hello"
    assert await source.size("a.txt") == 5
    assert await source.exists("a.txt")
    assert await source.read_range("a.txt", 1, 3) == b"el"
    assert await source.local_path("a.txt") == str(tmp_path / "a.txt")


async def test_reads_a_zip_member(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"hello world"})
    assert await _nested(tmp_path).read_bytes("a.zip!inner.txt") == b"hello world"


async def test_reads_a_member_two_containers_deep(tmp_path: Path) -> None:
    """The §1.1 shape, at the source layer."""
    inner = tmp_path / "build"
    inner.mkdir()
    _targz(inner, "nested.tar.gz", {"notes.txt": b"deep"})
    _zip(tmp_path, "docs.zip", {"nested.tar.gz": (inner / "nested.tar.gz").read_bytes()})
    source = _nested(tmp_path)
    try:
        assert await source.read_bytes("docs.zip!nested.tar.gz!notes.txt") == b"deep"
    finally:
        await source.aclose()


async def test_size_is_the_uncompressed_size(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"x" * 5000})
    assert await _nested(tmp_path).size("a.zip!inner.txt") == 5000


async def test_exists_is_true_for_a_present_member(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"hi"})
    source = _nested(tmp_path)
    assert await source.exists("a.zip!inner.txt")
    assert not await source.exists("a.zip!nope.txt")


async def test_exists_is_false_when_the_container_is_missing(tmp_path: Path) -> None:
    """`exists` answers a question; it does not raise about a bad guess."""
    assert not await _nested(tmp_path).exists("gone.zip!inner.txt")


async def test_read_range_slices_a_member(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"hello world"})
    assert await _nested(tmp_path).read_range("a.zip!inner.txt", 6, 11) == b"world"


async def test_stream_chunks_a_member(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"abcdef"})
    source = _nested(tmp_path)
    chunks = [c async for c in source.stream("a.zip!inner.txt", chunk_size=2)]
    assert b"".join(chunks) == b"abcdef"


async def test_local_path_materialises_a_member(tmp_path: Path) -> None:
    """The one place the cost is acknowledged rather than hidden."""
    _zip(tmp_path, "a.zip", {"inner.txt": b"hello"})
    source = _nested(tmp_path)
    try:
        path = Path(await source.local_path("a.zip!inner.txt"))
        assert path.read_bytes() == b"hello"
    finally:
        await source.aclose()


async def test_local_path_is_stable_across_calls(tmp_path: Path) -> None:
    """`stat_key` compares inodes; a new temp file per call would thrash."""
    _zip(tmp_path, "a.zip", {"inner.txt": b"hello"})
    source = _nested(tmp_path)
    try:
        first = await source.local_path("a.zip!inner.txt")
        assert await source.local_path("a.zip!inner.txt") == first
    finally:
        await source.aclose()


async def test_aclose_removes_materialised_members(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"hello"})
    source = _nested(tmp_path)
    path = Path(await source.local_path("a.zip!inner.txt"))
    await source.aclose()
    assert not path.exists()


async def test_a_missing_member_raises(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"hi"})
    with pytest.raises(SourceUnreadableError, match="nope.txt"):
        await _nested(tmp_path).read_bytes("a.zip!nope.txt")


async def test_a_container_with_no_opener_raises(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"not an archive")
    with pytest.raises(SourceUnreadableError, match="not a container"):
        await _nested(tmp_path).read_bytes("a.txt!inner.txt")


async def test_depth_beyond_the_limit_is_refused(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"hi"})
    source = _nested(tmp_path, ContainerLimits(max_depth=1))
    with pytest.raises(ContainerLimitExceededError, match="max_depth"):
        await source.read_bytes("a.zip!inner.txt")


async def test_a_traversing_member_is_refused(tmp_path: Path) -> None:
    """A member path is never resolved against the filesystem, so the root
    guard in `LocalFileSource` cannot see this one. It needs its own."""
    _zip(tmp_path, "a.zip", {"inner.txt": b"hi"})
    with pytest.raises(SourceUnreadableError, match="traversal"):
        await _nested(tmp_path).read_bytes("a.zip!../../etc/passwd")


async def test_an_absolute_member_is_refused(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"hi"})
    with pytest.raises(SourceUnreadableError, match="absolute"):
        await _nested(tmp_path).read_bytes("a.zip!/etc/passwd")


async def test_a_windows_drive_member_is_refused(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"hi"})
    with pytest.raises(SourceUnreadableError, match="absolute"):
        await _nested(tmp_path).read_bytes("a.zip!C:/windows/system32/config/sam")


async def test_a_member_over_the_byte_limit_is_refused(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"inner.txt": b"x" * 4096})
    source = _nested(tmp_path, ContainerLimits(max_member_bytes=64))
    with pytest.raises(ContainerLimitExceededError, match="max_member_bytes"):
        await source.read_bytes("a.zip!inner.txt")


async def test_the_expansion_ratio_fires_mid_stream_on_a_bomb(tmp_path: Path) -> None:
    """The check that matters. A bomb lies in its header, so the guard runs
    against bytes ACTUALLY WRITTEN, not against a declared size."""
    _zip(tmp_path, "bomb.zip", {"payload": b"\0" * (1 << 22)})
    source = _nested(tmp_path, ContainerLimits(max_expansion_ratio=2.0))
    with pytest.raises(ContainerLimitExceededError, match="expansion"):
        await source.read_bytes("bomb.zip!payload")


async def test_a_container_with_too_many_members_is_refused(tmp_path: Path) -> None:
    _zip(tmp_path, "many.zip", {f"f{n}.txt": b"x" for n in range(20)})
    source = _nested(tmp_path, ContainerLimits(max_members=5))
    with pytest.raises(ContainerLimitExceededError, match="max_members"):
        await source.read_bytes("many.zip!f0.txt")


async def test_a_container_whose_total_exceeds_the_limit_is_refused(tmp_path: Path) -> None:
    _zip(tmp_path, "big.zip", {f"f{n}.txt": b"x" * 1000 for n in range(10)})
    source = _nested(tmp_path, ContainerLimits(max_total_bytes=500))
    with pytest.raises(ContainerLimitExceededError, match="max_total_bytes"):
        await source.read_bytes("big.zip!f0.txt")


async def test_a_symlink_member_of_a_tar_is_refused_on_read(tmp_path: Path) -> None:
    path = tmp_path / "l.tar"
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("passwd")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)
    with pytest.raises(SourceUnreadableError, match="symlink"):
        await _nested(tmp_path).read_bytes("l.tar!passwd")


def test_the_composite_reports_no_opener_for_an_unknown_mime() -> None:
    from readeverything.domain.identity import MimeType

    assert _openers().opener_for(MimeType.parse("text/plain")) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --all-extras pytest tests/unit/adapters/test_nested_source.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'readeverything.adapters.nested_source'`

- [ ] **Step 3: Write the implementation**

Create `src/readeverything/adapters/nested_source.py`. Write `walk` as a plain delegation for now — Task 6 replaces it.

```python
"""Reading something inside something else.

A DECORATOR over another `FileSource`, not a replacement for one. Every method
splits the uri and, **if there is exactly one segment, delegates verbatim to
`inner` and returns**. That is the whole compatibility story: a perception over
a directory of loose files behaves identically whether or not this is
installed, and every existing `LocalFileSource` test keeps exercising the real
code path rather than a lookalike.

For a multi-segment uri it resolves left to right -- open the outermost
container from `inner`, open each subsequent container from the member bytes of
the one before, answer the request against the final member.

The reason this is the source layer and not an archive handler with a
`read_entry` affordance: an affordance returning member bytes gives an agent
bytes, while a nested uri gives it a PERCEPTION -- a card, an outline, page
affordances, OCR, provenance. Every handler in this repository already reads
through `SourceReader` and is forbidden from touching a filesystem, so the PDF
handler reads a PDF inside a tarball inside a zip without one line of it
changing. This file is the adapter collecting on a bill the architecture
already paid.

Two guards live here that `LocalFileSource`'s root check cannot cover, because
a member path is never resolved against a filesystem at all: a `..` component
and an absolute member are both refused outright, and a member the container
declares as a symlink is reported but never followed.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator, Sequence
from pathlib import Path, PurePosixPath

from readeverything.domain.container_uri import join_uri, split_uri
from readeverything.domain.errors import ContainerLimitExceededError, SourceUnreadableError
from readeverything.domain.identity import MimeType
from readeverything.ports.containers import ArchiveEntry, ArchiveOpener, ContainerLimits
from readeverything.ports.detection import MimeDetector
from readeverything.ports.source import FileSource

#: How much of a container is read to detect its mimetype. Matches
#: `pipeline.perception._HEAD_BYTES`, so a container is typed by exactly the
#: bytes the pipeline would have typed it by.
_HEAD_BYTES = 4096

_CHUNK = 1 << 20


class CompositeOpener:
    """Dispatches to whichever opener claims the mimetype.

    The extension point the spec promises: a caller who wants `.7z` or `.rar`
    supplies their own `ArchiveOpener` here, and this repository never grows a
    dependency on either.
    """

    def __init__(self, *, openers: Sequence[ArchiveOpener]) -> None:
        self._openers = tuple(openers)

    def opener_for(self, mime: MimeType) -> ArchiveOpener | None:
        for opener in self._openers:
            if opener.claims(mime):
                return opener
        return None

    def claims(self, mime: MimeType) -> bool:
        return self.opener_for(mime) is not None

    async def entries(self, path: str) -> Sequence[ArchiveEntry]:
        raise NotImplementedError("dispatch through `opener_for`; a composite has no one format")

    def open_member(self, path: str, member: str) -> AsyncIterator[bytes]:
        raise NotImplementedError("dispatch through `opener_for`; a composite has no one format")

    async def aclose(self) -> None:
        for opener in self._openers:
            closer = getattr(opener, "aclose", None)
            if closer is not None:
                await closer()


def _checked_member(member: str, uri: str) -> str:
    """A member path, or a refusal.

    `LocalFileSource` guards its root with `resolve()` and a parent check.
    That guard cannot see any of this, because a member path is never resolved
    against the filesystem -- it is looked up in a container's own directory,
    which will happily hand back whatever the archive's author wrote there. So
    the check is textual and it is strict.
    """
    if member.startswith("/") or member.startswith("\\"):
        raise SourceUnreadableError(f"member {member!r} of {uri!r} is absolute; refused")
    if len(member) >= 2 and member[1] == ":" and member[0].isalpha():
        raise SourceUnreadableError(
            f"member {member!r} of {uri!r} is absolute (a drive letter); refused"
        )
    parts = PurePosixPath(member.replace("\\", "/")).parts
    if ".." in parts:
        raise SourceUnreadableError(f"member {member!r} of {uri!r} contains a traversal; refused")
    return member


class NestedSource:
    """A `FileSource` that can see inside containers."""

    def __init__(
        self,
        inner: FileSource,
        *,
        limits: ContainerLimits,
        archives: ArchiveOpener,
        detector: MimeDetector,
    ) -> None:
        self._inner = inner
        self._limits = limits
        self._archives = archives
        self._detector = detector
        self._temp = tempfile.TemporaryDirectory(prefix="readeverything-nested-")
        #: uri -> the temp file holding its bytes. Keyed on the full nested uri
        #: so `local_path` is STABLE: `pipeline.resolution.stat_key` compares
        #: inodes, and a fresh temp file per call would make every member look
        #: like a different file on every access.
        self._materialised: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        """Remove every temp file this source and its openers created."""
        await asyncio.to_thread(self._temp.cleanup)
        self._materialised.clear()
        closer = getattr(self._archives, "aclose", None)
        if closer is not None:
            await closer()

    # -- resolution ---------------------------------------------------------

    async def _opener_for_path(self, path: str, uri: str) -> ArchiveOpener:
        """The opener for the container at local `path`, or a refusal."""
        try:
            head = await asyncio.to_thread(_read_head, path)
        except OSError as exc:
            raise SourceUnreadableError(f"cannot read container {uri!r}: {exc}") from exc
        mime = await self._detector.detect(uri, head)
        chooser = getattr(self._archives, "opener_for", None)
        opener = chooser(mime) if chooser is not None else None
        if opener is None and self._archives.claims(mime):
            opener = self._archives
        if opener is None:
            raise SourceUnreadableError(f"{uri!r} is not a container this source can open ({mime})")
        return opener

    async def _entries(self, opener: ArchiveOpener, path: str, uri: str) -> Sequence[ArchiveEntry]:
        entries = await opener.entries(path)
        if len(entries) > self._limits.max_members:
            raise ContainerLimitExceededError(
                f"{uri!r} declares {len(entries)} members, over max_members "
                f"({self._limits.max_members})"
            )
        total = sum(entry.size_bytes for entry in entries)
        if total > self._limits.max_total_bytes:
            raise ContainerLimitExceededError(
                f"{uri!r} expands to {total} bytes, over max_total_bytes "
                f"({self._limits.max_total_bytes})"
            )
        return entries

    def _entry(self, entries: Sequence[ArchiveEntry], member: str, uri: str) -> ArchiveEntry:
        for entry in entries:
            if entry.path == member or entry.path.rstrip("/") == member:
                return entry
        raise SourceUnreadableError(f"no member {member!r} in {uri!r}")

    async def _container_path(self, segments: Sequence[str]) -> str:
        """A local filesystem path for the container named by `segments`.

        The outermost comes straight from `inner`. Anything deeper has to be
        materialised, because an `ArchiveOpener` takes a path -- which is the
        same honest cost `ports/source.py` already names for `local_path`, and
        the same thing that lets ffmpeg and pypdfium2 work on archive members
        without changing.
        """
        if len(segments) == 1:
            return await self._inner.local_path(segments[0])
        return await self._materialise(join_uri(segments))

    async def _member_bytes(self, uri: str) -> bytes:
        """The decompressed bytes of the member `uri` names."""
        segments = split_uri(uri)
        depth = len(segments) - 1
        if depth > self._limits.max_depth:
            raise ContainerLimitExceededError(
                f"{uri!r} is {depth} container(s) deep, over max_depth "
                f"({self._limits.max_depth})"
            )
        member = _checked_member(segments[-1], uri)
        container_uri = join_uri(segments[:-1])
        path = await self._container_path(segments[:-1])
        opener = await self._opener_for_path(path, container_uri)
        entries = await self._entries(opener, path, container_uri)
        entry = self._entry(entries, member, container_uri)
        if entry.is_symlink:
            raise SourceUnreadableError(
                f"member {member!r} of {container_uri!r} is a symlink; "
                "links inside containers are reported but never followed"
            )
        if entry.size_bytes > self._limits.max_member_bytes:
            raise ContainerLimitExceededError(
                f"member {member!r} of {container_uri!r} is {entry.size_bytes} bytes, "
                f"over max_member_bytes ({self._limits.max_member_bytes})"
            )
        return await self._drain(opener, path, entry, container_uri)

    async def _drain(
        self, opener: ArchiveOpener, path: str, entry: ArchiveEntry, container_uri: str
    ) -> bytes:
        """Decompress one member, guarding as the bytes actually arrive.

        Both guards run MID-STREAM. A zip bomb lies in its header, so checking
        `entry.size_bytes` alone would be reading the bomb's own paperwork:
        `written` is the only number here that cannot be forged.
        """
        ceiling = self._limits.max_member_bytes
        ratio_ceiling = (
            entry.compressed_bytes * self._limits.max_expansion_ratio
            if entry.compressed_bytes
            else None
        )
        buffer = bytearray()
        async for chunk in opener.open_member(path, entry.path):
            buffer += chunk
            if len(buffer) > ceiling:
                raise ContainerLimitExceededError(
                    f"member {entry.path!r} of {container_uri!r} exceeds max_member_bytes "
                    f"({ceiling}) while decompressing"
                )
            if ratio_ceiling is not None and len(buffer) > ratio_ceiling:
                raise ContainerLimitExceededError(
                    f"member {entry.path!r} of {container_uri!r} exceeds its expansion "
                    f"ratio limit ({self._limits.max_expansion_ratio}) while decompressing; "
                    f"{len(buffer)} bytes out of {entry.compressed_bytes} compressed"
                )
        return bytes(buffer)

    async def _materialise(self, uri: str) -> str:
        async with self._lock:
            existing = self._materialised.get(uri)
            if existing is not None:
                return existing
            data = await self._member_bytes(uri)
            target = Path(self._temp.name) / f"{len(self._materialised)}-{Path(uri).name}"
            await asyncio.to_thread(target.write_bytes, data)
            self._materialised[uri] = str(target)
            return str(target)

    # -- the FileSource surface ---------------------------------------------

    async def exists(self, uri: str) -> bool:
        segments = split_uri(uri)
        if len(segments) == 1:
            return await self._inner.exists(uri)
        try:
            container_uri = join_uri(segments[:-1])
            member = _checked_member(segments[-1], uri)
            path = await self._container_path(segments[:-1])
            opener = await self._opener_for_path(path, container_uri)
            entries = await self._entries(opener, path, container_uri)
        except SourceUnreadableError:
            # `exists` answers a question rather than raising about a guess,
            # matching `LocalFileSource.exists` on a missing path.
            return False
        return any(e.path == member or e.path.rstrip("/") == member for e in entries)

    async def size(self, uri: str) -> int:
        """The member's UNCOMPRESSED size, from the container's directory.

        The directory, not a decompression: this is the number a card reports
        and a card must stay within probe cost.
        """
        segments = split_uri(uri)
        if len(segments) == 1:
            return await self._inner.size(uri)
        container_uri = join_uri(segments[:-1])
        member = _checked_member(segments[-1], uri)
        path = await self._container_path(segments[:-1])
        opener = await self._opener_for_path(path, container_uri)
        entries = await self._entries(opener, path, container_uri)
        return self._entry(entries, member, container_uri).size_bytes

    async def read_bytes(self, uri: str) -> bytes:
        segments = split_uri(uri)
        if len(segments) == 1:
            return await self._inner.read_bytes(uri)
        return await self._member_bytes(uri)

    async def read_range(self, uri: str, start: int, end: int) -> bytes:
        segments = split_uri(uri)
        if len(segments) == 1:
            return await self._inner.read_range(uri, start, end)
        return (await self._member_bytes(uri))[start : max(start, end)]

    async def stream(self, uri: str, *, chunk_size: int = 1 << 20) -> AsyncIterator[bytes]:
        segments = split_uri(uri)
        if len(segments) == 1:
            async for chunk in self._inner.stream(uri, chunk_size=chunk_size):
                yield chunk
            return
        data = await self._member_bytes(uri)
        for offset in range(0, len(data), chunk_size):
            yield data[offset : offset + chunk_size]

    async def local_path(self, uri: str) -> str:
        segments = split_uri(uri)
        if len(segments) == 1:
            return await self._inner.local_path(uri)
        return await self._materialise(uri)

    async def walk(self, uri: str) -> Sequence[str]:
        return await self._inner.walk(uri)


def _read_head(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read(_HEAD_BYTES)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --all-extras pytest tests/unit/adapters/test_nested_source.py -x -q`
Expected: PASS

Note on `test_the_expansion_ratio_fires_mid_stream_on_a_bomb`: 4 MiB of zeros deflates to well under 1% of its size, so a ratio of 2.0 fires within the first chunk. If it does not, raise the payload size — never lower the assertion to a non-mid-stream check.

Note on `test_a_container_with_no_opener_raises`: the message asserted is "not a container", which the `_opener_for_path` refusal contains.

- [ ] **Step 5: Lint and type-check**

Run:
```bash
uv run --all-extras ruff check src/readeverything/adapters/nested_source.py tests/unit/adapters/test_nested_source.py
uv run --all-extras ruff format src/readeverything/adapters/nested_source.py tests/unit/adapters/test_nested_source.py
uv run --all-extras mypy src/readeverything/adapters/nested_source.py
```
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/readeverything/adapters/nested_source.py tests/unit/adapters/test_nested_source.py
git commit -m "feat(adapters): NestedSource resolves members, with limits and guards"
```

---

### Task 6: `walk` returns members inline

The single decision that makes every downstream feature free (spec §3.1). Because `pipeline.perception` walks and then inspects, and inspection dispatches on detected mimetype, an archived PDF reaches the PDF handler with **no registry change, no handler change, and no special case above the adapter layer**.

And the rule that keeps a `.docx` from becoming a folder: descend only when nothing claims the container's specific mimetype above the archive handler. Until Spec 9 lands, that is spelled as the explicit opt-out sets from Task 2.

**Files:**
- Modify: `src/readeverything/adapters/nested_source.py` (replace `walk`)
- Test: `tests/unit/adapters/test_nested_source.py` (append)

**Interfaces:**
- Consumes: everything from Task 5, plus `NOT_A_FOLDER_MIMES` and `NOT_A_FOLDER_SUFFIXES` from `ports.containers`.
- Produces: no new names; `NestedSource.walk` gains inline members.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/adapters/test_nested_source.py`:

```python
async def test_walk_returns_members_inline(tmp_path: Path) -> None:
    (tmp_path / "loose.txt").write_bytes(b"x")
    _zip(tmp_path, "docs.zip", {"report.txt": b"y"})
    assert sorted(await _nested(tmp_path).walk(".")) == [
        "docs.zip",
        "docs.zip!report.txt",
        "loose.txt",
    ]


async def test_walk_recurses_into_nested_containers(tmp_path: Path) -> None:
    """The §1.1 acceptance addressing, produced by `walk` alone."""
    build = tmp_path / "build"
    build.mkdir()
    _targz(build, "nested.tar.gz", {"notes.txt": b"deep"})
    _zip(tmp_path, "docs.zip", {"nested.tar.gz": (build / "nested.tar.gz").read_bytes()})
    (build / "nested.tar.gz").unlink()
    build.rmdir()
    source = _nested(tmp_path)
    try:
        assert sorted(await source.walk(".")) == [
            "docs.zip",
            "docs.zip!nested.tar.gz",
            "docs.zip!nested.tar.gz!notes.txt",
        ]
    finally:
        await source.aclose()


async def test_walk_does_not_return_directory_entries(tmp_path: Path) -> None:
    path = tmp_path / "d.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sub/", b"")
        archive.writestr("sub/a.txt", b"hi")
    assert sorted(await _nested(tmp_path).walk(".")) == ["d.zip", "d.zip!sub/a.txt"]


async def test_walk_escapes_a_literal_separator_in_a_member_name(tmp_path: Path) -> None:
    _zip(tmp_path, "a.zip", {"od!d.txt": b"x"})
    assert "a.zip!od!!d.txt" in await _nested(tmp_path).walk(".")


async def test_walk_stops_at_max_depth(tmp_path: Path) -> None:
    build = tmp_path / "build"
    build.mkdir()
    _targz(build, "nested.tar.gz", {"notes.txt": b"deep"})
    _zip(tmp_path, "docs.zip", {"nested.tar.gz": (build / "nested.tar.gz").read_bytes()})
    (build / "nested.tar.gz").unlink()
    build.rmdir()
    source = _nested(tmp_path, ContainerLimits(max_depth=1))
    try:
        assert sorted(await source.walk(".")) == ["docs.zip", "docs.zip!nested.tar.gz"]
    finally:
        await source.aclose()


async def test_walk_members_false_restores_the_old_behavior(tmp_path: Path) -> None:
    """Exactly today's output, and exactly today's number of opens."""
    _zip(tmp_path, "docs.zip", {"report.txt": b"y"})
    source = _nested(tmp_path, ContainerLimits(walk_members=False))
    assert sorted(await source.walk(".")) == ["docs.zip"]


async def test_walk_does_not_descend_into_a_docx(tmp_path: Path) -> None:
    """A .docx is a zip and is not a folder. Descending would list a dozen XML
    parts and bury the document itself."""
    with zipfile.ZipFile(tmp_path / "report.docx", "w") as archive:
        archive.writestr("[Content_Types].xml", b"<Types/>")
        archive.writestr("word/document.xml", b"<w:document/>")
    assert sorted(await _nested(tmp_path).walk(".")) == ["report.docx"]


async def test_walk_does_not_descend_into_an_epub_or_a_jar(tmp_path: Path) -> None:
    for name in ("book.epub", "lib.jar"):
        with zipfile.ZipFile(tmp_path / name, "w") as archive:
            archive.writestr("inner.txt", b"x")
    assert sorted(await _nested(tmp_path).walk(".")) == ["book.epub", "lib.jar"]


async def test_walk_survives_a_corrupt_archive(tmp_path: Path) -> None:
    """One unreadable archive must not blind the agent to its neighbours."""
    (tmp_path / "broken.zip").write_bytes(b"PK\x03\x04 and then garbage")
    (tmp_path / "fine.txt").write_bytes(b"x")
    assert sorted(await _nested(tmp_path).walk(".")) == ["broken.zip", "fine.txt"]


async def test_walk_survives_an_archive_over_its_limits(tmp_path: Path) -> None:
    _zip(tmp_path, "many.zip", {f"f{n}.txt": b"x" for n in range(20)})
    (tmp_path / "fine.txt").write_bytes(b"x")
    source = _nested(tmp_path, ContainerLimits(max_members=5))
    assert sorted(await source.walk(".")) == ["fine.txt", "many.zip"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --all-extras pytest tests/unit/adapters/test_nested_source.py -x -q -k walk`
Expected: FAIL — `walk` returns only the top-level files.

- [ ] **Step 3: Write the implementation**

In `src/readeverything/adapters/nested_source.py`, add the import:

```python
from readeverything.ports.containers import (
    ArchiveEntry,
    ArchiveOpener,
    ContainerLimits,
    NOT_A_FOLDER_MIMES,
    NOT_A_FOLDER_SUFFIXES,
)
```

and replace `walk` with:

```python
    async def _is_a_folder(self, uri: str, path: str) -> ArchiveOpener | None:
        """The opener to descend with, or None because this is not a folder.

        A container is not always a folder. A `.docx`, `.pptx`, `.xlsx`,
        `.odt`, `.epub` and `.jar` are all zip files, and descending into one
        would list `report.docx!word/document.xml` as a source -- which is
        worse than useless, because it buries the document itself under a
        dozen XML parts.

        The general rule is that `walk` descends only when no handler claims
        the container's specific mimetype above the archive handler. Spec 9
        makes detection report OOXML and ODF as their own types, at which
        point those handlers claim the file and it stops being a folder for
        the general reason. The explicit sets consulted here are what keep the
        behavior CORRECT in the interim rather than briefly wrong, and the
        suffix set is needed alongside the mimetype set because detection is
        content-first: the bytes of a `.docx` genuinely are a zip.
        """
        if PurePosixPath(uri).suffix.lower() in NOT_A_FOLDER_SUFFIXES:
            return None
        try:
            head = await asyncio.to_thread(_read_head, path)
        except OSError:
            return None
        mime = await self._detector.detect(uri, head)
        if str(mime) in NOT_A_FOLDER_MIMES:
            return None
        chooser = getattr(self._archives, "opener_for", None)
        opener = chooser(mime) if chooser is not None else None
        if opener is None and self._archives.claims(mime):
            opener = self._archives
        return opener

    async def _members_of(self, uri: str, path: str, depth: int) -> list[str]:
        """Every source inside the container at `uri`, recursively.

        Every failure here is swallowed and returns what was found so far.
        One corrupt or oversized archive in a directory must not blind the
        agent to its neighbours -- that is the §1.1 acceptance, and it is the
        difference between a degraded listing and no listing.
        """
        if depth >= self._limits.max_depth:
            return []
        opener = await self._is_a_folder(uri, path)
        if opener is None:
            return []
        try:
            entries = await self._entries(opener, path, uri)
        except SourceUnreadableError:
            return []
        found: list[str] = []
        for entry in entries:
            if entry.is_dir or entry.is_symlink:
                # A directory is not a source (matching `LocalFileSource.walk`)
                # and a symlink is reported by the archive card, never walked
                # into -- following one is the tar-specific hole.
                continue
            try:
                member_uri = join_uri((*split_uri(uri), entry.path))
            except ValueError:
                continue
            found.append(member_uri)
            try:
                inner_path = await self._materialise(member_uri)
            except (SourceUnreadableError, OSError):
                continue
            found.extend(await self._members_of(member_uri, inner_path, depth + 1))
        return found

    async def walk(self, uri: str) -> Sequence[str]:
        """Everything under `uri`, with container members listed inline.

        This is the decision that makes every downstream feature free. Because
        `pipeline.perception` walks and then inspects, and inspection
        dispatches on detected mimetype, an archived PDF reaches the PDF
        handler with no registry change, no handler change and no special case
        anywhere above this layer.

        The cost, stated rather than hidden: this now reads every archive's
        central directory -- a seek and a small read per archive, not a
        decompression, but on a directory of ten thousand zips it is ten
        thousand extra opens. `ContainerLimits.walk_members` turns it off.
        """
        found = list(await self._inner.walk(uri))
        if not self._limits.walk_members:
            return found
        results = list(found)
        for entry_uri in found:
            try:
                path = await self._inner.local_path(entry_uri)
            except SourceUnreadableError:
                continue
            results.extend(await self._members_of(entry_uri, path, 0))
        return results
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --all-extras pytest tests/unit/adapters/test_nested_source.py -x -q`
Expected: PASS (all of Task 5's tests included)

- [ ] **Step 5: Lint and type-check**

Run:
```bash
uv run --all-extras ruff check src/readeverything/adapters/nested_source.py tests/unit/adapters/test_nested_source.py
uv run --all-extras ruff format src/readeverything/adapters/nested_source.py tests/unit/adapters/test_nested_source.py
uv run --all-extras mypy src/readeverything/adapters/nested_source.py
```
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/readeverything/adapters/nested_source.py tests/unit/adapters/test_nested_source.py
git commit -m "feat(adapters): walk lists container members inline, docx excepted"
```

---

### Task 7: The archive handler

The container still deserves a card: an agent that lists a directory should learn what is in `release.tar.gz` without descending into it.

**`list_entries` is paged and there is no `read_entry`.** Paging because a 40,000-entry tarball is not one response. No `read_entry` because reading a member is spelled `inspect("a.zip!inner.txt")`, and two ways to reach the same bytes would mean two provenance stories for one citation — the failure this library exists to prevent.

**Files:**
- Create: `src/readeverything/handlers/archive.py`
- Test: `tests/unit/handlers/test_archive.py`

**Interfaces:**
- Consumes: `ArchiveEntry`, `ArchiveOpener`, `ARCHIVE_MIMES` from `ports.containers`; `SourceReader` from `ports.source`.
- Produces:
  - `class ListEntriesParams(BaseModel)` with `offset: int = 0` (ge=0) and `limit: int = 200` (ge=1, le=2000)
  - `class ArchiveHandler` with `__init__(self, *, source: SourceReader, archives: ArchiveOpener, observer: Observer | None = None) -> None`, `mime_patterns`, `priority = 0`, `handler_id = "archive"`, `handler_version = 1`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/handlers/test_archive.py`:

```python
"""The archive card, its paging, and the compliance laws."""

import zipfile
from pathlib import Path

import pytest

from readeverything.adapters.nested_source import CompositeOpener
from readeverything.adapters.tar_archive import TarArchiveOpener
from readeverything.adapters.zip_archive import ZipArchiveOpener
from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
from readeverything.domain.rendition import Budget
from readeverything.handlers.archive import ArchiveHandler, ListEntriesParams
from readeverything.testing.handler_compliance import MediaHandlerCompliance


def _archive_bytes(tmp_path: Path, members: dict[str, bytes]) -> bytes:
    path = tmp_path / "built.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path.read_bytes()


class _PathSource:
    """A `SourceReader` that writes whatever it is asked for to a real file.

    The archive handler needs a PATH, because `ArchiveOpener` takes one -- so
    a fake that only serves bytes cannot exercise it. Serving the same content
    at any uri is also what `MediaHandlerCompliance` requires.
    """

    def __init__(self, *, content: bytes, root: Path) -> None:
        self._content = content
        self._root = root
        self._paths: dict[str, str] = {}

    async def read_bytes(self, uri: str) -> bytes:
        return self._content

    async def read_range(self, uri: str, start: int, end: int) -> bytes:
        return self._content[start:end]

    async def stream(self, uri: str, *, chunk_size: int = 1 << 20):  # noqa: ANN201
        yield self._content

    async def local_path(self, uri: str) -> str:
        existing = self._paths.get(uri)
        if existing is None:
            target = self._root / f"{len(self._paths)}.bin"
            target.write_bytes(self._content)
            existing = str(target)
            self._paths[uri] = existing
        return existing


def _openers() -> CompositeOpener:
    return CompositeOpener(openers=[ZipArchiveOpener(), TarArchiveOpener()])


def _ref(size: int) -> SourceRef:
    return SourceRef(
        uri="release.zip",
        mime=MimeType.parse("application/zip"),
        content_hash=ContentHash("0" * 8),
        size_bytes=size,
    )


@pytest.fixture
def built(tmp_path: Path) -> bytes:
    return _archive_bytes(tmp_path, {"a.txt": b"hello", "sub/b.txt": b"world"})


@pytest.fixture
def handler(tmp_path: Path, built: bytes) -> ArchiveHandler:
    return ArchiveHandler(source=_PathSource(content=built, root=tmp_path), archives=_openers())


async def test_the_card_is_binary_and_counts_entries(
    handler: ArchiveHandler, built: bytes
) -> None:
    card = await handler.describe(_ref(len(built)))
    assert card.kind is MediaKind.BINARY
    assert card.facts["entry_count"] == 2
    assert card.facts["format"] == "application/zip"


async def test_the_card_reports_the_expansion_ratio(
    handler: ArchiveHandler, built: bytes
) -> None:
    card = await handler.describe(_ref(len(built)))
    assert float(card.facts["expansion_ratio"]) > 0


async def test_the_card_outlines_every_entry(handler: ArchiveHandler, built: bytes) -> None:
    card = await handler.describe(_ref(len(built)))
    assert [segment.label for segment in card.outline] == ["a.txt", "sub/b.txt"]


async def test_the_card_excerpts_the_first_member_paths(
    handler: ArchiveHandler, built: bytes
) -> None:
    card = await handler.describe(_ref(len(built)))
    assert card.excerpt is not None
    assert "a.txt" in card.excerpt


async def test_the_only_affordance_is_list_entries(handler: ArchiveHandler) -> None:
    """No `read_entry`: a member is reached as `inspect('a.zip!inner.txt')`,
    and two ways to the same bytes means two provenance stories."""
    assert [a.name for a in handler.affordances()] == ["list_entries"]


async def test_list_entries_pages(tmp_path: Path) -> None:
    built = _archive_bytes(tmp_path, {f"f{n:02d}.txt": b"x" for n in range(10)})
    handler = ArchiveHandler(
        source=_PathSource(content=built, root=tmp_path), archives=_openers()
    )
    rendition = await handler.invoke(
        _ref(len(built)), "list_entries", ListEntriesParams(offset=2, limit=3)
    )
    body = rendition.content.text  # type: ignore[union-attr]
    assert "f02.txt" in body and "f04.txt" in body
    assert "f05.txt" not in body and "f01.txt" not in body


async def test_list_entries_past_the_end_degrades_rather_than_raising(
    handler: ArchiveHandler, built: bytes
) -> None:
    rendition = await handler.invoke(
        _ref(len(built)), "list_entries", ListEntriesParams(offset=999, limit=10)
    )
    assert rendition.degraded


async def test_a_corrupt_archive_degrades_rather_than_raising(tmp_path: Path) -> None:
    """A handler never raises about its input, however broken."""
    handler = ArchiveHandler(
        source=_PathSource(content=b"PK\x03\x04 garbage", root=tmp_path), archives=_openers()
    )
    card = await handler.describe(_ref(15))
    assert card.facts["readable"] == "no"
    rendered = await handler.represent(_ref(15), Budget())
    assert rendered.degradations


async def test_represent_maps_every_character_to_an_entry(
    handler: ArchiveHandler, built: bytes
) -> None:
    rendered = await handler.represent(_ref(len(built)), Budget())
    assert rendered.locator_map.length == len(rendered.text)
    assert rendered.barriers == ()


async def test_represent_honours_the_budget(handler: ArchiveHandler, built: bytes) -> None:
    rendered = await handler.represent(_ref(len(built)), Budget(max_chars=5))
    assert len(rendered.text) == 5
    assert any(d.what == "text truncated" for d in rendered.degradations)


async def test_an_unknown_affordance_raises(handler: ArchiveHandler, built: bytes) -> None:
    from readeverything.domain.errors import UnknownAffordanceError

    with pytest.raises(UnknownAffordanceError):
        await handler.invoke(_ref(len(built)), "read_entry", ListEntriesParams())


class TestArchiveHandlerCompliance(MediaHandlerCompliance):
    @pytest.fixture
    def content(self, built: bytes) -> bytes:
        return built

    @pytest.fixture
    def handler(self, handler: ArchiveHandler) -> ArchiveHandler:  # noqa: F811
        return handler
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --all-extras pytest tests/unit/handlers/test_archive.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'readeverything.handlers.archive'`

Before implementing, read `src/readeverything/testing/handler_compliance.py` in full and check what the `handler`/`content` fixtures are named and what `MediaHandlerCompliance` requires of a source (it describes a ref at `"somewhere/else"`, which `_PathSource` above serves). If the fixture override shape above conflicts with the base class, rename the local fixture and reference it — do not weaken the compliance suite.

- [ ] **Step 3: Write the implementation**

Create `src/readeverything/handlers/archive.py`:

```python
"""Containers, as things in their own right.

An agent that lists a directory should learn what is in `release.tar.gz`
without descending into it, so the container gets a card of its own even
though `adapters/nested_source.py` is what makes its members readable.

`kind` is `BINARY` for the reason the README already gives for PDF:
`MediaKind` names how bytes are SHAPED, and a container's shape is binary.
What it *is* is carried by its facts and its affordances.

There is exactly one affordance and it is paged. Paging because a
40,000-entry tarball is not one response. One, because reading a member is
spelled `inspect("a.zip!inner.txt")` -- adding a `read_entry` here would give
one sequence of bytes two ways to be reached, and therefore two provenance
stories for one citation, which is the failure this library exists to prevent.

Card cost stays inside the contract: reading a zip's central directory or
walking tar headers is a probe, not a decompression.
"""

from __future__ import annotations

import time
from typing import ClassVar

from pydantic import BaseModel, Field

from readeverything.domain.affordance import Affordance, DetailLevel
from readeverything.domain.capability import Capability
from readeverything.domain.card import Card, Segment
from readeverything.domain.errors import SourceUnreadableError, UnknownAffordanceError
from readeverything.domain.identity import MediaKind, SourceRef
from readeverything.domain.locator_map import LocatorMap, LocatorSegment
from readeverything.domain.locators import ByteRange, CharSpan, Locator
from readeverything.domain.observation import OperationFinished, OperationStarted
from readeverything.domain.rendition import (
    Budget,
    Degradation,
    Rendered,
    Rendition,
    TextContent,
)
from readeverything.ports.containers import ARCHIVE_MIMES, ArchiveEntry, ArchiveOpener
from readeverything.ports.observation import Observer, emit
from readeverything.ports.source import SourceReader

#: What `represent` calls itself when it narrates. Matches the name every
#: other handler uses, per `video.py`.
_OPERATION = "represent"

#: How many member paths the card shows. A card is what a human skims, and a
#: skim of a 40,000-entry tarball is its first few names, not all of them.
_EXCERPT_ENTRIES = 20


class ListEntriesParams(BaseModel):
    offset: int = Field(default=0, ge=0, description="0-indexed entry to start from.")
    limit: int = Field(default=200, ge=1, le=2000, description="How many entries to return.")


def _line(entry: ArchiveEntry) -> str:
    kind = "dir " if entry.is_dir else ("link" if entry.is_symlink else "file")
    return f"{kind} {entry.size_bytes:>12}  {entry.path}"


class ArchiveHandler:
    """Describes a container without descending into it."""

    mime_patterns: ClassVar[tuple[str, ...]] = tuple(sorted(ARCHIVE_MIMES))
    priority: ClassVar[int] = 0
    handler_id: ClassVar[str] = "archive"
    handler_version: ClassVar[int] = 1

    def __init__(
        self,
        *,
        source: SourceReader,
        archives: ArchiveOpener,
        observer: Observer | None = None,
    ) -> None:
        self._source = source
        self._archives = archives
        self._observer = observer

    def requires(self) -> frozenset[Capability]:
        """Nothing. zipfile and tarfile are stdlib; there is no binary here."""
        return frozenset()

    def affordances(self) -> tuple[Affordance, ...]:
        return (
            Affordance(
                name="list_entries",
                description=(
                    "List the members of this container, a page at a time. "
                    "To READ a member, inspect it directly at "
                    "'<this uri>!<member path>' — that is the only way to reach "
                    "its bytes, and it gives you the member's own card and "
                    "affordances rather than a blob."
                ),
                params=ListEntriesParams,
                requires=frozenset(),
                level=DetailLevel.SEGMENT,
            ),
        )

    async def _entries(self, ref: SourceRef) -> tuple[ArchiveEntry, ...] | None:
        """The container's directory, or None when it could not be opened.

        None rather than an exception: this handler never raises about its
        input, matching `PdfHandler._open`. The source layer is where an
        unreadable container raises; a handler reports.
        """
        try:
            path = await self._source.local_path(ref.uri)
            return tuple(await self._archives.entries(path))
        except (SourceUnreadableError, OSError, NotImplementedError):
            return None

    async def describe(self, ref: SourceRef) -> Card:
        entries = await self._entries(ref)
        if entries is None:
            return Card(
                ref=ref,
                kind=MediaKind.BINARY,
                facts={"readable": "no", "size_bytes": ref.size_bytes},
                outline=(),
                excerpt=None,
                affordances=self.affordances(),
            )
        uncompressed = sum(entry.size_bytes for entry in entries)
        compressed = sum(entry.compressed_bytes for entry in entries)
        # A solid container is one whose members have no seekable place in the
        # file. That is the fact a caller needs to predict what reading three
        # members will cost, so it is on the card rather than inferred.
        solid = all(entry.byte_offset is None for entry in entries) if entries else False
        return Card(
            ref=ref,
            kind=MediaKind.BINARY,
            facts={
                "readable": "yes",
                "format": str(ref.mime),
                "entry_count": len(entries),
                "uncompressed_bytes": uncompressed,
                "compressed_bytes": compressed,
                "expansion_ratio": round(uncompressed / compressed, 2) if compressed else 0.0,
                "solid": "yes" if solid else "no",
                "size_bytes": ref.size_bytes,
            },
            outline=tuple(
                Segment(self._locator(entry, index), entry.path)
                for index, entry in enumerate(entries)
            ),
            excerpt="\n".join(entry.path for entry in entries[:_EXCERPT_ENTRIES]) or None,
            affordances=self.affordances(),
        )

    def _locator(self, entry: ArchiveEntry, index: int) -> Locator:
        """Where an entry is, in whatever terms the format actually supports.

        A `ByteRange` when the container gives an offset. A solid container
        gives none -- an offset into a gzip stream is not somewhere anyone can
        seek to -- so those fall back to the entry's line in the listing this
        handler itself produces, which is a place that genuinely exists rather
        than an invented byte range.
        """
        if entry.byte_offset is None:
            return CharSpan(index, index + 1)
        return ByteRange(entry.byte_offset, entry.byte_offset + max(1, entry.compressed_bytes))

    async def invoke(self, ref: SourceRef, name: str, params: BaseModel) -> Rendition:
        if name != "list_entries":
            raise UnknownAffordanceError(name, (a.name for a in self.affordances()))
        if not isinstance(params, ListEntriesParams):
            raise TypeError(f"expected ListEntriesParams, got {type(params).__name__}")
        entries = await self._entries(ref)
        if entries is None:
            return Rendition(
                locator=ByteRange(0, max(1, ref.size_bytes)),
                content=TextContent(f"{ref.uri} could not be opened as an archive"),
                degraded=True,
            )
        page = entries[params.offset : params.offset + params.limit]
        if not page:
            return Rendition(
                locator=ByteRange(0, max(1, ref.size_bytes)),
                content=TextContent(
                    f"offset {params.offset} is past the end; "
                    f"this container has {len(entries)} entry(ies)"
                ),
                degraded=True,
            )
        header = f"{len(page)} of {len(entries)} entries, from offset {params.offset}\n"
        body = header + "\n".join(_line(entry) for entry in page)
        return Rendition(locator=ByteRange(0, max(1, ref.size_bytes)), content=TextContent(body))

    async def represent(self, ref: SourceRef, budget: Budget) -> Rendered:
        """Narrated start to finish, matching every other handler."""
        emit(self._observer, OperationStarted(operation=_OPERATION, ref=ref))
        started = time.perf_counter()
        try:
            return await self._represent(ref, budget)
        finally:
            emit(
                self._observer,
                OperationFinished(
                    operation=_OPERATION, ref=ref, elapsed_s=time.perf_counter() - started
                ),
            )

    async def _represent(self, ref: SourceRef, budget: Budget) -> Rendered:
        """The entry listing as text, with a locator over every line.

        `barriers` stays empty: an entry listing has no natural chunk
        boundary, and inventing one would tell a chunker something about this
        text that is not true of it.
        """
        entries = await self._entries(ref)
        if not entries:
            summary = (
                f"Unreadable archive {ref.uri}, {ref.size_bytes} bytes."
                if entries is None
                else f"Archive {ref.uri} contains no entries."
            )
            detail = (
                "the file could not be opened as an archive; no entries were listed"
                if entries is None
                else "the archive opened and declares no members"
            )
            return self._fit(
                summary,
                (LocatorSegment(CharSpan(0, len(summary)), ByteRange(0, max(1, ref.size_bytes))),),
                budget,
                (Degradation(what="archive unlistable", detail=detail),),
            )
        chunks: list[str] = []
        segments: list[LocatorSegment] = []
        cursor = 0
        for index, entry in enumerate(entries):
            # The trailing newline is INSIDE the segment, exactly as
            # `pdf.PAGE_SEPARATOR` is inside a page's: `LocatorMap` demands
            # gapless zero-start coverage and `CharSpan` rejects a zero-width
            # span, so a separator owned by nobody is what breaks the map.
            chunk = _line(entry) + "\n"
            segments.append(
                LocatorSegment(CharSpan(cursor, cursor + len(chunk)), self._locator(entry, index))
            )
            cursor += len(chunk)
            chunks.append(chunk)
        return self._fit("".join(chunks), tuple(segments), budget, ())

    def _fit(
        self,
        full: str,
        segments: tuple[LocatorSegment, ...],
        budget: Budget,
        degradations: tuple[Degradation, ...],
    ) -> Rendered:
        """Apply the budget, pruning the map along with the text.

        `Rendered` rejects a map that does not cover its text exactly, so
        truncation cannot touch the text alone. A budget of zero still keeps
        one character, because `CharSpan(0, 0)` raises.
        """
        if budget.max_chars is None or len(full) <= budget.max_chars:
            return Rendered(
                text=full,
                locator_map=LocatorMap.build(segments),
                barriers=(),
                degradations=degradations,
            )
        keep = max(1, budget.max_chars)
        kept = tuple(
            LocatorSegment(CharSpan(s.span.start, min(s.span.end, keep)), s.locator)
            for s in segments
            if s.span.start < keep
        )
        return Rendered(
            text=full[:keep],
            locator_map=LocatorMap.build(kept),
            barriers=(),
            degradations=(
                *degradations,
                Degradation(
                    what="text truncated",
                    detail=f"kept {keep} of {len(full)} characters",
                ),
            ),
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --all-extras pytest tests/unit/handlers/test_archive.py -x -q`
Expected: PASS

- [ ] **Step 5: Lint and type-check**

Run:
```bash
uv run --all-extras ruff check src/readeverything/handlers/archive.py tests/unit/handlers/test_archive.py
uv run --all-extras ruff format src/readeverything/handlers/archive.py tests/unit/handlers/test_archive.py
uv run --all-extras mypy src/readeverything/handlers/archive.py
```
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/readeverything/handlers/archive.py tests/unit/handlers/test_archive.py
git commit -m "feat(handlers): the archive card, with paged list_entries and no read_entry"
```

---

### Task 8: Composition, exports, README and the acceptance test

Wires it together and proves §1.1 end to end. **Every edit to `composition.py` must be additive** — a Spec 9 agent is editing the same file concurrently.

**Files:**
- Modify: `src/readeverything/composition.py`
- Modify: `src/readeverything/__init__.py`
- Modify: `README.md`
- Test: `tests/integration/test_containers.py`

**Interfaces:**
- Consumes: everything from Tasks 1–7.
- Produces: `build_perception(..., containers: ContainerLimits | None = ContainerLimits(), archives: ArchiveOpener | None = None, ...)`.

**Note on the default:** the spec §7 signature writes `containers: ContainerLimits | None = None` with prose saying "`None` disables descent entirely" and "The default is a `ContainerLimits()` with §3.3's values — descent on by default". Those two sentences cannot both be satisfied by a `None` default, so the plan follows the *behavioral* requirement: the parameter defaults to `ContainerLimits()`, and passing `None` explicitly disables descent and yields today's behavior exactly.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_containers.py`:

```python
"""The §1.1 acceptance, end to end, with fixtures built in a tmpdir.

The point of this file is the thing it does NOT do: no handler was modified to
make any of it pass. The PDF handler descends into a tarball inside a zip
because it reads through `SourceReader` and cannot tell where its bytes came
from.
"""

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from readeverything.composition import build_perception
from readeverything.domain.capability import CapabilitySet
from readeverything.domain.rendition import Budget, TextContent
from readeverything.ports.containers import ContainerLimits

pytestmark = pytest.mark.integration

pdfium = pytest.importorskip("pypdfium2")


def _pdf(pages: int) -> bytes:
    document = pdfium.PdfDocument.new()
    try:
        for _ in range(pages):
            document.new_page(200, 200)
        buffer = io.BytesIO()
        document.save(buffer)
        return buffer.getvalue()
    finally:
        document.close()


def _targz(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """`docs.zip` holding `report.pdf` and `nested.tar.gz` holding `notes.txt`."""
    with zipfile.ZipFile(tmp_path / "docs.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.pdf", _pdf(9))
        archive.writestr("nested.tar.gz", _targz({"notes.txt": b"the note body\n"}))
    return tmp_path


async def test_list_returns_members_addressed_by_the_grammar(root: Path) -> None:
    perception = await build_perception(root, capabilities=CapabilitySet())
    listed = sorted(await perception.list("."))
    assert listed == [
        "docs.zip",
        "docs.zip!nested.tar.gz",
        "docs.zip!nested.tar.gz!notes.txt",
        "docs.zip!report.pdf",
    ]


async def test_inspecting_an_archived_pdf_gives_a_pdf_card(root: Path) -> None:
    """No registry change, no handler change. This is the whole spec."""
    perception = await build_perception(root, capabilities=CapabilitySet())
    card = await perception.inspect("docs.zip!report.pdf")
    assert card.facts["page_count"] == 9
    assert "read_page" in card.affordance_names()


async def test_inspecting_a_doubly_nested_text_member_gives_a_text_card(root: Path) -> None:
    perception = await build_perception(root, capabilities=CapabilitySet())
    card = await perception.inspect("docs.zip!nested.tar.gz!notes.txt")
    assert "read_range" in card.affordance_names()
    assert card.excerpt is not None and "the note body" in card.excerpt


async def test_page_seven_of_the_nested_pdf_cites_the_full_nested_path(root: Path) -> None:
    perception = await build_perception(root, capabilities=CapabilitySet())
    rendition = await perception.invoke("docs.zip!report.pdf", "read_page", {"page": 7})
    assert isinstance(rendition.content, TextContent)
    card = await perception.inspect("docs.zip!report.pdf")
    assert card.ref.uri == "docs.zip!report.pdf"


async def test_the_container_itself_still_gets_an_archive_card(root: Path) -> None:
    perception = await build_perception(root, capabilities=CapabilitySet())
    card = await perception.inspect("docs.zip")
    assert card.facts["entry_count"] == 2
    assert card.affordance_names() == ("list_entries",)


async def test_a_zip_bomb_is_refused_with_a_bounded_error(tmp_path: Path) -> None:
    """Refused, not truncated: half a file reported on as whole is the harm."""
    from readeverything.domain.errors import ContainerLimitExceededError

    with zipfile.ZipFile(tmp_path / "bomb.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("payload", b"\0" * (1 << 22))
    perception = await build_perception(
        tmp_path,
        capabilities=CapabilitySet(),
        containers=ContainerLimits(max_expansion_ratio=2.0),
    )
    with pytest.raises(ContainerLimitExceededError):
        await perception.inspect("bomb.zip!payload")


async def test_a_corrupt_member_does_not_blind_the_agent_to_its_neighbours(
    tmp_path: Path,
) -> None:
    (tmp_path / "broken.zip").write_bytes(b"PK\x03\x04 and then garbage")
    (tmp_path / "fine.txt").write_bytes(b"readable\n")
    perception = await build_perception(tmp_path, capabilities=CapabilitySet())
    assert sorted(await perception.list(".")) == ["broken.zip", "fine.txt"]
    card = await perception.inspect("fine.txt")
    assert card.excerpt is not None and "readable" in card.excerpt


async def test_containers_none_yields_todays_behavior(root: Path) -> None:
    perception = await build_perception(root, capabilities=CapabilitySet(), containers=None)
    assert sorted(await perception.list(".")) == ["docs.zip"]


async def test_representing_the_container_lists_its_entries(root: Path) -> None:
    perception = await build_perception(root, capabilities=CapabilitySet())
    rendered = await perception.represent("docs.zip", Budget())
    assert "report.pdf" in rendered.text
    assert rendered.locator_map.length == len(rendered.text)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --all-extras pytest tests/integration/test_containers.py -x -q`
Expected: FAIL — `build_perception() got an unexpected keyword argument 'containers'`

- [ ] **Step 3: Write the implementation**

In `src/readeverything/composition.py`, add these imports alongside the existing ones (alphabetical position, nothing reordered):

```python
from readeverything.adapters.nested_source import CompositeOpener, NestedSource
from readeverything.adapters.tar_archive import TarArchiveOpener
from readeverything.adapters.zip_archive import ZipArchiveOpener
from readeverything.handlers.archive import ArchiveHandler
from readeverything.ports.containers import ArchiveOpener, ContainerLimits
from readeverything.ports.source import FileSource
```

Add a helper above `build_perception`:

```python
def _source_for(
    root: Path | str,
    containers: ContainerLimits | None,
    archives: ArchiveOpener | None,
) -> FileSource:
    """The local source, wrapped in `NestedSource` unless descent is off.

    `containers=None` yields today's behavior EXACTLY, including no extra
    opens during `walk` -- the decorator is not constructed at all, rather
    than constructed and told to do nothing, so there is no code path to
    regress.
    """
    source = LocalFileSource(root=root)
    if containers is None:
        return source
    openers = (
        CompositeOpener(
            openers=[
                ZipArchiveOpener(),
                TarArchiveOpener(max_materialised_bytes=containers.max_materialised_bytes),
            ]
        )
        if archives is None
        else archives
    )
    return NestedSource(
        source,
        limits=containers,
        archives=openers,
        detector=PuremagicDetector(),
    )
```

Add the two keyword arguments to `build_perception`'s signature, after `probe_binaries`:

```python
    containers: ContainerLimits | None = ContainerLimits(),
    archives: ArchiveOpener | None = None,
```

Append to `build_perception`'s docstring:

```
    `containers` controls descent into archives. It defaults to
    `ContainerLimits()` -- descent ON, because a library whose promise is
    "read everything" should read the tarball. Passing `None` disables it and
    yields today's behavior exactly, including no extra opens during `walk`.
    `archives` overrides the bundled zip and tar openers, which is the
    extension point for `.7z` or `.rar` without this repository growing a
    dependency on either.
```

Replace the `source = LocalFileSource(root=root)` line in `build_perception` with:

```python
    source = _source_for(root, containers, archives)
```

and add `ArchiveHandler` to the `handlers` list, immediately **before** `BinaryHandler` (it must stay last, per its own comment):

```python
        ArchiveHandler(source=source, archives=_archive_opener(containers, archives)),
```

To avoid constructing the openers twice, restructure minimally: have `_source_for` return `tuple[FileSource, ArchiveOpener]` and unpack it. Concretely, rename it `_source_and_openers`, returning `(source, openers)` where `openers` is the composite even when `containers is None` (the handler still describes archives when descent is off — a card is not a descent). Use:

```python
    source, openers = _source_and_openers(root, containers, archives)
```

and register `ArchiveHandler(source=source, archives=openers, observer=observer)`.

In `src/readeverything/__init__.py`, add to the `TYPE_CHECKING` block and to `_LAZY` (both alphabetically; `__all__` is `sorted(_LAZY)` and `test_all_is_sorted_and_unique` will fail otherwise):

```python
    "ArchiveEntry": "readeverything.ports.containers",
    "ArchiveHandler": "readeverything.handlers.archive",
    "ArchiveOpener": "readeverything.ports.containers",
    "CompositeOpener": "readeverything.adapters.nested_source",
    "ContainerLimitExceededError": "readeverything.domain.errors",
    "ContainerLimits": "readeverything.ports.containers",
    "NestedSource": "readeverything.adapters.nested_source",
    "TarArchiveOpener": "readeverything.adapters.tar_archive",
    "ZipArchiveOpener": "readeverything.adapters.zip_archive",
    "container_of": "readeverything.domain.container_uri",
    "join_uri": "readeverything.domain.container_uri",
    "split_uri": "readeverything.domain.container_uri",
```

with matching `from ... import X as X` lines in the `TYPE_CHECKING` block.

In `README.md`, add a section after whichever section covers listing a directory:

````markdown
### Descending into containers

A zip, a tarball and a `.tar.gz` are directories as far as the library is
concerned. Members are addressed with `!`:

```python
perception = await build_perception("./corpus")
await perception.list(".")
# ['docs.zip', 'docs.zip!report.pdf', 'docs.zip!nested.tar.gz',
#  'docs.zip!nested.tar.gz!notes.txt']

card = await perception.inspect("docs.zip!report.pdf")
card.facts["page_count"]        # 9 — a real PDF card, from inside the zip
await perception.invoke("docs.zip!report.pdf", "read_page", {"page": 7})
```

Nothing in the PDF handler knows it is inside an archive: every handler reads
bytes through a port and cannot tell where they came from.

A literal `!` in a member name is escaped `!!`. Descent is bounded by
`ContainerLimits` — depth, member size, total size, member count and, the one
that matters, an expansion ratio checked *while* decompressing, so a zip bomb
is refused rather than filling a disk:

```python
from readeverything import ContainerLimits

await build_perception("./corpus", containers=ContainerLimits(max_depth=1))
await build_perception("./corpus", containers=None)  # today's behavior, no descent
```

A `.docx`, `.epub` or `.jar` is a zip too, and is deliberately *not* treated as
a folder: descending would bury the document under a dozen XML parts.
`.7z` and `.rar` are not supported, because each needs a dependency this
library does not take — supply your own `ArchiveOpener` via `archives=`.
````

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
uv run --all-extras pytest tests/integration/test_containers.py tests/unit/test_public_surface.py tests/unit/test_composition.py tests/unit/test_reads_no_environment.py -x -q
```
Expected: PASS

- [ ] **Step 5: Lint and type-check**

Run:
```bash
uv run --all-extras ruff check src/readeverything/composition.py src/readeverything/__init__.py tests/integration/test_containers.py
uv run --all-extras ruff format src/readeverything/composition.py src/readeverything/__init__.py tests/integration/test_containers.py
uv run --all-extras mypy src/readeverything/composition.py src/readeverything/__init__.py
```
Expected: clean

- [ ] **Step 6: Re-run every test this plan created**

Run:
```bash
uv run --all-extras pytest \
  tests/unit/domain/test_container_uri.py \
  tests/unit/ports/test_containers.py \
  tests/unit/adapters/test_zip_archive.py \
  tests/unit/adapters/test_tar_archive.py \
  tests/unit/adapters/test_nested_source.py \
  tests/unit/handlers/test_archive.py \
  tests/integration/test_containers.py \
  tests/unit/adapters/test_local_source.py \
  -q
```
Expected: PASS. `test_local_source.py` is included deliberately: the compatibility promise is that it still passes untouched.

- [ ] **Step 7: Commit**

```bash
git add src/readeverything/composition.py src/readeverything/__init__.py README.md tests/integration/test_containers.py
git commit -m "feat: wire container descent into build_perception, with the acceptance test"
```

---

## Notes for the executor

- **`pipeline/resolution.py` is not touched.** A member has no inode, so `stat_key` already returns `None` and members are already never memoized. Spec §6 says its existing rule was written for exactly this case. If you find yourself wanting a member memo, stop and ask — a memo keyed on a member uri needs an invalidation rule for the *containing* file changing, which is how the resolution memo and the artifact store start to blur.
- **Hashing needs no change.** A member's `content_hash` is the blake2b of its *decompressed* bytes, because `ContentHasher` reads through the `SourceReader` port and `NestedSource` serves decompressed bytes. That is what makes the artifact store warm across the boundary: extract a PDF from a zip and its cached OCR still hits.
- **If any spec requirement turns out to be unimplementable as written, stop and report it.** Do not redesign silently. Two shape deviations are already recorded above (Task 2's `open_member`, Task 8's `containers` default) — anything beyond those is a new decision and is not yours to make alone.
