# readeverything: Descending Into Containers

**Date:** 2026-08-15
**Status:** Approved for planning
**Predecessors:** Spec 1 (perception core), Spec 4 (the document family)
**Successors:** Spec 9 (the office family), Spec 10 (faithful rendering)

---

## 1. Why this, and why now

Two files in `src/` already describe this feature, in prose, as though it
existed.

`ports/source.py`:

> `uri` is opaque. A local path, an object-store key and an archive member
> `"/a.zip!inner.txt"` are all just strings here; only the adapter interprets
> them.

`domain/identity.py`:

> `SourceRef` carries no filesystem path semantics on purpose: `uri` is opaque
> to the domain, so an archive member addressed as `/a.zip!inner.txt` and an
> object-store key are the same kind of thing.

No adapter interprets it. Point the library at a directory holding
`release.tar.gz` and it reports `binary` and offers a hex dump — of gzip
framing, which is to say, of nothing. The most common way a set of documents
actually arrives is the one shape the library cannot see into.

The structural argument is stronger than the feature argument. Every handler in
this repository reads bytes through `SourceReader` and is forbidden from
touching a filesystem. That constraint was written down to keep ffmpeg confined
to one adapter — but what it *bought*, without anyone spending it, is that a
handler cannot tell where its bytes came from. **The PDF handler will read a
PDF inside a tarball inside a zip without one line of it changing.** This spec
is the adapter that collects on a bill the architecture already paid.

That is why this is the source layer and not an archive handler with a
`read_entry` affordance. An affordance returning member bytes gives the agent
bytes; nested URIs give it a *perception* — a card, an outline, page
affordances, OCR, provenance. The second is the product.

### 1.1 Acceptance

> Point the library at a directory containing `docs.zip`, which holds
> `report.pdf` and `nested.tar.gz`, which in turn holds `notes.txt`.
> `list_paths` returns all of them, members included, addressed as
> `docs.zip!report.pdf` and `docs.zip!nested.tar.gz!notes.txt`. `inspect` on the
> first returns a PDF card with a page count and `read_page`. `inspect` on the
> second returns a text card with `read_range`. Asking page 7 of the nested PDF
> returns page 7, and its citation names the full nested path. A zip bomb is
> refused with a bounded error rather than exhausting the disk, and a corrupt
> member fails on read without blinding the agent to its neighbours.

---

## 2. The URI grammar

This section is normative and is frozen before implementation, because Spec 9's
fixtures reference it.

```
uri     := segment ( "!" segment )*
segment := <opaque string, no unescaped "!">
```

- The **first** segment is a path relative to the perception root, interpreted
  exactly as `LocalFileSource` interprets it today. Nothing about existing
  behavior changes for a uri containing no `!`.
- Each **subsequent** segment is a member path *within* the container named by
  everything to its left, always `/`-separated regardless of host platform,
  because that is what both the zip central directory and the tar header store.
- A literal `!` in a member name is escaped `!!`. Splitting is therefore a scan,
  not `str.split("!")`. This is unusual enough in practice that getting it wrong
  would be invisible for a year; it is specified now precisely because it is
  rare.

**Why `!`:** it is the convention Java's `jar:` URLs have used for two decades,
it is already written into two docstrings in this repository, and it is legal in
POSIX filenames but vanishingly rare — which is why the escape exists rather
than a claim that collision is impossible.

Grammar handling lives in `domain/container_uri.py`: pure functions, no I/O.
Under the import-linter layered contract it belongs in `domain` because both an
adapter and (in Spec 9) test fixtures parse these strings, and `domain` is the
only layer both may import.

```python
def split_uri(uri: str) -> tuple[str, ...]:      # "a!b!c" -> ("a", "b", "c")
def join_uri(segments: Sequence[str]) -> str:    # inverse, escaping "!"
def container_of(uri: str) -> str | None:        # "a!b!c" -> "a!b"
```

---

## 3. `NestedSource`: a decorator, not a replacement

`adapters/nested_source.py` provides a `FileSource` that wraps another
`FileSource`.

```python
class NestedSource:
    def __init__(
        self,
        inner: FileSource,
        *,
        limits: ContainerLimits,
        archives: ArchiveOpener,
    ) -> None: ...
```

Every method splits the uri. **If there is exactly one segment, it delegates
verbatim to `inner` and returns.** That is the whole compatibility story: a
perception over a directory of loose files behaves identically whether or not
this decorator is installed, and the existing `LocalFileSource` tests continue
to exercise the real path.

For a multi-segment uri, it resolves left to right: open the outermost
container from `inner`, open each subsequent container from the member bytes of
the one before it, and answer the requested operation against the final member.

| Method | Behavior on a member |
| --- | --- |
| `exists` | True when the member is present in its container's directory |
| `size` | The member's *uncompressed* size, from the container's directory |
| `read_bytes` | The member's decompressed bytes, subject to `ContainerLimits` |
| `read_range` | Slices the member; see §3.2 on why this is not always cheap |
| `stream` | Chunks the member without materialising it where the format allows |
| `local_path` | Materialises the member to a temp file and returns its path |
| `walk` | See §3.1 |

`local_path` is the honest one. `ports/source.py` already says a non-local
adapter "must materialise a temporary file" and calls it "the one place that
cost is acknowledged rather than hidden". A member of a solid archive is
exactly that case, and it is what lets ffmpeg and pypdfium2 — which take paths,
not streams — work on archive members with no changes.

### 3.1 `walk` returns members inline

`walk` on a directory returns what it returns today, and additionally, for each
file whose mimetype is a container format, the members within it, addressed by
the grammar in §2. Members that are themselves containers recurse, up to
`max_depth`.

This is the single decision that makes every downstream feature free. Because
`pipeline.perception` walks and then inspects, and because inspection dispatches
on detected mimetype, an archived PDF reaches the PDF handler with no
registry change, no handler change, and no special case anywhere above the
adapter layer.

**A container is not always a folder.** A `.docx`, `.pptx`, `.xlsx`, `.odt`,
`.epub` and `.jar` are all zip files, and descending into them would list
`report.docx!word/document.xml` as a source — which is worse than useless,
because it buries the document itself in a dozen XML parts.

The rule: **`walk` descends into a container only when no handler claims that
container's specific mimetype at a higher priority than the archive handler.**
Detection reports OOXML and ODF as their own mimetypes rather than
`application/zip` (Spec 9 §3 adds that refinement), so once those handlers
exist they claim the file and it stops being a folder. Until Spec 9 lands,
`NestedSource` carries an explicit opt-out set of these mimetypes so the
behavior is correct in the interim rather than briefly wrong.

This rule is owned here, by the layer that does the descending, and Spec 9 §2
records it as a dependency rather than restating it as a decision.

**Cost, stated rather than hidden:** walking a directory now reads every
archive's central directory. That is a seek and a small read per archive, not a
decompression — but it is not free, and on a directory of ten thousand zips it
is ten thousand extra opens. `ContainerLimits.walk_members` (default `True`)
turns it off for callers who want the old behavior.

### 3.2 Seekable versus solid containers

Two container shapes, and the difference is not cosmetic:

- **Seekable** (`.zip`, uncompressed `.tar`): a central directory or header
  chain gives each member's offset. `read_range` on a member is a genuine
  ranged read. Nothing is materialised.
- **Solid / streaming** (`.tar.gz`, `.tar.bz2`, `.tar.xz`): the compression
  wraps the whole archive, so member *n* cannot be reached without
  decompressing members 0..*n*-1. Reading three members naively is three full
  decompressions.

The adapter decompresses a solid container **once**, to a temp file, and reuses
it for every member read for the lifetime of the `NestedSource`. The temp file
is removed when the perception is closed. This is a cache with a bounded size
(`ContainerLimits.max_materialised_bytes`), and when the bound is hit it evicts
least-recently-used rather than failing — the alternative, failing, would make a
directory of large tarballs unreadable rather than slow.

### 3.3 Limits, and the zip bomb

```python
@dataclass(frozen=True, slots=True)
class ContainerLimits:
    max_depth: int = 3
    max_member_bytes: int = 1 << 30          # 1 GiB, any single member
    max_total_bytes: int = 4 << 30           # 4 GiB, one container expanded
    max_members: int = 10_000
    max_expansion_ratio: float = 200.0       # uncompressed / compressed
    max_materialised_bytes: int = 8 << 30    # the §3.2 temp cache
    walk_members: bool = True
```

Every one of these raises `ContainerLimitExceededError` (a new subclass of
`SourceUnreadableError`, so existing `except` clauses keep working) rather than
returning truncated data. Truncation would hand a handler a half a PDF, which
it would then report on as though it were whole.

`max_expansion_ratio` is checked *during* decompression against bytes written so
far, not afterwards against the declared size — a zip bomb lies in its header.
This is the check that matters; the others are belt.

Defaults are conservative and every one is an explicit constructor argument,
per the library's standing rule that nothing configures itself from the
environment.

### 3.4 Path traversal

`LocalFileSource` guards the root with `resolve()` and a parent check. Members
need a second, different guard, because a member path is never resolved against
the filesystem at all:

- A member segment containing `..` as any component is refused.
- A member segment that is absolute (leading `/`, or a Windows drive) is refused.
- A member that the container declares as a symlink is not followed; it is
  reported in the outline with its target as a fact, and reading it raises.

That last one is the tar-specific hole — a tarball can carry a symlink to
`/etc/passwd`, and "materialise this member" would otherwise happily follow it.
Refusing to follow is the only defensible default for a library whose entire
sandboxing story is "nothing outside the root".

---

## 4. `ArchiveOpener`: the port

Handlers do not shell out and adapters own format knowledge, so container
formats sit behind a port in `ports/containers.py`:

```python
@runtime_checkable
class ArchiveOpener(Protocol):
    def claims(self, mime: MimeType) -> bool: ...
    async def entries(self, path: str) -> Sequence[ArchiveEntry]: ...
    async def open_member(self, path: str, member: str) -> AsyncIterator[bytes]: ...
```

```python
@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    path: str
    size_bytes: int
    compressed_bytes: int
    is_dir: bool
    is_symlink: bool
    modified_epoch_s: float | None
    byte_offset: int | None   # None for solid containers
```

Two adapters ship: `adapters/zip_archive.py` (stdlib `zipfile`) and
`adapters/tar_archive.py` (stdlib `tarfile`, covering `.tar`, `.tar.gz`,
`.tgz`, `.tar.bz2`, `.tar.xz`). Both are stdlib-only, so **this entire spec adds
no new dependency** — which is the second reason to do it before the office
family, whose extra is heavy.

A `CompositeOpener` dispatches on mimetype, and a caller can supply their own
for `.7z` or `.rar` without this repository growing a dependency on either.

---

## 5. The archive handler

The container itself still deserves a card, because an agent that lists a
directory should learn what is in `release.tar.gz` without descending into it.

`handlers/archive.py` claims the container mimetypes and produces:

- `kind`: `BINARY`. Same reasoning the README already gives for PDF — `MediaKind`
  names how bytes are shaped, and a container's shape is binary. What it *is*
  is carried by its affordances.
- `facts`: entry count, total compressed and uncompressed bytes, expansion
  ratio, format, and whether it is solid.
- `outline`: one `Segment` per entry — label is the member path, locator is the
  entry's `ByteRange` within the archive where the format gives one.
- `excerpt`: the first several member paths, which is what a human skims for.
- `affordances`: `list_entries(offset, limit)` and nothing else.

**`list_entries` is paged and there is no `read_entry`.** Paging because a
40,000-entry tarball is not one response. No `read_entry` because reading a
member is spelled `inspect("a.zip!inner.txt")` — and two ways to reach the same
bytes would mean two provenance stories for one citation, which is the failure
this library exists to prevent.

`represent` renders the entry listing as text with a `LocatorMap` over it, so a
claim about a manifest cites the line it came from. `barriers` are empty; an
entry listing has no natural chunk boundary.

Card cost stays within the contract: reading a zip central directory or walking
tar headers is a probe, not a decompression.

---

## 6. Identity, hashing, caching

- **Hashing.** A member's `content_hash` is the blake2b of its *decompressed*
  bytes, identical to the same file loose on disk. That is what makes the
  content-addressed artifact store work across the boundary: extract a PDF from
  a zip, and its cached OCR is still warm.
- **Resolution memo.** `pipeline/resolution.py` already declines to memoize
  anything it cannot stat, and documents why. A member has no inode, so
  `stat_key` returns `None` and members are simply never memoized. **No change
  to that file.** Its existing rule was written for exactly this case.
- **Artifact cache key.** Contains the content hash, so it is correct for
  members without modification.

The cost is that every `inspect` of a member re-reads and re-hashes it. For a
solid archive that is cheap after §3.2's materialisation. It is left as-is
rather than given a bespoke memo, because a memo keyed on a member uri would
need an invalidation rule for the *containing* file changing, and inventing one
here is how the resolution memo and the artifact store would start to blur —
the exact conflation `resolution.py`'s docstring warns against.

---

## 7. Composition

`build_perception` gains:

```python
async def build_perception(
    root, *,
    containers: ContainerLimits | None = ContainerLimits(),  # None disables descent
    archives: ArchiveOpener | None = None,                   # defaults to zip + tar
    ...
) -> Perception
```

**Corrected during planning: the signature said `= None` while the prose below
said descent is on by default.** The prose is the intended behavior and the
default is now a `ContainerLimits()`; an explicit `None` opts out. Recorded
rather than silently amended, per this repository's habit in Spec 4 §1.1.

Passing `containers=None` yields today's behavior exactly, including no extra
opens during `walk`. The default is a `ContainerLimits()` with §3.3's values —
descent on by default, because a library whose promise is "read everything"
should read the tarball.

---

## 8. Testing

- **Unit, `tests/unit/domain/test_container_uri.py`:** the grammar. Round-trip
  property test with Hypothesis over member names including `!`, since that
  escape is the part most likely to rot.
- **Unit, `tests/unit/adapters/test_nested_source.py`:** against fakes — single
  segment delegates verbatim; depth limits; traversal and symlink refusals; the
  expansion-ratio check firing mid-stream on a synthetic bomb.
- **Unit, `tests/unit/handlers/test_archive.py`:** the existing
  `handler_compliance` suite from `readeverything.testing`, plus paging and the
  corrupt-member case.
- **Integration, `tests/integration/test_containers.py`:** the §1.1 acceptance
  scenario end to end, building the fixture archives in a tmpdir rather than
  committing binaries. This is the test that proves the PDF handler descended
  without being modified.

No live tests. Nothing here touches a model.

---

## 9. What this deliberately does not do

- **No `.7z`, `.rar`, or `.iso`.** Each needs a dependency or a binary. The
  `ArchiveOpener` port is the extension point, and it is public.
- **No writing.** The library reads.
- **No cross-container dedup.** The same file in twelve zips is hashed twelve
  times. Content-addressed artifacts already mean the *expensive* work happens
  once; hashing again is cheap and a dedup layer here would need a cache with an
  invalidation rule, per §6.
- **No `walk` streaming.** `SourceLister.walk` returns a `Sequence`, and
  changing it to an iterator is a breaking port change that should be its own
  decision, not a side effect of this one.
