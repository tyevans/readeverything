# RFC 0001 — Source spans carry caller provenance

**Target:** `redstring`
**Raised by:** `deepagents-read-everything` (see
`docs/superpowers/specs/2026-08-14-readeverything-perception-core-design.md` §9)
**Status:** Proposed
**Blocking:** yes — the perception core's citation path depends on it

---

## Summary

Add one concept to redstring: a `SourceDocument` may carry **spans** — sorted,
non-overlapping character ranges over its own text, each with an opaque payload
and an optional barrier flag.

From that single addition, three things follow that redstring cannot do today:

1. A caller can attach arbitrary provenance to a *region* of a document and have
   it survive chunking.
2. A caller can declare positions the chunker must not merge across.
3. A retrieval hit arrives carrying the provenance of the regions it overlaps.

No new dependencies. No new ports. redstring stays pure and text-only, and never
inspects a payload.

---

## Why this is redstring's problem, not the caller's

The test applied throughout is: **can the change be justified with a purely-text
example?** All three can.

- A PDF ingested as text needs a page number per region. Today the page map is
  lost the moment the text is concatenated.
- An HTML document needs a section anchor per region, so a quotation can link
  back to `#installation` rather than to the page.
- A concatenated mailbox needs a message id per region.
- "Do not merge two chapters into one chunk" is a chunk-quality rule with no
  media content whatsoever.

None of these are media features. Media is only the forcing function that made
the gap unavoidable — the same way `research-team`'s private
`BoundaryPreferenceChunker` copy (B120) was upstreamed because *citation
quality* turned out to be a general concern rather than one project's taste.
This RFC is the same shape of contribution, from the same direction, for the
same reason.

---

## What the current code actually does

Three findings, in increasing order of surprise.

### `StoredChunk.metadata` is plumbed end to end and never filled

`domain/chunk.py` declares it. `chunks/adapters/postgres.py` gives it a `jsonb`
column, writes it in `_INCOMING`, and reads it back in the row mapper. The
compliance suite exercises it.

And `extraction/corpus.py::build_stored_chunks` — the single function both write
paths share — constructs `StoredChunk` without it:

```python
StoredChunk(
    id=ident,
    tenant_id=tenant_id,
    source_id=source_id,
    text=chunk.text,
    chunk_index=chunk.chunk_index,
    start_char=chunk.start_char,
    end_char=chunk.end_char,
    entity_ids=found[ident],
    #  chunk.metadata is dropped here
)
```

`extraction.chunking.Chunk.metadata` exists too, and no chunker ever writes to
it. So there is a fully persisted metadata channel that is structurally
unreachable. This is the "one fact stored in two places with nothing that fails
when the copies disagree" failure mode, in its quieter form: a field that
exists, is tested for round-tripping, and can never be non-empty in production.

### Retrieval would already carry provenance

`ChunkRetrievalResult.matches` is `list[ScoredChunk]`, and a `ScoredChunk` wraps
a `StoredChunk`. Once `metadata` is populated, provenance reaches the caller
with **no change to any retrieval code**. What is needed is a test asserting it,
not an implementation.

### The real gap is that a caller cannot annotate a chunk it did not create

This is the blocker. redstring chunks internally, from `SourceDocument.text`.
The caller never sees the split, so it cannot attach anything per-chunk, and it
cannot re-derive the offsets without reimplementing the chunker.

`SourceDocument.metadata` does not help: it is document-scoped, so it can say
"this came from lecture.mp4" but never "this passage is 00:42:15–00:42:31,
spoken by Priya". Document-level metadata is exactly the granularity that makes
a citation useless.

Hence: the annotation must be attached to *ranges of the text*, by the caller,
before chunking — and intersected with chunks by redstring afterwards.

---

## Proposal

### R1 — `SourceSpan` on `SourceDocument`

```python
class SourceSpan(BaseModel):
    """A region of a document's text, and what the caller knows about it."""

    start_char: int          # inclusive
    end_char: int            # exclusive
    payload: Mapping[str, JsonSafe]
    barrier: bool = False
```

```python
class SourceDocument(BaseModel):
    ...
    spans: tuple[SourceSpan, ...] = ()
```

Validation, at construction, where the offending value is in hand:

- `0 <= start_char < end_char <= len(text)`
- spans are sorted by `start_char` and non-overlapping
- spans need not be gapless; an unannotated region simply has no provenance
- `payload` passes `reject_unstorable_text`, because it lands in `jsonb` and in
  the durable event log — same rule already applied per-field on `Provenance`
  and `Entity`

**redstring never inspects a payload.** It is carried, intersected, unioned, and
handed back. Any interpretation is the caller's. This is what keeps a timestamp,
a page number and a message id the same feature.

### R2 — Barriers constrain the chunker

`Chunker.chunk` gains one optional argument:

```python
def chunk(
    self,
    text: str,
    max_chunk_size: int | None = None,
    overlap_size: int | None = None,
    barriers: Sequence[int] = (),
) -> ChunkingResult: ...
```

A barrier is an offset at which a cut is **mandatory**: no chunk may contain
text from both sides of it. Barriers are derived by the caller-facing layer from
spans with `barrier=True` (their `start_char`), so a caller never passes them
directly.

This fits both existing chunkers with no change to their character:

- `BoundaryPreferenceChunker` already computes candidate boundaries once for the
  document and locates them by bisection. Barriers become a mandatory subset:
  the window is capped at the next barrier, and the cascade
  (paragraph → sentence → word → hard cut) runs only within that cap.
- The two rules the implementation holds to are preserved exactly, and are the
  acceptance criteria for this change:
  **chunk text is never stripped**, and
  **boundary detection may only choose among split points, never rewrite text.**
  A barrier removes candidate split points; it never introduces or edits a
  character. The partition stays lossless.

**Overlap must not cross a barrier.** Overlap exists so a sentence spanning a
boundary survives intact somewhere; a barrier asserts the opposite — that the two
sides are different things. Where they conflict, the barrier wins and the
overlap is truncated. A chunk that straddles a speaker change produces a
mis-attributed claim, which is a correctness defect, not a recall trade.

A barrier that makes `max_chunk_size` unsatisfiable yields a **shorter** chunk,
never a violated barrier — the same posture `BoundaryPreferenceChunker` already
takes for documents with sparse punctuation.

No effect on chunking signatures: `extraction/corpus.py` digests the boundaries
produced rather than the chunker's declared settings, so a barrier that changes
the split changes the signature for free, and one that does not, does not. That
existing decision makes this change cost nothing here.

### R3 — Intersect spans onto chunks, and let them reach the caller

In `build_stored_chunks`, each chunk's `[start_char, end_char)` is intersected
with the document's spans, and the overlapping payloads are attached under a
single reserved key:

```python
metadata={"spans": [span.payload for span in overlapping]}
```

Reserved key rather than a merge into the top level, so a caller's own metadata
keys can never collide with this mechanism.

`ChunkRetrievalResult` then already carries them. The work here is a test, not a
change.

---

## The design question this exposes: identity is content-addressed, so provenance is multi-valued

`chunk_id(source_id, text)` hashes text, deliberately (`domain/chunk.py`). Two
passages with identical text are therefore **one chunk with one id**, and
`build_stored_chunks` already handles that case for entity links by accumulating
rather than overwriting:

```python
seen = found[ident]
seen.extend(entity_id for entity_id in links.get(...) if entity_id not in seen)
```

Provenance must behave the same way, and this is not an edge case:

- a boilerplate footer repeated on all 40 pages of a PDF
- a licence header at the top of every file in a concatenated corpus
- a filler cue — "yeah", "right" — recurring throughout a transcript

Under first-wins, the footer would be attributed to page 1 and to nowhere else,
which is a citation that is confidently wrong. **Payloads accumulate, in
document order, deduplicated.** A chunk that occurs three times honestly has
three origins, and the list says so.

This mirrors `entity_ids` exactly, which is the strongest argument that it is
right: the same identity decision produced the same multiplicity, and redstring
already chose accumulation once.

**Consequence for callers:** `metadata["spans"]` is a list and may hold more
than one payload even for a single-origin document, because a chunk can overlap
several spans. Callers must not assume length 1. For the perception core this is
correct behaviour, not a wart — a chunk spanning 00:42:15–00:43:02 genuinely
covers several transcript cues, and the citation is the union of their spans.

---

## Alternatives rejected

**Extend `Provenance`.** It describes the *claiming* of an entity — who said it,
when, how sure — and is documented as such under ADR 0035. Source location is a
property of the text, not of an observation about an entity, and `Relationship`
already demonstrates that forcing one provenance shape onto two concepts
produces fields that are always absent.

**One `SourceDocument` per region** (a document per transcript cue, per page).
No upstream change at all, and it was seriously considered. Rejected because it
destroys cross-region context for the extractor — the entity introduced in one
paragraph and referenced in the next becomes two unlinked mentions — and because
it multiplies document count by two to three orders of magnitude for a long
recording.

**A caller-side offset map, with redstring untouched.** The caller keeps
`char_span → locator` privately and resolves retrieval hits against it. This
works and needs nothing here, but it makes every consumer of the corpus
re-implement the resolution, and it silently breaks whenever anything reflows
text. It also leaves `StoredChunk.metadata` dead, which was a defect before this
RFC existed.

**Multimodal extraction** (`LlmProvider.extract` accepting content blocks).
Explicitly out of scope. It changes redstring's core port to serve one caller,
and the described-into-text path must be shown insufficient first.

---

## Compatibility

Additive. `spans` defaults to `()`, `barriers` defaults to `()`, and
`metadata["spans"]` is absent when no spans are supplied. Every existing caller
is unaffected and every existing chunk id is unchanged.

The one visible behaviour change is that `StoredChunk.metadata` can now be
non-empty. Code reading it as always-empty was reading a field that had no
reason to exist.

`Chunker` is a Protocol, so a third-party chunker that does not accept
`barriers` stops structurally satisfying it. This is intended — silently
ignoring a barrier would produce mis-attributed claims with no signal — and it
is why the compliance suite gains a barrier case (below).

---

## Acceptance criteria

- [ ] `SourceSpan` validates bounds, ordering, non-overlap and payload
      storability at construction, with a test per rule.
- [ ] Property test: for any text and any valid barrier set, no chunk contains
      text from both sides of a barrier.
- [ ] Property test: the partition remains lossless with barriers — concatenating
      chunks minus overlap reproduces the input exactly, character for character.
- [ ] Property test: barriers never introduce, drop or rewrite a character;
      CRLF survives.
- [ ] Overlap is truncated at a barrier rather than crossing it.
- [ ] A barrier that makes `max_chunk_size` unsatisfiable yields a shorter chunk
      and does not raise.
- [ ] Spans intersect onto chunks correctly, including partial overlap at both
      chunk edges.
- [ ] Repeated identical text accumulates payloads in document order, without
      duplicates — the footer case, tested explicitly.
- [ ] `build_stored_chunks` propagates `Chunk.metadata` to `StoredChunk.metadata`.
- [ ] Round-trip through both `ChunkStore` adapters preserves payloads.
- [ ] A retrieval hit carries the provenance of the spans it overlaps.
- [ ] The chunk-store compliance suite gains a barrier case, so third-party
      adapters and chunkers inherit the contract.
- [ ] An ADR records the accumulation decision and the reserved `"spans"` key.

## Suggested ADRs

- *Spans are the caller's, and redstring does not read them* — why the payload is
  opaque, and why this is the boundary that keeps redstring text-only.
- *A repeated passage has repeated provenance* — content-addressed identity makes
  provenance multi-valued; accumulation mirrors `entity_ids`.

---

## Sequencing

Land in redstring, cut a release, pin it, then build against it. No development
against an unreleased sibling.

R1 and R3 are independent of R2 and could ship first: together they make
`StoredChunk.metadata` reachable and provenance available at retrieval, which is
already useful to text-only callers. R2 is what makes attribution *correct* for
multi-speaker and multi-page sources, and the perception core needs all three.
