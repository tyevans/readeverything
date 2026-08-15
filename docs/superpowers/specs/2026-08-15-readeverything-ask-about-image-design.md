# Asking a question about a picture

**Status:** design
**Date:** 2026-08-15

## The problem

Every part of asking a vision model a question already exists. Reaching it does
not.

An agent that wants to know what is in a photograph must first call
`inspect_path` to discover that `describe_image` exists, read the JSON schema
that comes back, and then call `invoke_affordance` with the affordance name and
a `params` dict shaped to that schema. Two round trips and a schema-reading step
before a single question gets asked. The same question about a PDF page is
`ocr_page`; about a video frame it is `describe_frame`. Three names, three
schemas, one intent.

Asking about *part* of a picture is worse: it cannot be done. `crop_region`
returns PNG bytes, `page_region` returns PNG bytes, `frame_at` returns PNG
bytes — and every one of them renders to the agent as

```
[image/png image, 41203 bytes — call invoke_affordance with describe_image or ocr to read it]
```

which is advice the agent cannot take. `describe_image` reads a *uri*. There is
no path from returned bytes back into a model. The hint names a door that is not
there.

So the three failures are one failure wearing three coats:

1. **The two-step dance.** Discovery is mandatory even when the intent is
   obvious.
2. **No region-scoped asking.** The coordinates exist for cropping and for
   nothing else.
3. **Bytes are a dead end.** Every image-producing affordance terminates in a
   message telling the agent to do something impossible.

## What this is not

It is not a fourth way to describe an image. `describe_image`, `ocr`,
`ocr_page`, `describe_frame`, `crop_region`, `page_region`, `page_image` and
`frame_at` all keep their current names and signatures. Nothing is deprecated
and nothing is rewritten. What follows adds one tool and one affordance name,
and changes one line of rendering.

## The shape

One new agent tool:

```
ask_about_image(uri, question, where={...}) -> str
```

Its entire body is

```python
await perception.invoke(uri, "ask_about_image", {**where, "question": question})
```

There is no kind switch and no mimetype knowledge in the tool layer. `where` is
forwarded untyped, and `Perception.invoke` validates it against whatever schema
the resolved handler declared for that affordance — machinery that already
exists and is already the only place params are validated. When the handler for
a file offers no such affordance, `UnknownAffordanceError` renders a message
naming the affordances the file *does* have, so pointing this at a `.txt`
produces a redirect rather than a dead end.

### Why a tool at all, when `agent/tools.py` argues against them

That module's docstring makes a specific argument, and it is a good one:

> Three tools rather than one per affordance. Affordances are per-mimetype and
> therefore per-file, so a tool per affordance would mean a tool list that
> changes with whatever the agent last looked at.

`ask_about_image` does not violate this. The tool list stays fixed at four for
every file in every deployment. What makes that possible is that the tool is
*not* bound to an affordance — it is bound to a **name convention**. Any handler
may claim the name; the tool never learns which ones did. The docstring's
argument survives intact, and the rule it protects — the tool layer knows
nothing about kinds — is still literally true after this change.

The alternative considered and rejected was a router: a tool that inspects the
file's kind and dispatches to `describe_image` / `ocr_page` / `describe_frame`
itself. It needs no handler changes, and it puts a mimetype switch and a
union-of-all-coordinates schema inside the one module whose stated discipline is
having neither. Rejected on that basis.

## The convention

Three handlers grow an affordance named `ask_about_image`. Each declares its own
coordinate schema. All share a `question` field and an optional `region`:

| Handler | `where` fields | Renders from |
|---|---|---|
| image | `region` | the file itself |
| pdf | `page`, `dpi`, `region` | pdfium page render |
| video | `seconds`, `region` | ffmpeg frame extract |

Each is offered only when a vision model is wired, exactly as `describe_image`
and `describe_frame` are today — the `if self._vision is None` guard already in
each handler, and the registry's capability filtering, need no changes to drop
these on a vision-less deployment.

`question` has no default. `describe_image` defaults its prompt to "Describe
this image in two or three sentences" because it is a *describe* affordance and
the description is the point. `ask_about_image` with no question is a call with
no intent, and defaulting it would quietly turn it into a slower `describe_image`.

## Region

A shared `RegionParams` base and a `crop_to_region(image, region) -> bytes`
helper, extracted from `CropParams` in `handlers/image.py` — including its
unit-square validator, which today lives in one handler and would otherwise be
copy-pasted into three.

That validator exists for a documented reason worth preserving verbatim: `BBox`
catches an out-of-bounds crop too, but only once the crop is already running,
so the caller sees a bare `ValueError` from inside the domain rather than a
rejection at the boundary where the mistake was made. Extracting it keeps that
property for PDF and video, which currently do not have it at all.

Cropping happens *before* the vision call, in all three handlers:

- image — crop the decoded `Image`
- pdf — render the page, then crop
- video — extract the frame, then crop

The model receives only the requested rectangle. It is never sent a whole frame
with a described sub-area, which would ask it to do the cropping in prose and
make the locator a claim rather than a fact.

Omitting `region` means the whole image, so the common case stays
`ask_about_image(uri, question)` with nothing else to fill in.

## The rendering change

One line. The `ImageContent` branch of `_render_rendition` currently routes to
`invoke_affordance with describe_image or ocr`, which is the dead end described
above. It should route to `ask_about_image` instead, and — because the hint is
generated from `_IMAGE_READING_AFFORDANCES` matched against the card — the
constant gains the new name.

The no-vision branch of that same rendering stays exactly as it is. When no
vision capability is registered, saying so remains more useful than naming a
tool the registry filtered out.

## Caching

Free, and worth stating so nobody adds it. `artifact_key` already keys on
content hash, handler id, handler version, affordance name and params, so two
identical questions about the same region of the same file is a cache hit
without a line of new code. Two *different* questions about the same region are
correctly two different calls.

## Cost

This makes the expensive path easy to reach. That is the entire point of the
change and also its main risk, and it should be stated plainly rather than
discovered.

On the live server, one still image plus a short question measured 1,140 prompt
tokens, and a single call took 19-34s (n=5, varying with completion length).
Nothing here makes an individual call cheaper. What changes is that an agent
which previously needed two round trips and a schema-reading step to reach a
vision model now reaches it in one.

Two existing mechanisms bound this, and both apply unchanged: `ask_about_image`
runs under the same limiter as every other vision affordance, and the
transcript-first guidance in `read_transcript`'s description still steers agents
to text before pixels on video. No new budget or cap is proposed here. If a cap
turns out to be needed it should be added on measurement, not on the suspicion
that this document is creating a problem it has not yet observed.

**A region-scoped ask is not cheaper than the whole-image ask.** Measured
2026-08-15 against `qwen3.8-27b-mtp`, on a 720x480 frame and crops of it:

| Input | Pixels | `prompt_tokens` |
|---|---|---|
| whole frame 720x480 | 345,600 | 1,140 |
| centre crop 360x240 | 86,400 | 1,140 |
| centre crop 72x48 | 3,456 | 1,140 |

One hundredth of the area costs exactly the same. The server resizes every
image to a fixed grid before encoding it, so image cost is per *image*, not per
pixel, and it cannot be turned down from the client — the same shape as the
already-recorded finding that video cost is a function of duration alone. Wall
time tracked completion length, not input size (26.8s whole vs 26.2s at
quarter-area, across two reps each — noise).

So `region` is a **precision feature, not an economy one**. It is worth having
because it puts the crop in the locator rather than in prose, and because it
stops an agent from having to describe which part of the picture it means. It
does not reduce cost, and this document must not be read as claiming it does.
The corollary is that "ask about four regions separately" costs four times "ask
about the whole image once" — agents should be pointed at a region because the
question is about that region, never as a saving.

## What gets built

1. `handlers/regions.py` — `RegionParams` and `crop_to_region`, moved out of
   `image.py`. `CropParams` becomes a `RegionParams` subclass; `crop_region`'s
   behaviour does not change.
2. `handlers/image.py` — `AskAboutImageParams(RegionParams)` with `question`;
   affordance and `invoke` case.
3. `handlers/pdf.py` — the same, plus `page` and `dpi`, reusing the render path
   `ocr_page` already uses.
4. `handlers/video.py` — the same, plus `seconds`, reusing the extract path
   `describe_frame` already uses.
5. `agent/tools.py` — the fourth tool, the affordance-name constant, and the
   one-line hint change.

## Testing

Per handler, against the existing fake vision model in `testing/fakes.py`:

- the affordance is present when a vision model is wired and absent when it is
  not;
- a `region` reaches the model as cropped bytes — asserted on what the fake
  received, not on the prose that came back;
- an out-of-bounds region is rejected at the boundary, with the parameter error,
  not as a `ValueError` from inside `BBox`;
- an empty completion still raises `InfrastructureError` rather than indexing
  silence, matching `_see`.

At the tools layer:

- the tool forwards `where` untyped and never calls `inspect`;
- a file whose handler has no `ask_about_image` renders the affordance list
  rather than a traceback.

On the shared helper: crop arithmetic, including the existing
`max(..., + 1)` degenerate-rectangle guard that keeps a sliver from rounding to
zero width.

## Open question

`ocr` stays a separate affordance. It carries a tuned prompt, takes no prompt
parameter, and is a name an agent finds by reading the card — folding it into
`ask_about_image` with a canned question would lose that discoverability in
exchange for one fewer name. This is a judgement call made in the absence of an
explicit decision, and it is cheap to reverse in either direction.
