# readeverything

Give an agent eyes into a filesystem. `readeverything` turns a directory of
mixed files into mimetype-dispatched media representations — text spans,
image crops, hex dumps — each carrying a locator back to exactly where it
came from, so an agent's answer can point at its source instead of just
asserting one.

## Install

```bash
pip install readeverything
```

## Use it

<!-- readeverything:tested -->
<!-- The block below is compiled and executed by
     tests/integration/test_readme_example.py, which injects `root` (a
     temporary directory holding notes.txt) into its namespace and asserts on
     the names it leaves behind (`card`, `tools`). Exactly one block in this
     file may carry the marker above. Edit the example freely — but it has to
     keep running. -->

```python
from readeverything import (
    Budget,
    Capability,
    SemaphoreLimiter,
    build_perception,
    build_tools,
)


class Narrate:
    """An observer: anything with `observe(event)`. Yours can do better than print."""

    def observe(self, event):
        print(f"{type(event).__name__}: {event.operation} on {event.ref.uri}")


perception = await build_perception(
    root,
    # Watch a long read as it happens — started, progressed, finished — and
    # never let more than four vision calls run at once.
    observer=Narrate(),
    limiter=SemaphoreLimiter({Capability.VISION: 4}),
)
card = await perception.inspect("notes.txt")
tools = build_tools(perception)

# Narrate() sees this read start and finish; a video would report each frame.
rendered = await perception.represent("notes.txt", Budget(max_chars=None))
```

Drop the `observer` and `limiter` arguments and it is three lines; with them,
a caller can see which file a slow read is on and bound how hard it leans on a
vision endpoint. An observer never changes what a read returns, and one that
raises cannot fail the read.

`build_perception` walks `root` and wires up
detection, hashing, and the handler registry. `card` describes what the file
is (`card.kind`, e.g. `"text"`) and what you can do with it (`card.affordances`,
a tuple of `Affordance` objects — `[a.name for a in card.affordances]` gives
e.g. `["read_range"]`). `build_tools` turns the whole perception surface into
four LangChain-compatible tools an agent can call directly:
`inspect_path`, `list_paths`, `invoke_affordance`, and `ask_about_image`.

Calling an affordance yourself works the same way an agent's tool call does:

```python
result = await perception.invoke("notes.txt", "read_range", {"start": 4, "end": 9})
```

## Give it to an agent

`build_tools` returns plain LangChain `BaseTool`s, so it drops straight into
[`deepagents`](https://pypi.org/project/deepagents/) with no extra glue:

```python
from deepagents import create_deep_agent
from readeverything import build_perception, build_tools

perception = await build_perception(root)
agent = create_deep_agent(tools=build_tools(perception))
```

Now the agent can look at a directory of mixed files — including images —
and answer questions about them with locators back to the source.

## Add vision

Image affordances beyond a raw crop need a model. Point `readeverything` at
any OpenAI-compatible vision endpoint and the extra affordances appear:

```python
from readeverything import build_openai_vision_model, build_perception, build_tools

vision = build_openai_vision_model(base_url="http://localhost:8000/v1", model="qwen2-vl")
perception = await build_perception(root, vision=vision)
tools = build_tools(perception)
```

With no vision model supplied, images still work — `crop_region` is always
available — they just offer fewer affordances.

## The library reads the filesystem, never the environment

Every input — the root directory, the vision endpoint, the API key — is an
explicit argument. `readeverything` never reads an environment variable to
configure itself. That means two differently-configured `Perception`
instances can run side by side in one process: point one at a local vision
server and leave the other with none, in the same test run or the same
service.

## What's supported today

| Media | `card.kind` | Affordances | Needs |
|---|---|---|---|
| Text, JSON, XML | `text` | `read_range` | nothing extra |
| Images | `image` | `crop_region` always; `describe_image` and `ocr` when a vision model is supplied | `images` extra (Pillow) for image handling; a vision model for description and OCR |
| PDF | `binary` | `read_page`, `page_region`, `page_image`; `ocr_page` when a vision model is supplied | `documents` extra (pypdfium2); a vision model for `ocr_page` |
| Audio | `audio` | `read_span`, when a transcriber is supplied | `transcription` extra (faster-whisper) and an `ffmpeg` binary |
| Video | `video` | `frame_at`; `describe_frame` when a vision model is supplied | an `ffmpeg` binary; a vision model for `describe_frame` |
| Everything else | `binary` | `hexdump` | nothing extra |

A PDF reports `card.kind == "binary"`, not a kind of its own. `MediaKind` names
how bytes are *shaped*, and a PDF is a container; the fact that it has pages is
carried by its affordances, which is where a caller acts on it anyway.

Office documents and archives have no handlers yet — files of those kinds fall
through to the binary fallback above (a hex dump), not a dedicated
representation.

## Extras

```bash
pip install "readeverything[images]"    # Pillow, for image handling
pip install "readeverything[vision]"     # langchain-openai, for vision models
pip install "readeverything[langchain]"  # langchain-core only, no OpenAI client
```

On a machine with none of these installed — no Pillow, no vision client, no
model server running anywhere — the example at the top still works: text is still read, and every other file still gets a locator-carrying
hex dump.
