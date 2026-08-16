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
| HTML (`.html`, `.xhtml`) | `text` | `read_section`, `read_range` | nothing extra |
| EPUB (`.epub`) | `binary` | `read_chapter`, `read_range` | nothing extra |
| Images | `image` | `crop_region` always; `describe_image` and `ocr` when a vision model is supplied | `images` extra (Pillow) for image handling; a vision model for description and OCR |
| PDF | `binary` | `read_page`, `page_region`, `page_image`; `ocr_page` when a vision model is supplied | `documents` extra (pypdfium2); a vision model for `ocr_page` |
| Word (`.docx`, `.odt`) | `binary` | `read_section`, `read_range`, `list_comments`, `read_table`; `page_image` when a converter is available | `office` extra (python-docx, lxml); a `soffice` binary for `page_image` |
| Slides (`.pptx`, `.odp`) | `binary` | `read_slide`, `list_media`; `describe_slide_image` when a vision model is supplied; `page_image` when a converter is available; `describe_slide` when both are | `office` extra (python-pptx, lxml); a vision model, a `soffice` binary, or both |
| Spreadsheets (`.xlsx`, `.ods`) | `binary` | `read_sheet`, `read_cells`, `list_sheets`; `page_image` of the print layout when a converter is available | `office` extra (openpyxl, lxml); a `soffice` binary for `page_image` |
| Legacy office (`.doc`, `.ppt`, `.xls`) | `binary` | `read_page`, `page_image` — only when a converter is available, otherwise the hex dump | a `soffice` binary and the `documents` extra (pypdfium2) |
| Audio | `audio` | `read_span`, when a transcriber is supplied | `transcription` extra (faster-whisper) and an `ffmpeg` binary |
| Video | `video` | `frame_at`; `describe_frame` when a vision model is supplied | an `ffmpeg` binary; a vision model for `describe_frame` |
| Archives (zip, tar, tar.gz, tar.bz2, tar.xz) | `binary` | `list_entries`; members are addressed directly, see below | nothing extra |
| Everything else | `binary` | `hexdump` | nothing extra |

A PDF reports `card.kind == "binary"`, not a kind of its own. `MediaKind` names
how bytes are *shaped*, and a PDF is a container; the fact that it has pages is
carried by its affordances, which is where a caller acts on it anyway.

Office documents are detected by their **content**, not their extension: the
zip container's part names are what distinguish a `.docx` from a `.pptx` from a
plain `.zip`, so a deck renamed `report.bin` is still read as a deck.

A spreadsheet shows cached **values** in `represent`, because that is what the
sheet means; `read_cells(..., formulas=true)` shows the formulas, because that
is what an auditor needs. When a workbook was saved by a tool that stores no
cached values, the formula text is shown in their place and a `Degradation`
says so — a sheet full of arithmetic is never reported as empty.

Legacy `.doc`, `.ppt` and `.xls` are read only when a converter is available.
They are OLE2 compound files, a different container format entirely, and their
pure-Python support is poor — so rather than read them badly, the library reads
them through LibreOffice or not at all. With no `soffice` on the machine they
fall through to the hex dump exactly as they always did.

The three are handled as **one family**, not three, and that is a limitation
worth knowing about rather than a design choice. All three share the OLE2
compound-file header, so content detection reports `application/msword` for a
real `.doc`, `.ppt` and `.xls` alike; telling them apart needs a full OLE2
directory walk, which this library does not do. It costs nothing in practice,
because the converter detects the real format itself and a `.ppt` still opens
in Impress. What it costs is the right to *say* which application made the
file, so the card does not: it reports the page count it observed from the
conversion. A caller with a better detector can supply one, and the handler
already claims all three mimetypes.

## Faithful rendering

Reading a deck structurally answers most questions and cannot answer one kind
at all: *which quarter's bar is taller?*, *is the disclaimer inside the box or
below it?* A slide is a visual artifact, and its meaning is often in
arrangement and imagery that no text extraction recovers.

With a `soffice` binary on the machine, `page_image` appears on the slide, Word
and spreadsheet handlers, and the page it returns goes to the same vision path
that already reads PDFs and photographs:

```python
perception = await build_perception("./corpus")   # nothing else to configure
png = await perception.invoke("deck.pptx", "page_image", {"page": 4, "dpi": 150})
```

With a vision model configured as well, a deck gains `describe_slide`, which
does both halves in one call:

```python
answer = await perception.invoke(
    "deck.pptx", "describe_slide", {"page": 4, "question": "Which quarter's bar is taller?"}
)
answer.locator     # PageRef(page=4) — the answer cites the slide
```

That is a different question from `describe_slide_image`, and both exist:
`describe_slide_image` asks about a picture the author embedded, and
`describe_slide` asks about the slide as the audience saw it.

The document is converted to PDF **once**, cached in the artifact store under
its content hash, and every page after the first is rendered from that — so a
four-hundred-slide deck pays the conversion once per machine rather than once
per slide. Conversion runs with macro execution disabled, in a private
LibreOffice profile, under a bounded timeout that kills the process; a document
from an untrusted directory is the reason all three of those are not optional.

**A rendering is not the document.** Fonts substitute when the original's are
not installed, and layout engines differ. The library says so rather than
glossing it: every rendered page carries a `Degradation` naming the converter,
and so does the text of a converted legacy file, where wording is the
original's but ordering is the importer's.

Rendering is negotiated, not required. With no `soffice` the affordance does
not appear at all — there is no tool that exists and returns an apology.
`build_perception(..., renderer=NullRenderer())` turns it off even on a machine
that has LibreOffice, which is how a test gets determinism without uninstalling
software, and `renderer=` accepts any `DocumentRenderer` if you would rather
convert some other way.

## Descending into containers

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
bytes through a port and cannot tell where they came from. A member hashes to
the same value as the same file loose on disk, so a cached OCR stays warm
across the boundary.

A literal `!` in a member name is escaped `\!`, and a literal `\` is escaped
`\\` — so a zip written on Windows, whose member names separate with `\`,
arrives with those doubled. Descent is bounded by
`ContainerLimits` — depth, member size, total size, member count and, the one
that matters, an expansion ratio checked *while* decompressing, so a zip bomb
is refused rather than filling a disk:

```python
from readeverything import ContainerLimits

await build_perception("./corpus", containers=ContainerLimits(max_depth=1))
await build_perception("./corpus", containers=None)  # no descent at all
```

A `.docx`, `.epub` or `.jar` is a zip too, and is deliberately *not* treated as
a folder: descending would bury the document under a dozen XML parts. An EPUB
is read as a book instead — chapters in the spine's order, each citation naming
the part it came from, so `book.epub` reads as a novel rather than a manifest. `.7z` and
`.rar` are not supported, because each needs a dependency this library does not
take — supply your own `ArchiveOpener` via `archives=`.

An office document is a zip, but it is a *document*: the part-name detection
above claims it before the archive handler does, so `report.docx` gets a Word
card rather than a folder of XML parts.

## Extras

```bash
pip install "readeverything[images]"    # Pillow, for image handling
pip install "readeverything[vision]"     # langchain-openai, for vision models
pip install "readeverything[langchain]"  # langchain-core only, no OpenAI client
pip install "readeverything[office]"     # python-docx, python-pptx, openpyxl, lxml
```

On a machine with none of these installed — no Pillow, no vision client, no
model server running anywhere — the example at the top still works: text is still read, and every other file still gets a locator-carrying
hex dump.
