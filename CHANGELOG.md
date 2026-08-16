# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

A slide is a visual artifact, and asking what a chart shows no longer means
reading the words around it. With LibreOffice installed, office documents hand
back page images — and the legacy formats that used to hex-dump hand back
words.

### Added

- **An EPUB handler.** A book used to hex-dump. `.epub` is in
  `NOT_A_FOLDER_MIMES` — descending into one would bury the novel under a
  manifest, a stylesheet and a dozen numbered parts — and until now nothing
  claimed the mimetype, so the refusal to treat it as a folder left it with
  nowhere to go. `EpubHandler` reads the spine, in the spine's order, which is
  the only thing in the file that says what order the book is read in.
  `read_chapter` and `read_range`; chapters are named from the table of
  contents, EPUB 2's NCX and EPUB 3's nav document alike.

  Prose comes from the same `html_text` reader the HTML handler uses, so a
  chapter and a saved web page are read by identical code. A DRM-protected
  book says it is encrypted rather than returning a book of noise, and one
  unreadable part costs a reader that chapter rather than the other nine.

  Stdlib only: `zipfile` and `xml.etree`, no new dependency.

- **`PartSpan`, a locator for a document made of files.** An EPUB's text lives
  in a dozen XHTML files inside a zip, and none of the six existing locators
  could say where a sentence came from: `CharSpan` names offsets with no file,
  which in a book of twelve chapters is twelve possible answers, and a
  `ByteRange` into the epub addresses compressed bytes — checkable only by a
  reader who reimplements DEFLATE, which is the opposite of what a citation is
  for. `PartSpan` names the member and the offsets within it: extract
  `OEBPS/ch3.xhtml`, slice, and the quoted sentence is there.

- **An HTML handler.** A saved article, a scraped page or an emailed report
  used to match `kind:text` and come back as raw markup — tags, inline scripts
  and stylesheets, which on a real page is most of the bytes and none of the
  answer. `HtmlHandler` claims `text/html` and `application/xhtml+xml` at the
  registry's exact-mimetype step and returns prose, with `read_section` and
  `read_range` reading exactly as they do on a Word document.

  What it does not give up is the citation. Every locator points into the
  ORIGINAL markup, not into the stripped text, so slicing the file at a
  citation and collapsing its whitespace reproduces the quoted sentence — a
  property test pins that down. Stripping tags is precisely the operation that
  usually breaks provenance, and this is the handler that does not.

  Costs no new dependency: the offsets come from stdlib `html.parser`, which
  reports the source position of every event.
- **`page_image` on the Word, slide and spreadsheet handlers.** One page as a
  PNG, at whatever dpi you ask for, straight into the vision path that already
  reads PDFs and photographs. A Word document gains pagination this way that
  the structural reader genuinely does not have — a `.docx` has no pages until
  something lays it out — and a spreadsheet's page is its print layout rather
  than its cell grid.
- **`describe_slide` on the slide handler**, when a vision model and a
  converter are both available. It renders the slide and asks the model about
  it in one call, and the answer is located at the slide — where chaining
  `page_image` into `ask_about_image` costs two round trips and a schema read
  to express one intent. It is a different question from
  `describe_slide_image`, which asks about a picture the author *embedded*;
  this one asks about the slide as an audience saw it.
- **Legacy `.doc`, `.ppt` and `.xls` are readable**, where 0.3.0 declared them
  out of scope. They get a card, page text and page images. They arrive as a
  capability rather than as a dependency: with no converter they fall through
  to the hex dump exactly as before, so nothing regresses on a machine that
  does not have one.
- **A `DOCUMENT_RENDER` capability and a `DocumentRenderer` port.** The fourth
  thing this library negotiates rather than requires, after vision,
  transcription and ffmpeg. `build_perception(renderer=...)` points it at a
  converter of your own; `renderer=NullRenderer()` turns rendering off even on
  a machine that has LibreOffice, which is how a test gets determinism without
  uninstalling software.
- **A `soffice` adapter.** It converts a document to PDF **once**, stores that
  in the artifact store under the source's content hash, and renders every page
  from it — so a four-hundred-slide deck pays the conversion once per machine
  rather than once per slide.
- **`Rendition.degradations`.** A rendition could previously only say *that*
  something was wrong with it, via a boolean. A converted page image is not
  wrong; it is a rendering, and that is a third thing. The agent tool pack
  prints these as a trailing `note:` line.

### Fixed

- **A container uri whose member name contains `!` addressed the wrong file.**
  The escape was a doubled separator, and doubling a separator cannot be
  unambiguous: `("a.zip", "!inner")` and `("a.zip!", "inner")` both encoded to
  `a.zip!!!inner`, so one of the two could never be read back — and a member
  named just `!` encoded to something that failed to parse at all. The escape
  is now `\`, which escapes itself: `\!` is a literal separator and `\\` a
  literal backslash.

  **This changes the uri text** for the small number of members whose names
  contain `!` or `\`. Nothing else moves — a uri with neither is byte-for-byte
  what it was, which is every uri anyone has. A zip written on Windows
  separates member names with `\`, so those now arrive doubled; `join_uri` and
  `walk` already spell them correctly, and only a hand-written uri notices.

  Found by the round-trip property test that was written for exactly this and
  had been passing on easier examples for a year.

### Changed

- **A rendered page says it is a rendering.** LibreOffice's rendering of a
  PowerPoint is a rendering and not the thing itself: fonts substitute when the
  original's are not installed, and layout engines differ. Every rendered page
  carries a `Degradation` naming the converter, and so does the text of a
  converted legacy file, where the wording is the original's but the ordering
  is the importer's. An agent reading type off a slide should not report the
  substitute as the author's choice.
- **PDF page rendering moved into `adapters/pdfium_render.py`**, which the PDF
  handler now delegates to. No behavior changed; the renderer needed the same
  code and an adapter cannot import a handler.

### Notes

- **Rendering never happens during `inspect` or a directory listing.** Every
  rendering affordance is `DEEP`. Listing a folder must not convert a deck.
- **Conversion runs with macro execution disabled**, in a private LibreOffice
  profile, under a bounded timeout that kills the process. This is the only
  place in the library where a document could execute code, and LibreOffice has
  no command-line flag for it — macro policy is a profile setting, so the
  adapter writes one before the first launch. Conversions also serialise, one
  at a time: `soffice` is not concurrency-safe across processes sharing a
  profile.
- **The legacy OLE2 formats are detected as a single family, not three.** All
  three share the compound-file header, and it is indistinguishable within the
  4096 bytes detection reads — so a real `.doc`, `.ppt` and `.xls` all report
  `application/msword`. Telling them apart needs a full OLE2 directory walk,
  which this library does not do. It costs nothing in practice, because it is
  the **converter**, not the mimetype, that determines how the file is read:
  a `.ppt` labelled `application/msword` still opens in Impress. What it costs
  is the right to say which application made the file, so the card does not —
  it reports the page count it observed from the conversion. Same shape as
  0.3.0's note about a plain zip under `word/`/`ppt/`/`xl/`: a real
  misdetection, degrading honestly rather than silently.
- **No bundled binary and no new Python dependency.** LibreOffice is ~500MB and
  not pip-installable, so there is no new extra for rendering. pypdfium2, which
  turns the converted PDF into images, already ships in `documents`.
- Rendering a deck **inside a tarball** works, and nothing was written to make
  it: a converter takes a path, and 0.3.0's container descent already
  materialises a member on request.

## [0.3.0] - 2026-08-15

An archive is a directory now, and an office document is a document. The two
compose without either knowing about the other.

### Added

- **Container descent.** A zip, tar, `.tar.gz`, `.tar.bz2` or `.tar.xz` is
  walked as a directory, and its members are addressed with `!` —
  `docs.zip!nested.tar.gz!report.pdf`. A member gets a real card from whichever
  handler claims its mimetype: an archived PDF has a page count and
  `read_page`, an archived deck has slides. **No handler was modified to make
  this work.** Every handler reads bytes through the `SourceReader` port and
  cannot tell where they came from, which is a property the architecture has
  had since 0.1.0 with nothing spending it.
- **`ContainerLimits`**, bounding depth, member size, total size, member count,
  and an expansion ratio checked *while* decompressing rather than against the
  header — a zip bomb lies in its own paperwork, so only bytes actually written
  are trustworthy. Members with `..` components, absolute paths, or symlink
  targets are refused outright.
- **`ArchiveOpener`**, a port with stdlib zip and tar adapters. Supply your own
  for `.7z` or `.rar`; this library takes no dependency for either.
- **An archive handler**, giving the container itself a card: entry count,
  compressed and uncompressed totals, expansion ratio, and a per-entry outline.
  `list_entries` is paged. There is deliberately no `read_entry` — a member is
  reached by inspecting its uri, and two routes to the same bytes would mean
  two provenance stories for one citation.
- **Word, slides and spreadsheets** (`.docx`, `.pptx`, `.xlsx` and the ODF
  `.odt`/`.odp`/`.ods`), in a new `office` extra. A document reports a heading
  outline, a deck reports slides with **speaker notes included and labelled**,
  and a workbook reports sheets and used ranges. Legacy OLE2 `.doc`/`.ppt`/
  `.xls` remain out of scope.
- **`CellRange`**, a locator addressing a rectangle of cells in a named sheet.
  None of the five existing locators could address a cell: a `CharSpan` into
  rendered text is an artifact of the handler's delimiter choice and does not
  survive changing it.

### Changed

- **Office documents are detected by content, not extension.** A `.docx` is a
  zip, so detection classifies by the container's part-name prefixes (`word/`,
  `ppt/`, `xl/`); a deck renamed `report.bin` is still read as a deck. This is
  also what stops descent from dissolving every office document into a folder
  of XML parts — detection claims the file before the archive handler sees it.
- **An unopenable document no longer reports itself as an empty one.**
  `list_comments` said "no comments" and `list_media` said "no pictures" about
  files they had failed to open. Both check readability first. This is the same
  defect shape as reporting a scanned page as empty.
- **A spreadsheet shows values and formulas both.** `represent` shows cached
  values, because that is what the sheet means; `read_cells(..., formulas=true)`
  shows formulas, because that is what an auditor needs. Where a workbook
  carries no cached values, the formula text appears with a `Degradation`
  saying so, rather than a sheet of arithmetic reading as blank.

### Notes

- A member hashes to the same value as the same file loose on disk, so a
  cached artifact — an OCR, a transcript — stays warm across the container
  boundary.
- A solid archive (`.tar.gz` and friends) cannot seek to a member, so it is
  decompressed once and reused for the lifetime of the perception, bounded by
  an LRU cache rather than by failing.
- Descent is on by default; pass `containers=None` to `build_perception` for
  0.2.0's behavior exactly, including no extra opens during `walk`.
- A plain zip whose first entries happen to sit under a top-level `word/`,
  `ppt/` or `xl/` directory is misread as an office document. It then degrades
  honestly rather than hex-dumping, so it is not a regression — but it is a
  real if unlikely misdetection.

## [0.2.0] - 2026-08-15

Asking a vision model about a picture takes one call now, and may be scoped to
a rectangle.

### Added

- **`ask_about_image`**, an agent tool and a same-named affordance on the
  image, PDF and video handlers. Previously the same intent needed two round
  trips — `inspect_path` to discover the affordance and read its schema, then
  `invoke_affordance` — and went by three different names (`describe_image`,
  `ocr_page`, `describe_frame`) depending on the medium. The tool dispatches on
  the affordance *name*, never on mimetype, so the tool list stays a fixed size
  for every file.
- **Region-scoped asking.** All three handlers accept `x`/`y`/`w`/`h` as
  fractions of the frame and send the vision model only that rectangle.
  Previously the coordinates existed for cropping and nothing else: the bytes a
  crop returned could not be put to a model.
- **`handlers.regions`**, a shared `RegionParams` and `crop_to_region`. Its
  unit-square validator lived in the image handler alone, so PDF and video page
  and frame regions had no boundary validation at all.

### Changed

- **`describe_frame`'s vision call is now bounded by the vision limiter**,
  matching `watch_segment`, `represent` and the new `ask_about_image`. It was
  the only unbounded vision path in the video handler. Under concurrency, many
  simultaneous `describe_frame` calls now queue against the semaphore instead
  of reaching the model all at once.
- **An image-bearing rendition no longer offers advice that cannot be taken.**
  `crop_region`, `page_region` and `frame_at` used to render as "call
  `invoke_affordance` with `describe_image`" — impossible, since that reads a
  uri and nothing accepts returned bytes. The hint now names `ask_about_image`
  on the file with the same coordinates.
- **`readeverything.testing`'s affordance-invocability law is stricter.** It
  synthesizes minimal valid parameters for required fields and genuinely
  invokes affordances that cannot be constructed with no arguments, rather than
  passing over them. Handlers outside this repository declaring a
  required-parameter affordance are now held to it too.

### Notes

- A region is a precision feature, not an economy one. Measured against
  `qwen3.8-27b-mtp`: a 720x480 frame, a 360x240 crop of it and a 72x48 crop of
  it all cost 1,140 prompt tokens. The server resizes to a fixed grid, so cost
  is per image rather than per pixel and cannot be reduced from the client.
- On video the region narrows what the model sees but does not appear in the
  locator: the domain has no composite locator, and a `TimeSpan` cannot carry a
  `BBox`. The affordance's own description says so, so an agent is not misled.

## [0.1.0] - 2026-08-15

First published release. The perception surface has been exercised against
real files but not against other people's, and the handler protocol is the
part most likely to move once it has -- hence `Development Status :: Alpha`
on a `0.x` version, where a minor bump may still break you.

### Added

- **Perception core.** A filesystem root becomes mimetype-dispatched media
  representations — text spans, image crops, hex dumps — each carrying a
  locator back to the byte range, page, or timestamp it came from, so an
  answer can point at its source instead of asserting one.
- **Handler dispatch by mimetype**, with content sniffing (`puremagic`) and
  encoding detection (`charset-normalizer`) rather than trust in file
  extensions.
- **Budgets and capabilities.** `Budget`, `Capability`, and `SemaphoreLimiter`
  bound what a perception pass may spend and which adapters it may reach, so
  an expensive handler cannot be entered by accident.
- **Optional adapters, each behind its own extra**: `images` (Pillow),
  `documents` (pypdfium2), `vision` (an OpenAI-compatible vision model),
  `transcription` (local faster-whisper weights),
  `remote-transcription` (an HTTP client against a whisper.cpp server), and
  `agents` (deepagents tool wiring).
- **`readeverything.testing`**: port compliance suites, shipped inside the
  package so handler authors outside this repository can run the same
  conformance tests the built-in adapters are held to.
- **Layered architecture contract** enforced in CI by import-linter, and a
  test asserting optional dependencies stay confined to their adapters.

### Notes

- Requires Python 3.13.
- Video is read transcript-first: the words are cheaper and more informative
  than the frames, and frames are sampled only when asked for.

[Unreleased]: https://github.com/tyevans/readeverything/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/tyevans/readeverything/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/tyevans/readeverything/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tyevans/readeverything/releases/tag/v0.1.0
