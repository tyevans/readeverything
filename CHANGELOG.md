# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
