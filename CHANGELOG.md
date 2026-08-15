# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/tyevans/readeverything/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tyevans/readeverything/releases/tag/v0.1.0
