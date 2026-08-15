"""Detecting a mimetype from bytes, with the filename as a tiebreak.

Order is deliberate and is the whole point of the module:

1. A magic signature in the content. Authoritative when present.
2. The filename's extension. A claim, consulted only when the bytes are silent.
3. Decodable as UTF-8 with no control characters, therefore `text/plain`.
4. `application/octet-stream`, which the binary fallback handler always accepts.

Getting 1 and 2 the wrong way round is the classic defect: a PNG named
`photo.txt` would dispatch to a text handler and produce mojibake that looks
like a corrupt file rather than a misidentified one.

One narrow exception sits inside step 1: `.m4a` and `.mp4` are the same ISO
BMFF container, byte-identical at the header, so the magic signature reports
`video/mp4` for both — whether a given file actually holds a video track
lives in its stream table, not its magic bytes. That is not "the bytes are
silent" (the case step 2 exists for); it is the bytes being ambiguous by
construction within one container family. So when the magic signature
reports an mp4-family type *and* the filename's extension guesses an
`audio/*` type, the extension wins for that single container ambiguity only
— content still beats filename in general. An unregistered extension such as
`.m4b` (audiobook) guesses `None` from `mimetypes` and therefore stays
`video/mp4` under this rule: that is a genuinely unknown claim, not one this
module should invent an answer for.
"""

from __future__ import annotations

import mimetypes

import puremagic

from readeverything.domain.identity import MimeType

_OCTET_STREAM = MimeType.parse("application/octet-stream")

#: mp4-family mime types puremagic reports for the ISO BMFF container, which
#: `.m4a` (audio-only) and `.mp4` (commonly video) share byte-for-byte at the
#: header.
_MP4_FAMILY = frozenset({"video/mp4", "audio/mp4"})

#: Bytes that never appear in text a handler could usefully read. Tab, newline
#: and carriage return are excluded because they obviously do.
_CONTROL = frozenset(range(0, 9)) | frozenset(range(14, 32))


class PuremagicDetector:
    """Content-first mimetype detection."""

    async def detect(self, uri: str, head: bytes) -> MimeType:
        if not head:
            return _OCTET_STREAM

        try:
            matches = puremagic.magic_string(head)
        except Exception:
            # The octet-stream tail below is what makes "there is no
            # unsupported file" true, so nothing may prevent reaching it.
            matches = []
        guessed, _ = mimetypes.guess_type(uri)

        for match in matches:
            if match.mime_type:
                try:
                    signature = MimeType.parse(match.mime_type)
                except ValueError:
                    continue
                if match.mime_type in _MP4_FAMILY and guessed and guessed.startswith("audio/"):
                    # The mp4 container is ambiguous by construction here, not
                    # silent: `.m4a` and `.mp4` share a header, and only the
                    # extension distinguishes an audio-only file from one that
                    # also holds video.
                    try:
                        return MimeType.parse(guessed)
                    except ValueError:
                        pass
                return signature

        if guessed:
            try:
                return MimeType.parse(guessed)
            except ValueError:
                pass

        try:
            head.decode("utf-8")
        except UnicodeDecodeError:
            return _OCTET_STREAM
        if any(byte in _CONTROL for byte in head):
            return _OCTET_STREAM
        return MimeType.parse("text/plain")
