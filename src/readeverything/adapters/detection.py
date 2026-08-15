"""Detecting a mimetype from bytes, with the filename as a tiebreak.

Order is deliberate and is the whole point of the module:

1. A magic signature in the content. Authoritative when present.
2. The filename's extension. A claim, consulted only when the bytes are silent.
3. Decodable as UTF-8 with no control characters, therefore `text/plain`.
4. `application/octet-stream`, which the binary fallback handler always accepts.

Getting 1 and 2 the wrong way round is the classic defect: a PNG named
`photo.txt` would dispatch to a text handler and produce mojibake that looks
like a corrupt file rather than a misidentified one.
"""

from __future__ import annotations

import mimetypes

import puremagic

from readeverything.domain.identity import MimeType

_OCTET_STREAM = MimeType.parse("application/octet-stream")

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
        for match in matches:
            if match.mime_type:
                try:
                    return MimeType.parse(match.mime_type)
                except ValueError:
                    continue

        guessed, _ = mimetypes.guess_type(uri)
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
