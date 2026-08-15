"""Detecting a mimetype from bytes, with the filename as a tiebreak.

Order is deliberate and is the whole point of the module:

0. An ISO BMFF `ftyp` box, checked directly. See `_is_iso_bmff`.
1. A magic signature in the content, *above a confidence floor*. Authoritative
   when present. See `_SIGNATURE_FLOOR` for why the floor is not optional.
2. The filename's extension. A claim, consulted only when the bytes are silent.
3. Decodable as UTF-8 with no control characters, therefore `text/plain`.
4. `application/octet-stream`, which the binary fallback handler always accepts.

Step 0 and the floor were both added after a 44-minute video was detected as
`audio/x-sndr` — a 0.20-confidence guess at a Macintosh sound resource — and
dispatched to the audio handler, which cost it every visual affordance while
reporting nothing wrong. "Detected as audio" and "is audio" are
indistinguishable downstream, so a detection this module gets wrong is a
capability the caller silently never had.

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

#: Below this, a puremagic match is not a signature — it is a coincidence.
#:
#: Measured across this project's video corpus: every genuine match reports
#: 0.80 or higher, and EVERY mp4 in it also matches "Macintosh SNDR Resource"
#: (`audio/x-sndr`) at 0.20. That junk match normally loses to the real one and
#: is invisible. On a container whose brand puremagic has no signature for it
#: was the only match, and a 44-minute video was dispatched to the audio
#: handler — which cost it every visual affordance without reporting anything,
#: because "detected as audio" and "is audio" are indistinguishable downstream.
#:
#: A match this weak means the bytes are SILENT, which is the case the filename
#: exists to answer. The floor sits between the two observed populations rather
#: than at a round number.
_SIGNATURE_FLOOR = 0.5

#: Offset and value of the box type every ISO BMFF file carries: a 4-byte size,
#: then the literal `ftyp`.
_FTYP_OFFSET = 4
_FTYP = b"ftyp"


def _is_iso_bmff(head: bytes) -> bool:
    """Whether `head` begins with an ISO BMFF `ftyp` box.

    puremagic identifies this container by enumerating BRANDS — `isom`, `mp42`,
    `M4V ` and so on — so it misses any brand its table predates. `iso5`, the
    DASH-derived brand, is one such, and that omission is what misdetected a
    44-minute video as a Macintosh sound resource.

    The box itself is the stable fact: brands come and go, `ftyp` at offset 4
    does not. Checking it directly keeps this a CONTENT decision, which is the
    order this module promises; falling back to the extension would have fixed
    the same file while leaving one named `.bin` broken.
    """
    return len(head) >= _FTYP_OFFSET + len(_FTYP) and head[_FTYP_OFFSET:8] == _FTYP


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

        # An ISO BMFF container, recognised by its own box rather than by a
        # brand table. Checked before the signature list because puremagic's
        # answer for an unlisted brand is not a weaker version of this one — it
        # is unrelated noise, and there is no confidence at which noise about a
        # Macintosh sound resource should outrank the container actually here.
        if _is_iso_bmff(head):
            # The `.m4a`/`.mp4` ambiguity is unchanged: one container family
            # holds both, and only the extension separates audio-only from
            # video. See the module docstring.
            if guessed and guessed.startswith("audio/"):
                try:
                    return MimeType.parse(guessed)
                except ValueError:
                    pass
            return MimeType.parse("video/mp4")

        for match in matches:
            if match.mime_type:
                if match.confidence < _SIGNATURE_FLOOR:
                    # Not a signature, a coincidence. Treat the bytes as silent
                    # and let the filename — or the text/binary tail below —
                    # answer, rather than asserting a type on 0.20 confidence.
                    continue
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
