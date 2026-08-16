"""Reading an office package's zip container, without a zip library's help.

Two jobs that look alike and are not.

`office_mimetype` classifies a container from its FIRST 4096 BYTES, because
that is all `Perception` ever hands the detector (`pipeline/perception.py`).
`zipfile` cannot help here: a zip's central directory lives at the END of the
file, so a bounded head has no index at all. What a head does have is a run of
local file headers, each naming one part, and the part names are enough — an
OOXML package keeps its family's parts under `word/`, `ppt/` or `xl/`, and an
ODF package stores a `mimetype` entry first and uncompressed for exactly this
purpose.

Spec §3 originally proposed reading `[Content_Types].xml` instead. Measured,
that cannot work: openpyxl writes `[Content_Types].xml` as the LAST entry of a
workbook, so it is unreachable from any bounded head, and a rule built on it
would detect Word and PowerPoint while silently failing every Excel file. The
override it carries is also `…spreadsheetml.sheet.main+xml` rather than the
document mimetype, so it would need unpicking even where it is reachable. The
spec now specifies the part-name rule implemented here.

`read_part` and `part_names` are the other job: they receive the WHOLE bytes
and may use `zipfile` normally.

No I/O here and no environment: every function takes bytes and returns data.
That is what lets the handlers import this module directly without breaking
`ports/handler.py`'s rule that a handler never touches a filesystem.
"""

from __future__ import annotations

import io
import struct
import zipfile

WORD_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
SLIDES_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
SHEETS_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
ODF_TEXT_MIME = "application/vnd.oasis.opendocument.text"
ODF_SLIDES_MIME = "application/vnd.oasis.opendocument.presentation"
ODF_SHEETS_MIME = "application/vnd.oasis.opendocument.spreadsheet"

#: Every mimetype this spec's three handlers claim. Keeping one frozenset means
#: adding a format cannot leave a consumer behind.
OFFICE_MIMETYPES = frozenset(
    {
        WORD_MIME,
        SLIDES_MIME,
        SHEETS_MIME,
        ODF_TEXT_MIME,
        ODF_SLIDES_MIME,
        ODF_SHEETS_MIME,
    }
)

#: The part-name prefix each OOXML family stores its own parts under. Order is
#: irrelevant: a package holds exactly one of these trees.
_OOXML_PREFIXES = {"word/": WORD_MIME, "ppt/": SLIDES_MIME, "xl/": SHEETS_MIME}

_ODF_MIMETYPES = frozenset({ODF_TEXT_MIME, ODF_SLIDES_MIME, ODF_SHEETS_MIME})

#: Local file header: b"PK\x03\x04", then 26 more bytes before the name begins.
_LOCAL_HEADER = b"PK\x03\x04"
_LOCAL_HEADER_SIZE = 30

#: The entry an ODF package must store first and uncompressed, whose bytes are
#: the document's own mimetype.
_ODF_MIMETYPE_ENTRY = "mimetype"

#: Set in a local header's flag word when the sizes are written AFTER the data,
#: in a trailing descriptor. The header's own size field is then zero, so where
#: the next header begins is unknowable and the walk stops rather than looping.
_STREAMED_SIZES = 0x08


def _entries(head: bytes) -> list[tuple[str, int, int, int]]:
    """`(name, method, data_offset, compressed_size)` for every complete header.

    Stops at the first thing that is not a complete local file header: the end
    of the buffer, a truncated name, a central-directory signature, or an entry
    whose size was streamed. Never reads past `head`, and cannot loop — every
    iteration advances the cursor by at least `_LOCAL_HEADER_SIZE`.
    """
    found: list[tuple[str, int, int, int]] = []
    offset = 0
    while offset + _LOCAL_HEADER_SIZE <= len(head):
        if head[offset : offset + 4] != _LOCAL_HEADER:
            break
        flags, method = struct.unpack_from("<HH", head, offset + 6)
        compressed, _uncompressed = struct.unpack_from("<II", head, offset + 18)
        name_length, extra_length = struct.unpack_from("<HH", head, offset + 26)
        name_start = offset + _LOCAL_HEADER_SIZE
        name_end = name_start + name_length
        if name_end > len(head):
            # The name is cut in half. Half a name is not evidence.
            break
        try:
            name = head[name_start:name_end].decode("utf-8")
        except UnicodeDecodeError:
            break
        data_offset = name_end + extra_length
        found.append((name, method, data_offset, compressed))
        if flags & _STREAMED_SIZES and compressed == 0:
            break
        offset = data_offset + compressed
    return found


def zip_part_names(head: bytes) -> tuple[str, ...]:
    """Part names readable from the local file headers inside `head`.

    A prefix of the package's contents, in stored order — never the whole
    listing, because the head is bounded. Use `part_names` when you hold the
    whole file.
    """
    return tuple(name for name, _method, _offset, _size in _entries(head))


def office_mimetype(head: bytes) -> str | None:
    """The specific office mimetype these leading bytes describe, or None.

    None means "not an office document as far as the head can tell", which
    leaves the caller on whatever it would otherwise have concluded. A plain
    `.zip` and a `.jar` must land here, because Spec 8 descends into those and
    must not descend into a `.docx`.

    Residual risk, recorded rather than hidden: a plain zip whose first entries
    happen to sit under a top-level `word/`, `ppt/` or `xl/` directory is read
    as an office document. The handler then fails to parse it and degrades with
    an honest report, which is this library's contract for a misdetection — and
    is strictly better than the hex dump such a file gets today.
    """
    if not head.startswith(_LOCAL_HEADER):
        return None
    entries = _entries(head)
    if not entries:
        return None

    name, method, data_offset, compressed = entries[0]
    if name == _ODF_MIMETYPE_ENTRY and method == zipfile.ZIP_STORED:
        declared = head[data_offset : data_offset + compressed]
        # A short read means the head ended mid-value. Half a mimetype is not
        # evidence of a whole one.
        if len(declared) == compressed:
            value = declared.decode("ascii", errors="replace")
            if value in _ODF_MIMETYPES:
                return value

    for part, _method, _offset, _size in entries:
        for prefix, mime in _OOXML_PREFIXES.items():
            if part.startswith(prefix):
                return mime
    return None


def read_part(data: bytes, name: str) -> bytes | None:
    """One member of a whole package, or None if it is not there.

    Never raises. Every caller is on a handler's path, and a handler degrades.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as package:
            return package.read(name)
    except (KeyError, OSError, zipfile.BadZipFile):
        return None


def part_names(data: bytes) -> tuple[str, ...]:
    """Every member of a whole package, or an empty tuple if it is unreadable."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as package:
            return tuple(package.namelist())
    except (OSError, zipfile.BadZipFile):
        return ()
