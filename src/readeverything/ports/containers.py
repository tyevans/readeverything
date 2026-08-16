"""Opening a container, and what it may cost.

A container format is adapter knowledge, exactly like a video codec: handlers
do not shell out and nothing above this layer learns what a central directory
is. Two adapters ship against this port, both stdlib -- which is why this
entire feature adds no dependency -- and a caller who wants `.7z` or `.rar`
supplies their own opener without this repository growing a dependency on
either.

`ContainerLimits` sits here rather than in `domain` because it describes what
an opener's CALLER will enforce while consuming this port, not a rule about
what a source is. Every field is an explicit constructor argument, per the
library's standing rule that nothing configures itself from the environment.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from readeverything.domain.identity import MimeType


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    """One member, as its container's own directory describes it.

    `compressed_bytes` is carried alongside `size_bytes` because the
    expansion-ratio guard needs both, and only the container knows the former.
    `byte_offset` is `None` for a solid container, which is the single fact
    that tells a caller whether a ranged read is genuinely ranged -- see the
    `NestedSource` docstring on seekable versus solid.
    """

    path: str
    size_bytes: int
    compressed_bytes: int
    is_dir: bool
    is_symlink: bool
    modified_epoch_s: float | None
    byte_offset: int | None

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("an archive entry's path must not be empty")
        if self.size_bytes < 0:
            raise ValueError(f"size_bytes must not be negative, got {self.size_bytes}")
        if self.compressed_bytes < 0:
            raise ValueError(f"compressed_bytes must not be negative, got {self.compressed_bytes}")


@runtime_checkable
class ArchiveOpener(Protocol):
    """Reads a container's directory, and one member's bytes at a time."""

    def claims(self, mime: MimeType) -> bool:
        """Whether this opener understands `mime`."""
        ...

    async def entries(self, path: str) -> Sequence[ArchiveEntry]:
        """Every member the container declares.

        A probe, not a decompression: reading a zip central directory or
        walking tar headers is a seek and a small read. Anything that costs
        more than that belongs behind `open_member`.
        """
        ...

    def open_member(self, path: str, member: str) -> AsyncIterator[bytes]:
        """The member's decompressed bytes, chunked.

        Declared non-`async` returning an `AsyncIterator`, matching
        `SourceReader.stream`, so an implementation is a plain async
        generator. Chunked rather than whole because the caller's expansion
        guard has to fire MID-STREAM: a zip bomb lies in its header, so
        checking a declared size after the fact is checking the bomb's own
        paperwork.
        """
        ...


@dataclass(frozen=True, slots=True)
class ContainerLimits:
    """What descending into a container is allowed to cost.

    Conservative by default, because the failure this bounds is a zip bomb
    filling a disk. `max_expansion_ratio` is the one that matters and the rest
    are belt: it is checked DURING decompression against bytes written so far,
    never afterwards against a declared size.

    `walk_members` exists because §3.1's inline listing is not free -- walking
    a directory now reads every archive's central directory, which on ten
    thousand zips is ten thousand extra opens. A caller who wants the old
    behavior turns it off rather than losing the feature entirely.
    """

    max_depth: int = 3
    max_member_bytes: int = 1 << 30
    max_total_bytes: int = 4 << 30
    max_members: int = 10_000
    max_expansion_ratio: float = 200.0
    max_materialised_bytes: int = 8 << 30
    walk_members: bool = True


#: Containers that ARE folders: descending into one yields sources a caller
#: wanted. Kept as strings rather than `MimeType` so a membership test costs a
#: `str(mime)` and no parsing.
ARCHIVE_MIMES: frozenset[str] = frozenset(
    {
        "application/zip",
        "application/x-zip-compressed",
        "application/x-tar",
        "application/x-gtar",
        "application/gzip",
        "application/x-gzip",
        "application/x-bzip2",
        "application/x-xz",
    }
)

#: Containers that are NOT folders, and the whole reason §3.1 needed a rule.
#:
#: A `.docx`, `.pptx`, `.xlsx`, `.odt`, `.epub` and `.jar` are all zip files.
#: Descending into one would list `report.docx!word/document.xml` as a source,
#: which is worse than useless: it buries the document itself under a dozen XML
#: parts. Once Spec 9's handlers claim these mimetypes they stop being folders
#: for the general reason -- no handler claims a plain zip, and these are
#: claimed -- and this explicit set is what keeps the behavior correct in the
#: interim rather than briefly wrong.
NOT_A_FOLDER_MIMES: frozenset[str] = frozenset(
    {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.oasis.opendocument.text",
        "application/vnd.oasis.opendocument.presentation",
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/epub+zip",
        "application/java-archive",
        "application/vnd.android.package-archive",
    }
)

#: The same rule, spelled as filenames.
#:
#: Not redundant with `NOT_A_FOLDER_MIMES`: detection is content-first, and the
#: bytes of a `.docx` ARE a zip -- puremagic can and does report
#: `application/zip` for one, which would sail straight past the mimetype set
#: and bury the document. Until Spec 9 §3 teaches the detector to report OOXML
#: and ODF as their own types, the filename is the only signal that survives,
#: and this is one of the narrow places where consulting it is correct rather
#: than lazy.
NOT_A_FOLDER_SUFFIXES: frozenset[str] = frozenset(
    {
        ".docx",
        ".docm",
        ".pptx",
        ".pptm",
        ".xlsx",
        ".xlsm",
        ".odt",
        ".odp",
        ".ods",
        ".epub",
        ".jar",
        ".war",
        ".apk",
        ".whl",
    }
)
