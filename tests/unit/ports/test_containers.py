"""The container port's own invariants."""

import pytest

from readeverything.ports.containers import (
    ARCHIVE_MIMES,
    NOT_A_FOLDER_MIMES,
    NOT_A_FOLDER_SUFFIXES,
    ArchiveEntry,
    ContainerLimits,
)


def test_default_limits_match_the_spec() -> None:
    limits = ContainerLimits()
    assert limits.max_depth == 3
    assert limits.max_member_bytes == 1 << 30
    assert limits.max_total_bytes == 4 << 30
    assert limits.max_members == 10_000
    assert limits.max_expansion_ratio == 200.0
    assert limits.max_materialised_bytes == 8 << 30
    assert limits.walk_members is True


def test_limits_are_frozen() -> None:
    """Configuration a caller passed must not drift under them mid-walk."""
    limits = ContainerLimits()
    with pytest.raises(AttributeError):
        limits.max_depth = 9  # type: ignore[misc]


def test_an_entry_rejects_a_negative_size() -> None:
    with pytest.raises(ValueError, match="size_bytes"):
        ArchiveEntry(
            path="a.txt",
            size_bytes=-1,
            compressed_bytes=0,
            is_dir=False,
            is_symlink=False,
            modified_epoch_s=None,
            byte_offset=None,
        )


def test_an_entry_rejects_a_negative_compressed_size() -> None:
    with pytest.raises(ValueError, match="compressed_bytes"):
        ArchiveEntry(
            path="a.txt",
            size_bytes=0,
            compressed_bytes=-1,
            is_dir=False,
            is_symlink=False,
            modified_epoch_s=None,
            byte_offset=None,
        )


def test_an_entry_rejects_an_empty_path() -> None:
    with pytest.raises(ValueError, match="path"):
        ArchiveEntry(
            path="",
            size_bytes=0,
            compressed_bytes=0,
            is_dir=False,
            is_symlink=False,
            modified_epoch_s=None,
            byte_offset=None,
        )


def test_zip_based_documents_are_not_folders() -> None:
    """A .docx is a zip and is emphatically not a directory of XML parts."""
    docx = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert docx in NOT_A_FOLDER_MIMES
    assert "application/epub+zip" in NOT_A_FOLDER_MIMES
    assert ".docx" in NOT_A_FOLDER_SUFFIXES
    assert ".jar" in NOT_A_FOLDER_SUFFIXES


def test_plain_archives_are_claimed() -> None:
    assert "application/zip" in ARCHIVE_MIMES
    assert "application/x-tar" in ARCHIVE_MIMES
    assert "application/gzip" in ARCHIVE_MIMES


def test_the_two_sets_do_not_overlap() -> None:
    """An overlap would make "descend into this" ambiguous by construction."""
    assert not (ARCHIVE_MIMES & NOT_A_FOLDER_MIMES)
