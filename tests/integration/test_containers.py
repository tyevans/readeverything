"""The §1.1 acceptance, end to end, with fixtures built in a tmpdir.

The point of this file is the thing it does NOT do: no handler was modified to
make any of it pass. The PDF handler descends into a tarball inside a zip
because it reads through `SourceReader` and cannot tell where its bytes came
from.
"""

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from readeverything.composition import build_perception
from readeverything.domain.capability import CapabilitySet
from readeverything.domain.errors import ContainerLimitExceededError
from readeverything.domain.rendition import Budget, TextContent
from readeverything.ports.containers import ContainerLimits

pytestmark = pytest.mark.integration

pdfium = pytest.importorskip("pypdfium2")


def _pdf(pages: int) -> bytes:
    document = pdfium.PdfDocument.new()
    try:
        for _ in range(pages):
            document.new_page(200, 200)
        buffer = io.BytesIO()
        document.save(buffer)
        return buffer.getvalue()
    finally:
        document.close()


def _targz(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """`docs.zip` holding `report.pdf` and `nested.tar.gz` holding `notes.txt`."""
    with zipfile.ZipFile(tmp_path / "docs.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.pdf", _pdf(9))
        archive.writestr("nested.tar.gz", _targz({"notes.txt": b"the note body\n"}))
    return tmp_path


async def test_list_returns_members_addressed_by_the_grammar(root: Path) -> None:
    perception = await build_perception(root, capabilities=CapabilitySet.empty())
    listed = sorted(await perception.list("."))
    assert listed == [
        "docs.zip",
        "docs.zip!nested.tar.gz",
        "docs.zip!nested.tar.gz!notes.txt",
        "docs.zip!report.pdf",
    ]


async def test_inspecting_an_archived_pdf_gives_a_pdf_card(root: Path) -> None:
    """No registry change, no handler change. This is the whole spec."""
    perception = await build_perception(root, capabilities=CapabilitySet.empty())
    card = await perception.inspect("docs.zip!report.pdf")
    assert card.facts["page_count"] == 9
    assert "read_page" in card.affordance_names()


async def test_inspecting_a_doubly_nested_text_member_gives_a_text_card(root: Path) -> None:
    perception = await build_perception(root, capabilities=CapabilitySet.empty())
    card = await perception.inspect("docs.zip!nested.tar.gz!notes.txt")
    assert "read_range" in card.affordance_names()
    assert card.excerpt is not None
    assert "the note body" in card.excerpt


async def test_page_seven_of_the_nested_pdf_cites_the_full_nested_path(root: Path) -> None:
    perception = await build_perception(root, capabilities=CapabilitySet.empty())
    rendition = await perception.invoke("docs.zip!report.pdf", "read_page", {"page": 7})
    assert isinstance(rendition.content, TextContent)
    card = await perception.inspect("docs.zip!report.pdf")
    assert card.ref.uri == "docs.zip!report.pdf"


async def test_a_members_hash_is_of_its_decompressed_bytes(root: Path, tmp_path: Path) -> None:
    """What makes the artifact cache warm across the boundary.

    Extract a file from a zip and its cached OCR still hits, because a member
    and the same file loose on disk hash identically.
    """
    loose = tmp_path / "loose"
    loose.mkdir()
    (loose / "notes.txt").write_bytes(b"the note body\n")
    inside = await build_perception(root, capabilities=CapabilitySet.empty())
    outside = await build_perception(loose, capabilities=CapabilitySet.empty())
    member = await inside.inspect("docs.zip!nested.tar.gz!notes.txt")
    plain = await outside.inspect("notes.txt")
    assert member.ref.content_hash == plain.ref.content_hash


async def test_the_container_itself_still_gets_an_archive_card(root: Path) -> None:
    perception = await build_perception(root, capabilities=CapabilitySet.empty())
    card = await perception.inspect("docs.zip")
    assert card.facts["entry_count"] == 2
    assert card.affordance_names() == ("list_entries",)


async def test_a_zip_bomb_is_refused_with_a_bounded_error(tmp_path: Path) -> None:
    """Refused, not truncated: half a file reported on as whole is the harm."""
    with zipfile.ZipFile(tmp_path / "bomb.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("payload", b"\0" * (1 << 22))
    perception = await build_perception(
        tmp_path,
        capabilities=CapabilitySet.empty(),
        containers=ContainerLimits(max_expansion_ratio=2.0),
    )
    with pytest.raises(ContainerLimitExceededError):
        await perception.inspect("bomb.zip!payload")


async def test_a_corrupt_member_does_not_blind_the_agent_to_its_neighbours(
    tmp_path: Path,
) -> None:
    (tmp_path / "broken.zip").write_bytes(b"PK\x03\x04 and then garbage")
    (tmp_path / "fine.txt").write_bytes(b"readable\n")
    perception = await build_perception(tmp_path, capabilities=CapabilitySet.empty())
    assert sorted(await perception.list(".")) == ["broken.zip", "fine.txt"]
    card = await perception.inspect("fine.txt")
    assert card.excerpt is not None
    assert "readable" in card.excerpt


async def test_containers_none_yields_todays_behavior(root: Path) -> None:
    perception = await build_perception(root, capabilities=CapabilitySet.empty(), containers=None)
    assert sorted(await perception.list(".")) == ["docs.zip"]


async def test_representing_the_container_lists_its_entries(root: Path) -> None:
    perception = await build_perception(root, capabilities=CapabilitySet.empty())
    rendered = await perception.represent("docs.zip", Budget(max_chars=None))
    assert "report.pdf" in rendered.text
    assert rendered.locator_map.length == len(rendered.text)
