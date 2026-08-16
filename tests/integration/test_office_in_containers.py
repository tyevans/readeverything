"""An office document inside an archive.

The composition of Spec 8 and Spec 9, which neither could test on its own:
they were built concurrently, in separate worktrees, sharing no code. Spec 9's
handlers read bytes through `SourceReader` and cannot tell whether those bytes
came from disk or from three containers deep; Spec 8's `NestedSource` makes a
member reachable without knowing what a member holds. Nothing was written to
join them, so this file is the evidence that the seam holds rather than a
feature in its own right.

The second half is the sharper test. A `.docx` IS a zip, so descent and
detection make opposite claims about it: descent says "this is a folder of XML
parts", detection says "this is a Word document". Detection must win, or every
office document in the corpus dissolves into `word/document.xml` fragments —
and the one place that can go wrong is exactly here, where both features are
installed at once.
"""

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from readeverything.composition import build_perception
from readeverything.domain.capability import CapabilitySet
from readeverything.domain.rendition import TextContent
from tests.fixtures_office import docx_bytes, pptx_bytes, xlsx_bytes

pytestmark = pytest.mark.integration


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
    """`corpus.zip` holds a deck and a tarball; the tarball holds a document.

    Deliberately two containers deep for the `.docx`, and a solid archive at
    that: a `.tar.gz` cannot seek to a member, so reaching the document exercises
    the materialisation path rather than a ranged read.
    """
    with zipfile.ZipFile(tmp_path / "corpus.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("quarterly.pptx", pptx_bytes())
        archive.writestr("books.tar.gz", _targz({"policy.docx": docx_bytes()}))
    (tmp_path / "loose.xlsx").write_bytes(xlsx_bytes())
    return tmp_path


async def test_an_archived_deck_gets_a_slide_card(root: Path) -> None:
    perception = await build_perception(root, capabilities=CapabilitySet.empty())

    card = await perception.inspect("corpus.zip!quarterly.pptx")

    assert card.facts["slide_count"] == 3
    assert "read_slide" in card.affordance_names()


async def test_a_doubly_nested_document_gets_a_word_card(root: Path) -> None:
    """Two containers deep, the outer one solid. No handler knows any of that."""
    perception = await build_perception(root, capabilities=CapabilitySet.empty())

    card = await perception.inspect("corpus.zip!books.tar.gz!policy.docx")

    assert "read_section" in card.affordance_names()
    assert [segment.label for segment in card.outline] == ["Alpha", "Bravo", "Charlie"]


async def test_reading_a_nested_section_cites_the_full_nested_path(root: Path) -> None:
    uri = "corpus.zip!books.tar.gz!policy.docx"
    perception = await build_perception(root, capabilities=CapabilitySet.empty())

    rendition = await perception.invoke(uri, "read_section", {"index": 0})

    # A `Rendition` carries a `locator`, not a ref: the uri lives on the card,
    # which is where a citation gets the "which file" half of its provenance.
    assert isinstance(rendition.content, TextContent)
    assert "alpha" in rendition.content.text.lower()
    card = await perception.inspect(uri)
    assert card.ref.uri == uri


async def test_an_office_document_is_a_document_not_a_folder(root: Path) -> None:
    """The one place descent and detection disagree, and detection must win.

    Without this, every `.docx` in a corpus is listed as a dozen XML parts and
    the document itself is never inspected.
    """
    perception = await build_perception(root, capabilities=CapabilitySet.empty())

    listed = await perception.list(".")

    assert "loose.xlsx" in listed
    assert not [uri for uri in listed if uri.startswith("loose.xlsx!")]
    assert not [uri for uri in listed if "quarterly.pptx!" in uri]


async def test_an_archived_member_hashes_like_the_same_file_loose(root: Path) -> None:
    """What keeps a cached artifact warm across the container boundary."""
    loose = root / "quarterly.pptx"
    loose.write_bytes(pptx_bytes())
    perception = await build_perception(root, capabilities=CapabilitySet.empty())

    inside = await perception.inspect("corpus.zip!quarterly.pptx")
    outside = await perception.inspect("quarterly.pptx")

    assert inside.ref.content_hash == outside.ref.content_hash
