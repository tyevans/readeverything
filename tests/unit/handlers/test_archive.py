"""The archive card, its paging, and the compliance laws."""

import zipfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from readeverything.adapters.nested_source import CompositeOpener
from readeverything.adapters.tar_archive import TarArchiveOpener
from readeverything.adapters.zip_archive import ZipArchiveOpener
from readeverything.domain.errors import UnknownAffordanceError
from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
from readeverything.domain.locators import ByteRange
from readeverything.domain.rendition import Budget, TextContent
from readeverything.handlers.archive import ArchiveHandler, ListEntriesParams
from readeverything.testing.handler_compliance import MediaHandlerCompliance


def _archive_bytes(tmp_path: Path, members: dict[str, bytes]) -> bytes:
    path = tmp_path / "built.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path.read_bytes()


class _PathSource:
    """A `SourceReader` that writes whatever it is asked for to a real file.

    The archive handler needs a PATH, because `ArchiveOpener` takes one -- so
    a fake that only serves bytes cannot exercise it. Serving the same content
    at any uri is also what `MediaHandlerCompliance` requires.
    """

    def __init__(self, *, content: bytes, root: Path) -> None:
        self._content = content
        self._root = root
        self._paths: dict[str, str] = {}

    async def read_bytes(self, uri: str) -> bytes:
        return self._content

    async def read_range(self, uri: str, start: int, end: int) -> bytes:
        return self._content[start:end]

    async def stream(self, uri: str, *, chunk_size: int = 1 << 20) -> AsyncIterator[bytes]:
        yield self._content

    async def local_path(self, uri: str) -> str:
        existing = self._paths.get(uri)
        if existing is None:
            target = self._root / f"{len(self._paths)}.bin"
            target.write_bytes(self._content)
            existing = str(target)
            self._paths[uri] = existing
        return existing


def _openers() -> CompositeOpener:
    return CompositeOpener(openers=[ZipArchiveOpener(), TarArchiveOpener()])


def _ref(size: int) -> SourceRef:
    return SourceRef(
        uri="release.zip",
        mime=MimeType.parse("application/zip"),
        content_hash=ContentHash("0" * 8),
        size_bytes=size,
    )


def _handler_over(tmp_path: Path, content: bytes) -> ArchiveHandler:
    return ArchiveHandler(source=_PathSource(content=content, root=tmp_path), archives=_openers())


@pytest.fixture
def built(tmp_path: Path) -> bytes:
    return _archive_bytes(tmp_path, {"a.txt": b"hello", "sub/b.txt": b"world"})


@pytest.fixture
def handler(tmp_path: Path, built: bytes) -> ArchiveHandler:
    return _handler_over(tmp_path, built)


async def test_the_card_is_binary_and_counts_entries(handler: ArchiveHandler, built: bytes) -> None:
    card = await handler.describe(_ref(len(built)))
    assert card.kind is MediaKind.BINARY
    assert card.facts["entry_count"] == 2
    assert card.facts["format"] == "application/zip"


async def test_the_card_reports_the_expansion_ratio(handler: ArchiveHandler, built: bytes) -> None:
    card = await handler.describe(_ref(len(built)))
    assert float(card.facts["expansion_ratio"]) > 0


async def test_the_card_reports_whether_the_container_is_solid(
    handler: ArchiveHandler, built: bytes
) -> None:
    """A zip is seekable, so reading three members is not three inflations."""
    card = await handler.describe(_ref(len(built)))
    assert card.facts["solid"] == "no"


async def test_the_card_outlines_every_entry(handler: ArchiveHandler, built: bytes) -> None:
    card = await handler.describe(_ref(len(built)))
    assert [segment.label for segment in card.outline] == ["a.txt", "sub/b.txt"]


async def test_the_card_excerpts_the_first_member_paths(
    handler: ArchiveHandler, built: bytes
) -> None:
    card = await handler.describe(_ref(len(built)))
    assert card.excerpt is not None
    assert "a.txt" in card.excerpt


async def test_the_only_affordance_is_list_entries(handler: ArchiveHandler) -> None:
    """No `read_entry`: a member is reached as `inspect('a.zip!inner.txt')`,
    and two ways to the same bytes means two provenance stories."""
    assert [a.name for a in handler.affordances()] == ["list_entries"]


async def test_list_entries_pages(tmp_path: Path) -> None:
    built = _archive_bytes(tmp_path, {f"f{n:02d}.txt": b"x" for n in range(10)})
    handler = _handler_over(tmp_path, built)
    rendition = await handler.invoke(
        _ref(len(built)), "list_entries", ListEntriesParams(offset=2, limit=3)
    )
    assert isinstance(rendition.content, TextContent)
    body = rendition.content.text
    assert "f02.txt" in body and "f04.txt" in body
    assert "f05.txt" not in body and "f01.txt" not in body


async def test_list_entries_past_the_end_degrades_rather_than_raising(
    handler: ArchiveHandler, built: bytes
) -> None:
    rendition = await handler.invoke(
        _ref(len(built)), "list_entries", ListEntriesParams(offset=999, limit=10)
    )
    assert rendition.degraded


async def test_list_entries_rejects_the_wrong_params_type(
    handler: ArchiveHandler, built: bytes
) -> None:
    """A params-type mismatch is a wiring bug, not bad input from an agent."""
    with pytest.raises(TypeError):
        await handler.invoke(_ref(len(built)), "list_entries", Budget())  # type: ignore[arg-type]


async def test_a_corrupt_archive_degrades_rather_than_raising(tmp_path: Path) -> None:
    """A handler never raises about its input, however broken."""
    handler = _handler_over(tmp_path, b"PK\x03\x04 garbage")
    card = await handler.describe(_ref(15))
    assert card.facts["readable"] == "no"
    rendered = await handler.represent(_ref(15), Budget(max_chars=None))
    assert rendered.degradations
    rendition = await handler.invoke(_ref(15), "list_entries", ListEntriesParams())
    assert rendition.degraded


async def test_an_empty_archive_is_not_reported_as_unreadable(tmp_path: Path) -> None:
    """It opened. Saying otherwise is a false claim about a file that read."""
    handler = _handler_over(tmp_path, _archive_bytes(tmp_path, {}))
    rendered = await handler.represent(_ref(22), Budget(max_chars=None))
    assert "no entries" in rendered.text
    assert "could not be opened" not in rendered.text


async def test_represent_maps_every_character_to_an_entry(
    handler: ArchiveHandler, built: bytes
) -> None:
    rendered = await handler.represent(_ref(len(built)), Budget(max_chars=None))
    assert rendered.locator_map.length == len(rendered.text)
    assert rendered.barriers == ()


async def test_represent_honours_the_budget(handler: ArchiveHandler, built: bytes) -> None:
    rendered = await handler.represent(_ref(len(built)), Budget(max_chars=5))
    assert len(rendered.text) == 5
    assert any(d.what == "text truncated" for d in rendered.degradations)


async def test_a_caller_can_supply_a_single_opener_of_their_own(
    tmp_path: Path, built: bytes
) -> None:
    """§9's extension point. A bare opener needs no composite to wrap it."""
    handler = ArchiveHandler(
        source=_PathSource(content=built, root=tmp_path), archives=ZipArchiveOpener()
    )
    card = await handler.describe(_ref(len(built)))
    assert card.facts["entry_count"] == 2


async def test_a_format_no_opener_claims_is_reported_unreadable(
    tmp_path: Path, built: bytes
) -> None:
    """Not a crash and not a silent empty listing: an honest "no"."""
    handler = ArchiveHandler(
        source=_PathSource(content=built, root=tmp_path),
        archives=CompositeOpener(openers=[TarArchiveOpener()]),
    )
    card = await handler.describe(_ref(len(built)))
    assert card.facts["readable"] == "no"


async def test_a_solid_container_locates_entries_without_inventing_offsets(
    tmp_path: Path,
) -> None:
    """A gzip stream has no seekable offsets, so an outline must not claim any."""
    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("a.txt")
        info.size = 5
        archive.addfile(info, io.BytesIO(b"hello"))
    content = buffer.getvalue()
    handler = _handler_over(tmp_path, content)
    ref = SourceRef(
        uri="release.tar.gz",
        mime=MimeType.parse("application/gzip"),
        content_hash=ContentHash("0" * 8),
        size_bytes=len(content),
    )
    card = await handler.describe(ref)
    assert card.facts["solid"] == "yes"
    assert not any(isinstance(segment.locator, ByteRange) for segment in card.outline)


async def test_an_unknown_affordance_raises(handler: ArchiveHandler, built: bytes) -> None:
    with pytest.raises(UnknownAffordanceError):
        await handler.invoke(_ref(len(built)), "read_entry", ListEntriesParams())


class TestArchiveHandlerCompliance(MediaHandlerCompliance):
    @pytest.fixture
    def ref(self, content: bytes) -> SourceRef:
        """Overridden to say `application/zip`.

        The base fixture says `application/octet-stream`, which this handler's
        composite opener does not claim -- so every law would run against the
        unreadable-archive path and prove nothing about a real one.
        """
        return SourceRef(
            uri="compliance-subject",
            mime=MimeType.parse("application/zip"),
            content_hash=ContentHash("0" * 64),
            size_bytes=len(content),
        )

    @pytest.fixture
    def content(self, tmp_path: Path) -> bytes:
        return _archive_bytes(tmp_path, {"a.txt": b"hello", "sub/b.txt": b"world"})

    @pytest.fixture
    def handler(self, tmp_path: Path, content: bytes) -> ArchiveHandler:
        return _handler_over(tmp_path, content)
