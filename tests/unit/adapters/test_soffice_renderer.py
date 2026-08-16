"""`SofficeRenderer` against stand-in executables. No LibreOffice required.

The stand-ins are shell scripts, matching how `test_binary_probe.py` tests a
probe: what is under test is the adapter's contract — which flags it passes,
what it does when the child hangs, when it converts a document twice — not
LibreOffice's PDF output, which `tests/integration/test_soffice_renderer.py`
covers against the real binary.
"""

from __future__ import annotations

import asyncio
import shlex
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from readeverything.adapters.artifact_store import InMemoryArtifactStore
from readeverything.adapters.soffice_renderer import (
    MACRO_SECURITY_SETTINGS,
    RENDERABLE_MIMETYPES,
    SofficeRenderer,
)
from readeverything.domain.capability import Capability
from readeverything.domain.errors import RenditionFailedError
from readeverything.domain.identity import MimeType
from readeverything.ports.rendering import DocumentRenderer
from tests.fixtures_pdf import born_digital


def _script(path: Path, body: str) -> str:
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def argv_log(tmp_path: Path) -> Path:
    return tmp_path / "argv.log"


@pytest.fixture
def converter(tmp_path: Path, argv_log: Path) -> str:
    """A stand-in that records its argv and writes a real 2-page PDF out.

    It finds `--outdir` positionally the way the adapter passes it, so a
    reordering of the argument vector breaks this test rather than passing
    quietly.
    """
    pdf = tmp_path / "canned.pdf"
    pdf.write_bytes(born_digital(["alpha", "beta"]))
    return _script(
        tmp_path / "fake-soffice",
        f'echo "$@" >> {shlex.quote(str(argv_log))}\n'
        "while [ $# -gt 0 ]; do\n"
        '  if [ "$1" = "--outdir" ]; then shift; out="$1"; fi\n'
        "  shift\n"
        "done\n"
        f'cp {shlex.quote(str(pdf))} "$out/converted.pdf"\n',
    )


@pytest.fixture
def deck(tmp_path: Path) -> str:
    path = tmp_path / "deck.pptx"
    path.write_bytes(b"not really a deck, and it does not matter: the stand-in ignores it")
    return str(path)


def _renderer(executable: str, **kwargs: object) -> SofficeRenderer:
    return SofficeRenderer(
        artifacts=InMemoryArtifactStore(),
        executable=executable,
        **kwargs,  # type: ignore[arg-type]
    )


# --- the port -------------------------------------------------------------


def test_the_adapter_satisfies_the_port(tmp_path: Path) -> None:
    assert isinstance(_renderer("soffice"), DocumentRenderer)


def test_it_claims_the_formats_pypdfium2_cannot_open(tmp_path: Path) -> None:
    renderer = _renderer("soffice")
    for mime in RENDERABLE_MIMETYPES:
        assert renderer.claims(MimeType.parse(mime)), mime


def test_it_does_not_claim_pdf(tmp_path: Path) -> None:
    """PDFs already have a handler that renders them directly. Claiming them
    would route a file through a subprocess it does not need."""
    assert not _renderer("soffice").claims(MimeType.parse("application/pdf"))


def test_it_does_not_claim_arbitrary_binaries(tmp_path: Path) -> None:
    assert not _renderer("soffice").claims(MimeType.parse("application/octet-stream"))


def test_the_revision_names_the_adapter_version(tmp_path: Path) -> None:
    """It keys the converted PDF. A converter change that produced different
    pages from the same bytes must not be served the old ones."""
    assert "soffice" in _renderer("soffice").revision


# --- security: this is the only place a document can execute code ---------


async def test_conversion_runs_with_macro_execution_disabled(
    converter: str, deck: str, argv_log: Path, tmp_path: Path
) -> None:
    """A document from an untrusted directory must not execute on conversion.

    LibreOffice has NO command-line flag for this — checked against `soffice
    --help` on 25.8.7.3, which offers `--safe-mode` and nothing else relevant.
    Macro policy is a *profile* setting, so the adapter seeds the private
    profile it is about to use. `DisableMacrosExecution` is the authoritative
    one: LibreOffice's own registry schema says it "will disable Basic,
    Beanshell, Javascript and Python scripts" and that `MacroSecurityLevel` is
    ignored when it is true. Its shipped default is FALSE, with
    `MacroSecurityLevel` 2 — so doing nothing here would leave macros governed
    by a prompt no headless process can answer.
    """
    profile = tmp_path / "profile"
    renderer = _renderer(converter, profile_root=profile)
    await renderer.page_count(deck)

    seeded = (profile / "user" / "registrymodifications.xcu").read_text(encoding="utf-8")
    assert "DisableMacrosExecution" in seeded
    assert "<value>true</value>" in seeded
    for name, value in MACRO_SECURITY_SETTINGS:
        assert f'oor:name="{name}"' in seeded
        assert f"<value>{value}</value>" in seeded


async def test_conversion_uses_a_private_profile_rather_than_the_users(
    converter: str, deck: str, argv_log: Path, tmp_path: Path
) -> None:
    """soffice is not concurrency-safe across processes sharing a profile, and
    a library must never write into the invoking user's LibreOffice settings.
    """
    profile = tmp_path / "profile"
    await _renderer(converter, profile_root=profile).page_count(deck)

    argv = argv_log.read_text().strip()
    assert f"-env:UserInstallation=file://{profile}" in argv


async def test_the_profile_is_a_file_url_not_a_bare_path(
    converter: str, deck: str, argv_log: Path, tmp_path: Path
) -> None:
    """`-env:UserInstallation` takes a URL. A bare path is silently ignored,
    which means the user's real profile is used and nothing says so."""
    await _renderer(converter, profile_root=tmp_path / "profile").page_count(deck)
    assert "-env:UserInstallation=file:///" in argv_log.read_text()


async def test_conversion_is_headless_and_never_restores_a_previous_session(
    converter: str, deck: str, argv_log: Path, tmp_path: Path
) -> None:
    argv = shlex.split(await _run_and_read(converter, deck, argv_log, tmp_path))
    for flag in ("--headless", "--invisible", "--nologo", "--nodefault", "--norestore"):
        assert flag in argv, flag


async def test_conversion_asks_for_pdf_and_names_no_shell(
    converter: str, deck: str, argv_log: Path, tmp_path: Path
) -> None:
    """The document's path is its own argv element, never formatted into a
    string a shell would parse — the rule every other spawning adapter here
    follows."""
    argv = shlex.split(await _run_and_read(converter, deck, argv_log, tmp_path))
    assert "--convert-to" in argv
    assert argv[argv.index("--convert-to") + 1] == "pdf"
    assert deck in argv


async def _run_and_read(converter: str, deck: str, argv_log: Path, tmp_path: Path) -> str:
    await _renderer(converter, profile_root=tmp_path / "profile").page_count(deck)
    return argv_log.read_text().strip()


# --- timeouts -------------------------------------------------------------


async def test_a_conversion_that_hangs_is_killed_and_reported(tmp_path: Path, deck: str) -> None:
    """A malformed document can hang soffice indefinitely. It must never block
    a perception forever, and the deadline is not optional."""
    hang = _script(tmp_path / "hang", "sleep 30")
    renderer = _renderer(hang, profile_root=tmp_path / "profile", timeout_s=0.3)
    with pytest.raises(RenditionFailedError):
        await renderer.page_count(deck)


async def test_a_hung_conversion_leaves_no_running_child(tmp_path: Path, deck: str) -> None:
    """Killed AND reaped. A timeout that only stops waiting accumulates
    zombies, which on a long-running agent is a slow leak rather than a
    visible failure."""
    marker = tmp_path / "still-alive"
    hang = _script(
        tmp_path / "hang",
        f"sleep 0.6\ntouch {shlex.quote(str(marker))}\n",
    )
    renderer = _renderer(hang, profile_root=tmp_path / "profile", timeout_s=0.2)
    with pytest.raises(RenditionFailedError):
        await renderer.page_count(deck)
    await asyncio.sleep(1.0)
    assert not marker.exists(), "the timed-out child kept running after the deadline"


async def test_a_converter_that_fails_is_reported_rather_than_swallowed(
    tmp_path: Path, deck: str
) -> None:
    fails = _script(tmp_path / "fails", "echo 'source file could not be loaded' >&2\nexit 1")
    with pytest.raises(RenditionFailedError):
        await _renderer(fails, profile_root=tmp_path / "profile").page_count(deck)


async def test_a_converter_that_exits_clean_but_writes_nothing_is_a_failure(
    tmp_path: Path, deck: str
) -> None:
    """Measured trap, and the same one `ffmpeg_frames` documents: exit status
    is not evidence of output. soffice exits 0 having written no PDF for
    inputs it declines, and treating that as success hands a caller zero pages
    for a document that has some."""
    quiet = _script(tmp_path / "quiet", "exit 0")
    with pytest.raises(RenditionFailedError):
        await _renderer(quiet, profile_root=tmp_path / "profile").page_count(deck)


async def test_a_missing_executable_is_reported_rather_than_raising_oserror(
    tmp_path: Path, deck: str
) -> None:
    renderer = _renderer("definitely-not-a-real-binary-xyz", profile_root=tmp_path / "profile")
    with pytest.raises(RenditionFailedError):
        await renderer.page_count(deck)


# --- caching --------------------------------------------------------------


async def test_a_document_is_converted_once_and_then_served_from_the_store(
    converter: str, deck: str, argv_log: Path, tmp_path: Path
) -> None:
    """The single most important performance decision in the design: conversion
    is seconds, and without this every page render would pay it again."""
    renderer = _renderer(converter, profile_root=tmp_path / "profile")
    assert await renderer.page_count(deck) == 2
    assert (await renderer.render_page(deck, 1, dpi=72)).startswith(b"\x89PNG")

    assert len(argv_log.read_text().strip().splitlines()) == 1


async def test_page_renders_after_the_first_call_do_not_reconvert(
    converter: str, deck: str, argv_log: Path, tmp_path: Path
) -> None:
    renderer = _renderer(converter, profile_root=tmp_path / "profile")
    await renderer.render_page(deck, 1, dpi=72)
    await renderer.render_page(deck, 2, dpi=72)
    await renderer.page_count(deck)

    assert len(argv_log.read_text().strip().splitlines()) == 1


async def test_a_second_renderer_sharing_the_store_does_not_reconvert(
    converter: str, deck: str, argv_log: Path, tmp_path: Path
) -> None:
    """Once per machine, ever — not once per process. The cache is the store,
    not an instance attribute."""
    store = InMemoryArtifactStore()
    first = SofficeRenderer(artifacts=store, executable=converter, profile_root=tmp_path / "p1")
    second = SofficeRenderer(artifacts=store, executable=converter, profile_root=tmp_path / "p2")
    await first.page_count(deck)
    await second.page_count(deck)

    assert len(argv_log.read_text().strip().splitlines()) == 1


async def test_a_different_document_is_a_cache_miss(
    converter: str, deck: str, argv_log: Path, tmp_path: Path
) -> None:
    other = tmp_path / "other.pptx"
    other.write_bytes(b"different bytes entirely")
    renderer = _renderer(converter, profile_root=tmp_path / "profile")
    await renderer.page_count(deck)
    await renderer.page_count(str(other))

    assert len(argv_log.read_text().strip().splitlines()) == 2


async def test_the_cache_key_carries_the_renderer_revision(
    converter: str, deck: str, tmp_path: Path
) -> None:
    """A converter upgrade must not serve pages the old one produced."""
    store = InMemoryArtifactStore()
    await SofficeRenderer(
        artifacts=store, executable=converter, profile_root=tmp_path / "p"
    ).page_count(deck)
    (key,) = store.keys()
    assert "soffice" in key


# --- rendering ------------------------------------------------------------


async def test_a_rendered_page_is_a_png(converter: str, deck: str, tmp_path: Path) -> None:
    png = await _renderer(converter, profile_root=tmp_path / "profile").render_page(deck, 1, dpi=72)
    assert png.startswith(b"\x89PNG")


async def test_a_page_past_the_end_fails_rather_than_returning_the_last_one(
    converter: str, deck: str, tmp_path: Path
) -> None:
    renderer = _renderer(converter, profile_root=tmp_path / "profile")
    with pytest.raises(RenditionFailedError):
        await renderer.render_page(deck, 99, dpi=72)


# --- serialisation --------------------------------------------------------


async def test_conversions_are_serialised_through_the_limiter(
    converter: str, deck: str, tmp_path: Path
) -> None:
    """soffice is not concurrency-safe across processes. The limiter is where
    that is enforced, under DOCUMENT_RENDER, exactly as VISION is bounded."""
    asked: list[Capability] = []

    class _Recording:
        @asynccontextmanager
        async def limit(self, capability: Capability) -> AsyncIterator[None]:
            asked.append(capability)
            yield

    renderer = SofficeRenderer(
        artifacts=InMemoryArtifactStore(),
        executable=converter,
        profile_root=tmp_path / "profile",
        limiter=_Recording(),
    )
    await renderer.page_count(deck)

    assert asked == [Capability.DOCUMENT_RENDER]


async def test_a_cache_hit_does_not_take_the_limiter(
    converter: str, deck: str, tmp_path: Path
) -> None:
    """Otherwise every page render of a converted document queues behind every
    other document's conversion, for no work at all."""
    asked: list[Capability] = []

    class _Recording:
        @asynccontextmanager
        async def limit(self, capability: Capability) -> AsyncIterator[None]:
            asked.append(capability)
            yield

    renderer = SofficeRenderer(
        artifacts=InMemoryArtifactStore(),
        executable=converter,
        profile_root=tmp_path / "profile",
        limiter=_Recording(),
    )
    await renderer.page_count(deck)
    await renderer.render_page(deck, 1, dpi=72)

    assert asked == [Capability.DOCUMENT_RENDER]


def test_the_default_limits_serialise_document_render() -> None:
    """One at a time, not four. Two soffice processes racing on one profile is
    the failure this bound exists for, so the default is 1 rather than the 4
    ffmpeg and vision use."""
    from readeverything.adapters.semaphore_limiter import DEFAULT_LIMITS

    assert DEFAULT_LIMITS[Capability.DOCUMENT_RENDER] == 1
