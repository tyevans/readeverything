"""`DocumentRenderer` over LibreOffice's `soffice`.

The verified command, measured against LibreOffice 25.8.7.3 on this machine:

    soffice -env:UserInstallation=file:///<profile> --headless --invisible \
            --nologo --nodefault --nolockcheck --norestore \
            --convert-to pdf --outdir <tmpdir> <path>

    exit 0, stdout "convert ... -> .../a.pdf using filter : impress_pdf_Export"

**One conversion, not one per page.** `--convert-to pdf` produces a PDF of the
whole document, and page images then come from pypdfium2 via
`adapters/pdfium_render.py` — the same code `handlers/pdf.py` renders through.
So this adapter is a *converter*, and rendering is not reimplemented here.

**The converted PDF is an artifact.** It is stored under the source's content
hash plus this adapter's version, so a document is converted once per machine
rather than once per page. Conversion is seconds; without this, a
four-hundred-slide deck would pay it four hundred times. It is also why
`render_page` after the first call is fast.

**Exit status is not evidence of output.** soffice exits 0 having written no
PDF for inputs it declines. The produced file is what is checked, exactly as
`ffmpeg_frames.py` checks output length rather than `returncode` — a different
binary, the same measured trap.

Security. This is the only place in the library where a document could execute
code, and it is treated as such:

* **Macros are disabled by configuration, because there is no flag.** `soffice
  --help` on 25.8.7.3 offers `--safe-mode` and nothing else relevant: macro
  policy lives in the user profile. LibreOffice's own registry schema
  (`share/registry/main.xcd`) says of `DisableMacrosExecution` — "Specifies
  whether the macro execution is disabled in general. This will disable Basic,
  Beanshell, Javascript and Python scripts. If it is set to true, the
  'MacroSecurityLevel' is ignored." Its shipped default is **false**, with
  `MacroSecurityLevel` 2 (which prompts, and a headless process cannot answer a
  prompt). So the adapter seeds `registrymodifications.xcu` in the private
  profile before the first conversion. Verified by running a real conversion
  against a seeded profile and reading the file back: LibreOffice parsed and
  preserved all three settings.
* **A private profile per instance**, via `-env:UserInstallation=`. It must be
  a `file://` URL; a bare path is silently ignored, which would fall back to
  the invoking user's real LibreOffice profile — a library must never write
  there, and soffice is not concurrency-safe across processes sharing one.
* **A bounded, mandatory timeout with a kill and a reap.** A malformed
  document can hang soffice indefinitely, and a perception must never block
  forever.
* **No shell.** `asyncio.create_subprocess_exec` takes an argument vector; the
  document's path is its own element and is never formatted into a string.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import tempfile
from pathlib import Path

from readeverything.adapters.ooxml import (
    ODF_SHEETS_MIME,
    ODF_SLIDES_MIME,
    ODF_TEXT_MIME,
    SHEETS_MIME,
    SLIDES_MIME,
    WORD_MIME,
)
from readeverything.adapters.pdfium_render import page_count, render_page_png
from readeverything.domain.capability import Capability
from readeverything.domain.errors import InfrastructureError, RenditionFailedError
from readeverything.domain.identity import MimeType
from readeverything.ports.artifacts import ArtifactStore
from readeverything.ports.limits import Limiter

#: Bumped when a change here would produce a different PDF from the same bytes.
#: It is half the artifact key — the source's content hash is the other half —
#: so bumping it invalidates exactly what it should and nothing else.
ADAPTER_VERSION = 1

#: The legacy OLE2 mimetypes. Only `application/msword` is reachable through
#: `PuremagicDetector` today: a compound-file header is identical across Word,
#: Excel and PowerPoint, puremagic reports `application/msword` for all three
#: at 0.80 confidence, and that is above the signature floor so the filename is
#: never consulted. The other two are listed because a caller with a real OLE2
#: directory walker plugs in a better detector and this is then already right.
#: It costs nothing today, because soffice detects the real format itself.
LEGACY_WORD_MIME = "application/msword"
LEGACY_SLIDES_MIME = "application/vnd.ms-powerpoint"
LEGACY_SHEETS_MIME = "application/vnd.ms-excel"

#: Everything this converter will hand to soffice. Deliberately NOT
#: `application/pdf`: `handlers/pdf.py` renders those directly, and claiming
#: them would route a file through a subprocess it does not need.
RENDERABLE_MIMETYPES: frozenset[str] = frozenset(
    {
        WORD_MIME,
        SLIDES_MIME,
        SHEETS_MIME,
        ODF_TEXT_MIME,
        ODF_SLIDES_MIME,
        ODF_SHEETS_MIME,
        LEGACY_WORD_MIME,
        LEGACY_SLIDES_MIME,
        LEGACY_SHEETS_MIME,
    }
)

#: What is written into the private profile before the first conversion, as
#: `(name, value)` under `/org.openoffice.Office.Common/Security/Scripting`.
#:
#: `DisableMacrosExecution` is the load-bearing one and the other two are
#: defence in depth: LibreOffice ignores `MacroSecurityLevel` while it is true,
#: but a future change here that dropped it would leave 3 ("very high": run
#: only from trusted locations, of which a fresh profile has none) rather than
#: the shipped default of 2, which prompts. `BlockUntrustedRefererLinks` stops
#: a document pulling a remote resource in during conversion.
MACRO_SECURITY_SETTINGS: tuple[tuple[str, str], ...] = (
    ("DisableMacrosExecution", "true"),
    ("MacroSecurityLevel", "3"),
    ("BlockUntrustedRefererLinks", "true"),
)

_SECURITY_PATH = "/org.openoffice.Office.Common/Security/Scripting"

_XCU_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry"\
 xmlns:xs="http://www.w3.org/2001/XMLSchema"\
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
{items}
</oor:items>
"""

_CHUNK = 1 << 20


def _security_profile() -> str:
    items = "\n".join(
        f'  <item oor:path="{_SECURITY_PATH}">'
        f'<prop oor:name="{name}" oor:op="fuse"><value>{value}</value></prop>'
        f"</item>"
        for name, value in MACRO_SECURITY_SETTINGS
    )
    return _XCU_TEMPLATE.format(items=items)


def _hash_file(path: str) -> str:
    digest = hashlib.blake2b(digest_size=32)
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


class SofficeRenderer:
    """Converts a document to PDF once, then renders its pages from that."""

    def __init__(
        self,
        *,
        artifacts: ArtifactStore,
        limiter: Limiter | None = None,
        executable: str = "soffice",
        timeout_s: float = 120.0,
        profile_root: Path | str | None = None,
        binary_revision: str | None = None,
    ) -> None:
        """`profile_root` is where the private LibreOffice profile is created.

        `None` means a temporary directory owned by this instance, cleaned up
        when the instance is collected. A caller passing a path keeps the
        profile across runs, which saves LibreOffice's first-start setup — and
        is what the unit tests read back to check what was seeded.

        `binary_revision` is the version string the capability probe already
        obtained, if the caller has it. It tightens the cache key so a
        LibreOffice upgrade is a miss rather than a hit. It is optional
        because obtaining it costs a subprocess, and the design's stated key is
        content hash plus adapter version; a cached PDF from an older
        LibreOffice is a valid rendering of the same source, not a wrong one.
        """
        self._artifacts = artifacts
        self._limiter = limiter
        self._executable = executable
        self._timeout_s = timeout_s
        self._binary_revision = binary_revision
        # Held on the instance so its finaliser cleans the directory up when
        # the renderer is collected. A bare mkdtemp would leak a profile per
        # process, which on a long-running agent is a slow disk leak.
        self._owned_profile: tempfile.TemporaryDirectory[str] | None = None
        if profile_root is None:
            self._owned_profile = tempfile.TemporaryDirectory(prefix="readeverything-soffice-")
            self._profile = Path(self._owned_profile.name)
        else:
            self._profile = Path(profile_root)
        self._seeded = False

    # -- the port --------------------------------------------------------

    @property
    def revision(self) -> str:
        if self._binary_revision is None:
            return f"soffice/{ADAPTER_VERSION}"
        return f"soffice {self._binary_revision}/{ADAPTER_VERSION}"

    def claims(self, mime: MimeType) -> bool:
        return str(mime) in RENDERABLE_MIMETYPES

    async def page_count(self, path: str) -> int:
        return await asyncio.to_thread(page_count, await self._pdf(path))

    async def render_page(self, path: str, page: int, *, dpi: int = 150) -> bytes:
        data = await self._pdf(path)
        try:
            return await asyncio.to_thread(render_page_png, data, page, dpi=dpi)
        except InfrastructureError as exc:
            # `RenditionFailedError` is what this port promises; a bad page
            # number arriving as a bare `InfrastructureError` would be a
            # failure a caller cannot distinguish from a converter crash.
            raise RenditionFailedError(str(exc)) from exc

    # -- conversion ------------------------------------------------------

    def _key(self, content_hash: str) -> str:
        return f"soffice-pdf/{self.revision}/{content_hash}"

    async def _pdf(self, path: str) -> bytes:
        """The converted PDF's bytes, from the store if it has been seen.

        The store is consulted BEFORE the limiter is taken. Queueing a cache
        hit behind another document's conversion would serialise reads that do
        no work at all, which on a deck read page by page is the whole cost of
        the feature reappearing.
        """
        try:
            content_hash = await asyncio.to_thread(_hash_file, path)
        except OSError as exc:
            raise RenditionFailedError(f"could not read {path!r} to convert it: {exc}") from exc
        key = self._key(content_hash)
        cached = await self._artifacts.get(key)
        if cached is not None:
            return cached
        if self._limiter is None:
            pdf = await self._convert(path)
        else:
            async with self._limiter.limit(Capability.DOCUMENT_RENDER):
                pdf = await self._convert(path)
        await self._artifacts.put(key, pdf)
        return pdf

    async def _seed_profile(self) -> None:
        """Write the macro policy into the profile, once per instance.

        Written BEFORE the first launch: LibreOffice reads
        `registrymodifications.xcu` at startup, and a policy applied after the
        process is running would apply to nothing.
        """
        if self._seeded:
            return

        def _write() -> None:
            user = self._profile / "user"
            user.mkdir(parents=True, exist_ok=True)
            (user / "registrymodifications.xcu").write_text(_security_profile(), encoding="utf-8")

        try:
            await asyncio.to_thread(_write)
        except OSError as exc:
            raise RenditionFailedError(
                f"could not prepare a private LibreOffice profile at {self._profile}: {exc}"
            ) from exc
        self._seeded = True

    def _argv(self, path: str, outdir: str) -> tuple[str, ...]:
        return (
            self._executable,
            # A URL, not a path: a bare path is silently ignored and the
            # invoking user's real profile is used instead, with nothing said.
            f"-env:UserInstallation={self._profile.resolve().as_uri()}",
            "--headless",
            "--invisible",
            "--nologo",
            "--nodefault",
            "--nolockcheck",
            "--norestore",
            "--convert-to",
            "pdf",
            "--outdir",
            outdir,
            path,
        )

    async def _convert(self, path: str) -> bytes:
        await self._seed_profile()
        with tempfile.TemporaryDirectory(prefix="readeverything-convert-") as outdir:
            await self._spawn(self._argv(path, outdir))
            produced = sorted(Path(outdir).glob("*.pdf"))
            if not produced:
                # soffice exits 0 having written nothing for inputs it
                # declines. Exit status is not evidence of output; see the
                # module docstring.
                raise RenditionFailedError(
                    f"soffice produced no PDF for {path!r}; it may not be a document "
                    f"LibreOffice can open"
                )
            return await asyncio.to_thread(produced[0].read_bytes)

    async def _spawn(self, argv: tuple[str, ...]) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise RenditionFailedError(f"could not start {argv[0]!r}: {exc}") from exc

        try:
            _stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self._timeout_s)
        except TimeoutError as exc:
            # Kill AND reap. Stopping the wait alone leaves the child running
            # and unclaimed, which on a long-running agent is a slow leak
            # rather than a visible failure. The child may have exited in the
            # gap, so `ProcessLookupError` is not an error here.
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            raise RenditionFailedError(
                f"conversion exceeded its {self._timeout_s}s deadline and was killed"
            ) from exc

        if process.returncode != 0:
            raise RenditionFailedError(
                f"conversion failed (exit {process.returncode}): "
                f"{stderr.decode('utf-8', errors='replace').strip()}"
            )
