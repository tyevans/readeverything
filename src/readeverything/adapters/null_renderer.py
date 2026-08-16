"""A renderer that renders nothing, on purpose.

Determinism in CI is the whole reason this exists. A test asserting "no
rendering affordance appears" must be able to get that outcome on a machine
that happens to have LibreOffice installed, and uninstalling software is not an
acceptable way to configure a library.

Passing this to `build_perception(renderer=...)` turns rendering off
completely: no `DOCUMENT_RENDER` is declared, so no rendering affordance is
published and `OfficeLegacyHandler` is not registered. That is negotiation, not
degradation — nothing appears and then apologises.
"""

from __future__ import annotations

from readeverything.domain.errors import RenditionFailedError
from readeverything.domain.identity import MimeType


class NullRenderer:
    """Claims nothing, renders nothing, and says so."""

    #: Never reaches a cache key — nothing this renderer produces is stored —
    #: but the port requires it, and a caller printing it should read something
    #: that explains itself.
    revision = "null"

    def claims(self, mime: MimeType) -> bool:
        return False

    async def page_count(self, path: str) -> int:
        return 0

    async def render_page(self, path: str, page: int, *, dpi: int = 150) -> bytes:
        raise RenditionFailedError(
            "rendering is disabled: this composition was given a NullRenderer"
        )
