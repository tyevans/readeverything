"""The `DocumentRenderer` port: path in, PNG bytes out."""

from __future__ import annotations

import pytest

from readeverything.adapters.null_renderer import NullRenderer
from readeverything.domain.errors import InfrastructureError, RenditionFailedError
from readeverything.domain.identity import MimeType
from readeverything.ports.rendering import DocumentRenderer


class _Renderer:
    """A minimal structural implementation. It inherits nothing."""

    revision = "fake-1"

    def claims(self, mime: MimeType) -> bool:
        return str(mime) == "application/vnd.ms-powerpoint"

    async def page_count(self, path: str) -> int:
        return 3

    async def render_page(self, path: str, page: int, *, dpi: int = 150) -> bytes:
        return b"\x89PNG"

    async def page_text(self, path: str, page: int) -> str:
        return "the fourth slide"


def test_a_structural_implementation_satisfies_the_port() -> None:
    assert isinstance(_Renderer(), DocumentRenderer)


def test_an_unrelated_object_does_not_satisfy_the_port() -> None:
    assert not isinstance(object(), DocumentRenderer)


def test_a_rendition_failure_is_an_infrastructure_error() -> None:
    """The world did not cooperate; the request made sense.

    A caller that already handles `InfrastructureError` handles a converter
    that timed out, without knowing this class exists.
    """
    assert issubclass(RenditionFailedError, InfrastructureError)


# --- the null renderer ----------------------------------------------------


def test_the_null_renderer_satisfies_the_port() -> None:
    """It has to. It is what a caller passes to turn rendering OFF, and a
    caller cannot pass something the signature rejects."""
    assert isinstance(NullRenderer(), DocumentRenderer)


def test_the_null_renderer_claims_nothing() -> None:
    assert not NullRenderer().claims(MimeType.parse("application/msword"))


async def test_the_null_renderer_reports_no_pages() -> None:
    assert await NullRenderer().page_count("/anything") == 0


async def test_the_null_renderer_refuses_text_too() -> None:
    """Symmetric with the image path. Returning "" would read as "this page is
    blank", which is a claim about a document nothing opened."""
    with pytest.raises(RenditionFailedError):
        await NullRenderer().page_text("/anything", 1)


async def test_the_null_renderer_raises_rather_than_returning_a_blank_image() -> None:
    """Never `b""`. An empty PNG handed onward as "page 1" is an observation
    nothing made — the same reasoning `ffmpeg_frames` returns None for."""
    with pytest.raises(RenditionFailedError):
        await NullRenderer().render_page("/anything", 1)
