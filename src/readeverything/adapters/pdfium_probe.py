"""`MediaProbe` over pypdfium2.

pypdfium2 wraps Google's PDFium under Apache-2.0/BSD and ships a bundled
binary, so this needs no OS dependency and no `Capability` member — it is a
Python import, gated like Pillow.

pymupdf is faster at some of this and is not used: it is AGPL-3.0 and this
library is MIT. That is a licensing conflict, not a performance trade-off, and
it is written here so it is not reopened as one.

pdfium is synchronous and CPU-bound, so every call runs in a thread.
"""

from __future__ import annotations

import asyncio

import pypdfium2 as pdfium  # type: ignore[import-untyped]  # pypdfium2 ships no py.typed marker

from readeverything.domain.errors import InfrastructureError
from readeverything.ports.probe_media import DocumentFacts


def open_document(data: bytes) -> pdfium.PdfDocument:
    """Open bytes as a PDF, or raise `InfrastructureError`.

    One place opens documents so the pdfium import stays in one adapter module
    and every caller gets the same translated failure.
    """
    try:
        return pdfium.PdfDocument(data)
    except Exception as exc:
        raise InfrastructureError(f"could not open as a PDF: {exc}") from exc


def _probe_sync(data: bytes) -> DocumentFacts:
    document = open_document(data)
    try:
        sizes = tuple(document[i].get_size() for i in range(len(document)))
        raw = document.get_metadata_dict()
        metadata = {str(k): str(v) for k, v in raw.items() if v}
        return DocumentFacts(page_count=len(document), page_sizes=sizes, metadata=metadata)
    finally:
        document.close()


class PdfiumProbe:
    """Page count, page sizes and metadata. No text."""

    async def probe(self, data: bytes) -> DocumentFacts:
        return await asyncio.to_thread(_probe_sync, data)
