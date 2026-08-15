from __future__ import annotations

import pytest

from readeverything.adapters.pdfium_probe import PdfiumProbe
from readeverything.domain.errors import InfrastructureError


async def test_a_three_page_document_reports_three_pages(three_page_pdf: bytes) -> None:
    facts = await PdfiumProbe().probe(three_page_pdf)
    assert facts.page_count == 3
    assert len(facts.page_sizes) == 3


async def test_page_sizes_come_back_in_points(three_page_pdf: bytes) -> None:
    """US Letter is 612x792 points. The handler normalises BBoxes by these
    numbers, so a wrong unit here produces citations off by a factor of 72."""
    facts = await PdfiumProbe().probe(three_page_pdf)
    assert facts.page_sizes[0] == pytest.approx((612.0, 792.0))


async def test_metadata_is_string_to_string(three_page_pdf: bytes) -> None:
    facts = await PdfiumProbe().probe(three_page_pdf)
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in facts.metadata.items())


async def test_a_probe_of_bytes_that_are_not_a_pdf_raises_infrastructure_error() -> None:
    """The probe may raise; the HANDLER is what must never raise. Keeping the
    failure loud here means the handler decides how to degrade, rather than
    receiving a fabricated zero-page document."""
    with pytest.raises(InfrastructureError):
        await PdfiumProbe().probe(b"this is not a pdf at all")


async def test_probing_does_not_extract_text(three_page_pdf: bytes) -> None:
    """The cheapness claim, asserted rather than trusted."""
    facts = await PdfiumProbe().probe(three_page_pdf)
    assert "alpha" not in repr(facts)
