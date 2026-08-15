from __future__ import annotations

import dataclasses

import pytest

from readeverything.ports.probe_media import DocumentFacts


def test_document_facts_rejects_a_page_count_that_disagrees_with_its_sizes() -> None:
    """The two fields describe the same document and must not contradict.

    A probe that reported 10 pages and 3 sizes would produce a card claiming a
    page count nothing measured — and `read_page(7)` would then fail on a
    document the card said had page 7.
    """
    with pytest.raises(ValueError, match="page_count"):
        DocumentFacts(page_count=10, page_sizes=((612.0, 792.0),), metadata={})


def test_document_facts_rejects_a_non_positive_page_size() -> None:
    with pytest.raises(ValueError):
        DocumentFacts(page_count=1, page_sizes=((0.0, 792.0),), metadata={})


def test_document_facts_carries_no_text() -> None:
    """The card path must stay cheap. A probe that extracted text to answer
    'how many pages' would defeat the progressive-disclosure design this
    library is built on."""
    assert not any("text" in f.name for f in dataclasses.fields(DocumentFacts))
