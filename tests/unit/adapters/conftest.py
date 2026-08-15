from __future__ import annotations

import pytest
from tests.fixtures_pdf import born_digital


@pytest.fixture
def three_page_pdf() -> bytes:
    return born_digital(["alpha", "beta", "gamma"])
