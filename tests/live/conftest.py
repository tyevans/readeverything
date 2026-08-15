"""Configuration for tests that touch the real model server.

Tests may read the environment; the library may not. `test_reads_no_environment`
scans `src/` only, which is the line: configuration reaches the library as
constructor arguments, and it is the caller's business where it came from.
"""

import os

import pytest

from readeverything.adapters.vision_langchain import (
    LangChainVisionModel,
    build_openai_vision_model,
)

DEFAULT_BASE_URL = "http://192.168.1.14:8080/v1/"
DEFAULT_MODEL = "qwen3.8-27b-mtp"


@pytest.fixture(scope="session")
def live_base_url() -> str:
    return os.environ.get("READEVERYTHING_LIVE_BASE_URL", DEFAULT_BASE_URL)


@pytest.fixture(scope="session")
def live_model_name() -> str:
    return os.environ.get("READEVERYTHING_LIVE_MODEL", DEFAULT_MODEL)


@pytest.fixture
def live_vision(live_base_url: str, live_model_name: str) -> LangChainVisionModel:
    return build_openai_vision_model(base_url=live_base_url, model=live_model_name)


@pytest.fixture(scope="session")
def live_whisper_model_dir() -> str | None:
    """Directory of a converted CTranslate2 whisper model, if configured.

    Unset on this machine, always: no weights are present and none are
    downloaded. `tests/live/test_transcription.py` skips cleanly when this is
    `None` rather than attempting to fetch anything.
    """
    return os.environ.get("READEVERYTHING_LIVE_WHISPER_MODEL_DIR")
