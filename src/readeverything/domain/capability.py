"""What this deployment can do.

Model capabilities and OS binaries are the same kind of thing. A missing
`ffmpeg` must degrade exactly like a missing vision model, because from a
handler's point of view both are "I cannot produce that". One mechanism means
there is one place degradation is decided, and no special cases.

Each capability carries a **revision string** — a model id plus revision, or a
binary version. It is not used for matching; it exists so the artifact cache
key changes when the thing behind a capability changes. Without it, swapping
the vision model silently serves a mixture of descriptions from two models.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Self


class Capability(StrEnum):
    VISION = "vision"
    ASR = "asr"
    DIARIZATION = "diarization"
    TEXT_LLM = "text_llm"
    FFMPEG = "ffmpeg"
    EXIFTOOL = "exiftool"
    LIBREOFFICE = "libreoffice"
    TESSERACT = "tesseract"


@dataclass(frozen=True, slots=True)
class CapabilitySet:
    """The capabilities available, each with the revision behind it."""

    revisions: Mapping[Capability, str]

    @classmethod
    def empty(cls) -> Self:
        return cls(revisions={})

    @classmethod
    def of(cls, revisions: Mapping[Capability, str]) -> Self:
        return cls(revisions=dict(revisions))

    def satisfies(self, required: frozenset[Capability] | set[Capability]) -> bool:
        return all(capability in self.revisions for capability in required)

    def fingerprint(self) -> str:
        """A stable digest of what is installed, for the artifact cache key."""
        digest = hashlib.blake2b(digest_size=16)
        for capability in sorted(self.revisions, key=str):
            digest.update(str(capability).encode("utf-8"))
            digest.update(b"\x00")
            digest.update(self.revisions[capability].encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()
