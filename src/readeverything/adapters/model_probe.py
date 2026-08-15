"""Deriving a capability's revision from the model object itself.

This closes a seam Plan 2's review flagged: the VISION revision in a
`CapabilitySet` and the `model_id` of the injected `VisionModel` were
independent inputs with nothing requiring them to agree. Deriving one from the
other makes disagreement impossible rather than merely discouraged — which
matters once artifacts are cached, because a key that misdescribes the model
that produced it serves a mixture of two models' output as though it were one.

There is deliberately no way for a caller to override the derived value; doing
so would reopen the seam this module exists to close.
"""

from __future__ import annotations

from readeverything.domain.capability import Capability
from readeverything.ports.transcription import Transcriber
from readeverything.ports.vision import VisionModel


class ModelProbe:
    """Reports model-backed capabilities from the models actually injected."""

    def __init__(
        self, *, vision: VisionModel | None = None, transcriber: Transcriber | None = None
    ) -> None:
        self._vision = vision
        self._transcriber = transcriber

    async def revision(self, capability: Capability) -> str | None:
        if capability is Capability.VISION and self._vision is not None:
            return self._vision.model_id
        if capability is Capability.ASR and self._transcriber is not None:
            return self._transcriber.model_id
        return None
