"""The laws every `MediaHandler` must obey.

Subclass, supply a `handler` and a `content` fixture, and inherit the contract.
These are the same bodies the bundled handlers are tested against.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
from readeverything.domain.rendition import Budget


class MediaHandlerCompliance:
    """Laws a handler must satisfy to be usable by the registry.

    Your source must be able to serve the SAME content at a second uri,
    "somewhere/else": `test_describe_depends_only_on_content` describes a ref
    pointing there to prove the card does not vary with the path. A source
    that cannot will raise rather than fail the law.
    """

    @pytest.fixture
    def handler(self) -> object:
        raise NotImplementedError("supply a `handler` fixture")

    @pytest.fixture
    def content(self) -> bytes:
        raise NotImplementedError("supply a `content` fixture")

    @pytest.fixture
    def ref(self, content: bytes) -> SourceRef:
        return SourceRef(
            uri="compliance-subject",
            mime=MimeType.parse("application/octet-stream"),
            content_hash=ContentHash("0" * 64),
            size_bytes=len(content),
        )

    async def test_describe_depends_only_on_content(self, handler, content, ref) -> None:  # type: ignore[no-untyped-def]
        """Same bytes at a different uri produce an identical card body.

        A card that varies with the path would make the artifact cache — which
        is keyed on content — serve one path's card for another's.
        """
        moved = SourceRef(
            uri="somewhere/else",
            mime=ref.mime,
            content_hash=ref.content_hash,
            size_bytes=ref.size_bytes,
        )
        first = await handler.describe(ref)
        second = await handler.describe(moved)
        assert first.kind == second.kind
        assert dict(first.facts) == dict(second.facts)
        assert first.outline == second.outline
        assert first.excerpt == second.excerpt

    async def test_the_card_kind_is_a_media_kind(self, handler, ref) -> None:  # type: ignore[no-untyped-def]
        card = await handler.describe(ref)
        assert isinstance(card.kind, MediaKind)

    async def test_declared_affordances_are_invocable(self, handler, content, ref) -> None:  # type: ignore[no-untyped-def]
        """Every zero-argument affordance can be invoked with default parameters.

        Drift between what a handler declares and what it implements would make
        capability negotiation a lie: the registry would expose a tool that
        cannot run. Some affordances take a required, caller-supplied parameter
        with no sensible default (a free-form question, say); those cannot be
        constructed with no arguments at all, so this law skips them rather than
        asserting a default that would defeat the point of requiring the field.
        """
        for affordance in handler.affordances():
            try:
                params = affordance.params()
            except ValidationError:
                continue
            await handler.invoke(ref, affordance.name, params)

    async def test_an_undeclared_affordance_raises(self, handler, ref) -> None:  # type: ignore[no-untyped-def]
        from readeverything.domain.errors import UnknownAffordanceError

        with pytest.raises(UnknownAffordanceError):
            await handler.invoke(ref, "definitely_not_an_affordance", None)

    async def test_represent_produces_a_map_covering_its_text(self, handler, ref) -> None:  # type: ignore[no-untyped-def]
        """`Rendered` validates this itself; this proves the handler builds one."""
        rendered = await handler.represent(ref, Budget(max_chars=None))
        assert rendered.locator_map.length == len(rendered.text)

    async def test_represent_respects_a_budget_or_reports_degradation(self, handler, ref) -> None:  # type: ignore[no-untyped-def]
        """Truncation must be announced, and announcements must be truthful.

        Comparing against an unbounded render closes both directions: text
        shorter than unbounded REQUIRES a degradation, and text equal to
        unbounded FORBIDS a spurious one. The previous `permits(...) or
        degradations` form was satisfiable by a handler that truncated
        silently and by one that cried wolf without truncating.
        """
        unbounded = await handler.represent(ref, Budget(max_chars=None))
        budget = Budget(max_chars=10)
        rendered = await handler.represent(ref, budget)
        assert budget.permits(len(rendered.text))
        if len(rendered.text) < len(unbounded.text):
            assert rendered.degradations, "truncated without reporting a degradation"
        else:
            assert not rendered.degradations, "reported a degradation without truncating"
