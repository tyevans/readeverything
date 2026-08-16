"""The laws every `MediaHandler` must obey.

Subclass, supply a `handler` and a `content` fixture, and inherit the contract.
These are the same bodies the bundled handlers are tested against.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from readeverything.domain.identity import ContentHash, MediaKind, MimeType, SourceRef
from readeverything.domain.rendition import Budget

#: A minimal value per builtin type, used to synthesize a required field this
#: law has no other way to fill. Deliberately small (empty string, zero): the
#: point is only that the affordance actually runs, not that the value means
#: anything.
_MINIMAL_VALUES: dict[type, object] = {str: "", int: 0, float: 0.0, bool: False}


def _synthesize_minimal(params_cls: type[BaseModel]) -> BaseModel | None:
    """The most trivial instance of `params_cls` that construction allows.

    Fills every required field with a minimal value for its annotated type.
    Returns `None` if a required field's type isn't one of the builtins above,
    or if even the minimal values fail the model's own validation — callers
    treat `None` as "this affordance can't be exercised generically" and skip
    it narrowly, rather than silently skipping every non-zero-argument
    affordance the way a bare `except ValidationError: continue` would.
    """
    kwargs: dict[str, object] = {}
    for name, field in params_cls.model_fields.items():
        if not field.is_required():
            continue
        if field.annotation not in _MINIMAL_VALUES:
            return None
        kwargs[name] = _MINIMAL_VALUES[field.annotation]
    try:
        return params_cls(**kwargs)
    except ValidationError:
        return None


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
        """Every declared affordance can actually be invoked.

        Drift between what a handler declares and what it implements would make
        capability negotiation a lie: the registry would expose a tool that
        cannot run. Most affordances construct their params with zero
        arguments; a few take a required, caller-supplied field with no
        sensible default (a free-form question, say). For those, a minimal
        valid instance is synthesized from the params model's own required
        fields, so the affordance is still exercised — only a required field
        of a type this law cannot synthesize (something other than str, int,
        float, or bool) causes that one affordance to be skipped, narrowly,
        rather than every non-zero-argument affordance being skipped wholesale.
        """
        for affordance in handler.affordances():
            try:
                params = affordance.params()
            except ValidationError:
                synthesized = _synthesize_minimal(affordance.params)
                if synthesized is None:
                    continue
                params = synthesized
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
