from readeverything.agent.results import ToolResult, never_raises
from readeverything.domain.errors import SourceUnreadableError


async def test_a_success_carries_its_value() -> None:
    @never_raises
    async def fine() -> str:
        return "ok"

    result = await fine()
    assert result.ok
    assert result.value == "ok"
    assert result.error is None


async def test_a_domain_error_becomes_a_structured_failure() -> None:
    """A traceback reaching a model is a wasted and unrecoverable turn."""

    @never_raises
    async def bad() -> str:
        raise SourceUnreadableError("no such file: /nope")

    result = await bad()
    assert not result.ok
    assert result.error is not None
    assert "no such file" in result.error
    assert result.error_type == "SourceUnreadableError"


async def test_an_unexpected_error_is_also_caught() -> None:
    """Not just our exceptions: an adapter bug must not reach the model either."""

    @never_raises
    async def worse() -> str:
        raise ZeroDivisionError("oops")

    result = await worse()
    assert not result.ok
    assert result.error_type == "ZeroDivisionError"


def test_a_result_renders_compactly_for_a_model() -> None:
    assert "ok" in ToolResult(ok=True, value="ok", error=None, error_type=None).render()
    rendered = ToolResult(ok=False, value=None, error="boom", error_type="X").render()
    assert "ERROR" in rendered and "boom" in rendered
