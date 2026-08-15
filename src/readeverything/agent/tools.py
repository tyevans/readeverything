"""The framework-agnostic tool pack.

Three tools rather than one per affordance. Affordances are per-mimetype and
therefore per-file, so a tool per affordance would mean a tool list that
changes with whatever the agent last looked at — which no agent framework
supports and no model handles well. Instead `inspect_path` *tells* the model
which affordances this file has, and `invoke_affordance` runs one by name. The
card is the discovery mechanism.

This module and `adapters/langchain_*.py` are the only places `langchain` may
be imported.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from readeverything.agent.results import never_raises
from readeverything.domain.card import Card
from readeverything.domain.rendition import (
    ImageContent,
    Rendition,
    StructuredContent,
    TextContent,
)
from readeverything.pipeline.perception import Perception


class InspectParams(BaseModel):
    uri: str = Field(description="Path to inspect, relative to the configured root.")


class ListParams(BaseModel):
    uri: str = Field(default=".", description="Directory to list, relative to the root.")


class InvokeParams(BaseModel):
    uri: str = Field(description="Path the affordance applies to.")
    affordance: str = Field(description="Affordance name, from the card's `affordances`.")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Arguments, matching that affordance's schema."
    )


def _render_card(card: Card) -> str:
    return json.dumps(
        {
            "uri": card.ref.uri,
            "mime": str(card.ref.mime),
            "kind": str(card.kind),
            "size_bytes": card.ref.size_bytes,
            "facts": dict(card.facts),
            "outline": [
                {"label": segment.label, "locator": repr(segment.locator)}
                for segment in card.outline
            ],
            "excerpt": card.excerpt,
            "affordances": [
                {
                    "name": affordance.name,
                    "description": affordance.description,
                    "params": affordance.params.model_json_schema(),
                }
                for affordance in card.affordances
            ],
        },
        indent=2,
    )


def _render_rendition(rendition: Rendition) -> str:
    match rendition.content:
        case TextContent(text=text):
            body = text
        case StructuredContent(rows=rows):
            body = json.dumps(list(rows), indent=2)
        case ImageContent(data=data, mime=mime):
            body = f"[{mime} image, {len(data)} bytes — pass to a vision tool to read it]"
    marker = " (degraded)" if rendition.degraded else ""
    return f"located at {rendition.locator!r}{marker}:\n{body}"


def build_tools(perception: Perception) -> list[BaseTool]:
    """The tool pack over one `Perception`."""

    @never_raises
    async def inspect_path(uri: str) -> str:
        return _render_card(await perception.inspect(uri))

    @never_raises
    async def list_paths(uri: str = ".") -> str:
        return "\n".join(await perception.list(uri))

    @never_raises
    async def invoke_affordance(
        uri: str, affordance: str, params: Mapping[str, Any] | None = None
    ) -> str:
        return _render_rendition(await perception.invoke(uri, affordance, params or {}))

    async def _inspect(uri: str) -> str:
        return (await inspect_path(uri)).render()

    async def _list(uri: str = ".") -> str:
        return (await list_paths(uri)).render()

    async def _invoke(uri: str, affordance: str, params: dict[str, Any] | None = None) -> str:
        return (await invoke_affordance(uri, affordance, params)).render()

    return [
        StructuredTool.from_function(
            coroutine=_inspect,
            name="inspect_path",
            description=(
                "Inspect any file and get a compact description of it: its type, size, "
                "key metadata, an outline, a short excerpt, and the list of affordances "
                "available for going deeper. Always call this before invoke_affordance. "
                "Cheap: it never runs a model over the whole file."
            ),
            args_schema=InspectParams,
        ),
        StructuredTool.from_function(
            coroutine=_list,
            name="list_paths",
            description="List every file under a directory, recursively.",
            args_schema=ListParams,
        ),
        StructuredTool.from_function(
            coroutine=_invoke,
            name="invoke_affordance",
            description=(
                "Run one of the affordances named in a file's card — for example reading "
                "a character range, a page, a transcript segment, or a video frame. "
                "The affordance name and its parameter schema come from inspect_path. "
                "Results are always accompanied by a locator saying where in the file "
                "they came from."
            ),
            args_schema=InvokeParams,
        ),
    ]
