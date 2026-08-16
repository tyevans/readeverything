"""The framework-agnostic tool pack.

A fixed handful of tools rather than one per affordance. Affordances are
per-mimetype and therefore per-file, so a tool per affordance would mean a
tool list that changes with whatever the agent last looked at — which no
agent framework supports and no model handles well. Instead `inspect_path`
*tells* the model which affordances this file has, and `invoke_affordance`
runs one by name. The card is the discovery mechanism.

`ask_about_image` is the one considered exception, and it survives the
argument above only because it does not violate it: it dispatches on a NAME
CONVENTION (the `ask_about_image` affordance, which three handlers happen to
declare), never on mimetype or media kind, and it is present in the returned
list for every file regardless of what that file is. This module must never
learn about mimetypes or media kinds — if a change here needs an `if kind ==`
or a handler import, it is the wrong change.

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


class AskAboutImageParams(BaseModel):
    model_config = {"populate_by_name": True}

    uri: str = Field(description="Path to the image, PDF or video to ask about.")
    question: str = Field(description="What you want to know.")
    params: dict[str, Any] = Field(
        default_factory=dict,
        alias="where",
        description=(
            "Where to look, when the file needs it: page/dpi for a PDF, seconds for a "
            "video, and x/y/w/h as fractions 0-1 for a region of any of them. "
            "Omit for the whole image."
        ),
    )


def _render_validation_error(exc: Exception) -> str:
    """Malformed tool arguments must read like any other failure, not a traceback."""
    return f"ERROR ({type(exc).__name__}): {exc}"


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


#: Affordances that turn image bytes into something a text model can read.
#: Named here so the rendering and the handler cannot drift apart silently.
_IMAGE_READING_AFFORDANCES = ("ask_about_image", "describe_image", "ocr")


def _render_rendition(rendition: Rendition, affordances: tuple[str, ...] = ()) -> str:
    match rendition.content:
        case TextContent(text=text):
            body = text
        case StructuredContent(rows=rows):
            body = json.dumps(list(rows), indent=2)
        case ImageContent(data=data, mime=mime):
            usable = [a for a in _IMAGE_READING_AFFORDANCES if a in affordances]
            if "ask_about_image" in usable:
                body = (
                    f"[{mime} image, {len(data)} bytes — call ask_about_image on this "
                    f"file, with the same coordinates, to ask about it]"
                )
            elif usable:
                route = " or ".join(usable)
                body = (
                    f"[{mime} image, {len(data)} bytes — "
                    f"call invoke_affordance with {route} to read it]"
                )
            else:
                # No vision capability is registered here. Saying so is true and
                # actionable; pointing at an affordance the registry filtered
                # out would send the model after a tool call that cannot work.
                body = (
                    f"[{mime} image, {len(data)} bytes — "
                    f"cannot be read here: no vision capability is configured]"
                )
        case _:
            body = f"[unrenderable content: {type(rendition.content).__name__}]"
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
        rendition = await perception.invoke(uri, affordance, params or {})
        if isinstance(rendition.content, ImageContent):
            # Only the ImageContent branch of `_render_rendition` needs the
            # card, to say which affordances can read the image back as text.
            # Fetching it unconditionally would pay `inspect`'s full
            # decode/describe cost on every call, for a hint used nowhere
            # else — and would fail invocations whose `describe` breaks even
            # when `invoke` itself would have succeeded.
            card = await perception.inspect(uri)
            affordance_names = tuple(a.name for a in card.affordances)
        else:
            affordance_names = ()
        return _render_rendition(rendition, affordance_names)

    @never_raises
    async def ask_about_image(
        uri: str, question: str, params: Mapping[str, Any] | None = None
    ) -> str:
        rendition = await perception.invoke(
            uri, "ask_about_image", {**(params or {}), "question": question}
        )
        return _render_rendition(rendition)

    async def _inspect(uri: str) -> str:
        return (await inspect_path(uri)).render()

    async def _list(uri: str = ".") -> str:
        return (await list_paths(uri)).render()

    async def _invoke(uri: str, affordance: str, params: dict[str, Any] | None = None) -> str:
        return (await invoke_affordance(uri, affordance, params)).render()

    async def _ask(uri: str, question: str, params: dict[str, Any] | None = None) -> str:
        return (await ask_about_image(uri, question, params)).render()

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
            handle_validation_error=_render_validation_error,
        ),
        StructuredTool.from_function(
            coroutine=_list,
            name="list_paths",
            description="List every file under a directory, recursively.",
            args_schema=ListParams,
            handle_validation_error=_render_validation_error,
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
            handle_validation_error=_render_validation_error,
        ),
        StructuredTool.from_function(
            coroutine=_ask,
            name="ask_about_image",
            description=(
                "Ask a vision model a question about a picture — a photograph, a page "
                "of a PDF, or a frame of a video. Give `where` to say which page, which "
                "moment, or which rectangular region; omit it for the whole image. "
                "You do not need to call inspect_path first. This runs a vision model — "
                "for a video, read the transcript before you look at frames."
            ),
            args_schema=AskAboutImageParams,
            handle_validation_error=_render_validation_error,
        ),
    ]
