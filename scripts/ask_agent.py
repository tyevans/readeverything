"""Give a deepagent the media/ directory and ask what a file is about.

A harness, not a test: it costs real model calls against a real endpoint.
Used to measure what a strategy change is worth end to end — the numbers in
the transcript-first spec came from here.

    uv run python scripts/ask_agent.py "What is mystery_subject.mp4 about?"
"""

import asyncio
import sys
import time
from pathlib import Path

from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from readeverything import (
    Capability,
    SemaphoreLimiter,
    build_openai_vision_model,
    build_perception,
    build_tools,
)

BASE_URL = "http://192.168.1.14:8080/v1/"
MODEL = "qwen3.8-27b-mtp"
WHISPER_DIR = "models/faster-whisper-small"
QUESTION = sys.argv[1] if len(sys.argv) > 1 else "What is mystery_subject.mp4 about?"


_T0 = time.monotonic()


def _ts() -> str:
    elapsed = time.monotonic() - _T0
    return f"{int(elapsed // 60):02d}:{elapsed % 60:04.1f}"


class Narrate:
    def observe(self, event):
        name = type(event).__name__
        print(f"[{_ts()}]   perception {name} {event.operation} {event.ref.uri}", flush=True)


async def main() -> None:
    vision = build_openai_vision_model(
        base_url=BASE_URL,
        model=MODEL,
        max_tokens=1500,  # reasoning eats the budget before answering below ~1k
        timeout_s=300.0,
    )
    # Wired only if the weights are present. They are a large download the
    # project does not make on its own, so a machine without them still runs
    # this harness — it just falls back to frames on a caption-less file.
    transcriber = None
    if Path(WHISPER_DIR).is_dir():
        from readeverything import WhisperTranscriber

        transcriber = WhisperTranscriber(model_dir=WHISPER_DIR)
        print(f"transcriber: {transcriber.model_id}", flush=True)

    perception = await build_perception(
        "media",
        vision=vision,
        transcriber=transcriber,
        observer=Narrate(),
        limiter=SemaphoreLimiter({Capability.VISION: 3}),
    )
    tools = build_tools(perception)
    print("tools:", [t.name for t in tools], flush=True)

    agent = create_deep_agent(
        model=ChatOpenAI(
            base_url=BASE_URL,
            model=MODEL,
            api_key="not-needed",
            timeout=600.0,
            max_completion_tokens=4000,
            # Thinking off. The model reasons by default, and in an agent loop
            # that reasoning is paid for on every turn while contributing
            # nothing a tool call needs — the decision "call inspect_path next"
            # does not improve with deliberation.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
        tools=tools,
        system_prompt=(
            "You inspect media files through the provided tools. "
            "Start with inspect_path to see what a file is and which affordances it has, "
            "then invoke_affordance to look at specific moments. "
            "A video's card gives its duration: sample frames ACROSS the whole timeline "
            "(beginning, middle, end), not just the opening seconds, before concluding. "
            "Cite the timestamps your conclusion rests on."
        ),
    )

    print(f"\n===== QUESTION =====\n{QUESTION}\n", flush=True)
    seen = 0
    last = None
    async for chunk in agent.astream(
        {"messages": [HumanMessage(content=QUESTION)]},
        {"recursion_limit": 60},
        stream_mode="values",
    ):
        messages = chunk.get("messages", [])
        for m in messages[seen:]:
            kind = type(m).__name__
            for call in getattr(m, "tool_calls", []) or []:
                print(f"[{_ts()}] CALL {call['name']}({call['args']})", flush=True)
            text = m.content if isinstance(m.content, str) else str(m.content)
            if kind == "ToolMessage":
                head = text.replace("\n", " ")[:300]
                print(f"[{_ts()}] RESULT {head}{'…' if len(text) > 300 else ''}", flush=True)
            elif kind == "AIMessage" and text.strip():
                print(f"[{_ts()}] SAYS {text.strip()[:600]}", flush=True)
            last = m
        seen = len(messages)

    print("\n===== ANSWER =====", flush=True)
    print(last.content if last is not None else "(no messages)")


asyncio.run(main())
