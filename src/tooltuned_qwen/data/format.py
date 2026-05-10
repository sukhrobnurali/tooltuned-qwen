"""Render function-calling samples through the Qwen 3.5 chat template.

The template must come from `tokenizer.apply_chat_template` -- hand-written
templating silently breaks training (brief section 11 q6).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

Source = Literal["xlam", "hermes"]


def _as_obj(v: Any) -> Any:
    return json.loads(v) if isinstance(v, str) else v


def _xlam_to_messages(
    sample: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tools = _as_obj(sample["tools"]) or []
    answers = _as_obj(sample["answers"]) or []

    tool_calls: list[dict[str, Any]] = []
    for ans in answers:
        args = ans.get("arguments", {})
        # Qwen 3.5's chat template iterates `arguments` as a mapping (calls .items()),
        # so keep it parsed -- the OpenAI-style JSON-string form blows up `do_items`.
        if isinstance(args, str):
            args = json.loads(args)
        tool_calls.append(
            {
                "type": "function",
                "function": {"name": ans["name"], "arguments": args},
            }
        )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": sample["query"]},
    ]
    if tool_calls:
        messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
    return messages, tools


_HERMES_ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system", "tool": "tool"}


def _hermes_to_messages(
    sample: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    convo = sample.get("conversations") or []
    messages: list[dict[str, Any]] = []
    for turn in convo:
        role = _HERMES_ROLE_MAP.get(turn.get("from", ""), turn.get("from", "user"))
        messages.append({"role": role, "content": turn.get("value", "")})
    # Hermes embeds tool schemas inline in the system message; no separate tools list.
    return messages, []


def to_messages(
    sample: Mapping[str, Any], source: Source
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert a raw xlam/hermes row into (messages, tools).

    Tools are returned separately so they can be passed to
    `tokenizer.apply_chat_template(..., tools=tools)`, which is the
    Qwen-supported path for tool-aware templating.
    """
    if source == "xlam":
        return _xlam_to_messages(sample)
    if source == "hermes":
        return _hermes_to_messages(sample)
    raise ValueError(f"unknown source: {source}")


def format_sample(sample: Mapping[str, Any], tokenizer: Any, *, source: Source = "xlam") -> str:
    """Format one sample as a single training string via the Qwen chat template.

    Routes through `tokenizer.apply_chat_template(..., tokenize=False)` so the
    exact template shipped with the Qwen 3.5 tokenizer is used.
    """
    messages, tools = to_messages(sample, source)
    return tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=False,
    )
