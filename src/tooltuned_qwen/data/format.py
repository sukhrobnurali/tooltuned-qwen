"""Render function-calling samples through the Qwen 3.5 chat template.

The template must come from `tokenizer.apply_chat_template` -- hand-written
templating silently breaks training (brief section 11 q6).
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Mapping
from typing import Any, Literal

Source = Literal["xlam", "hermes"]
ThinkingMode = Literal["preserve", "strip", "mix-75-25"]

_TOOLS_BLOCK = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_RESPONSE_BLOCK = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


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


def _normalize_hermes_tools(parsed: Any) -> list[dict[str, Any]]:
    """Hermes tool schemas come bare ({name, parameters}) or wrapped
    ({type: 'function', function: {...}}). The Qwen template wants the wrapped
    shape -- coerce so xLAM and Hermes formatters produce structurally identical
    `tools` lists, keeping the Phase 2 ablation a fair comparison.
    """
    if not isinstance(parsed, list):
        return []
    out: list[dict[str, Any]] = []
    for t in parsed:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            out.append(t)
        elif "name" in t:
            out.append({"type": "function", "function": t})
    return out


def _parse_hermes_tool_calls(content: str) -> tuple[list[dict[str, Any]], str]:
    """Pull `<tool_call>{...}</tool_call>` blocks out of an assistant turn.

    Returns (tool_calls, residual_content). A turn that is *only* tool calls
    yields an empty residual; mixed turns keep their prose.
    """
    tool_calls: list[dict[str, Any]] = []
    for m in _TOOL_CALL_BLOCK.finditer(content):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        tool_calls.append(
            {
                "type": "function",
                "function": {"name": obj["name"], "arguments": args},
            }
        )
    residual = _TOOL_CALL_BLOCK.sub("", content).strip()
    return tool_calls, residual


def _hermes_to_messages(
    sample: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert a Hermes function-calling row into (messages, tools).

    Hermes inlines tool schemas inside `<tools>...</tools>` in the system turn
    and tool calls inside `<tool_call>...</tool_call>` in assistant turns.
    The Qwen 3.5 chat template wants the schemas as a separate `tools=` arg
    and the calls under a structured `tool_calls` field, so we extract.
    """
    convo = sample.get("conversations") or []
    tools: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []

    for turn in convo:
        raw_role = turn.get("from", "")
        role = _HERMES_ROLE_MAP.get(raw_role, raw_role or "user")
        content = turn.get("value", "") or ""

        if role == "system":
            m = _TOOLS_BLOCK.search(content)
            if m:
                try:
                    tools = _normalize_hermes_tools(json.loads(m.group(1)))
                except json.JSONDecodeError:
                    tools = []
                content = _TOOLS_BLOCK.sub("", content).strip()
            if content:
                messages.append({"role": "system", "content": content})
            continue

        if role == "assistant":
            tool_calls, residual = _parse_hermes_tool_calls(content)
            msg: dict[str, Any] = {"role": "assistant", "content": residual}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
            continue

        if role == "tool":
            tr = _TOOL_RESPONSE_BLOCK.search(content)
            if tr:
                content = tr.group(1).strip()
            messages.append({"role": "tool", "content": content})
            continue

        messages.append({"role": role, "content": content})

    return messages, tools


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


def _strip_thinking(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove `<think>...</think>` spans from every message's content.

    Used by `thinking_mode == "strip"` so the Qwen template can't restore a
    reasoning chain that the training run is supposed to suppress.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str) and "<think>" in content:
            new = dict(m)
            new["content"] = _THINK_BLOCK.sub("", content).strip()
            out.append(new)
        else:
            out.append(m)
    return out


def enable_thinking_for(mode: ThinkingMode, *, index: int, seed: int) -> bool:
    """Resolve the per-sample `enable_thinking` flag for a given mode.

    Mix mode is deterministic: `(seed, index)` seeds a stdlib RNG so the same
    config always produces the same 75/25 split, regardless of dataset order.
    """
    if mode == "preserve":
        return True
    if mode == "strip":
        return False
    if mode == "mix-75-25":
        # `random.Random` only accepts hashable scalar seeds, not tuples; mix
        # `seed` and `index` into a single 64-bit int via a splitmix-style step
        # so different (seed, index) pairs produce uncorrelated streams.
        mixed = (seed * 0x9E3779B97F4A7C15 + index) & 0xFFFFFFFFFFFFFFFF
        return random.Random(mixed).random() < 0.75
    raise ValueError(f"unknown thinking_mode: {mode}")


def format_sample(
    sample: Mapping[str, Any],
    tokenizer: Any,
    *,
    source: Source = "xlam",
    enable_thinking: bool = True,
) -> str:
    """Format one sample as a single training string via the Qwen chat template.

    Routes through `tokenizer.apply_chat_template(..., tokenize=False)` so the
    exact template shipped with the Qwen 3.5 tokenizer is used.

    `enable_thinking=False` both forwards to the template (which suppresses any
    `<think>` placeholder it would otherwise emit) and pre-strips inline
    `<think>...</think>` spans from message content -- the template alone can't
    rewrite content that's already in the data.
    """
    messages, tools = to_messages(sample, source)
    if not enable_thinking:
        messages = _strip_thinking(messages)
    return tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
