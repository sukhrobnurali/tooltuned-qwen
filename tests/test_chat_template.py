"""Chat-template round-trip: formatter must route through `apply_chat_template`.

The actual Qwen 3.5 tokenizer can't run in CI (heavy ML deps live in the
`[colab]` extra), so this test uses a stub tokenizer that mimics the
HF `apply_chat_template` interface and asserts the formatter:

  - Calls `apply_chat_template` with the right messages and tools.
  - Passes `tokenize=False` (we want a string for the SFTTrainer text field).
  - Round-trips xlam fields (query / tools / answers) into the messages.
  - Extracts Hermes `<tools>` and `<tool_call>` markup into the structured
    Qwen-template fields (separate `tools` arg and `tool_calls` on the
    assistant turn) -- bare role mapping would silently render half-rendered
    Hermes against fully rendered xLAM and bias the Phase 2 dataset ablation.
  - Honors the three `thinking_mode` strategies (preserve / strip / mix-75-25).

This guards brief section 11 q6: hand-written templating silently breaks training.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tooltuned_qwen.data.format import (
    enable_thinking_for,
    format_sample,
    to_messages,
)


class _StubTokenizer:
    """Minimal stand-in for a HF tokenizer with a working chat template."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        parts: list[str] = []
        if tools:
            parts.append(f"[tools={len(tools)}]")
        if enable_thinking:
            parts.append("[thinking]")
        for m in messages:
            content = m.get("content", "") or ""
            if m.get("tool_calls"):
                names = ",".join(tc["function"]["name"] for tc in m["tool_calls"])
                content = f"<tool_call:{names}>"
            parts.append(f"<|{m['role']}|>{content}<|end|>")
        return "\n".join(parts)


@pytest.fixture
def xlam_sample() -> dict[str, Any]:
    return {
        "query": "What's the weather in Tokyo?",
        "tools": json.dumps(
            [
                {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ]
        ),
        "answers": json.dumps(
            [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
        ),
    }


def test_xlam_to_messages_shape(xlam_sample: dict[str, Any]) -> None:
    messages, tools = to_messages(xlam_sample, "xlam")
    assert messages[0] == {"role": "user", "content": "What's the weather in Tokyo?"}
    assert messages[1]["role"] == "assistant"
    tool_calls = messages[1]["tool_calls"]
    assert tool_calls[0]["function"]["name"] == "get_weather"
    # Qwen 3.5's chat template expects arguments as a dict (it calls .items()),
    # not the OpenAI-style JSON string -- see format.py for the empirical finding.
    assert tool_calls[0]["function"]["arguments"] == {"city": "Tokyo"}
    assert tools[0]["name"] == "get_weather"


def test_format_sample_routes_through_apply_chat_template(
    xlam_sample: dict[str, Any],
) -> None:
    tok = _StubTokenizer()
    out = format_sample(xlam_sample, tok, source="xlam")

    assert len(tok.calls) == 1
    call = tok.calls[0]
    assert call["tokenize"] is False
    assert call["add_generation_prompt"] is False
    assert call["tools"] is not None and len(call["tools"]) == 1
    assert call["enable_thinking"] is True

    # Messages survive the round-trip into the templated string.
    assert "What's the weather in Tokyo?" in out
    assert "<tool_call:get_weather>" in out
    assert "[tools=1]" in out


def test_format_sample_without_answers_omits_assistant_turn() -> None:
    sample = {
        "query": "hi",
        "tools": "[]",
        "answers": "[]",
    }
    messages, _ = to_messages(sample, "xlam")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_unknown_source_rejected() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        to_messages({}, "bogus")  # type: ignore[arg-type]


def test_hermes_to_messages_maps_roles() -> None:
    sample = {
        "conversations": [
            {"from": "system", "value": "you have a tool"},
            {"from": "human", "value": "call it"},
            {"from": "gpt", "value": "ok"},
        ]
    }
    messages, tools = to_messages(sample, "hermes")
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert tools == []


# --------- Hermes tool-call extraction ----------


def _hermes_tool_call_sample() -> dict[str, Any]:
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    return {
        "conversations": [
            {
                "from": "system",
                "value": (
                    "You are a function-calling AI.\n"
                    f"<tools>{json.dumps(schemas)}</tools>\n"
                    "Use them when appropriate."
                ),
            },
            {"from": "human", "value": "What's the weather in Tokyo?"},
            {
                "from": "gpt",
                "value": (
                    '<tool_call>\n{"name": "get_weather", '
                    '"arguments": {"city": "Tokyo"}}\n</tool_call>'
                ),
            },
            {
                "from": "tool",
                "value": '<tool_response>\n{"temp_c": 19}\n</tool_response>',
            },
            {"from": "gpt", "value": "It's 19 C in Tokyo."},
        ]
    }


def test_hermes_extracts_tools_block_into_separate_arg() -> None:
    messages, tools = to_messages(_hermes_tool_call_sample(), "hermes")
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "get_weather"
    # The system message keeps its persona prose minus the <tools> block.
    sys_msg = messages[0]
    assert sys_msg["role"] == "system"
    assert "<tools>" not in sys_msg["content"]
    assert "function-calling" in sys_msg["content"]


def test_hermes_extracts_tool_call_block_into_structured_field() -> None:
    messages, _ = to_messages(_hermes_tool_call_sample(), "hermes")
    assistant_call = next(
        m for m in messages if m["role"] == "assistant" and m.get("tool_calls")
    )
    tool_calls = assistant_call["tool_calls"]
    assert len(tool_calls) == 1
    fn = tool_calls[0]["function"]
    assert fn["name"] == "get_weather"
    # Same dict-not-string convention as xLAM (Qwen template calls .items()).
    assert fn["arguments"] == {"city": "Tokyo"}
    # The raw `<tool_call>...</tool_call>` markup is gone from the visible content.
    assert "<tool_call>" not in assistant_call["content"]


def test_hermes_unwraps_tool_response_markup() -> None:
    messages, _ = to_messages(_hermes_tool_call_sample(), "hermes")
    tool_msg = next(m for m in messages if m["role"] == "tool")
    assert "<tool_response>" not in tool_msg["content"]
    assert tool_msg["content"].startswith("{")


def test_hermes_normalizes_bare_tool_schemas() -> None:
    """Hermes occasionally lists bare {name, parameters} dicts; coerce to
    the wrapped Qwen-template shape so xLAM and Hermes pipelines line up."""
    sample = {
        "conversations": [
            {
                "from": "system",
                "value": (
                    "<tools>"
                    + json.dumps(
                        [
                            {
                                "name": "ping",
                                "description": "ping",
                                "parameters": {"type": "object", "properties": {}},
                            }
                        ]
                    )
                    + "</tools>"
                ),
            },
            {"from": "human", "value": "ping?"},
        ]
    }
    _, tools = to_messages(sample, "hermes")
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "ping"


# --------- thinking-mode plumbing ----------


def _thinking_sample() -> dict[str, Any]:
    return {
        "query": "ping?",
        "tools": "[]",
        "answers": json.dumps([{"name": "ping", "arguments": {}}]),
    }


def test_format_sample_preserve_passes_enable_thinking_true() -> None:
    tok = _StubTokenizer()
    format_sample(_thinking_sample(), tok, source="xlam", enable_thinking=True)
    assert tok.calls[0]["enable_thinking"] is True


def test_format_sample_strip_passes_enable_thinking_false_and_strips_inline_think() -> None:
    tok = _StubTokenizer()
    sample = {
        "conversations": [
            {"from": "human", "value": "ping?"},
            {
                "from": "gpt",
                "value": "<think>thinking aloud...</think>pong",
            },
        ]
    }
    out = format_sample(sample, tok, source="hermes", enable_thinking=False)
    assert tok.calls[0]["enable_thinking"] is False
    # Inline reasoning must be removed from message content too -- the template
    # alone can't rewrite data, only suppress its own placeholder.
    assert "<think>" not in out
    assert "thinking aloud" not in out
    assert "pong" in out


def test_enable_thinking_for_preserve_and_strip_are_constant() -> None:
    assert all(enable_thinking_for("preserve", index=i, seed=42) for i in range(20))
    assert not any(enable_thinking_for("strip", index=i, seed=42) for i in range(20))


def test_enable_thinking_for_mix_75_25_split_is_deterministic() -> None:
    n = 1000
    flags = [enable_thinking_for("mix-75-25", index=i, seed=42) for i in range(n)]
    fraction = sum(flags) / n
    # Tolerate sampling slack but reject a clearly miscalibrated split.
    assert 0.70 < fraction < 0.80, f"mix-75-25 produced {fraction:.3f} of True"
    # Same seed + index → same flag, every time.
    again = [enable_thinking_for("mix-75-25", index=i, seed=42) for i in range(n)]
    assert flags == again


def test_enable_thinking_for_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown thinking_mode"):
        enable_thinking_for("nope", index=0, seed=0)  # type: ignore[arg-type]
