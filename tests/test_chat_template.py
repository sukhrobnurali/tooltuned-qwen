"""Chat-template round-trip: formatter must route through `apply_chat_template`.

The actual Qwen 3.5 tokenizer can't run in CI (heavy ML deps live in the
`[colab]` extra), so this test uses a stub tokenizer that mimics the
HF `apply_chat_template` interface and asserts the formatter:

  - Calls `apply_chat_template` with the right messages and tools.
  - Passes `tokenize=False` (we want a string for the SFTTrainer text field).
  - Round-trips xlam fields (query / tools / answers) into the messages.

This guards brief section 11 q6: hand-written templating silently breaks training.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tooltuned_qwen.data.format import format_sample, to_messages


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
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        parts: list[str] = []
        if tools:
            parts.append(f"[tools={len(tools)}]")
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
    # Arguments must be a JSON string (HF tool-call convention).
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "Tokyo"}
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
