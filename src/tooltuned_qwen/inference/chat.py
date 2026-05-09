"""Interactive tool-calling chat demo."""

from typing import Any


def chat(model: Any, tokenizer: Any, query: str, *, tools: list[dict[str, Any]]) -> str:
    raise NotImplementedError("Phase 6.2")
