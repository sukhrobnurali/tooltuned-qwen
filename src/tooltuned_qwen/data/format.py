"""Render function-calling samples through the Qwen 3.5 chat template.

The template must come from `tokenizer.apply_chat_template` — hand-written
templating silently breaks training (brief §11 q6).
"""

from collections.abc import Mapping
from typing import Any


def format_sample(sample: Mapping[str, Any], tokenizer: Any) -> str:
    raise NotImplementedError("Phase 1.1")
