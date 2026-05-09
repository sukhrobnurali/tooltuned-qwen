"""Load base + adapter for inference."""

from typing import Any


def load_model(*, adapter_repo: str | None = None, dtype: str = "bf16") -> tuple[Any, Any]:
    """Returns (model, tokenizer)."""
    raise NotImplementedError("Phase 6.2")
