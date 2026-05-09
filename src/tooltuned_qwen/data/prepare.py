"""High-level dataset prep helpers used by the Colab launcher notebook.

Anything more elaborate than a one-liner over `load_xlam` / `load_hermes`
goes here so the notebook stays "no logic in cells" (brief section 6.2 item 3).
"""

from __future__ import annotations

from typing import Any

from .load_xlam import load_xlam


def smoke_dataset(n: int = 8) -> Any:
    """Tiny xlam slice for the Phase 1.3 end-to-end smoke run.

    Returns raw rows; the chat template is applied inside `train()` once the
    Qwen tokenizer is loaded.
    """
    return load_xlam().select(range(n))
