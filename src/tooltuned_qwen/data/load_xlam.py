"""Load Salesforce/xlam-function-calling-60k from the HF hub."""

from __future__ import annotations

from typing import Any

REPO_ID = "Salesforce/xlam-function-calling-60k"


def load_xlam(*, cache_dir: str | None = None, split: str = "train") -> Any:
    """Return the xLAM split as a HuggingFace `Dataset`."""
    from datasets import load_dataset

    return load_dataset(REPO_ID, split=split, cache_dir=cache_dir)
