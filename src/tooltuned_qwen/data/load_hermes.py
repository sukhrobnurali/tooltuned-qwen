"""Load NousResearch/hermes-function-calling-v1 from the HF hub."""

from __future__ import annotations

import os
from typing import Any

REPO_ID = "NousResearch/hermes-function-calling-v1"


def load_hermes(
    *,
    cache_dir: str | None = None,
    subset: str = "func_calling",
    split: str = "train",
    token: str | None = None,
) -> Any:
    """Return a Hermes function-calling split as a HuggingFace `Dataset`.

    Hermes is ungated, but the `token` kwarg keeps the API symmetrical with
    `load_xlam` and lets a caller force-bypass the implicit auth chain.

    Hermes ships several configs; `func_calling` is the single-turn slice used
    for the Phase 2 dataset comparison. Multi-turn `func_calling_singleturn` and
    `glaive_func_calling` are also available -- pass `subset=` to switch.
    """
    from datasets import load_dataset

    auth = token if token is not None else os.environ.get("HF_TOKEN")
    return load_dataset(REPO_ID, subset, split=split, cache_dir=cache_dir, token=auth)
