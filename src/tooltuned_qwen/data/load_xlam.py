"""Load Salesforce/xlam-function-calling-60k from the HF hub."""

from __future__ import annotations

import os
from typing import Any

REPO_ID = "Salesforce/xlam-function-calling-60k"


def load_xlam(
    *, cache_dir: str | None = None, split: str = "train", token: str | None = None
) -> Any:
    """Return the xLAM split as a HuggingFace `Dataset`.

    xLAM is gated; `token` defaults to `HF_TOKEN` from the env. Passing it
    explicitly here (rather than relying on `huggingface_hub.get_token()`)
    sidesteps Colab's userdata cache, which can grandfather another
    account's token into the implicit auth chain (Phase 1.3 finding).
    """
    from datasets import load_dataset

    auth = token if token is not None else os.environ.get("HF_TOKEN")
    return load_dataset(REPO_ID, split=split, cache_dir=cache_dir, token=auth)
