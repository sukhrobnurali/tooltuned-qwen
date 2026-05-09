"""Main training entrypoint — Unsloth FastLanguageModel + TRL SFTTrainer."""

from typing import Any


def train(*, config_path: str, dataset: Any | None = None) -> str:
    """Run a fine-tune. Returns the path of the saved adapter."""
    raise NotImplementedError("Phase 1.2")
