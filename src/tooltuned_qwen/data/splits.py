"""Deterministic train / val / holdout splitting."""

from typing import Any


def split_dataset(
    dataset: Any,
    *,
    val_frac: float = 0.05,
    holdout_frac: float = 0.05,
    seed: int = 0,
) -> tuple[Any, Any, Any]:
    raise NotImplementedError("Phase 1.1")
