"""Deterministic train / val / holdout splitting.

Permutation is computed with a stdlib RNG seeded by `seed`, so the index
order is reproducible without importing `datasets` at module load.
"""

from __future__ import annotations

import random
from typing import Any


def permutation(n: int, seed: int) -> list[int]:
    """Return a deterministic shuffle of `range(n)` for the given seed."""
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    return indices


def split_dataset(
    dataset: Any,
    *,
    val_frac: float = 0.05,
    holdout_frac: float = 0.05,
    seed: int = 0,
) -> tuple[Any, Any, Any]:
    """Split a HF `Dataset` into (train, val, holdout) via a seeded permutation."""
    n = len(dataset)
    holdout_n = int(n * holdout_frac)
    val_n = int(n * val_frac)
    train_n = n - val_n - holdout_n
    if train_n <= 0:
        raise ValueError(
            f"split leaves no train rows: n={n}, val={val_n}, holdout={holdout_n}"
        )

    perm = permutation(n, seed)
    train_idx = perm[:train_n]
    val_idx = perm[train_n : train_n + val_n]
    holdout_idx = perm[train_n + val_n :]
    return dataset.select(train_idx), dataset.select(val_idx), dataset.select(holdout_idx)
