"""W&B logging + periodic eval callbacks for SFTTrainer."""

from typing import Any


def build_callbacks(*, eval_every: int = 100, wandb_project: str | None = None) -> list[Any]:
    raise NotImplementedError("Phase 3.2")
