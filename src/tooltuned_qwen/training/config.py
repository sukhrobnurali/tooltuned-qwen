"""Pydantic v2 schema for the training run config (single source of truth).

The YAML files under `configs/` are parsed into `TrainingConfig`; the trainer
never reads YAML directly. New knobs go here first, then into the YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DatasetSource = Literal["xlam", "hermes"]
ThinkingMode = Literal["preserve", "strip", "mix-75-25"]


class LoRAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(default=16, ge=1)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[DatasetSource] = Field(default_factory=lambda: ["xlam"])
    max_samples: int | None = None
    val_frac: float = Field(default=0.05, ge=0.0, lt=0.5)
    holdout_frac: float = Field(default=0.05, ge=0.0, lt=0.5)


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "adamw_8bit"
    learning_rate: float = Field(default=2e-4, gt=0.0)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_name: str
    base_model: str = "Qwen/Qwen3.5-4B"
    seed: int = 42

    data: DataConfig = Field(default_factory=DataConfig)
    max_seq_len: int = Field(default=2048, ge=128)
    packing: bool = True

    lora: LoRAConfig = Field(default_factory=LoRAConfig)

    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    batch_size: int = Field(default=4, ge=1)
    grad_accum_steps: int = Field(default=4, ge=1)
    epochs: int | None = 1
    max_steps: int | None = None

    thinking_mode: ThinkingMode = "preserve"

    output_dir: str = "outputs"

    # When set, the trainer reports to W&B under this project name. Smoke and
    # ablation YAMLs leave it null so their runs don't pollute the public
    # dashboard; only `default.yaml` (the Phase 3 main run) flips it on.
    wandb_project: str | None = None

    @model_validator(mode="after")
    def _exactly_one_of_epochs_or_max_steps(self) -> Self:
        if (self.epochs is None) == (self.max_steps is None):
            raise ValueError("Set exactly one of `epochs` or `max_steps`.")
        return self


def load_config(path: str | Path) -> TrainingConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TrainingConfig.model_validate(raw)
