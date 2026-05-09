"""YAML <-> TrainingConfig round-trip + invariant checks."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tooltuned_qwen.training.config import TrainingConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO_ROOT / "configs" / "default.yaml"


def test_default_yaml_loads() -> None:
    cfg = load_config(DEFAULT_YAML)
    assert cfg.base_model == "Qwen/Qwen3.5-4B"
    assert cfg.lora.rank == 16
    assert cfg.data.sources == ["xlam"]
    assert cfg.thinking_mode == "preserve"


def test_round_trip(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_YAML)
    out = tmp_path / "cfg.yaml"
    out.write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False), encoding="utf-8")
    assert load_config(out) == cfg


def test_epochs_xor_max_steps_required() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(run_name="x", epochs=None, max_steps=None)

    with pytest.raises(ValidationError):
        TrainingConfig(run_name="x", epochs=1, max_steps=100)


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(run_name="x", bogus_field=True)  # type: ignore[call-arg]
