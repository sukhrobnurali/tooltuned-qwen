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


def test_train_requires_exactly_one_config_argument() -> None:
    """train() takes either a YAML path or a TrainingConfig, not both/neither.
    Phase 2 ablations need the in-memory form to swap data.sources without
    writing temp YAMLs; guard the contract so neither caller can pass nothing."""
    from tooltuned_qwen.training.train import train

    with pytest.raises(ValueError, match="exactly one"):
        train()
    with pytest.raises(ValueError, match="exactly one"):
        train(config_path="x", config=TrainingConfig(run_name="x"))


def test_ablation_yamls_load() -> None:
    """All Phase 2 ablation YAMLs must validate; a typo here means a wasted
    Colab run, so catch it in CI before the GPU spins."""
    ablation_dir = REPO_ROOT / "configs"
    yamls = sorted(ablation_dir.glob("ablation_*.yaml"))
    assert len(yamls) == 5, f"expected 5 ablation YAMLs, found {len(yamls)}"
    for path in yamls:
        cfg = load_config(path)
        assert cfg.lora.rank == 16
        assert cfg.optimizer.learning_rate == pytest.approx(2e-4)
        assert cfg.data.max_samples == 1000
        assert cfg.epochs == 1
