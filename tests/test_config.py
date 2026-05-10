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
    # Phase 3 main-run knobs -- effective batch stays at 16 (bs * ga), but the
    # bs=16/ga=1 form trades VRAM for fewer steps (faster on A100). xLAM is
    # 60k rows; 10k is the first-attempt subset that fits the ~25-unit slice.
    assert cfg.batch_size == 16
    assert cfg.grad_accum_steps == 1
    assert cfg.data.max_samples == 10000
    assert cfg.wandb_project == "tooltuned-qwen"
    # Pinned to the public team workspace where the Phase 3 run lives.
    # Shipping with a hardcoded entity is what produced the broken-link bug
    # in the first Phase 3 push; the field is now explicit + tested so a
    # rename (or a flip back to private) trips the test.
    assert cfg.wandb_entity == "sukhrob-production"


def test_smoke_and_ablation_yamls_leave_wandb_off() -> None:
    """Only the main run reports to W&B. Ablation/smoke runs must not pollute
    the public dashboard, so their YAMLs must leave wandb_project unset."""
    for path in sorted(REPO_ROOT.glob("configs/ablation_*.yaml")):
        assert load_config(path).wandb_project is None, path
    assert load_config(REPO_ROOT / "configs" / "smoke.yaml").wandb_project is None


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
