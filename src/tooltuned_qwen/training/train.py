"""Main training entrypoint -- Unsloth FastLanguageModel + TRL SFTTrainer.

Heavy ML imports are deferred into the function body so the module is cheap
to import on a CPU machine (CI, lint, type-check) where the `[colab]` extra
isn't installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..data.format import Source, enable_thinking_for, format_sample
from .config import TrainingConfig, load_config


def train(
    *,
    config_path: str | None = None,
    config: TrainingConfig | None = None,
    dataset: Any | None = None,
) -> str:
    """Run a fine-tune. Returns the saved adapter path.

    Pass exactly one of `config_path` (load from YAML) or `config` (an in-memory
    TrainingConfig). The config-object form lets the Phase 2 ablation notebook
    point a thinking-mode YAML at the dataset winner without writing a temp
    file each run.
    """
    if (config_path is None) == (config is None):
        raise ValueError("Pass exactly one of `config_path` or `config`.")
    cfg = config if config is not None else load_config(config_path)  # type: ignore[arg-type]
    return _run(cfg, dataset=dataset)


def _run(cfg: TrainingConfig, *, dataset: Any | None) -> str:
    import unsloth  # noqa: F401  Unsloth must load before trl/transformers per its load-order docs.
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    # Unsloth loads with `device_map='auto'`; Accelerate's `prepare()` then refuses
    # to wrap it because it conservatively treats auto-mapped models as distributed.
    # Single-process single-GPU L4 path is fine -- bypass the check.
    os.environ.setdefault("ACCELERATE_BYPASS_DEVICE_MAP", "true")

    if cfg.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = cfg.wandb_project
        if cfg.wandb_entity is not None:
            os.environ["WANDB_ENTITY"] = cfg.wandb_entity
        os.environ.setdefault("WANDB_RUN_NAME", cfg.run_name)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_len,
        load_in_4bit=False,  # bf16 LoRA per ADR 0001 (Unsloth advises against 4-bit Qwen 3.5).
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        random_state=cfg.seed,
        use_gradient_checkpointing="unsloth",
    )

    if dataset is None:
        dataset = _load_dataset(cfg)
    train_ds = _ensure_text_column(dataset, tokenizer, cfg)

    run_dir = Path(cfg.output_dir) / cfg.run_name
    sft_args = SFTConfig(
        output_dir=str(run_dir),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.optimizer.learning_rate,
        warmup_ratio=cfg.optimizer.warmup_ratio,
        weight_decay=cfg.optimizer.weight_decay,
        optim=cfg.optimizer.name,
        max_steps=cfg.max_steps if cfg.max_steps is not None else -1,
        num_train_epochs=cfg.epochs if cfg.epochs is not None else 1,
        max_seq_length=cfg.max_seq_len,
        packing=cfg.packing,
        seed=cfg.seed,
        report_to="wandb" if cfg.wandb_project is not None else "none",
        run_name=cfg.run_name,
        save_strategy="no",
        logging_steps=1,
        bf16=True,
        fp16=False,
        dataset_text_field="text",
        # Single-process tokenization: Unsloth's monkey-patches make the
        # tokenizer non-picklable, which breaks the default multiprocess
        # `dataset.map` path inside SFTTrainer.
        dataset_num_proc=1,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_args,
        train_dataset=train_ds,
    )
    trainer.train()

    adapter_dir = run_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    return str(adapter_dir)


def _load_dataset(cfg: TrainingConfig) -> Any:
    if len(cfg.data.sources) != 1:
        raise NotImplementedError(
            "Phase 1 supports a single dataset source; multi-source mixing lands in Phase 3."
        )
    source = cfg.data.sources[0]
    if source == "xlam":
        from ..data.load_xlam import load_xlam

        ds = load_xlam()
    elif source == "hermes":
        from ..data.load_hermes import load_hermes

        ds = load_hermes()
    else:  # pragma: no cover - schema enum guards this.
        raise ValueError(f"unknown source: {source}")

    if cfg.data.max_samples is not None:
        ds = ds.select(range(min(cfg.data.max_samples, len(ds))))
    return ds


def _ensure_text_column(dataset: Any, tokenizer: Any, cfg: TrainingConfig) -> Any:
    if "text" in dataset.column_names:
        return dataset
    source: Source = cfg.data.sources[0]
    mode = cfg.thinking_mode
    seed = cfg.seed
    keep = list(dataset.column_names)

    def _render(sample: Any, idx: int) -> dict[str, str]:
        et = enable_thinking_for(mode, index=idx, seed=seed)
        return {
            "text": format_sample(sample, tokenizer, source=source, enable_thinking=et)
        }

    # `with_indices=True` lets the mix-75-25 mode produce a deterministic
    # 75/25 split keyed on (seed, position) rather than wall-clock RNG state.
    return dataset.map(_render, with_indices=True, remove_columns=keep)
