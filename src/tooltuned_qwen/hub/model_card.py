"""Render MODEL_CARD.md from training config + (optional) BFCL results.

The card is regenerated programmatically so it never drifts from the run
that produced the weights. Phase 3.3 calls this with `bfcl_results=None`
so the BFCL section renders as "TBD (Phase 4)"; Phase 4 calls it again
with the full results dict.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from ..training.config import TrainingConfig, load_config

# Schema Phase 4 fills (kept here so the producer/consumer agree):
# bfcl_results = {
#     "overall_base":  float,        # accuracy in [0, 1]
#     "overall_tuned": float,
#     "delta":         float,        # tuned - base, in [-1, 1]
#     "n_total":       int,          # number of BFCL prompts evaluated
#     "evaluated_at":  str,          # ISO date, e.g. "2026-05-12"
#     "rows":          list[dict] | None,  # optional per-category breakdown
#                                          # each row: {"category", "base", "tuned"}
# }


def generate_card(
    *,
    bfcl_results: dict[str, Any] | None,
    training_config_path: str,
    out_path: str,
    hf_repo: str = "sukhrobnurali/tooltuned-qwen-3.5-4b",
    github_repo: str = "https://github.com/sukhrobnurali/tooltuned-qwen",
    wandb_dashboard: str | None = None,
) -> str:
    """Render the model card to `out_path` and return the absolute path."""
    cfg = load_config(training_config_path)
    if wandb_dashboard is None and cfg.wandb_project is not None:
        wandb_dashboard = f"https://wandb.ai/sukhrobnurali/{cfg.wandb_project}"

    body = _render(
        cfg=cfg,
        bfcl_results=bfcl_results,
        hf_repo=hf_repo,
        github_repo=github_repo,
        wandb_dashboard=wandb_dashboard,
    )
    out = Path(out_path).resolve()
    out.write_text(body, encoding="utf-8")
    return str(out)


def _render(
    *,
    cfg: TrainingConfig,
    bfcl_results: dict[str, Any] | None,
    hf_repo: str,
    github_repo: str,
    wandb_dashboard: str | None,
) -> str:
    parts: list[str] = []
    parts.append(_frontmatter(cfg))
    parts.append("# tooltuned-qwen-3.5-4b\n")
    parts.append(_tldr(cfg, bfcl_results, hf_repo))
    parts.append(_bfcl_section(bfcl_results))
    parts.append(_training_data_section(cfg))
    parts.append(_procedure_section(cfg))
    parts.append(_hyperparams_section(cfg))
    parts.append(_intended_use_section())
    parts.append(_out_of_scope_section())
    parts.append(_limitations_section())
    parts.append(_license_section())
    parts.append(_reproduction_section(cfg, github_repo, wandb_dashboard))
    parts.append(_citation_section(hf_repo))
    parts.append(_author_section())
    return "\n".join(parts).rstrip() + "\n"


def _frontmatter(cfg: TrainingConfig) -> str:
    return (
        "---\n"
        "license: apache-2.0\n"
        f"base_model: {cfg.base_model}\n"
        "library_name: peft\n"
        "tags:\n"
        "  - tool-calling\n"
        "  - function-calling\n"
        "  - lora\n"
        "  - unsloth\n"
        "  - qwen\n"
        "datasets:\n"
        "  - Salesforce/xlam-function-calling-60k\n"
        "language:\n"
        "  - en\n"
        "---\n"
    )


def _tldr(
    cfg: TrainingConfig, bfcl_results: dict[str, Any] | None, hf_repo: str
) -> str:
    headline = _headline_number(bfcl_results)
    return (
        "## TL;DR\n\n"
        f"LoRA (rank {cfg.lora.rank}) fine-tune of `{cfg.base_model}` for tool-calling, "
        "trained on Salesforce/xlam-function-calling-60k with Unsloth + TRL.\n\n"
        f"- **Result on BFCL V4:** {headline}\n"
        f"- **Adapter:** [{hf_repo}](https://huggingface.co/{hf_repo})\n"
        f"- **Format:** bf16 LoRA adapter (Unsloth advises against 4-bit "
        "quant for Qwen 3.5)\n"
    )


def _headline_number(bfcl_results: dict[str, Any] | None) -> str:
    if bfcl_results is None:
        return "TBD (Phase 4)"
    base = bfcl_results.get("overall_base")
    tuned = bfcl_results.get("overall_tuned")
    delta = bfcl_results.get("delta")
    if base is None or tuned is None or delta is None:
        return "TBD (Phase 4)"
    sign = "+" if delta >= 0 else ""
    return f"{tuned * 100:.1f}% (base {base * 100:.1f}%, {sign}{delta * 100:.1f}pp)"


def _bfcl_section(bfcl_results: dict[str, Any] | None) -> str:
    if bfcl_results is None:
        return (
            "## BFCL V4 results\n\n"
            "TBD (Phase 4). Numbers and the comparison chart land once the full BFCL\n"
            "evaluation completes; this card is regenerated then.\n"
        )

    lines: list[str] = ["## BFCL V4 results\n"]
    base = bfcl_results.get("overall_base")
    tuned = bfcl_results.get("overall_tuned")
    delta = bfcl_results.get("delta")
    n = bfcl_results.get("n_total")
    when = bfcl_results.get("evaluated_at")

    if base is not None and tuned is not None and delta is not None:
        sign = "+" if delta >= 0 else ""
        lines.append("| Model | Overall accuracy |")
        lines.append("| --- | --- |")
        lines.append(f"| Base ({_safe_str(bfcl_results.get('base_model'))}) | {base * 100:.1f}% |")
        lines.append(f"| **This adapter** | **{tuned * 100:.1f}%** |")
        lines.append(f"| Delta | **{sign}{delta * 100:.1f}pp** |")
        lines.append("")

    rows = bfcl_results.get("rows")
    if isinstance(rows, list) and rows:
        lines.append("### Per-category breakdown\n")
        lines.append("| Category | Base | Tuned | Delta |")
        lines.append("| --- | --- | --- | --- |")
        for r in rows:
            cat = _safe_str(r.get("category"))
            rb = r.get("base")
            rt = r.get("tuned")
            if rb is None or rt is None:
                continue
            d = rt - rb
            s = "+" if d >= 0 else ""
            lines.append(f"| {cat} | {rb * 100:.1f}% | {rt * 100:.1f}% | {s}{d * 100:.1f}pp |")
        lines.append("")

    if n is not None or when is not None:
        meta = []
        if n is not None:
            meta.append(f"n={n}")
        if when is not None:
            meta.append(f"evaluated {when}")
        lines.append(f"_{', '.join(meta)}_")
        lines.append("")
    return "\n".join(lines)


def _training_data_section(cfg: TrainingConfig) -> str:
    sources = ", ".join(f"`{s}`" for s in cfg.data.sources)
    samples = (
        f"{cfg.data.max_samples:,}" if cfg.data.max_samples is not None else "full"
    )
    return (
        "## Training data\n\n"
        f"- Sources: {sources} (Salesforce/xlam-function-calling-60k)\n"
        f"- Samples used: {samples}\n"
        f"- Validation fraction: {cfg.data.val_frac}\n"
        f"- Held-out fraction: {cfg.data.holdout_frac}\n"
        f"- Thinking-mode strategy: `{cfg.thinking_mode}` (preserves Qwen 3.5's "
        "default reasoning trace; xLAM rows have no `<think>` content so the "
        "three strategies converge in practice)\n"
    )


def _procedure_section(cfg: TrainingConfig) -> str:
    return (
        "## Training procedure\n\n"
        "Supervised fine-tuning via Unsloth's `FastLanguageModel` + TRL's "
        "`SFTTrainer`. Adapter only — base weights are frozen. Single A100 "
        "(40 GB), bf16, gradient checkpointing on.\n"
    )


def _hyperparams_section(cfg: TrainingConfig) -> str:
    rows = [
        ("base_model", cfg.base_model),
        ("lora.rank", cfg.lora.rank),
        ("lora.alpha", cfg.lora.alpha),
        ("lora.dropout", cfg.lora.dropout),
        ("lora.target_modules", ", ".join(cfg.lora.target_modules)),
        ("optimizer", cfg.optimizer.name),
        ("learning_rate", cfg.optimizer.learning_rate),
        ("warmup_ratio", cfg.optimizer.warmup_ratio),
        ("weight_decay", cfg.optimizer.weight_decay),
        ("batch_size", cfg.batch_size),
        ("grad_accum_steps", cfg.grad_accum_steps),
        ("effective_batch_size", cfg.batch_size * cfg.grad_accum_steps),
        ("epochs", cfg.epochs if cfg.epochs is not None else "n/a"),
        ("max_steps", cfg.max_steps if cfg.max_steps is not None else "n/a"),
        ("max_seq_len", cfg.max_seq_len),
        ("packing", cfg.packing),
        ("seed", cfg.seed),
    ]
    lines = ["## Hyperparameters\n", "| Knob | Value |", "| --- | --- |"]
    lines.extend(f"| `{k}` | {v} |" for k, v in rows)
    lines.append("")
    return "\n".join(lines)


def _intended_use_section() -> str:
    return (
        "## Intended use\n\n"
        "Function calling / tool use in chat agents. The adapter pairs with "
        "the base Qwen 3.5 4B chat template; pass tool schemas in the system "
        "prompt and the model emits `<tool_call>` blocks (or XML-tagged "
        "`<function=...>` calls; the inference helper parses both).\n"
    )


def _out_of_scope_section() -> str:
    return (
        "## Out of scope\n\n"
        "- Non-English instruction following (xLAM is English-only).\n"
        "- Long-context tool dialogues beyond 2,048 tokens — the adapter was "
        "trained at that sequence length.\n"
        "- Safety-critical decisions. The adapter inherits Qwen 3.5's safety "
        "profile, no additional alignment was applied.\n"
    )


def _limitations_section() -> str:
    return (
        "## Limitations\n\n"
        "- LoRA rank 16 is a known-safe default, not an ablated optimum. "
        "Higher ranks may move the BFCL number further; rank ablations are a "
        "stretch goal.\n"
        "- BFCL holds out one slice of tool-calling behavior; performance on "
        "task families outside that distribution (multi-turn agentic loops, "
        "fully novel APIs) is not directly measured.\n"
    )


def _license_section() -> str:
    return (
        "## License\n\n"
        "Apache-2.0, matching the base model `Qwen/Qwen3.5-4B`.\n"
    )


def _reproduction_section(
    cfg: TrainingConfig, github_repo: str, wandb_dashboard: str | None
) -> str:
    lines = [
        "## Reproduction\n",
        f"Source: [{github_repo}]({github_repo}). Pinned versions live in "
        "`pyproject.toml`; the lockfile (`uv.lock`) is the reproducibility "
        "contract.\n",
        "```bash",
        f"git clone {github_repo}",
        "cd tooltuned-qwen",
        "uv sync",
        "# Run on Colab Pro A100; see notebooks/colab_main.ipynb",
        "```",
        "",
    ]
    if wandb_dashboard is not None:
        lines.append(f"Training curves: [{wandb_dashboard}]({wandb_dashboard}).")
        lines.append("")
    return "\n".join(lines)


def _citation_section(hf_repo: str) -> str:
    year = date.today().year
    return (
        "## Citation\n\n"
        "```bibtex\n"
        "@misc{nurali_tooltuned_qwen_2026,\n"
        "  author       = {Sukhrob Nurali},\n"
        "  title        = {tooltuned-qwen-3.5-4b: a tool-calling LoRA for Qwen 3.5 4B},\n"
        f"  year         = {{{year}}},\n"
        f"  howpublished = {{\\url{{https://huggingface.co/{hf_repo}}}}}\n"
        "}\n"
        "```\n"
    )


def _author_section() -> str:
    return (
        "## Author\n\n"
        "- Sukhrob Nurali — `sukhrobnurali@gmail.com`\n"
        "- Hugging Face: [sukhrobnurali](https://huggingface.co/sukhrobnurali)\n"
        "- GitHub: [sukhrobnurali](https://github.com/sukhrobnurali)\n"
    )


def _safe_str(x: Any) -> str:
    return "?" if x is None else str(x)
