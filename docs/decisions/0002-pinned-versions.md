# ADR 0002 — Library version pins

- **Status:** Accepted (provisional — frozen for Phase 0.4 scaffolding; revisit if S3 smoke test fails)
- **Date:** 2026-05-10
- **Decision driver:** Brief §9 ("latest stable") + §11 q2 + Plan Phase 0.2.

## Context

Qwen 3.5 4B requires `transformers>=5` per Unsloth's own documentation ("Older versions will not work"). Transformers v5 broke a number of trainer-internal APIs that TRL and PEFT had been depending on, so the "latest of everything" naive choice is not safe. We need a mutually-compatible set verified against an Unsloth release that supports v5.

## Decision

Pin the following floor versions for v1.0. Lockfile (`uv.lock`) is the source of truth at install time; `pyproject.toml` carries the human-readable constraints.

| Package | Pin | Why |
|---|---|---|
| `python` | `>=3.12,<3.13` | Brief §9 + plan; Colab L4 supports 3.12; some Unsloth wheels lag 3.13. |
| `unsloth` | `>=2026.2.1` | First Unsloth release with documented `transformers>=5` support (off by default — must be opted-in). |
| `unsloth_zoo` | `>=2026.2.1` | Tracks `unsloth` in lockstep; Unsloth's own update command keeps both pinned together. |
| `transformers` | `>=5.1.0,<6` | 5.1.0 is the validated pair with `trl==0.27.1` ("supports >80% of Unsloth's 120 notebooks" per Unsloth issue tracker). 5.0 has known regressions in `Trainer.training_step`. |
| `trl` | `>=0.27.1,<0.28` | Pair with transformers 5.1; `SFTTrainer` API stable in 0.27.x. Earlier 0.24.x line is transformers-v4 only. |
| `peft` | `>=0.18,<0.20` | LoRA API stable; 0.18+ supports transformers v5 adapter hooks. |
| `bitsandbytes` | `>=0.48` | Required by `adamw_8bit` optimizer (still used despite bf16 LoRA per Unsloth recipe). 4-bit quant path unused, but bnb is still the optimizer host. |
| `accelerate` | `>=1.5,<2` | Stable launcher; transformers v5 minimum. |
| `datasets` | `>=3.5,<4` | xLAM and Hermes both load on the 3.x line. |
| `huggingface_hub` | `>=0.30,<0.40` | `ModelCard`, `upload_folder`, snapshot APIs stable. |
| `torch` | `>=2.6,<2.8` | Last-known-good with Unsloth Triton kernels on L4 (CUDA 12.x). |
| `wandb` | `>=0.18` | Stable run-logging API; nothing exotic needed. |
| `pydantic` | `>=2.8,<3` | Plan Phase 0.6 config schema. |
| `pyyaml` | `>=6` | YAML config loader. |

Dev/CI deps (separate `[dev]` extra):

| Package | Pin |
|---|---|
| `pytest` | `>=8` |
| `ruff` | `>=0.7` |
| `pyright` | `>=1.1.380` (or `mypy>=1.11` if we go that way; decide in S2) |

## Strategy

1. **Compatible floors, not exact pins**, in `pyproject.toml`. The `uv.lock` will record the exact resolved versions and is the reproducibility contract.
2. **Freeze the lock at the end of Phase 0.4** (S2). After that, no upgrades inside a session unless required to fix a real bug. If an upgrade is required, it gets an ADR amendment.
3. **Unsloth v5-mode is opt-in.** Per the Unsloth issue tracker, transformers v5 support is not enabled by default in `unsloth>=2026.2.1` "due to possible instability." We will set the documented env-var / install-extra at the top of each notebook and in the `pyproject.toml` install command. (Exact env-var name to be re-checked at Phase 0.4 — Unsloth's docs are the source.)
4. **Smoke test in S3 is the final acceptance gate** for these pins. If the 1-step run on 8 samples crashes due to a version mismatch, we treat that as a pin bug, fix the offending pin, and write an ADR-0002 amendment.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Pin to exact versions in `pyproject.toml` | Loses the patch-level fixes that Unsloth ships frequently; reproducibility comes from the lockfile, not from `pyproject.toml`. |
| Stay on `transformers<5` (v4 line) | Unsloth doc explicitly states older transformers "will not work" for Qwen 3.5. Hard blocker. |
| Pin to bleeding-edge `transformers==5.2.x` | No Unsloth notebook coverage yet at the time of this ADR; high risk of silent breakage. |
| Use `mypy` instead of `pyright` | Either works; defer to S2 when we wire CI. |

## Risks

1. **Unsloth's v5 mode is gated as "possibly unstable."** Mitigation: Phase 1 smoke test catches this for ~5 Colab units, before any real training.
2. **TRL 0.27 → 0.28 may break `SFTTrainer` again.** Mitigation: pin `<0.28` so an `uv sync` upgrade can't quietly bump us into a breaking minor.
3. **Transformers v5 silently changed dataset-handling defaults in `Trainer`.** Mitigation: explicit `Trainer` args in our config; do not rely on defaults.
4. **Colab's preinstalled torch may conflict with our pin.** Mitigation: notebook cell #1 does `pip install -e .[colab]` with `--upgrade` to overwrite the runtime's torch.

## Consequences

- `pyproject.toml` (Phase 0.4) carries these floors as `dependencies`.
- `pip install -e .[colab]` is the canonical Colab install (Phase 1.2 notebook cell #1).
- CI (Phase 0.4) installs the same floors and runs lint + type + tests; CI does **not** train.
- Lockfile (`uv.lock`) is committed and is the reproducibility source of truth.

## References

- Unsloth doc — `unsloth.ai/docs/models/qwen3.5/fine-tune` ("transformers v5 required")
- Unsloth issue #4022 — "Provide official way to install with transformers 5.x"
- Unsloth release notes — `2026.2.x` series
- Brief: [03_tooltuned_qwen_brief.md](../../03_tooltuned_qwen_brief.md) — §9, §11 q2
- Plan: [read-the-project-description-cached-lightning.md](../../../../Users/User/.claude/plans/read-the-project-description-cached-lightning.md) — Phase 0.2, Phase 0.4
