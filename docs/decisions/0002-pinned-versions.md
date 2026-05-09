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
| `trl` | `>=0.20,<0.25` | Floor matches the lower edge of `unsloth>=2026.2.1`'s supported range; `<0.25` matches its upper cap. The original `>=0.27.1,<0.28` row turned out to be unsatisfiable against any released `unsloth` (see Amendment 2026-05-10 #2). `SFTTrainer` API still stable in this band. |
| `peft` | `>=0.18,<0.20` | LoRA API stable; 0.18+ supports transformers v5 adapter hooks. |
| `bitsandbytes` | `>=0.48` | Required by `adamw_8bit` optimizer (still used despite bf16 LoRA per Unsloth recipe). 4-bit quant path unused, but bnb is still the optimizer host. |
| `accelerate` | `>=1.5,<2` | Stable launcher; transformers v5 minimum. |
| `datasets` | `>=3.5,<4` | xLAM and Hermes both load on the 3.x line. |
| `huggingface_hub` | `>=1.3,<2` | Forced floor by `transformers>=5.1` (sub-5.4 needs `>=1.3`, 5.4+ needs `>=1.5`). Provisional `>=0.30,<0.40` would not resolve — see Amendment 2026-05-10. `ModelCard`, `upload_folder`, snapshot APIs stable on the 1.x line. |
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

## Amendment 2026-05-10 (Phase 0.4 / S2)

Caught at first `uv sync --extra dev`: the `huggingface_hub` floor `>=0.30,<0.40` is incompatible with the `transformers>=5.1.0` floor. uv's resolver reported:

- `transformers` in `[5.1.0, 5.3.0]` requires `huggingface_hub>=1.3.0`.
- `transformers` in `[5.4.0+]` requires `huggingface_hub>=1.5.0`.

Both ranges sit above the original `<0.40` ceiling — the v0.x line shipped `ModelCard` years ago, but transformers v5 picked up enough of the new hub APIs that the `0.x` shim was retired. We did not catch this in 0.2 because we read Unsloth's compat matrix and not transformers' own.

**Resolution:** widen to `huggingface_hub>=1.3,<2`. Public APIs we use (`ModelCard`, `upload_folder`, `snapshot_download`) all carry forward through the 1.x major. Pin updated in `pyproject.toml` and the table above.

No other pin moved. The lockfile generated at the end of S2 is the new acceptance baseline; if a smoke test in S3 surfaces a second forced floor, ADR amendment #2 will follow.

## Amendment 2026-05-10 #2 (Phase 0.4 / S2)

Caught at the *next* `uv sync --extra dev` after Amendment #1: every released `unsloth` (up through `2026.5.2`) pins `trl` to one of `>=0.18.2,<0.19` or `>0.19,<=0.24`. The original `trl>=0.27.1,<0.28` floor was therefore unsatisfiable against *any* version of unsloth, not just a specific one — the rationale ("0.27 is the validated pair with transformers 5.1") was traced to the wrong source (an unsloth issue thread describing a future intent, not a shipped support contract). Since the brief's hard constraint is "use Unsloth as-is" (§3 non-goal: not a new training framework), unsloth's pin envelope is what we have to live with.

**Resolution:**
- Widen `trl` to `>=0.20,<0.25`. Newest unsloth picks the top of that range; older unslothes pick lower.
- Keep `transformers>=5.1.0,<6`. trl 0.24 was originally written for transformers v4, but the empirical pair (transformers 5.x + trl 0.24 + unsloth 2026.5.x) is what unsloth ships against; **S3 smoke is the acceptance gate** for whether `SFTTrainer` actually runs end-to-end on this stack. If it doesn't, the next amendment will either (a) drop transformers to `<5` (which contradicts unsloth's "transformers v5 required" doc for Qwen 3.5 — would force a base-model rethink) or (b) wait for a newer unsloth release.
- Add `[tool.uv].conflicts` between extras `colab` and `dev` so resolution remains separable and CI doesn't have to satisfy the GPU stack.

The lesson: pin from *the resolver's perspective* (what installs cleanly together right now), not from advertised compat-matrix prose. Amendments #1 and #2 cost ~15 min total — cheap because we caught both at first sync, before any code touched these libraries.
