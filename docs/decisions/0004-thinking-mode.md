# ADR 0004 — Thinking-mode strategy for the LoRA fine-tune

- **Status:** Accepted
- **Date:** 2026-05-10
- **Decision driver:** Brief §11 q4 + Plan Phase 2.2. "Train three small adapters with reasoning preserved / stripped / 75-25 mixed, compare on BFCL holdout, pick the strategy that maximises BFCL."

## Context

Qwen 3.5 ships with reasoning enabled by default — the chat template emits a `<think>...</think>` placeholder before each assistant turn, and the model fills it with a chain-of-thought before producing the actual answer. For tool-calling, this is a double-edged sword: reasoning helps the model pick the right function but eats `max_new_tokens` budget that would otherwise carry the call itself.

Three training-time strategies were proposed by the brief, mirroring Unsloth's published recipe:

- **(a) preserve.** Render every training row with `enable_thinking=True`. Reasoning content survives if the data has any.
- **(b) strip.** Render with `enable_thinking=False` *and* delete `<think>...</think>` blocks from message content (the template alone can't rewrite data already in the rows). The model is trained to skip reasoning entirely.
- **(c) mix-75-25.** Per-sample, deterministic 75/25 split between (a) and (b), seeded by `(seed, sample_index)` so re-runs are reproducible.

Phase 2.2 ran the comparison on the dataset winner from Phase 2.1 (xLAM — see ADR 0003).

## Empirical setup

- Same 1k xLAM samples per arm, 1 epoch, identical hyperparameters: LoRA `r=16 / alpha=32`, `lr=2e-4`, `batch_size=8`, `grad_accum=2` (effective 16), `max_seq_len=1024`, `seed=42`. Only `thinking_mode` differs across arms. Configs at [configs/ablation_thinking_preserve.yaml](../../configs/ablation_thinking_preserve.yaml), [configs/ablation_thinking_strip.yaml](../../configs/ablation_thinking_strip.yaml), [configs/ablation_thinking_mix.yaml](../../configs/ablation_thinking_mix.yaml).
- Eval: same 50-item BFCL V3 simple slice as Phase 2.1, scored by `parse_tool_call` + `score_prediction`. Implementation at [src/tooltuned_qwen/eval/bfcl_holdout.py](../../src/tooltuned_qwen/eval/bfcl_holdout.py).
- Thinking-mode plumbing at [src/tooltuned_qwen/data/format.py](../../src/tooltuned_qwen/data/format.py): `enable_thinking_for(mode, index, seed)` resolves the per-sample flag, `_strip_thinking` rewrites message content, `format_sample(..., enable_thinking=...)` forwards to the chat template.

## Empirical results

| Thinking mode | BFCL holdout accuracy | Notes |
|---|---|---|
| preserve | **46/50 = 0.920** | Default Qwen behaviour at training and at inference. |
| strip | **46/50 = 0.920** | `<think>` placeholder suppressed, plus inline `<think>` removal. |
| mix-75-25 | **46/50 = 0.920** | 75% preserve / 25% strip per sample, seeded. |

All three modes scored **identically** — same number of correct predictions, almost certainly on the same items (greedy decoding + identical seed + nearly identical training data → near-identical model behaviour).

## Why all three converged

The result looks suspicious at first glance, but it's consistent with the data:

1. **xLAM has no `<think>...</think>` content.** Each row is a clean `(query, tools, answers)` triple. There's no reasoning chain to preserve or strip.
2. **The chat template's `<think>` placeholder is a small wrapper, not training signal.** With `enable_thinking=True` the template emits `<think></think>` between user and assistant turns; with `enable_thinking=False` it doesn't. Either way, the model is trained on the same `(query → tool_call)` mapping. The presence/absence of an empty thinking placeholder is a few extra tokens of context, not a semantic shift.
3. **Mix-75-25 is therefore a 75/25 split between two near-identical training distributions.** Same outcome.

This is a real, useful empirical finding: **the thinking-mode setting is a no-op for clean tool-calling datasets like xLAM at this training scale.** If we'd picked Hermes (which carries reasoning prose in some assistant turns), preserve and strip would have produced measurably different models.

## Decision

**Use `thinking_mode: preserve` in `configs/default.yaml`** — the script's selection from the comparison and the default Qwen 3.5 runtime mode.

Tie-breakers, given accuracy is identical:

1. **Matches base-model defaults.** Users who load the adapter and call `apply_chat_template` without arguments get the expected behaviour. No surprises.
2. **No data rewriting at training time.** `_strip_thinking` is a regex pass over message content; not a problem at 1k samples but unnecessary work for the Phase 3 main run.
3. **Inference flexibility.** A model trained with reasoning preserved can be invoked in either mode at inference (`enable_thinking=False` to suppress at runtime). A model trained with reasoning stripped is comfortable in only one mode.

## Inference-time implication for BFCL

The Phase 2.2 eval used `max_new_tokens=768` precisely because the model reasons before emitting `<tool_call>` — at the original 256 budget, completions were getting truncated mid-thought (head: "I can use the `calculate_triangle_area` function for this. Looking at the function parameters:" — and that was the *first 200 chars* of an unfinished completion). 768 leaves ~150 tokens of reasoning headroom plus the call itself.

For Phase 4's full BFCL V4 run we should keep `max_new_tokens` ≥ 768 and consider letting it grow if multi-step categories show truncation. Phase 4's `bfcl` CLI handles this internally; we just need to make sure our config matches.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| `strip` as default | Same accuracy on xLAM, but the model becomes inflexible at inference (can't easily turn reasoning back on). No measured upside. |
| `mix-75-25` as default | Same accuracy, more code paths exercised, no upside on this dataset. Worth revisiting if Phase 3 changes datasets to one with reasoning content (we won't — see ADR 0003). |
| Train at `enable_thinking=False` *and* increase `max_new_tokens` to the prior 256 cap | Tempting because it's "cheaper at inference," but the eval-time diagnosis showed reasoning-then-tool-call is the model's natural mode. Forcing it into a non-reasoning lane on data it was trained with reasoning-on would fight the base model. Defer to a future ablation if we ever care about inference latency. |
| Skip Phase 2.2 entirely and pick `preserve` by default | Tempting in hindsight, but the empirical result *is* the finding: "thinking-mode doesn't matter for clean tool-call data" is a non-obvious claim that's worth documenting in the ADR for future contributors. |

## Consequences

- `configs/default.yaml` keeps `thinking_mode: preserve` (already the provisional default; Phase 2.2 confirmed it empirically).
- The thinking-mode plumbing (`enable_thinking_for`, `_strip_thinking`, `format_sample(..., enable_thinking=...)`) stays in the codebase — it's correct, tested, and load-bearing if Phase 3 ever picks a thinking-aware dataset.
- BFCL evaluation must allow ≥768 `max_new_tokens` (current default in `run_bfcl_holdout`); 256 was empirically too small for reasoning + call to fit.
- The mix-75-25 mode is verified to produce the documented split (test: `test_enable_thinking_for_mix_75_25_split_is_deterministic`). Phase 3 may revisit if we expand datasets.

## Caveats

1. The 50-item BFCL slice is small. Three arms producing identically-correct predictions on the same items is consistent with the explanation above, but a larger slice could surface 1-2pp wobble. Phase 4's full BFCL V4 will be the precise measurement.
2. **The result generalises to clean tool-call data, not to all training data.** A future contributor swapping in a Hermes-trained or reasoning-trace-augmented dataset should re-run Phase 2.2 — the no-op finding does not transfer.
3. Inference-time `max_new_tokens` matters more than training-time thinking mode for this model; the diagnostic budget change (256 → 768) had a far larger effect on observed accuracy than any thinking-mode toggle.

## References

- Brief: [03_tooltuned_qwen_brief.md](../../03_tooltuned_qwen_brief.md) — §11 q4
- ADR 0003 — dataset choice (xLAM)
- Thinking-mode plumbing — [src/tooltuned_qwen/data/format.py](../../src/tooltuned_qwen/data/format.py) (`enable_thinking_for`, `_strip_thinking`, `format_sample`)
- Tests — [tests/test_chat_template.py](../../tests/test_chat_template.py) (`test_format_sample_strip_passes_enable_thinking_false_and_strips_inline_think`, `test_enable_thinking_for_mix_75_25_split_is_deterministic`)
- Result JSONs — `results/ablations/ablation-thinking-{preserve,strip,mix}.json` (saved on Colab; regenerable)
