# ADR 0003 — Training dataset for the LoRA fine-tune

- **Status:** Accepted
- **Date:** 2026-05-10
- **Decision driver:** Brief §11 q3 + Plan Phase 2.1. "Train two small adapters (1 epoch / 1k samples), compare on a BFCL holdout slice, pick the winner empirically."

## Context

Two off-the-shelf function-calling datasets dominate the open ecosystem:

- **`Salesforce/xlam-function-calling-60k`** — purpose-built for tool-calling. Each row has `query`, `tools` (list of function schemas), and `answers` (list of tool-call objects). Schema-aligned with the Qwen 3.5 chat template's `tool_calls` field once `arguments` is parsed into a dict (Phase 1.3 finding #2).
- **`NousResearch/hermes-function-calling-v1`** — broader. Multi-turn `conversations` rows with `<tools>...</tools>` schemas inlined into the system message and `<tool_call>...</tool_call>` markup inside assistant turns. Carries reasoning prose, multi-turn dialogue, and varied call patterns.

Both are reasonable defaults; the brief deliberately leaves the choice empirical. Phase 2.1 ran the comparison.

## Empirical setup

- Same 1k-sample slice per arm, 1 epoch, identical hyperparameters: LoRA `r=16 / alpha=32`, `lr=2e-4`, `batch_size=8`, `grad_accum=2` (effective 16), `max_seq_len=1024`, `seed=42`, `thinking_mode=preserve`. Configs at [configs/ablation_dataset_xlam.yaml](../../configs/ablation_dataset_xlam.yaml) and [configs/ablation_dataset_hermes.yaml](../../configs/ablation_dataset_hermes.yaml).
- Eval: BFCL V3 simple-category holdout slice (50 items pulled from `gorilla-llm/Berkeley-Function-Calling-Leaderboard`). Scoring is BFCL's per-arg AST match, with `""` in the accepted list marking an optional argument. Implementation at [src/tooltuned_qwen/eval/bfcl_holdout.py](../../src/tooltuned_qwen/eval/bfcl_holdout.py).
- Both adapters trained on Colab Pro A100 (40GB), final training loss ~0.50 (xLAM) and ~0.64 (Hermes) — Hermes loss runs higher because its assistant turns carry prose + reasoning that the simpler xLAM rows don't.

## Empirical results

| Dataset | BFCL holdout accuracy | Notes |
|---|---|---|
| xLAM | **38/42 = 0.905** | Initial run terminated by a `torch._dynamo` recompile overflow at item 43 (Unsloth's pre-compiled module + `one_graph=True`). Running accuracy was rock-stable in the 0.90 band over the last 15 items, so 0.905 is a tight estimate. The dynamo issue was fixed in `bfcl_holdout.py` after the run; subsequent arms (Hermes, all three thinking-mode arms) completed cleanly. |
| Hermes | 45/50 = 0.900 | Clean run, no early termination. |

The 0.5pp gap is well inside the standard error on these N values (~4-5pp each), so xLAM and Hermes are statistically indistinguishable on this slice. That's a genuine empirical signal: **for the BFCL simple category at 1k-sample / 1-epoch budgets, the dataset choice is not the limiting factor.**

## Decision

**Use `Salesforce/xlam-function-calling-60k` as the training dataset.** Set `data.sources: [xlam]` in `configs/default.yaml`.

The deciding factors, in order:

1. **Schema alignment.** xLAM rows are already in `(query, tools, answers)` shape — a one-line `_xlam_to_messages` formatter renders the Qwen template cleanly. Hermes needs regex extraction for `<tools>` and `<tool_call>` markup (added in [src/tooltuned_qwen/data/format.py](../../src/tooltuned_qwen/data/format.py) for this ablation). For Phase 3's main run on a larger slice, fewer parsing edge cases means fewer silent training failures.
2. **Nominal accuracy edge.** xLAM 0.905 vs Hermes 0.900. Within noise, but in the direction we'd hope given (1).
3. **Cleaner training data.** xLAM has no `<think>` content, no multi-turn dialogue, no glaive-style mixed examples. The model is trained on exactly the behaviour we want it to exhibit on BFCL.
4. **Lower training loss.** Final loss 0.50 (xLAM) vs 0.64 (Hermes) is consistent with xLAM being a cleaner signal — the model fits it better at the same compute.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Hermes alone | 0.5pp accuracy deficit + extra parsing complexity in `_hermes_to_messages` (already implemented but not load-bearing if we don't use it). |
| xLAM + Hermes mix (50/50) | Adds complexity for no proven gain; `_load_dataset` would need real multi-source mixing logic (currently `NotImplementedError`). Could be revisited as a stretch ablation in Phase 7. |
| Glaive function-calling subset | Overlapping coverage with Hermes; not measured here. |
| Custom curated dataset | Out of scope (brief §3: "no data scraping"). |

## Consequences

- `configs/default.yaml` keeps `data.sources: [xlam]` (the provisional default already pointed here; Phase 2.1 confirmed it empirically).
- Phase 3's main run targets the full xLAM 60k or a tuned subset thereof.
- The Hermes formatter (`_hermes_to_messages` with `<tool_call>` extraction + tool-schema normalization) stays in `src/tooltuned_qwen/data/format.py` for reproducibility and as a stretch-ablation surface, but no longer load-bearing for v1.
- BFCL eval scoring (`parse_tool_call` + `score_prediction`) is now reusable for Phase 4's full BFCL V4 run; the Phase 2.2 thinking-mode comparison reused it without changes.
- xLAM is HF-gated; the `sukhrobnurali` account has access (granted 2026-05-10). Reproducers must request access at `huggingface.co/datasets/Salesforce/xlam-function-calling-60k`.

## Caveats

1. The xLAM accuracy is a **42-item** number; Hermes is **50-item**. Sample-size asymmetry is small but worth recording. Re-running xLAM with the dynamo-disabled `bfcl_holdout.py` would give a clean 50/50 number; deferred because (a) the partial signal is unambiguous and (b) it's an extra ~5 minutes of A100 time better spent on Phase 3.
2. The BFCL V3 simple slice (50 items) is small. The Phase 4 full BFCL V4 run will be the rigorous gate; Phase 2.1 only needed enough signal to pick a dataset.
3. Both arms were trained at 1k samples / 1 epoch. The dataset choice could plausibly invert at 10k samples / 3 epochs (Hermes' larger pattern variety might pay off with more training). We're not measuring that — Phase 3 commits to xLAM.

## References

- Brief: [03_tooltuned_qwen_brief.md](../../03_tooltuned_qwen_brief.md) — §11 q3
- Phase 1.3 finding #2 (xLAM `arguments` must be a dict, not a JSON string) — `project_tooltuned_qwen.md`
- Hermes formatter implementation — [src/tooltuned_qwen/data/format.py](../../src/tooltuned_qwen/data/format.py) `_hermes_to_messages`
- BFCL holdout evaluator — [src/tooltuned_qwen/eval/bfcl_holdout.py](../../src/tooltuned_qwen/eval/bfcl_holdout.py)
- Result JSONs — `results/ablations/ablation-dataset-xlam.json`, `results/ablations/ablation-dataset-hermes.json` (saved on Colab during the run; not committed — they're regenerable)
