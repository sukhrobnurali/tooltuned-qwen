# ADR 0006 — Eval debugging: tuned LoRA over-calls on irrelevance categories

- **Status:** Accepted
- **Date:** 2026-05-14
- **Decision driver:** Phase 4 BFCL gate (brief §4.4) failed twice (S9, S10) with delta ≈ -8pp. This ADR locks the diagnosis and the remediation path.

## Context

S9 (2026-05-13) ran the first Phase 4 gate check at mixed token caps (cached base at 768, tuned at 768): tuned 363/458 = 79.3% vs base 87.3%, delta **-8.08pp**, gate FAILED. Per-category for the tuned arm was lost when the Colab runtime disconnected before download — only the printed overall survived.

S10 (this ADR) re-ran the tuned arm with two fixes baked into the notebook (commit `23bbb72`): a `max_new_tokens=512` cap that recovered ~30% throughput, and an auto-save of `tuned_results` to a private HF smoke repo as a network backup against disconnect. The S10 plan correctly insisted on re-running the base arm at the matching 512-token cap rather than short-circuiting from the S8 cache — the two caps would have biased the delta against the tuned arm because Qwen 3.5 reasons before emitting `<tool_call>` and shorter caps mean more truncated thoughts → parser misses.

S10 numbers (matching 512-token caps for both arms):

| Arm | Correct | Total | Overall | Source |
| --- | --- | --- | --- | --- |
| Base (Qwen 3.5 4B, S10 fresh) | 397 | 458 | 86.68% | Printed by cell 3 (per-category lost on disconnect) |
| Tuned (LoRA, S10 fresh) | 362 | 458 | 79.04% | [results/bfcl_intree/tuned/results.json](../../results/bfcl_intree/tuned/results.json) (recovered from HF smoke-repo backup) |
| **Delta** | -35 | — | **-7.64pp** | Gate FAILED (≥ +3.0pp required) |

The S10 base re-run at 512 tokens (86.68%) was 0.6pp lower than the S8 cached base at 768 tokens (87.34%, 400/458). That 0.6pp drop maps to roughly three items lost to the tighter cap — below sampling noise on a 50-item category. So the per-category contrast against the S8 cached base remains load-bearing for the diagnosis below, with the caveat that ±1 item per category from the cap difference is in the noise floor.

## Evidence — per-category breakdown

Tuned at 512 tokens (S10 fresh) vs base at 768 tokens (S8 cached, see [results/bfcl_intree/base/results.json](../../results/bfcl_intree/base/results.json)):

| Category | Base | Tuned | Delta | Items | Scoring mode |
| --- | --- | --- | --- | --- | --- |
| `simple` | 90.0% (45) | 88.0% (44) | -2.0pp / -1 | 50 | AST match |
| `multiple` | 92.0% (46) | 90.0% (45) | -2.0pp / -1 | 50 | AST match |
| `parallel` | 88.0% (44) | 88.0% (44) | 0.0pp / 0 | 50 | AST match (lenient) |
| `parallel_multiple` | 98.0% (49) | 92.0% (46) | -6.0pp / -3 | 50 | AST match (lenient) |
| `live_simple` | 80.0% (40) | 74.0% (37) | -6.0pp / -3 | 50 | AST match |
| `live_multiple` | 78.0% (39) | 78.0% (39) | 0.0pp / 0 | 50 | AST match |
| `live_parallel` | 81.2% (13) | 68.8% (11) | -12.5pp / -2 | 16 | AST match (lenient) |
| `live_parallel_multiple` | 95.8% (23) | 91.7% (22) | -4.2pp / -1 | 24 | AST match (lenient) |
| **`irrelevance`** | **80.0% (40)** | **42.0% (21)** | **-38.0pp / -19** | **50** | **must NOT call** |
| `live_relevance` | 66.7% (12) | 77.8% (14) | +11.1pp / +2 | 18 | must call |
| **`live_irrelevance`** | **98.0% (49)** | **78.0% (39)** | **-20.0pp / -10** | **50** | **must NOT call** |

Totals by scoring mode:

| Mode | Categories | Items | Tuned delta |
| --- | --- | --- | --- |
| AST match (8 cats) | 8 | 290 | -11 items / -3.8pp on AST-only |
| must call (`live_relevance`) | 1 | 18 | +2 items / +11.1pp |
| must NOT call (`irrelevance` + `live_irrelevance`) | 2 | 100 | **-29 items / -29.0pp** |
| **All** | **11** | **458** | **-38 items / -8.30pp** |

## Diagnosis (locked)

**76% of the regression (29 of 38 lost items) comes from the two `must_not_call` categories**: `irrelevance` (-19 items) and `live_irrelevance` (-10 items). The tuned model over-calls when tools are unrelated to the user's query.

Root cause is data composition. `Salesforce/xlam-function-calling-60k` is positive-only by design — every row pairs a user query with a function call. There is no `gold = "no call"` row anywhere in the 60k. The LoRA therefore learns a Bayesian prior of "the user wants a tool call" that is far stronger than the pretrained base's prior. When evaluated on prompts where the available tools genuinely don't help (BFCL's `irrelevance` set), the fine-tuned model confidently calls the closest-looking tool — exactly the failure mode `must_not_call` tests for.

Three supporting facts:

1. **`live_relevance` (must call) IMPROVED +11.1pp** (+2 items, 12 → 14). The LoRA *did* learn the positive signal — it correctly emits a call more often when tools are relevant. The model isn't broken; it's biased toward calling.
2. **AST categories regressed only -11 items across 290 items / 8 categories** (-3.8pp on AST-only), with no single AST category falling by more than -12.5pp on a 16-item slice. This is consistent with mild format drift / minor overfitting to xLAM's specific function signatures, not a structural failure. AST training succeeded; it's the *over-eagerness* that costs us.
3. **The base's 80% / 98% on `irrelevance` / `live_irrelevance`** shows pretrained Qwen 3.5 4B already had a reasonable "decline" instinct. The LoRA destroyed that instinct by training exclusively on positives.

## Decision

**Phase 3.5: re-train with a ~25% irrelevance mixin synthesised from xLAM.**

Mixin source: synthesise negatives from xLAM itself — for each candidate xLAM row, swap its `tools` field with a tools array sampled from a different xLAM row, mark the expected response as "no call" (a short prose decline rather than any function emission). This keeps the negative examples in the same domain as the positives, requires no new gated dataset access, and is contamination-free with respect to BFCL test items.

Proposed mix:

- 10,000 positive xLAM rows (unchanged from Phase 3).
- 2,500 synthesised irrelevance rows (10k × 0.25).
- 12,500 total, 1 epoch on A100, batch_size=16 / grad_accum=1, ~30-35 min training.

The 25% mixin number is loose — somewhere in 20-30% is the design space. Too little and the negative signal is washed out by 4× more positives; too much and AST throughput drops because the model is spending capacity learning the negative case (which is a single fixed response shape). 25% is enough to be felt without dominating.

### Realistic Phase 3.5 outcome

Honest gate math — base is 86.68% at the S10 cap, gate target is 89.68% (+3pp). Tuned needs to gain 38 items from its current 362/458 just to reach base, plus 14 more to reach the gate (52 items total = +11.4pp).

Plausible Phase 3.5 lifts:

| Category | Current | Plausible post-3.5 | Item gain |
| --- | --- | --- | --- |
| `irrelevance` | 21/50 (42%) | 35-40/50 (70-80%) | +14 to +19 |
| `live_irrelevance` | 39/50 (78%) | 45-48/50 (90-96%) | +6 to +9 |
| AST cats (8) | 269/290 (-11 vs base) | recovery in ±5 items either way | -5 to +5 |
| `live_relevance` | 14/18 | maintain | 0 |

Best case: +33 items → 395/458 = 86.2% overall = -0.5pp delta. **Still under the gate.**
Realistic case: +20 items → 382/458 = 83.4% = -3.3pp delta.
Worst case: +5 items if AST regresses → 367/458 = 80.1% = -6.6pp.

**Phase 3.5 alone is unlikely to clear the +3pp gate against this particular base, because the pretrained Qwen 3.5 4B is already a strong tool-caller in our in-tree evaluator (86.7%).** This is not surprising in hindsight: ADR 0001 anchored the gate against the brief's BFCL V4 baseline of 50.3% (a much weaker number), but our in-tree V3 evaluator with lenient parallel scoring inflates absolute numbers for both arms equally (see ADR 0005 §Limitations). The gate threshold should arguably have been calibrated against the in-tree base measurement once it was known (S8) — that's a process miss to acknowledge.

We're proceeding with Phase 3.5 anyway because:

- The regression on `must_not_call` is a genuine quality issue and worth fixing on its own merit, gate or not.
- Recovering 20+ items on `irrelevance` / `live_irrelevance` makes the fine-tune *better than base on its training task* (tool-calling when tools are relevant) without being *worse than base on irrelevance*. That's a defensible portfolio outcome even if the headline delta stays mildly negative.
- The ~30 unit Colab cost is well within the remaining budget (~50-80 of 130 remaining after S10).

If Phase 3.5 still doesn't clear the gate, ADR 0007 will record the v1.0 ship decision: most likely "ship with honest disclosure, recompute the gate threshold against the in-tree base, document the V3-vs-V4 baseline divergence". Brief §4.4 is the gate-of-record; deviating from it is brief-amendment territory not ADR-decision territory.

## Alternatives considered

| Alternative | Why rejected (or not chosen) |
| --- | --- |
| Train on BFCL V3 `irrelevance` items themselves | Test-set contamination. Hard rule. |
| ToolBench negatives | Possibly viable but requires new dataset access + schema mapping. Synth-from-xLAM is faster and contamination-free. Re-consider for Phase 3.6 if 3.5 underdelivers. |
| Bigger xLAM pool (e.g. 30k positives) | Doesn't address the over-calling root cause — the model would learn the positive signal harder, making irrelevance worse. |
| Stricter parallel-category scoring | Would tighten the AST numbers symmetrically (both arms lose lenient credit). Doesn't fix the must_not_call regression. Tracked in ADR 0005 §Limitations as a future tightening. |
| Lower the LoRA rank | Phase 3 used r=16. Lowering it would reduce model capacity to memorise the "always call" pattern but also hurt AST learning. Counter-productive for this specific failure mode. |
| Ship as-is | Brief §4.4 mandates the gate. Shipping a sub-gate v1.0 violates the brief without a written amendment. |
| Skip Phase 3.5 and rewrite the gate | Premature — Phase 3.5 is cheap and tests the diagnosis. If it lifts the must_not_call numbers as predicted, we have evidence for the diagnosis even if the gate doesn't clear. |

## Consequences

- **New module:** `src/tooltuned_qwen/data/synth_irrelevance.py` — generator function `build_irrelevance_rows(positive_rows, n_target, seed)` returning xLAM-shaped rows with swapped tools + a "no call" response. Tests: schema match, swap is from a different row, response is non-emitting, deterministic under seed.
- **Training config:** `configs/phase35.yaml` extends `default.yaml` with the synth mixin enabled and a slightly higher epoch count to account for the larger pool. Schema change: optional `irrelevance_pct: float | None` field on `TrainingConfig`.
- **Colab notebook:** `notebooks/colab_phase35.ipynb` mirrors `colab_main.ipynb` with the synth-step before training. Push to a NEW HF repo `tooltuned-qwen-3.5-4b-v2` to keep v1 immutable as a comparison reference.
- **Phase 4 re-run:** `notebooks/colab_bfcl_intree.ipynb` re-points to the v2 repo for the tuned cell. Base re-uses the S10 cached numbers (same 512-token cap, no re-run needed).
- **Budget:** Phase 3.5 training (~15-20 units) + Phase 4 re-run for the tuned arm only (~10-15 units) = ~25-35 units. Cumulative ~75-100 of 130.
- **Cumulative tests target:** +6-8 (synth generator + config schema + notebook smoke).
- **Model card:** stays in TBD on the live HF main repo until either (a) Phase 3.5 clears the gate, or (b) ADR 0007 codifies a sub-gate ship path.

## References

- Brief: [03_tooltuned_qwen_brief.md](../../03_tooltuned_qwen_brief.md) — §4.4 BFCL gate
- ADR 0001 — base model (V4 baseline 50.3, gate +3pp)
- ADR 0003 — dataset choice (xLAM, positive-only; the data property that drove this regression)
- ADR 0005 — in-tree eval methodology + scoring limitations
- Tuned per-category results: [results/bfcl_intree/tuned/results.json](../../results/bfcl_intree/tuned/results.json)
- Base per-category reference (S8 cache at 768 tokens): [results/bfcl_intree/base/results.json](../../results/bfcl_intree/base/results.json)
- Project memory entry "S10 — re-eval tuned arm for per-category diagnostics" — the run that produced these numbers.
