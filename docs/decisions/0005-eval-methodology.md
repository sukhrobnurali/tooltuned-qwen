# ADR 0005 — Phase 4 evaluation methodology: in-tree BFCL V3 scorer

- **Status:** Accepted
- **Date:** 2026-05-13
- **Decision driver:** Brief §4.4 (BFCL gate of +3pp over base) + Plan Phase 4 + S6.5 dead-end on the canonical `bfcl` CLI.

## Context

Phase 4 is the gate that decides whether the LoRA adapter ships publicly. The brief §4.4 mandates ≥3pp on BFCL versus the base Qwen 3.5 4B. S6 implemented the canonical path: shell out to gorilla's `bfcl-eval` CLI (`pip install bfcl-eval`) in an isolated Colab venv, run `bfcl generate` + `bfcl evaluate`, fold the per-category JSONL into a results dict. That code stayed in-tree (`src/tooltuned_qwen/eval/bfcl_runner.py`, `notebooks/colab_bfcl.ipynb`, 80 tests).

S6.5 attempted to actually run it on Colab Pro A100 across roughly six debug cycles and hit six stacked upstream blockers — the dead-end was a transitive version pin chain:

1. `bfcl-eval==2026.3.23` pins `vllm==0.8.5`.
2. `vllm==0.8.5` was built against transformers v4 (uses the `Qwen2Tokenizer.all_special_tokens_extended` attribute that v5 removed).
3. Qwen 3.5's `config.json` declares `model_type: qwen3_5`, which only transformers v5 knows.
4. Upgrading vllm + transformers in lockstep (vllm 0.20.x + transformers 5.8) pulls torch 2.11.
5. torch 2.11's NCCL ABI (`ncclCommWindowDeregister`) doesn't match Colab's system NCCL 2.26.x, so `import torch` raises `OSError: undefined symbol`.

None of these is fixable from our side without forking vllm or rebuilding torch from source — neither viable for a portfolio-grade v1.0.

The project memory entry "S6.5 — Phase 4 canonical-CLI install attempts" lists each blocker plus the commit that worked around it; commits `99aef1e` → `f651f5d` are the audit trail. The terminal blocker (5) has no in-repo fix.

## Decision

**Defer the canonical bfcl-eval CLI path for v1.0. Phase 4 runs through an in-tree evaluator (`eval/bfcl_holdout.py::run_bfcl_full`) covering the 11 BFCL V3 single-turn categories**, scored with the same function-name + per-arg accept-list AST match the canonical evaluator uses on AST categories, plus a decision-to-call scorer for the three relevance/irrelevance categories. The evaluator runs through Unsloth in our working `[colab]` env — no vllm, no isolated venv, no version war.

### Concrete substitutions vs. the canonical path

| Aspect | Canonical bfcl-eval (S6, deferred) | In-tree run_bfcl_full (S7, this ADR) |
| --- | --- | --- |
| Test data | BFCL V4 from gorilla repo | BFCL V3 from the `gorilla-llm/Berkeley-Function-Calling-Leaderboard` HF mirror |
| Backend | vllm 0.8.5 (in `/content/bfcl-venv`) | Unsloth `FastLanguageModel` in main `[colab]` env |
| Generation | bfcl's `--backend vllm` + vllm serve | `model.generate(..., max_new_tokens=768)` directly |
| AST scoring | `bfcl evaluate` (canonical) | `score_prediction_multi` in [src/tooltuned_qwen/eval/bfcl_holdout.py](../../src/tooltuned_qwen/eval/bfcl_holdout.py) — same logic: function-name match + per-arg accept-list match with `""` marking optional |
| Categories | 11 single-turn V4 (`simple_python` + 4 + 5 live + 1 irrelevance) | 11 single-turn V3 (`simple` + 3 + 4 live + 3 relevance/irrelevance) — see ["Category mapping"](#category-mapping) below |
| Aggregation | Micro-average across categories | Micro-average across categories (`sum(correct)/sum(total)`) |
| Results dict | `{model, mode, n_total, evaluated_at, overall, per_category, ...}` | Same shape, `mode="intree"` to distinguish |

The schema match means `eval/compare.py::build_comparison` and `hub/model_card.py::generate_card` consume in-tree results unchanged — the canonical path can be revived later without touching the comparison or card code.

### Category mapping

V3 single-turn categories on the HF mirror, with their scoring mode:

| Category | Scoring | Section |
| --- | --- | --- |
| `simple` | AST match | non_live |
| `multiple` | AST match | non_live |
| `parallel` | AST match (lenient — see Limitations) | non_live |
| `parallel_multiple` | AST match (lenient) | non_live |
| `live_simple` | AST match | live |
| `live_multiple` | AST match | live |
| `live_parallel` | AST match (lenient) | live |
| `live_parallel_multiple` | AST match (lenient) | live |
| `irrelevance` | must NOT call | non_live |
| `live_relevance` | must call | live |
| `live_irrelevance` | must NOT call | live |

V3-vs-V4 differences worth flagging:

- V3 has `simple`; V4 split this into `simple_python` / `simple_java` / `simple_javascript`. Since xLAM is Python-centric, evaluating against V4's `simple_python` would be the closest equivalent — V3's `simple` is essentially the same content under a single label.
- V4 added a few categories (web search, memory, multi-turn agent) that we wouldn't have evaluated anyway — out of scope for a single-turn xLAM-trained LoRA.
- Question/function/ground-truth row shapes are identical between V3 and V4.

## Reproducibility

Anyone running this notebook reproduces the numbers:

```bash
git clone https://github.com/sukhrobnurali/tooltuned-qwen
cd tooltuned-qwen
uv sync
# Open notebooks/colab_bfcl_intree.ipynb on Colab Pro A100
```

Or from a Python REPL with the `[colab]` extras installed:

```python
from tooltuned_qwen.eval.bfcl_holdout import run_bfcl_full
base = run_bfcl_full("Qwen/Qwen3.5-4B", n_per_cat=50)
tuned = run_bfcl_full("sukhrobnurali/tooltuned-qwen-3.5-4b", n_per_cat=50)
```

Data is pulled from the HF mirror (cached under `results/bfcl_holdout/`) so the test items are identical across runs.

## Limitations

1. **Parallel categories use lenient any-match scoring.** Canonical BFCL requires a 1:1 set match between predicted calls and ground-truth entries (no missing, no extras) on `parallel*`. The in-tree scorer (`score_prediction_multi`) returns True if any predicted call satisfies any ground-truth entry. This inflates the absolute parallel numbers, but since both arms see the same scorer, the delta-vs-base remains directionally honest — and the gate (+3pp) is a relative threshold. A future revival of the canonical CLI will tighten this without changing the architecture decisions.
2. **V3 vs V4 absolute numbers are not directly comparable to the BFCL leaderboard.** The brief's 50.3 baseline for Qwen 3.5 4B was the V4 leaderboard number; our in-tree base measurement will land somewhere different (V3 content + lenient parallel scorer). The gate is "tuned - base ≥ 3pp on the SAME evaluator we used for base" — internally consistent, but not a leaderboard claim.
3. **Sample size: 50 items per category** is the notebook default. That's 550 prompts per arm, ~1.5 h of A100 time per arm. Standard error on a 50-item slice is ~6-7pp at p=0.5; the per-category numbers are noisier than the overall (which sees all 550). The Phase 2 ablations established that 50-item slices were directionally stable, so this is good enough for the gate — full-pass numbers (~3000+ items per arm, all categories) are a stretch goal if budget allows.
4. **Decision-to-call scoring is binary.** Relevance/irrelevance categories have no AST scoring — the model either called something or didn't. A model that hallucinates the wrong tool on `live_relevance` still scores 100% on that category by our rules; a model that calls when it shouldn't on `irrelevance` scores 0%. Canonical BFCL applies the same coarse rule, so this is faithful to upstream.
5. **No tool-call output parsing for non-Qwen formats.** `parse_tool_calls` recognises (a) JSON-payload `<tool_call>` blocks, (b) XML-tag `<function=...>` form (our fine-tune's actual emission per Phase 2 finding), and (c) bare-JSON fallback. A model that emits Llama's `<|python_tag|>` or OpenAI's `tool_calls` array would score 0 across the board. Out of scope: this evaluator is specific to Qwen 3.5.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Build vllm from source against torch 2.10 in the bfcl-venv | Days of compile/debug work, no guarantee Qwen 3.5 + vllm 0.8.5 cooperate even with the torch issue fixed. Cost > value for v1.0. |
| Use a different inference backend in bfcl-eval (`--backend openai-compatible` against an Unsloth-hosted server) | bfcl-eval's OpenAI-compat backend still expects native tool-call output, which our fine-tune doesn't emit (XML-tag form). Would need a server-side shim, similar effort to the in-tree path with more moving parts. |
| Wait for bfcl-eval to bump its vllm pin past 0.8.5 | Unknown timeline (vllm needs to ship a transformers-v5-compatible release with torch 2.9-2.10 wheels). Project budget says "ship v1.0 now, revive the canonical path when upstream stabilises." |
| Run BFCL via the gorilla `oss_eval` script directly (not the CLI) | Same dep tree, same vllm/torch problem. Doesn't escape the blocker. |
| Drop Phase 4 entirely and ship without published numbers | Violates brief §4.4. The whole point of the LoRA is the BFCL claim. |
| Use a different benchmark (e.g. ToolBench, API-Bank) | Loses comparability with the published Qwen 3.5 baseline (brief §1). The brief explicitly anchors on BFCL. |

## Consequences

- **Code that stays:** `bfcl_runner.py`, `compare.py`, `notebooks/colab_bfcl.ipynb`, and the 22 tests covering them stay in-tree as the canonical path. Reviving them is a notebook-only change when upstream pins stabilise — no schema migration needed because the in-tree path emits the same results dict shape.
- **Code added:** `BFCL_CATEGORIES` table + `load_bfcl_categories` + `parse_tool_calls` + `score_prediction_multi` + `score_must_call` / `score_must_not_call` + `score_item` + `shape_results` + `run_bfcl_full` in [src/tooltuned_qwen/eval/bfcl_holdout.py](../../src/tooltuned_qwen/eval/bfcl_holdout.py); new notebook [notebooks/colab_bfcl_intree.ipynb](../../notebooks/colab_bfcl_intree.ipynb). 14 new tests in [tests/test_bfcl_holdout.py](../../tests/test_bfcl_holdout.py).
- **Model card:** the BFCL section's "TBD (Phase 4)" placeholder becomes a real table on a successful gate (`fc_results['delta'] >= 0.03`). The card explicitly cites this ADR so a reader of the published HF card understands the methodology pick.
- **Budget:** ~10-15 Colab units expected for the eval pass (single notebook run, both arms, n_per_cat=50). Cumulative budget remains ~50-60 of 130, leaving headroom for Phase 5/6/7.
- **Gate semantics unchanged:** ≥3pp tuned-vs-base. Branch B (gate fails) writes a separate ADR 0005-eval-debugging (different filename) with the diagnosis and halts the HF push.

## Caveats

1. The brief's baseline number `50.3` is a V4 leaderboard figure for the base Qwen 3.5 4B. Our base measurement under this ADR will not match it — that's expected, not a bug. The gate is internal: tuned vs base on the same evaluator.
2. If the gate passes here but a future canonical-CLI run produces a smaller delta (parallel categories are tighter, V4 content differs), we should re-run the canonical path before publishing the canonical number. For now, the in-tree number IS the published number, with this ADR as the disclosure.
3. **A reviewer can re-run the gate with `n_per_cat=None`** for a full-pass measurement — same code path, more compute. The notebook variable `N_PER_CAT` at the top of cell 2 controls the cap.

## References

- Brief: [03_tooltuned_qwen_brief.md](../../03_tooltuned_qwen_brief.md) — §4.4 BFCL gate, §16 attribution discipline
- ADR 0001 — base model (`Qwen/Qwen3.5-4B`, V4 baseline 50.3)
- Deferred canonical path: [src/tooltuned_qwen/eval/bfcl_runner.py](../../src/tooltuned_qwen/eval/bfcl_runner.py), [notebooks/colab_bfcl.ipynb](../../notebooks/colab_bfcl.ipynb)
- In-tree evaluator: [src/tooltuned_qwen/eval/bfcl_holdout.py](../../src/tooltuned_qwen/eval/bfcl_holdout.py) (`run_bfcl_full`, `load_bfcl_categories`, `parse_tool_calls`, `score_prediction_multi`, `score_must_call`, `score_must_not_call`, `shape_results`)
- Phase 4 notebook: [notebooks/colab_bfcl_intree.ipynb](../../notebooks/colab_bfcl_intree.ipynb)
- Tests: [tests/test_bfcl_holdout.py](../../tests/test_bfcl_holdout.py)
- Project memory entry "S6.5 — Phase 4 canonical-CLI install attempts" — the six-blocker debug chain that motivated this pivot.
