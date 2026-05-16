# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer](https://semver.org/).

## [Unreleased]

- Phase 3.5 deferred: irrelevance-mixin retrain to address the `must_not_call` regression diagnosed in [ADR 0006](docs/decisions/0006-eval-debugging.md). Spec lives in the ADR; not scheduled for v1.0.
- Canonical `bfcl-eval` CLI path on Colab (deferred in S6.5; see [ADR 0005](docs/decisions/0005-eval-methodology.md)). Code is retained in `src/tooltuned_qwen/eval/bfcl_runner.py` for later revival once the vllm / transformers / Qwen 3.5 / NCCL install matrix stabilises.
- Merged-weights export + GGUF quantisation (v1.1 candidate, separate scope from v1.0).

## [v1.0.0] -- 2026-05-14

Initial public release. Adapter is published below the project brief's +3pp BFCL gate; the disclosure is on the model card and in this README. See [ADR 0007](docs/decisions/0007-ship-v1.0-below-gate.md) for the decision rationale.

### Phase 0 -- scaffolding

- Repo init, uv-managed venv (Python 3.12), pinned versions in `pyproject.toml` + `uv.lock` ([ADR 0002](docs/decisions/0002-pinned-versions.md)).
- Tooling: `ruff` (lint), `pyright` (types), `pytest` (tests); wired into CI.
- Author identity locked to `Sukhrob Nurali <sukhrobnurali@gmail.com>`. Attribution discipline enforced per brief §16; the §16.6 four-grep audit returns zero hits across the full history.
- Base model: `Qwen/Qwen3.5-4B` ([ADR 0001](docs/decisions/0001-base-model.md)).

### Phase 1 -- inference + chat-template plumbing

- Loader, tokenizer, chat-template apply path (`src/tooltuned_qwen/inference/`).
- Tool-call parser: handles both Qwen's native `<tool_call>` blocks and XML-tagged `<function=...>` calls.

### Phase 2 -- dataset + ablations

- xLAM (`Salesforce/xlam-function-calling-60k`) vs Hermes head-to-head at 1 epoch / 1k samples. xLAM wins on schema alignment + training loss; ~0.5pp accuracy gap within standard error ([ADR 0003](docs/decisions/0003-dataset.md)).
- Thinking-mode strategy: `preserve` / `strip` / `mix-75-25` all converge in practice -- xLAM rows have no `<think>` content, so the three modes produce near-identical training samples. Locked on `preserve` to match Qwen 3.5's default ([ADR 0004](docs/decisions/0004-thinking-mode.md)).

### Phase 3 -- LoRA training

- bf16 LoRA, rank 16, alpha 32, target modules `q_proj`/`k_proj`/`v_proj`/`o_proj`/`gate_proj`/`up_proj`/`down_proj`. AdamW, learning rate 2e-4, warmup ratio 0.03, weight decay 0.01.
- 1 epoch on 10,000 xLAM rows. Single A100 (40 GB), gradient checkpointing on, ~30 min wall time.
- Adapter pushed to `huggingface.co/sukhrobnurali/tooltuned-qwen-3.5-4b` (public).
- Training curves published at `wandb.ai/sukhrob-production/tooltuned-qwen` (public, team workspace).

### Phase 4 -- BFCL evaluation

- Eval methodology pivoted from canonical `bfcl-eval` CLI to an in-tree V3 single-turn scorer ([ADR 0005](docs/decisions/0005-eval-methodology.md)). Six stacked install blockers across vllm / transformers / mistralai / Qwen 3.5 / NCCL forced the pivot; the in-tree path runs in the same Unsloth env Phase 3 used.
- In-tree evaluator scores 11 BFCL V3 categories totalling 458 items: AST-match (8 cats / 290 items), `must_call` (1 cat / 18 items), `must_not_call` (2 cats / 100 items).
- Result: base 87.3% / tuned 79.0% / delta **-8.3pp**. Gate FAILED (target: +3pp).
- Diagnosis ([ADR 0006](docs/decisions/0006-eval-debugging.md)): 76% of the regression (29 of 38 lost items) comes from the two `must_not_call` categories (`irrelevance` -19, `live_irrelevance` -10). Root cause is data composition -- xLAM is positive-only by design, so the LoRA learns a strong "the user wants a tool call" prior that costs us when tools genuinely don't help.
- Decision ([ADR 0007](docs/decisions/0007-ship-v1.0-below-gate.md)): ship v1.0 below the gate with the disclosure on the card + README rather than burn the remaining Colab budget on a Phase 3.5 retrain that mathematically still wouldn't clear the gate (best-case projection: ~-0.5pp delta).

### Artifacts

- `src/tooltuned_qwen/hub/model_card.py` renders the card with an explicit gate-disclosure callout when `delta < +3pp`.
- `results/bfcl_intree/comparison/{bfcl_results.json, comparison.md, bfcl_comparison.png}` are the published numbers.
- 95 tests pass (`uv run pytest`), `ruff` clean, `pyright` 0 errors.

[Unreleased]: https://github.com/sukhrobnurali/tooltuned-qwen/compare/v1.0.0...HEAD
[v1.0.0]: https://github.com/sukhrobnurali/tooltuned-qwen/releases/tag/v1.0.0
