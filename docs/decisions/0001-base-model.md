# ADR 0001 — Base model selection

- **Status:** Accepted
- **Date:** 2026-05-10
- **Decision driver:** Brief §11 q1 + Plan Phase 0.1. "Stay on Qwen 3.5 4B unless a newer small model exists with an Unsloth-compatible recipe AND a credible BFCL baseline."

## Context

The brief was drafted assuming `Qwen/Qwen3.5-4B-Instruct` as the base. By 2026-05-10 we need to (a) confirm the handle still exists and is the right one, (b) check whether a newer small Qwen has shipped, (c) confirm Unsloth's recipe is current.

## Decision

Use **`Qwen/Qwen3.5-4B`** as the base model.

## Findings

1. **Latest small Qwen is still 3.5.** The Qwen 3.5 small lineup (0.8B / 2B / 4B / 9B) was released Feb–Mar 2026 (Apache-2.0). The Qwen 3.6 line that followed in April 2026 contains 27B and 35B-A3B variants — **no 4B in 3.6**. Per HF and third-party coverage, the 3.5-4B is currently the most capable open model under 5B.
2. **Native function-calling baseline exists.** Qwen 3.5 4B's own model card publishes a BFCL-V4 score of **50.3** (9B variant: 66.1). This is our baseline and the gate target is **≥53.3** (≥3pp improvement, brief §12).
3. **No separate `-Instruct` variant.** Unlike the older Qwen 3 line (e.g. `Qwen/Qwen3-4B-Instruct-2507`), Qwen 3.5 ships a single `Qwen/Qwen3.5-4B` repo. Instruction behaviour is controlled at runtime via `chat_template_kwargs={"enable_thinking": False}` for non-thinking mode (mirrors the brief's "thinking-mode strategy" question, §11 q4).
4. **Unsloth's recipe is current.** `unsloth.ai/docs/models/qwen3.5/fine-tune` exists and is maintained. Recommended starter knobs: `r=16`, `lora_alpha=16`, bf16 LoRA, `max_seq_len=2048`, batch 1, `adamw_8bit`. ~10 GB VRAM — fits Colab L4 (24 GB) easily.
5. **Tool-calling parser shipped.** vLLM/SGLang accept `--tool-call-parser qwen3_coder` for the 3.5 family; this matters for the §4 inference-time evaluator wiring.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Qwen 3.6 27B / 35B-A3B | Out-of-scope for the 4B portfolio piece; A100 territory, blows the 130-unit budget. |
| Qwen 3.5 9B | Tempting (BFCL 66.1 baseline) but ~2× compute. Brief §7 already lists 9B as a stretch ablation. Stay 4B for v1, revisit in Phase 7. |
| Qwen 3.5 2B | Smaller baseline; less headroom; less impressive portfolio narrative. |
| Qwen 3-4B-Instruct-2507 | Older series, superseded by 3.5; Unsloth recipe is for 3.5. |
| Gemma 4 small | Different ecosystem, would require rewriting half the brief's tooling-calling assumptions; rejected for scope discipline. |

## Deviations from the brief — flagged

Two small wording corrections vs. the brief / plan; not material to v1 scope but recorded so they don't surface as silent surprises later.

1. **HF handle.** Brief §11 q1 / plan §4.2 reference `Qwen/Qwen3.5-4B-Instruct`. Real handle is `Qwen/Qwen3.5-4B`. Update everywhere when scaffolding lands (S2).
2. **QLoRA framing.** Brief §2 calls the deliverable a "QLoRA fine-tune"; brief title and several sections echo "QLoRA". Unsloth's official guidance for Qwen 3.5 is **"It is not recommended to do QLoRA (4-bit) training on the Qwen3.5 models"** — bf16 LoRA is the documented recipe (10 GB VRAM, fits L4). We therefore plan a **bf16 LoRA** fine-tune, not QLoRA. The README and model card should describe the work as "LoRA fine-tune" (no Q). The portfolio narrative is unaffected.

A future ADR (post-S5, if we have spare budget) could rerun a 4-bit comparison purely as an ablation. Not in v1 scope.

## Consequences

- All scaffolding, configs, model card text, and README copy use `Qwen/Qwen3.5-4B` and "LoRA" (not "QLoRA").
- `transformers` must be **v5+** (Unsloth doc: "Older versions will not work" for Qwen 3.5). See ADR 0002 for the version-pin set.
- BFCL-V4 gate target locked at **≥53.3** (50.3 baseline + 3pp from §12).
- Default thinking-mode strategy stays "to be decided empirically in Phase 2.2"; nothing in this ADR pre-commits.
- License: Apache-2.0 inherited from the base. Unblocks brief §11 q9 — the LICENSE file just needs to reproduce Apache-2.0 verbatim, no Qwen-specific custom terms to copy.

## References

- HF model card — `huggingface.co/Qwen/Qwen3.5-4B`
- Unsloth fine-tune doc — `unsloth.ai/docs/models/qwen3.5/fine-tune`
- Brief: [03_tooltuned_qwen_brief.md](../../03_tooltuned_qwen_brief.md) — §1, §2, §11 q1, §11 q9, §12
- Plan: [read-the-project-description-cached-lightning.md](../../../../Users/User/.claude/plans/read-the-project-description-cached-lightning.md) — Phase 0.1, Phase 4.2, Phase 4.4
