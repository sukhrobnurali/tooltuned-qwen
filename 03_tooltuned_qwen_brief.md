# Project Brief: `tooltuned-qwen`

> **Fine-tuning Qwen 3.5 4B for measurably better tool-calling than the base model.**
> QLoRA via Unsloth on Colab Pro L4. Evaluated on the public Berkeley Function-Calling Leaderboard (BFCL V4). Ships a LoRA adapter, merged weights, and GGUF on Hugging Face, with a reproducible training pipeline on GitHub. Designed as a dual portfolio piece for both platforms.

---

## 1. Purpose & Problem

Most "AI/ML engineers" claim fine-tuning experience but cannot point to a single fine-tuned model they've shipped publicly with measurable gains over a base. That gap — between *talking about* fine-tuning and *demonstrating* it — is exactly where senior MLE candidates separate themselves.

This project closes the gap with a single, sharp deliverable: a Qwen 3.5 4B fine-tune that scores meaningfully higher than the base on BFCL V4, the public function-calling benchmark, with the entire pipeline open and reproducible.

Why tool-calling specifically:
- **Objectively measurable.** BFCL gives a single number per model. Improvement vs. base is undeniable.
- **Hot 2026 topic.** Every agent framework needs reliable tool-callers. Fine-tuned tool-callers are valuable.
- **Cross-promotes the other repos.** `pr-review-agent` and `llm-gateway` both benefit from better tool-calling models — this fine-tune feeds the same portfolio narrative.
- **Off-the-shelf datasets.** No data scraping required. xLAM and Hermes are on HuggingFace, ready to use.
- **Cheap to train.** 4B QLoRA fits a Colab L4 and trains in hours, not days.

The deliverable is not just a model. It is a **dual portfolio piece**: a polished GitHub repo and a polished Hugging Face model page, cross-linked, each engineered to make a recruiter or hiring engineer immediately register the author as someone who ships fine-tuned models, not just talks about them.

---

## 2. Goals

**Primary:**
1. **Train a Qwen 3.5 4B LoRA adapter that beats base Qwen 3.5 4B on BFCL V4.** Target: ≥3 percentage points overall accuracy improvement. If the gap is smaller, dig in and figure out why before shipping.
2. **Hugging Face artifact must be portfolio-grade.** Comprehensive model card, before/after BFCL table, training hyperparameters, intended use, limitations, license. Visually clean. Looks like a published research-style release.
3. **GitHub repo must be portfolio-grade.** Clean codebase, reproducible pipeline, 5-cell Colab launcher, clear README with results, methodology, ablations.
4. **Two artifacts, cross-linked.** HF page links to GitHub repo. GitHub README links to HF model. Both link to author profiles. Triple cross-promotion.
5. **Total Colab compute budget under 130 units** (35-unit safety buffer from the 150 available).

**Secondary:**
6. Reproduce-from-scratch: someone with a HF account and Colab Pro should be able to retrain the model by running the notebook.
7. Optional stretch artifact: Qwen 3.5 9B variant on A100 (only if 4B run is clean and budget permits).

---

## 3. Non-Goals (Explicit)

- **Not a new training framework.** We use Unsloth as-is. We don't reinvent QLoRA.
- **Not a SOTA chase.** We don't need to beat the BFCL leaderboard's top model — we need to beat the *base Qwen 3.5 4B*. That's the credibility statement.
- **Not multi-model.** One base model (Qwen 3.5 4B), one main artifact. Stretch 9B is optional.
- **Not a Hugging Face Space.** Per scope: no live demo. The model card, results table, and a small CLI inference example in the repo are sufficient.
- **Not a YouTube tutorial.** Per scope: written deliverables only this round.
- **Not full fine-tuning (FFT).** QLoRA only. We're optimizing for portfolio impact per compute hour, not parameter coverage.
- **Not a chat assistant tune.** This is *specifically* for tool-calling. We do not try to make the model better at general conversation.

---

## 4. Target Audience for the Artifacts

The HF model page and the GitHub repo serve different audiences:

| Surface | Primary audience | What they look for |
|---|---|---|
| **Hugging Face** | ML engineers searching "function calling Qwen", recruiters scanning HF profile | Model card quality, eval numbers, downloads, intended use |
| **GitHub** | Recruiters reviewing portfolio, engineers wanting to reproduce, OSS contributors | Code quality, clear README, reproduction instructions, ablations |

The Claude Code session must treat both as first-class deliverables. The HF page is **not** an afterthought.

---

## 5. Positioning vs. Existing Tool-Calling Fine-Tunes

Many tool-calling fine-tunes exist on Hugging Face. Our differentiation is not novelty — it's **execution and transparency**:

- **Specific recent base model** (Qwen 3.5 4B, latest open generation as of build date)
- **Public, reproducible eval** (BFCL V4 with logs published)
- **Full pipeline open-sourced** (most HF tool-tunes are opaque — push the weights, hide the data prep)
- **Honest before/after numbers** (not cherry-picked subsets)
- **Multilingual base** (Qwen supports 201 languages — flag this even if our training data is English; future contributors can extend)

The Claude Code session should re-survey HF for similar models before launch. Frame our positioning as "fully reproducible Qwen 3.5 4B tool-calling fine-tune with BFCL evaluation," not "best tool-calling model." Honesty is the moat.

---

## 6. Core Deliverables (MVP)

### 6.1. Hugging Face artifacts (under `sukhrobnurali/tooltuned-qwen-3.5-4b`)

1. **LoRA adapter** (~50–80MB). Drop-in for anyone running base Qwen 3.5 4B who wants the tool-calling boost.
2. **Merged weights** in bf16 (full model, ~8GB). Convenience artifact for users who don't want to manage adapters.
3. **GGUF quantized weights** (Q4_K_M and Q8_0 at minimum). For Ollama, llama.cpp, LM Studio users.
4. **Polished model card.** Sections: TL;DR, BFCL V4 results table (base vs. fine-tuned), training data, training procedure, hyperparameters, intended use, out-of-scope use, limitations, license, reproduction (link to GitHub), citation, author.

### 6.2. GitHub repo (`sukhrobnurali/tooltuned-qwen`)

1. **Clean Python package** with `src/tooltuned_qwen/` layout (data, training, eval, hub, inference modules).
2. **Configs in YAML** (default, dataset ablation, LoRA-rank ablation, stretch 9B).
3. **5-cell Colab launcher notebook** that clones repo, installs, prepares data, trains, evals, pushes — no logic in cells.
4. **Reproducible eval pipeline** that runs BFCL V4 against base and adapter and produces the comparison table.
5. **README as a research-style write-up.** Sections: TL;DR with results table, methodology, dataset, training, evaluation, ablations (if any), reproduction, links to HF artifacts, license, citation.
6. **Inference example script** (`scripts/inference.py`) — load model + adapter, demonstrate tool-calling on a sample query, print structured output.
7. **CI** (lint, type check, unit tests on data/format code; not on training).

### 6.3. Combined cross-linking

- HF model card has prominent badge/link to GitHub repo
- GitHub README has prominent badge/link to HF model
- Both link to author HF profile + GitHub profile + LinkedIn
- Submit to BFCL leaderboard (if scores warrant — author decision before launch)

---

## 7. Stretch Deliverables (post-MVP, only if budget allows)

1. **Qwen 3.5 9B variant** on A100 (~3hr, ~40 units). Same dataset, same procedure. Compare on BFCL. Adds a "scale ablation" section to the README.
2. **Dataset ablation:** train two adapters (xLAM-only and Hermes-only), compare BFCL scores, document which dataset wins. This is a real research signal recruiters appreciate.
3. **LoRA-rank ablation:** train at rank 8 vs. 16 vs. 32, compare. Quick experiments, big README payoff.
4. **MTEB-style multi-benchmark eval:** in addition to BFCL, run on a second function-calling benchmark (e.g., ToolBench eval) for triangulation.
5. **Adapter compatibility test:** show the LoRA loads cleanly with vLLM (ties to project #1 `llm-gateway`).

---

## 8. Technical Architecture (pipeline view)

```
   ┌────────────────────────────────┐
   │  HuggingFace Hub                │
   │  - xLAM-function-calling-60k   │
   │  - Hermes-Function-Calling-V1  │
   └─────────────┬───────────────────┘
                 │  download
                 ▼
   ┌──────────────────────────────────┐
   │  Data prep (src/data)            │
   │  - load + dedupe                 │
   │  - apply Qwen chat template      │
   │  - split: train / val / holdout  │
   └─────────────┬────────────────────┘
                 │  formatted JSONL
                 ▼
   ┌──────────────────────────────────┐
   │  Training (src/training)         │
   │  - Unsloth FastLanguageModel     │
   │  - QLoRA (4-bit + LoRA r=16)     │
   │  - SFTTrainer (TRL)              │
   │  - W&B logging                   │
   └─────────────┬────────────────────┘
                 │  LoRA adapter
                 ▼
   ┌──────────────────────────────────┐
   │  Eval (src/eval)                 │
   │  - BFCL V4 on base               │
   │  - BFCL V4 on adapter            │
   │  - holdout test set              │
   │  - results JSON + markdown table │
   └─────────────┬────────────────────┘
                 │  metrics + adapter
                 ▼
   ┌──────────────────────────────────┐
   │  Hub push (src/hub)              │
   │  - LoRA adapter                  │
   │  - merged bf16 weights           │
   │  - GGUF (Q4_K_M, Q8_0)           │
   │  - auto-generated model card     │
   └──────────────────────────────────┘
```

The pipeline runs as **two separate Colab notebooks**: `colab_train.ipynb` (train + push adapter) and `colab_eval.ipynb` (run BFCL on base and adapter, generate the model card, push final artifacts). Splitting reduces session-timeout risk.

---

## 9. Tech Stack

> **Per project preference: latest stable versions.** The Claude Code session must verify before pinning.

- **Language:** Python 3.12+
- **Training:** Unsloth (latest), TRL, transformers v5+ (Unsloth requires v5 for Qwen 3.5)
- **Quantization:** bitsandbytes 4-bit NF4
- **Datasets:** `datasets` (HuggingFace)
- **Hub interaction:** `huggingface_hub`
- **Quantization for GGUF:** Unsloth's built-in GGUF export, or `llama.cpp` quantize tool
- **Observability:** wandb (free tier, public dashboard linked from model card adds credibility)
- **Eval:** `bfcl` CLI from `gorilla` repo (Berkeley)
- **Config:** Pydantic v2 + YAML
- **Inference:** transformers + PEFT for adapter, vLLM compat as stretch
- **Dev tooling:** uv, ruff, pytest, mypy or pyright
- **Notebook:** Colab Pro (L4 primary, A100 stretch)

---

## 10. Suggested Repo Structure

```
tooltuned-qwen/
├── src/tooltuned_qwen/
│   ├── __init__.py
│   ├── data/
│   │   ├── load_xlam.py
│   │   ├── load_hermes.py
│   │   ├── format.py            # apply Qwen chat template
│   │   └── splits.py            # train / val / holdout
│   ├── training/
│   │   ├── config.py            # TrainingConfig (Pydantic)
│   │   ├── train.py             # main training entrypoint
│   │   └── callbacks.py         # wandb, eval-on-step
│   ├── eval/
│   │   ├── bfcl_runner.py       # invokes bfcl CLI
│   │   ├── holdout.py           # custom test set
│   │   └── compare.py           # base vs adapter table
│   ├── hub/
│   │   ├── push_adapter.py
│   │   ├── push_merged.py
│   │   ├── push_gguf.py
│   │   └── model_card.py        # programmatic card generator
│   └── inference/
│       ├── load.py
│       └── chat.py              # interactive demo
├── configs/
│   ├── default.yaml
│   ├── ablation_dataset_xlam.yaml
│   ├── ablation_dataset_hermes.yaml
│   ├── ablation_rank_8.yaml
│   ├── ablation_rank_32.yaml
│   └── stretch_9b.yaml
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── eval.py
│   ├── push_to_hub.py
│   └── inference.py             # demo script for README
├── notebooks/
│   ├── colab_train.ipynb        # 5-cell launcher, no logic
│   ├── colab_eval.ipynb
│   └── .gitkeep
├── tests/
│   ├── test_data_format.py
│   ├── test_chat_template.py
│   └── test_inference_load.py
├── results/                     # eval outputs, BFCL logs, charts
│   ├── base_qwen35_4b/
│   └── tooltuned_qwen35_4b/
├── docs/
│   ├── methodology.md
│   ├── reproduction.md
│   ├── benchmarks.md
│   └── decisions/               # ADRs
├── .env.example
├── pyproject.toml
├── README.md
├── MODEL_CARD.md                # source of truth for HF card
└── LICENSE                      # match Qwen base license
```

---

## 11. Open Questions for the Claude Code Session to Research First

These are deliberately unresolved.

1. **Is Qwen 3.5 4B still the right pick?** Check HF for any newer small Qwen open-weight release (Qwen 3.6 4B?) since this brief was written. If a newer small model exists, evaluate switching.
2. **Verify Unsloth + transformers + TRL versions** are mutually compatible and current. Pin in `pyproject.toml`.
3. **Dataset selection: xLAM vs. Hermes.** Train two small (1-epoch, 1k-sample) test runs on each dataset, compare on BFCL holdout. Pick the winner empirically. Document the choice in an ADR.
4. **Thinking mode strategy.** Qwen 3.5 has reasoning by default. For tool-calling, do we (a) train with reasoning preserved, (b) strip it entirely, or (c) train with 75/25 mix per Unsloth's recommendation? Test on small runs, decide based on BFCL impact.
5. **LoRA rank.** Default 16 is fine, but rank-8 is faster and rank-32 has more capacity. Quick ablation can answer this.
6. **Chat template correctness.** Qwen's chat template differs by version. Verify the dataset is formatted with the *correct* Qwen 3.5 template. Format errors silently destroy training.
7. **BFCL evaluator mode.** BFCL has "FC mode" (native function-calling) and "Prompt mode" (text-based). Run both? Just FC? Document.
8. **GGUF quantization choices.** Q4_K_M is standard. Q8_0 for higher quality. IQ4_XS for size. Recommend pushing Q4_K_M and Q8_0 minimum.
9. **License inheritance.** Qwen 3.5 4B has a specific license — match it on our adapter and merged weights. Verify before pushing.
10. **Reproducibility seed strategy.** Set seeds, log them in the model card. Document exact `transformers` / `unsloth` / `torch` versions used.
11. **W&B run visibility.** Public W&B dashboard adds credibility (anyone can verify our training curves). Recommend public; the CC session should confirm with author.
12. **Should we submit to BFCL leaderboard?** Submission process requires forking the gorilla repo and adding our model. Big credibility win if scores are strong. CC session should evaluate after eval results are in.
13. **Model card visual polish.** HF supports markdown + a hero image / chart. Generate a clean before/after BFCL bar chart (matplotlib → save as PNG → embed in card). This is the #1 thing recruiters notice.

---

## 12. Success Criteria

The project is "done" (v1.0 release-ready) when ALL of:

- [ ] Trained LoRA adapter beats base Qwen 3.5 4B on BFCL V4 by ≥3pp overall accuracy
- [ ] LoRA adapter pushed to HF with proper card and license
- [ ] Merged bf16 weights pushed to HF
- [ ] GGUF (Q4_K_M, Q8_0 minimum) pushed to HF
- [ ] HF model card includes: TL;DR, BFCL before/after table, hyperparameters, dataset details, intended use, limitations, license, GitHub link, before/after chart image
- [ ] GitHub README includes: TL;DR with chart, methodology, dataset, training, evaluation, reproduction steps, HF link, license
- [ ] Inference script works from a fresh clone with documented setup
- [ ] Colab launcher notebook reproduces the training from scratch
- [ ] Test coverage on data prep + chat template formatting (≥80%)
- [ ] CI runs on every push (lint, type check, tests)
- [ ] Total Colab compute used: ≤130 units
- [ ] All commits human-style, no AI attribution anywhere (see §16)

**Stretch (post-launch, optional):**
- Submitted to BFCL leaderboard
- Qwen 3.5 9B stretch artifact also published
- Dataset ablation results published
- 50+ HF downloads in first month
- Featured on r/LocalLLaMA or similar

---

## 13. Distribution Strategy

Lighter than projects #1 and #2 because the artifacts speak for themselves. Focus on quality over volume.

- **HF model page is the primary distribution channel.** People searching "Qwen 3.5 function calling fine-tune" land here. Model card quality determines whether they download.
- **BFCL leaderboard submission** if scores warrant. Single biggest credibility multiplier available — your name appears on a public benchmark next to GPT-4 and Claude.
- **GitHub README** as a research-paper-lite write-up. Recruiters click through from HF.
- **One LinkedIn post** at launch: chart of BFCL improvement, link to HF and GitHub. No fluff.
- **Awesome lists:** awesome-llm-fine-tuning, awesome-qwen, awesome-function-calling.
- **No HN, no Reddit, no Twitter blast** — let the artifacts find their audience organically through HF search and the leaderboard.

The author can layer YouTube content on top later if desired — the assets (clean repo, polished model card, BFCL numbers) are exactly what a long-form fine-tuning tutorial video would showcase. Out of scope for v1.

---

## 14. Handoff to Claude Code (do this FIRST in the new session)

When this brief is loaded into a fresh VS Code Claude Code session, **do not start training immediately**. In this exact order:

1. **Re-verify model selection** (§11, q1). Has a newer small Qwen open-weight model (3.6 4B?) been released? If yes, evaluate switching. Document choice in `docs/decisions/0001-base-model.md`.
2. **Verify all library versions** (§11, q2). Latest Unsloth, transformers v5+, TRL, bitsandbytes, datasets, huggingface_hub, peft, wandb. Pin in `pyproject.toml`. Confirm Qwen 3.5 4B Unsloth notebook works as-is on Colab L4.
3. **Read §16 (Git Workflow) before the first commit.** Configure git identity, disable any default Claude / AI co-authorship attribution, verify with a test commit.
4. **Set up the project skeleton on day one.** uv-managed `pyproject.toml`, ruff config, pytest harness, GitHub Actions CI, `.gitignore` (must exclude notebook checkpoints, `.env`, `wandb/`, `outputs/`, `results/*/raw/`).
5. **Write a smoke-test training run.** 1 step on 8 samples. Confirm the pipeline works end-to-end (data load → format → train → save adapter → load adapter → infer) before any real training. Catches 90% of bugs cheaply.
6. **Run dataset comparison** (§11, q3). 1-epoch on 1k samples, xLAM vs. Hermes, eval on BFCL holdout. Pick winner. Document.
7. **Run thinking-mode comparison** (§11, q4) similarly small-scale. Decide strategy.
8. **THEN execute the main run.** Track on W&B (public dashboard).
9. **Run full BFCL V4 evaluation** on base and adapter. Generate comparison table + chart.
10. **Generate the model card programmatically** (`src/tooltuned_qwen/hub/model_card.py`) so it's never out of sync with eval results.
11. **Push artifacts to HF** in this order: adapter → merged → GGUF → final model card. Verify each renders correctly on hf.co.
12. **Polish the GitHub README.** The README is the project — write it last, after results are in. Embed the chart, table, methodology, reproduction steps. Make it look like a 1-page paper.
13. **Decide on BFCL leaderboard submission** (§11, q12). If scores warrant, prepare PR to gorilla repo.
14. **Open the brainstorm wide.** Challenge any assumption in this brief. If the data says a different base model, dataset, or LoRA config performs better, change course. Don't ship to spec; ship to what the data shows.

This brief is a starting point, not a contract.

---

## 15. Author Context (for tone/style of public artifacts)

Senior AI/ML Engineer, Tashkent, Uzbekistan. 17K+ subscriber YouTube channel (not a launch channel for this project but worth mentioning in author bio on HF). HF profile: `huggingface.co/sukhrobnurali`. GitHub: `github.com/sukhrobnurali`.

The HF model card and GitHub README voice should be: technically sharp, honest about limitations, no hype. Sample tone: *"This adapter improves Qwen 3.5 4B's BFCL V4 score from X to Y. It was trained on xLAM-60k for one epoch using QLoRA rank 16 on a single Colab L4. Total compute: ~5 hours. Code and exact reproduction steps in the linked GitHub repo."*

That tone — concrete, honest, reproducible — is what marks the author as senior. No "revolutionary," "state-of-the-art," "groundbreaking." Just numbers and methodology.

---

## 16. Git Workflow & Commit Conventions

**Real-time mode this round** — no backdating. Commit as work progresses. Real current dates. The discipline is on commit *style* and *attribution*, not chronology.

### 16.1. Git Identity

Before the first commit:

```bash
git config user.name "Sukhrob Nurali"
git config user.email "sukhrobnurali@gmail.com"
```

Verify with `git config --get user.email`.

### 16.2. NO AI Attribution — Hard Rule

This stays non-negotiable across all three projects.

- **Do NOT** add `Co-authored-by: Claude <noreply@anthropic.com>` or any variant to commit messages, ever.
- **Do NOT** reference Claude, Anthropic, AI assistance, "generated by", "AI-assisted", etc. anywhere in commit messages, PR descriptions, release notes, model card, or README.
- **Do NOT** sign commits with Claude's identity.
- All commits authored solely by the user.

Claude Code may add this attribution by default — **disable it explicitly at session start** and verify with a throwaway test commit before doing real work. If a commit slips through with an attribution, fix it immediately with `git commit --amend` or `git rebase -i` before pushing.

### 16.3. Commit-as-You-Go Cadence

The project will be built in one or more focused sessions over the current period. Commit incrementally as work progresses — that's the natural rhythm.

- **Commit after each logical unit of work.** Finished a working data loader? Commit. Got the chat template formatting passing tests? Commit. First successful smoke-test training step? Commit.
- **Don't dump everything in one giant final commit.** That's the AI-built-it-all-in-one-shot tell.
- **It's fine to push as you go** (no contamination concern since dates are real).
- **Push to GitHub at meaningful checkpoints** — at least: initial scaffolding, before each major phase (data, training, eval, hub push), and at v1.0.
- **Expected total commits: 30–50** across the build. More if iterations are messy (which is realistic), fewer is suspicious.

### 16.4. Commit Message Style — Human, Not AI

Same standard as projects #1 and #2. Loose Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `wip:`), broken occasionally.

**Good (target):**

```
feat: data loader for xlam
fix chat template - was missing system role
add bfcl runner wrapper
wip: trying r=32
ok r=16 is better
chore: bump unsloth
typo
docs: training section in README
add model card generator
fix push_adapter - wrong repo path
that works finally
xlam wins, dropping hermes prep
chart for BFCL comparison
```

**Bad (AI tells, AVOID):**

```
feat: Implement comprehensive data loading utility for the xLAM function calling dataset
refactor: Restructure the chat template module to leverage Unsloth's optimized formatting
docs: Add a comprehensive training methodology section to the README with detailed hyperparameter explanations
fix: Resolve an issue with the LoRA adapter push to Hugging Face Hub where the repository path was incorrectly configured
```

**Specific rules (same as previous projects):**

- No em-dashes (`—`) in commit messages — regular hyphens.
- Avoid: "comprehensive", "robust", "leverages", "facilitates", "ensures", "enables", "streamline", "seamlessly", "thoroughly", "extensive".
- Vary verb structure across consecutive commits.
- Mix lowercase and capitalized — not every message a complete sentence.
- 1–2 typos across the project is realistic.
- `wip:` and `oops` commits are good — they show iteration.
- Emoji at most once across the entire history, if at all.

### 16.5. Commit Granularity

- One logical change per commit.
- Don't squash aggressively at the end.
- Tag the v1.0 release commit with proper release notes.

### 16.6. Pre-Push Sanity Check

Before each push (and definitely before the final v1.0 push), run:

```bash
# AI attribution scan — all four MUST return zero
git log --all --grep="claude" -i
git log --all --grep="anthropic" -i
git log --all --grep="co-authored" -i
git log --all --grep="generated" -i

# Author check
git log --pretty=format:"%an <%ae>" | sort -u
# Should show only "Sukhrob Nurali <sukhrobnurali@gmail.com>"

# Glance at messages
git log --pretty=format:"%s" | head -40
# Should look natural, varied, not AI-uniform
```

If any AI-attribution check returns non-zero results, **stop and fix before pushing**. Use `git rebase -i` and `git commit --amend` to clean up.

### 16.7. The One-Sentence Rule

> Every commit must look like Sukhrob wrote it himself, working through a focused build session, with no trace of AI assistance anywhere in the history.

---

*End of brief. All three projects scoped.*
