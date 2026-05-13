"""In-tree BFCL evaluator covering the 11 single-turn V3 categories.

Started life in Phase 2 as a 50-item ablation slice over BFCL V3 simple
(see `run_bfcl_holdout` below); extended in S7 to cover the full
single-turn V3 surface area when the canonical gorilla `bfcl` CLI proved
unshippable on Colab (vllm 0.8.5 + transformers v5 + Qwen 3.5 + torch
2.11 NCCL ABI -- see project memory). `run_bfcl_full` is what Phase 4
runs; it returns the same results-dict shape that `bfcl_runner.run_bfcl`
returns, so `eval/compare.py` and `hub/model_card.py` consume it
unchanged.

Scoring per category (BFCL V3 conventions):
- `simple`, `multiple`, `parallel*`, `live_simple`, `live_multiple`,
  `live_parallel*` -- AST match: function-name + per-arg accept-list.
- `live_relevance` -- correct iff model emitted any tool call (tools
  are relevant; model SHOULD call).
- `irrelevance`, `live_irrelevance` -- correct iff model emitted NO
  tool call (tools are unrelated; model should DECLINE).

See ADR 0005 for the methodology choice + the documented limitation on
`parallel*` (we use lenient any-call-matches scoring rather than BFCL's
1:1 set match -- the delta-vs-base remains directionally meaningful
since both arms see the same lenient scorer).

`score_prediction` and the parser are deliberately import-light so they
can be exercised in unit tests; the heavy `run_bfcl_full` /
`run_bfcl_holdout` defer Unsloth / transformers imports into the
function body, matching the pattern used by `training/train.py` and
`eval/holdout.py`.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

# HF mirror of the BFCL V3 test sets + ground-truth files. V4 is hosted
# only on the gorilla repo at the time of writing; V3 is the released-on-HF
# version we can pin against, and the V3-vs-V4 substitution is documented
# in ADR 0005 -- both AST-score the same way, so the gate is unaffected.
_HF_BASE = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/main"
QUESTIONS_URL = f"{_HF_BASE}/BFCL_v3_simple.json"
ANSWERS_URL = f"{_HF_BASE}/possible_answer/BFCL_v3_simple.json"

# 11 single-turn V3 categories, paired with their scoring mode + section.
# Section labels match BFCL's leaderboard split (non_live = curated static
# prompts; live = real-world prompts collected from the wild). The 3
# relevance/irrelevance categories have no possible_answer files because
# their scoring is decision-to-call, not AST match.
BFCL_CATEGORIES: list[tuple[str, str, str]] = [
    ("simple", "ast_match", "non_live"),
    ("multiple", "ast_match", "non_live"),
    ("parallel", "ast_match", "non_live"),
    ("parallel_multiple", "ast_match", "non_live"),
    ("live_simple", "ast_match", "live"),
    ("live_multiple", "ast_match", "live"),
    ("live_parallel", "ast_match", "live"),
    ("live_parallel_multiple", "ast_match", "live"),
    ("irrelevance", "must_not_call", "non_live"),
    ("live_relevance", "must_call", "live"),
    ("live_irrelevance", "must_not_call", "live"),
]
DEFAULT_CATEGORIES: list[str] = [cat for cat, _, _ in BFCL_CATEGORIES]
_MODES: dict[str, str] = {cat: mode for cat, mode, _ in BFCL_CATEGORIES}
_SECTIONS: dict[str, str] = {cat: section for cat, _, section in BFCL_CATEGORIES}


def _category_mode(category: str) -> str:
    if category not in _MODES:
        raise ValueError(f"unknown BFCL category: {category!r}")
    return _MODES[category]


def _category_urls(category: str) -> tuple[str, str | None]:
    """Return (questions_url, answers_url|None) for a category. Answers is
    None for relevance/irrelevance categories (BFCL has no GT file)."""
    q = f"{_HF_BASE}/BFCL_v3_{category}.json"
    if _category_mode(category) == "ast_match":
        return q, f"{_HF_BASE}/possible_answer/BFCL_v3_{category}.json"
    return q, None

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Bare-JSON fallback: post-fine-tune Qwen sometimes emits raw `{"name": ..., "arguments": ...}`
# without the `<tool_call>` wrapper. Catch the first balanced JSON object.
_BARE_JSON_RE = re.compile(r"\{[^{}]*?\"name\"[^{}]*?\"arguments\".*?\}\s*\}", re.DOTALL)
# XML-tag form observed in our fine-tuned Qwen 3.5 4B output:
# `<function=NAME>\n<parameter=K>\nV\n</parameter>\n...\n</function>`. This is
# what the model actually emits after our LoRA on xLAM/Hermes -- not the JSON
# payload form the Qwen template's official `<tool_call>` block specifies.
_FUNCTION_TAG_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
_PARAMETER_TAG_RE = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL
)


def _try_parse_json_call(s: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    args = obj.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        return None
    return {"name": obj["name"], "arguments": args}


def _try_parse_xml_call(text: str) -> dict[str, Any] | None:
    """Extract `<function=NAME>...<parameter=K>V</parameter>...</function>`.

    Each parameter value is JSON-decoded if possible (so `10` becomes int,
    `"units"` becomes string, `true` becomes bool); falls through to a raw
    string for unquoted scalars like `units`. BFCL's per-arg acceptance is
    string-coercive, so either form scores correctly downstream.
    """
    fn = _FUNCTION_TAG_RE.search(text)
    if not fn:
        return None
    name = fn.group(1).strip()
    body = fn.group(2)
    arguments: dict[str, Any] = {}
    for param in _PARAMETER_TAG_RE.finditer(body):
        key = param.group(1).strip()
        raw = param.group(2).strip()
        try:
            arguments[key] = json.loads(raw)
        except json.JSONDecodeError:
            arguments[key] = raw
    return {"name": name, "arguments": arguments}


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract every `{name, arguments}` call from a model completion.

    Recognised emission shapes (in order, deduplicated by position):
      1. JSON payload inside `<tool_call>...</tool_call>` blocks
         (canonical Qwen template).
      2. XML-tag form `<function=NAME>...<parameter=K>V</parameter>...
         </function>` (what our fine-tune actually emits -- the
         `<tool_call>` outer wrapper, when present, contains the XML form).
      3. Bare JSON `{"name": ..., "arguments": ...}` without any wrapper
         -- only used as a fallback when neither (1) nor (2) matched.

    Returns an empty list if no shape recognised -- the relevance scorers
    distinguish "model declined" (empty) from "model called the wrong
    thing" (non-empty). The singular helper `parse_tool_call` is a thin
    wrapper preserved for Phase 2 callers.
    """
    calls: list[dict[str, Any]] = []
    for m in _TOOL_CALL_RE.finditer(text):
        parsed = _try_parse_json_call(m.group(1))
        if parsed is not None:
            calls.append(parsed)
    for m in _FUNCTION_TAG_RE.finditer(text):
        name = m.group(1).strip()
        args: dict[str, Any] = {}
        for pm in _PARAMETER_TAG_RE.finditer(m.group(2)):
            key = pm.group(1).strip()
            raw = pm.group(2).strip()
            try:
                args[key] = json.loads(raw)
            except json.JSONDecodeError:
                args[key] = raw
        calls.append({"name": name, "arguments": args})
    if not calls:
        bm = _BARE_JSON_RE.search(text)
        if bm:
            parsed = _try_parse_json_call(bm.group(0))
            if parsed is not None:
                calls.append(parsed)
    return calls


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """Singular variant: return the first parsed call, or None if none.

    Preserved for Phase 2 ablation callers (`run_bfcl_holdout`); new code
    in `run_bfcl_full` uses `parse_tool_calls` so parallel/relevance
    categories see every call the model emitted.
    """
    calls = parse_tool_calls(text)
    return calls[0] if calls else None


def _values_equivalent(value: Any, accepted: list[Any]) -> bool:
    """BFCL's per-arg acceptance: structural equality OR string-coerced equality.

    BFCL's simple-category scorer accepts a small slack between numeric forms
    (`5` vs `5.0`) and case-insensitive enum strings; we approximate that with
    string-cast comparison after the strict equality check fails.
    """
    if value in accepted:
        return True
    s = str(value)
    return any(s == str(a) for a in accepted if a != "")


def score_prediction(
    predicted: dict[str, Any] | None, ground_truth: list[dict[str, Any]]
) -> bool:
    """Return True iff `predicted` matches one of the BFCL ground-truth entries.

    `ground_truth` is BFCL's `[{<func_name>: {<arg>: [accepted, ...]}}]` shape.
    A param whose accepted list contains `""` is optional -- the prediction
    can omit it. Extra args in the prediction are tolerated (BFCL doesn't
    penalise them at this category level).
    """
    if not predicted:
        return False
    pred_name = predicted.get("name")
    pred_args = predicted.get("arguments", {})
    if not isinstance(pred_args, dict):
        return False

    for gt_entry in ground_truth:
        if not isinstance(gt_entry, dict):
            continue
        for gt_name, gt_params in gt_entry.items():
            if pred_name != gt_name:
                continue
            if _params_match(pred_args, gt_params):
                return True
    return False


def _params_match(pred_args: dict[str, Any], gt_params: dict[str, Any]) -> bool:
    for param, accepted in gt_params.items():
        accepted_list = accepted if isinstance(accepted, list) else [accepted]
        optional = "" in accepted_list
        if param not in pred_args:
            if optional:
                continue
            return False
        if not _values_equivalent(pred_args[param], accepted_list):
            return False
    return True


def score_prediction_multi(
    predicted_calls: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> bool:
    """Lenient any-match: True iff some predicted call satisfies some GT
    entry. For `simple`/`multiple` (single GT entry) this reduces to
    `score_prediction` against the singleton. For `parallel*` (multiple
    GT entries) this is a lower bound -- BFCL's canonical scorer requires
    a 1:1 set match with no extras, which we don't replicate in-tree
    (ADR 0005 §Limitations). Both base + tuned see the same scorer, so
    the delta-vs-base remains directionally meaningful.
    """
    if not predicted_calls:
        return False
    return any(score_prediction(call, ground_truth) for call in predicted_calls)


def score_must_call(predicted_calls: list[dict[str, Any]]) -> bool:
    """Decision category `live_relevance`: correct iff model called any
    tool. Tools are relevant; declining is a miss."""
    return len(predicted_calls) > 0


def score_must_not_call(predicted_calls: list[dict[str, Any]]) -> bool:
    """Decision categories `irrelevance` + `live_irrelevance`: correct
    iff model emitted NO tool call. Tools are unrelated; hallucinating
    a call is a miss."""
    return len(predicted_calls) == 0


def score_item(
    item: dict[str, Any], predicted_calls: list[dict[str, Any]]
) -> bool:
    """Dispatch scoring based on the category's mode (set in BFCL_CATEGORIES)."""
    mode = _category_mode(item["category"])
    if mode == "ast_match":
        gt = item.get("ground_truth") or []
        return score_prediction_multi(predicted_calls, gt)
    if mode == "must_call":
        return score_must_call(predicted_calls)
    if mode == "must_not_call":
        return score_must_not_call(predicted_calls)
    raise ValueError(f"unknown scoring mode: {mode!r}")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:
        fh.write(resp.read())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_bfcl_simple(
    *, n: int = 50, cache_dir: str | Path = "results/bfcl_holdout"
) -> list[dict[str, Any]]:
    """Fetch the first `n` simple-category items + ground truth.

    Returns `[{"id", "question", "function", "ground_truth"}, ...]`. Files are
    cached under `cache_dir` so re-runs (across the five Phase 2 ablations)
    don't re-download.
    """
    cache = Path(cache_dir)
    q_path = cache / "BFCL_v3_simple.json"
    a_path = cache / "BFCL_v3_simple_answers.json"
    _download(QUESTIONS_URL, q_path)
    _download(ANSWERS_URL, a_path)

    questions = _load_jsonl(q_path)
    answers = {row["id"]: row["ground_truth"] for row in _load_jsonl(a_path)}

    sliced: list[dict[str, Any]] = []
    for row in questions:
        if row["id"] not in answers:
            continue
        sliced.append(
            {
                "id": row["id"],
                "question": row["question"],
                "function": row["function"],
                "ground_truth": answers[row["id"]],
            }
        )
        if len(sliced) >= n:
            break
    return sliced


def load_bfcl_categories(
    categories: list[str] | None = None,
    *,
    n_per_cat: int | None = None,
    cache_dir: str | Path = "results/bfcl_holdout",
) -> list[dict[str, Any]]:
    """Fetch test items + ground truth for the requested BFCL categories.

    Returns a flat list of `{id, category, question, function, ground_truth}`
    dicts. `n_per_cat` truncates each category independently; `None`
    loads every item the HF mirror has for that category. Categories
    without ground-truth answer files (relevance/irrelevance) carry
    `ground_truth=None` -- their scoring is decision-to-call, not AST.
    """
    cats = list(categories or DEFAULT_CATEGORIES)
    cache = Path(cache_dir)
    out: list[dict[str, Any]] = []
    for cat in cats:
        q_url, a_url = _category_urls(cat)
        q_path = cache / f"BFCL_v3_{cat}.json"
        _download(q_url, q_path)
        questions = _load_jsonl(q_path)

        answers: dict[str, Any] = {}
        if a_url is not None:
            a_path = cache / f"possible_answer__BFCL_v3_{cat}.json"
            _download(a_url, a_path)
            answers = {
                row["id"]: row["ground_truth"] for row in _load_jsonl(a_path)
            }

        added = 0
        for row in questions:
            # AST categories must have a matching GT row; skip stragglers
            # so a half-mirrored category can't silently score as 0.
            if a_url is not None and row["id"] not in answers:
                continue
            out.append(
                {
                    "id": row["id"],
                    "category": cat,
                    "question": row["question"],
                    "function": row["function"],
                    "ground_truth": answers.get(row["id"]),
                }
            )
            added += 1
            if n_per_cat is not None and added >= n_per_cat:
                break
    return out


def _build_qwen_tool(fn: dict[str, Any]) -> dict[str, Any]:
    """BFCL function defs already match the Qwen tool schema (name, description,
    parameters); just wrap in the `{type: "function", function: {...}}` envelope
    that `apply_chat_template(tools=...)` expects."""
    return {"type": "function", "function": fn}


def shape_results(
    *,
    model_name: str,
    per_item: list[dict[str, Any]],
    evaluated_at: str | None = None,
) -> dict[str, Any]:
    """Aggregate per-item results into the canonical results dict that
    `eval/compare.build_comparison` consumes.

    Output schema matches `bfcl_runner.run_bfcl`:
      `{model, mode, n_total, evaluated_at, overall, per_category}`
    with `mode="intree"` to distinguish this in-tree eval from the
    deferred canonical FC/Prompt CLI paths.
    """
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in per_item:
        by_cat.setdefault(r["category"], []).append(r)

    per_category: dict[str, dict[str, Any]] = {}
    correct_sum = total_sum = 0
    for cat, rows in by_cat.items():
        correct = sum(1 for r in rows if r.get("correct"))
        total = len(rows)
        per_category[cat] = {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "section": _SECTIONS.get(cat, "unknown"),
        }
        correct_sum += correct
        total_sum += total

    overall = {
        "accuracy": correct_sum / total_sum if total_sum else 0.0,
        "correct": correct_sum,
        "total": total_sum,
    }
    return {
        "model": model_name,
        "mode": "intree",
        "n_total": total_sum,
        "evaluated_at": evaluated_at or date.today().isoformat(),
        "overall": overall,
        "per_category": per_category,
    }


def run_bfcl_holdout(
    adapter_path: str,
    *,
    n: int = 50,
    max_new_tokens: int = 768,
    cache_dir: str | Path = "results/bfcl_holdout",
) -> dict[str, Any]:
    """Load the adapter, run on `n` BFCL simple items, return per-item + summary.

    Heavy ML deps stay deferred so this module imports cleanly on a CPU box.
    Mirrors the Unsloth load pattern from `eval/holdout.py::quick_eval`:
    pass the adapter path to `from_pretrained` so layer-name reconciliation
    happens in one shot (Phase 1.3 finding).

    `max_new_tokens` defaults to 768: Qwen 3.5 reasons before emitting
    `<tool_call>`, and 256 was getting cut off mid-thought ("Let me call
    the function with ... <" was a typical truncation point during
    diagnosis). 768 gives ~150 tokens of reasoning headroom plus the
    tool-call block.
    """
    # Dynamo config has to land BEFORE Unsloth loads its pre-compiled
    # `unsloth_compiled_module_qwen3_5.py` -- that module is built with
    # one_graph=True, so any shape miss after load raises FailOnRecompileLimitHit.
    # Even with cache bumps, the BFCL loop overflows around item 40 (different
    # shapes per item). The reliable answer is to flip dynamo off entirely:
    # the OptimizedModule wrapper checks `config.disable` at call time and
    # falls through to the eager forward when set.
    import torch._dynamo

    torch._dynamo.config.cache_size_limit = 4096
    torch._dynamo.config.accumulated_cache_size_limit = 65536
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
    os.environ["TORCH_COMPILE_DISABLE"] = "1"

    import unsloth  # noqa: F401  load before transformers/peft (Unsloth load-order docs).
    from unsloth import FastLanguageModel

    os.environ.setdefault("ACCELERATE_BYPASS_DEVICE_MAP", "true")

    items = load_bfcl_simple(n=n, cache_dir=cache_dir)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=2048,
        load_in_4bit=False,
        dtype=None,
    )
    # Skip `FastLanguageModel.for_inference` -- it routes generation through
    # Unsloth's `unsloth_base_fast_generate` which sets `one_graph=True` on
    # the pre-compiled Qwen 3.5 module, which overflows the recompile counter
    # on a multi-shape loop. Standard `eval()` keeps the transformers generate
    # path, which is recompile-tolerant; throughput is lower but predictions
    # come out correct (the padded fast path was emitting empty completions
    # because pad_token=eos_token biased greedy decode to halt immediately).
    model.eval()
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    per_item: list[dict[str, Any]] = []
    correct = 0
    total = len(items)
    crashed_at: int | None = None
    crash_reason: str | None = None
    for i, item in enumerate(items, start=1):
        try:
            # BFCL `question` is `[[{"role": "user", "content": ...}, ...]]` --
            # one nested conversation. Take the inner list verbatim.
            messages = item["question"][0]
            tools = [_build_qwen_tool(fn) for fn in item["function"]]
            prompt = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = text_tokenizer(prompt, return_tensors="pt").to(model.device)
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
            completion = text_tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )
        except Exception as exc:
            # Catch dynamo recompile overflows or any other inference fault
            # so a 90%-complete run still produces a usable accuracy number.
            # The Phase 2 ablation is a comparative measurement; partial-N
            # results stay informative as long as both arms see the same N.
            crash_reason = f"{type(exc).__name__}: {exc}"
            print(f"  ! crashed at item {i}: {crash_reason}", flush=True)
            crashed_at = i
            break

        predicted = parse_tool_call(completion)
        ok = score_prediction(predicted, item["ground_truth"])
        per_item.append(
            {
                "id": item["id"],
                "predicted": predicted,
                "completion": completion,
                "correct": ok,
            }
        )
        if ok:
            correct += 1
        # Per-item progress -- without it, eager-mode generation looks like a
        # hung kernel for 5-10 minutes on a 50-item slice. `flush=True` so the
        # output appears in Colab in real time, not buffered until the loop ends.
        running_acc = correct / i
        print(
            f"[{i:>2d}/{total}] {item['id']:<10s} {'OK' if ok else '..'} "
            f"pred={predicted['name'] if predicted else None}  "
            f"running_acc={running_acc:.3f}",
            flush=True,
        )
        # Show the head + tail + length of every missed completion. The tail
        # is what matters: if it ends with `</tool_call>` the parser regex is
        # the bug; if it ends mid-thought we're hitting `max_new_tokens`; if
        # it ends in plain prose the trained model isn't using the tool-call
        # format at all.
        if predicted is None:
            head = completion[:120].replace("\n", " | ")
            tail = completion[-160:].replace("\n", " | ")
            print(
                f"          len={len(completion)} head={head!r} tail={tail!r}",
                flush=True,
            )

    n_done = len(per_item)
    accuracy = correct / n_done if n_done else 0.0
    return {
        "adapter_path": adapter_path,
        "n": n_done,
        "n_planned": len(items),
        "crashed_at": crashed_at,
        "crash_reason": crash_reason,
        "correct": correct,
        "accuracy": accuracy,
        "per_item": per_item,
    }


def run_bfcl_full(
    model_path: str,
    *,
    model_label: str | None = None,
    categories: list[str] | None = None,
    n_per_cat: int | None = None,
    max_new_tokens: int = 768,
    cache_dir: str | Path = "results/bfcl_holdout",
    out_path: str | Path | None = None,
    progress_every: int = 10,
) -> dict[str, Any]:
    """Evaluate a model on the 11 V3 single-turn BFCL categories in-tree.

    `model_path` is what Unsloth's `FastLanguageModel.from_pretrained`
    consumes -- the HF repo of either the base (`Qwen/Qwen3.5-4B`) or
    a published adapter (`sukhrobnurali/tooltuned-qwen-3.5-4b`). The
    adapter form follows the Phase 1.3 #5 finding: pass the adapter
    path directly to `from_pretrained`, never via `load_adapter`.

    Returns the canonical results dict consumed by
    `eval/compare.build_comparison` -- same shape as
    `bfcl_runner.run_bfcl`, but `mode="intree"` distinguishes the
    scoring path. `out_path` (optional) gets the same dict as JSON.

    `n_per_cat`: per-category truncation. Default `None` loads all
    items, which is ~1500+ inferences and ~hours on A100. The notebook
    passes a small N (50-100) for portfolio-grade signal at a fraction
    of the cost; see ADR 0005 for the sample-size argument.
    """
    # Dynamo guards (Phase 2 finding #11): Unsloth's pre-compiled
    # `unsloth_compiled_module_qwen3_5.py` is built with one_graph=True,
    # which overflows the recompile counter on a multi-shape eval loop.
    # Flip dynamo off entirely before Unsloth's import; the OptimizedModule
    # wrapper checks `config.disable` at call time and falls through to
    # eager. Mirrors the setup in `run_bfcl_holdout`.
    import torch._dynamo

    torch._dynamo.config.cache_size_limit = 4096
    torch._dynamo.config.accumulated_cache_size_limit = 65536
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
    os.environ["TORCH_COMPILE_DISABLE"] = "1"

    import unsloth  # noqa: F401  Unsloth load-order: before transformers/peft.
    from unsloth import FastLanguageModel

    os.environ.setdefault("ACCELERATE_BYPASS_DEVICE_MAP", "true")

    cats = list(categories or DEFAULT_CATEGORIES)
    items = load_bfcl_categories(cats, n_per_cat=n_per_cat, cache_dir=cache_dir)
    print(
        f"loaded {len(items)} items across {len(cats)} categories: "
        f"{', '.join(cats)}",
        flush=True,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=False,
        dtype=None,
    )
    # See `run_bfcl_holdout` for the for_inference-skip rationale.
    model.eval()
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    per_item: list[dict[str, Any]] = []
    correct = 0
    total = len(items)
    last_print_cat: str | None = None
    for i, item in enumerate(items, start=1):
        if item["category"] != last_print_cat:
            print(f"-- category: {item['category']} --", flush=True)
            last_print_cat = item["category"]
        try:
            messages = item["question"][0]
            tools = [_build_qwen_tool(fn) for fn in item["function"]]
            prompt = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = text_tokenizer(prompt, return_tensors="pt").to(model.device)
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
            completion = text_tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )
        except Exception as exc:
            # Surface the per-item failure as a "miss" rather than aborting
            # the whole run -- one bad prompt can't tank a 1000-item eval.
            # The full crash_reason lands in per_item for post-hoc triage.
            crash_reason = f"{type(exc).__name__}: {exc}"
            print(f"  ! item {i} ({item['id']}) errored: {crash_reason}", flush=True)
            per_item.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "predicted_calls": [],
                    "completion": "",
                    "correct": False,
                    "error": crash_reason,
                }
            )
            continue

        predicted_calls = parse_tool_calls(completion)
        ok = score_item(item, predicted_calls)
        per_item.append(
            {
                "id": item["id"],
                "category": item["category"],
                "predicted_calls": predicted_calls,
                "completion": completion,
                "correct": ok,
            }
        )
        if ok:
            correct += 1
        if i % progress_every == 0 or i == total:
            running_acc = correct / i
            print(
                f"[{i:>4d}/{total}] {item['id']:<32s} "
                f"{'OK' if ok else '..'} running_acc={running_acc:.3f}",
                flush=True,
            )

    results = shape_results(
        model_name=model_label or model_path,
        per_item=per_item,
    )
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(results)
        # Strip completions from disk-written copy -- a full 1k-item run with
        # ~1k-char completions hits ~1 MB and floods git/HF; per-item triage
        # data lives in the in-memory return value only.
        payload["per_item_summary"] = [
            {"id": r["id"], "category": r["category"], "correct": r["correct"]}
            for r in per_item
        ]
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    return results
