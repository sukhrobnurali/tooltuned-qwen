"""Cheap BFCL holdout slice for the Phase 2 ablations.

The full BFCL V4 evaluation is Phase 4; this module is the small holdout
slice (default N=50) used for the dataset and thinking-mode ablations. It
pulls test items + ground truth from the HF mirror of BFCL V3 simple, runs
the model in tool-calling mode, parses `<tool_call>` blocks from the
completion, and scores each prediction against the simple-category AST
match (function-name + per-arg acceptable-value list, with `""` marking an
optional argument).

`score_prediction` and the parser are deliberately import-light so they can
be exercised in unit tests; the heavy `run_bfcl_holdout` defers Unsloth /
transformers imports into the function body, matching the pattern used by
`training/train.py` and `eval/holdout.py`.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

# HF mirror of the BFCL V3 simple-category test set + ground-truth file.
# V4 is hosted only on the gorilla repo at the time of writing; V3 is the
# released-on-HF version we can pin against. Both are AST-scored the same
# way, so the Phase 2 signal carries over. Phase 4 will switch to the
# canonical `bfcl` CLI against the latest BFCL release.
_HF_BASE = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/main"
QUESTIONS_URL = f"{_HF_BASE}/BFCL_v3_simple.json"
ANSWERS_URL = f"{_HF_BASE}/possible_answer/BFCL_v3_simple.json"

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Bare-JSON fallback: post-fine-tune Qwen sometimes emits raw `{"name": ..., "arguments": ...}`
# without the `<tool_call>` wrapper. Catch the first balanced JSON object.
_BARE_JSON_RE = re.compile(r"\{[^{}]*?\"name\"[^{}]*?\"arguments\".*?\}\s*\}", re.DOTALL)


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """Pull a `{name, arguments}` dict out of a model completion.

    Returns `None` if no tool call is recognisable -- that's a strict miss
    for scoring purposes (the model failed to call a function at all).
    """
    m = _TOOL_CALL_RE.search(text)
    if m:
        candidate = m.group(1)
    else:
        bm = _BARE_JSON_RE.search(text)
        if not bm:
            return None
        candidate = bm.group(0)
    try:
        obj = json.loads(candidate)
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


def _build_qwen_tool(fn: dict[str, Any]) -> dict[str, Any]:
    """BFCL function defs already match the Qwen tool schema (name, description,
    parameters); just wrap in the `{type: "function", function: {...}}` envelope
    that `apply_chat_template(tools=...)` expects."""
    return {"type": "function", "function": fn}


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
    # The relevant knob is `accumulated_cache_size_limit` (sum across all
    # tracked frames), not the per-function `cache_size_limit`. Bump it
    # generously and flip suppress_errors so an overflow falls back to eager
    # instead of crashing the whole run.
    import torch._dynamo

    torch._dynamo.config.cache_size_limit = 4096
    torch._dynamo.config.accumulated_cache_size_limit = 65536
    torch._dynamo.config.suppress_errors = True

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
    for i, item in enumerate(items, start=1):
        # BFCL `question` is `[[{"role": "user", "content": ...}, ...]]` -- one
        # nested conversation. Take the inner list verbatim.
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
        # Show the raw completion when parsing misses on the first few items --
        # if the model isn't emitting `<tool_call>` blocks at all we want to
        # see the actual output to diagnose chat-template / decoding bugs.
        if predicted is None and i <= 3:
            preview = completion[:200].replace("\n", " | ")
            print(f"          completion preview: {preview!r}", flush=True)

    accuracy = correct / len(items) if items else 0.0
    return {
        "adapter_path": adapter_path,
        "n": len(items),
        "correct": correct,
        "accuracy": accuracy,
        "per_item": per_item,
    }
