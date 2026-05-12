"""Wrap the gorilla `bfcl` CLI for BFCL V4 evaluation.

This module shells out to `bfcl generate` (model produces tool-call
predictions) and then `bfcl evaluate` (canonical AST scorer), then folds
the per-category `score/MODEL/<section>/BFCL_v4_<cat>_score.json` files
into a single results dict that's compatible with
`hub.model_card.generate_card(bfcl_results=...)`.

The two heavy CLI steps need vLLM + a GPU; only the parsing and command
construction are unit-testable. The Phase 4 Colab notebook drives the
generate + evaluate steps; this wrapper centralises the conventions
(model-name suffixing, default test categories, output dir layout) so the
notebook stays a thin launcher.

Mode (`fc` vs `prompt`) is selected by the model-name suffix BFCL uses
internally: `<name>-FC` for function-calling mode, plain `<name>` for the
prompt-shaped mode. We normalise the suffix here so callers can pass the
base name and just say `mode="fc"`.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

# Test categories used for the Phase 4 main eval. Single-turn coverage --
# multi-turn / agentic / web-search / memory categories are out of scope
# for v1.0 since our LoRA only trained on single-turn xLAM data, and the
# multi-turn family takes ~5-10x the compute. Each category here maps to
# a `BFCL_v4_<name>_score.json` file under `score/MODEL/<section>/`.
DEFAULT_TEST_CATEGORIES: list[str] = [
    # Non-live (curated, static prompts). BFCL V4 splits `simple` by language
    # -- we evaluate Python only since xLAM is Python-centric (Java + JS are
    # available as `simple_java` / `simple_javascript`; bfcl also offers the
    # `non_python` collection alias for the cross-language pair).
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    # Live (real-world prompts collected in the wild)
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_relevance",
    "live_irrelevance",
]

_VALID_MODES = ("fc", "prompt")
_FC_SUFFIX = "-FC"


def _resolve_model_name(model: str, mode: str) -> str:
    """Normalise BFCL's `<name>-FC` vs `<name>` model-name convention.

    Always emit the canonical name for the requested mode so the rest of
    the runner (and the score-file paths it parses) can rely on it.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
    if mode == "fc":
        return model if model.endswith(_FC_SUFFIX) else model + _FC_SUFFIX
    # prompt mode: strip the suffix if present
    if model.endswith(_FC_SUFFIX):
        return model[: -len(_FC_SUFFIX)]
    return model


def _build_generate_cmd(
    *,
    model: str,
    test_categories: list[str],
    backend: str,
    local_model_path: str | None,
    lora_modules: dict[str, str] | None,
    extra_args: list[str] | None,
    bfcl_executable: str = "bfcl",
) -> list[str]:
    """Construct `bfcl generate ...` argv. BFCL's CLI takes a single
    comma-joined test-category arg, not repeated flags.

    `bfcl_executable` defaults to the bare `bfcl` (assumes PATH-resolvable),
    but on Colab we install bfcl-eval into an isolated venv to avoid a
    torch/vllm pin conflict with the training stack -- so the notebook
    passes the venv's `/content/bfcl-venv/bin/bfcl` here.
    """
    cmd: list[str] = [
        bfcl_executable,
        "generate",
        "--model",
        model,
        "--test-category",
        ",".join(test_categories),
        "--backend",
        backend,
    ]
    if local_model_path is not None:
        cmd += ["--local-model-path", local_model_path]
    if lora_modules:
        cmd.append("--enable-lora")
        for name, path in lora_modules.items():
            cmd += ["--lora-modules", f"{name}={path}"]
    if extra_args:
        cmd += list(extra_args)
    return cmd


def _run_streaming(
    cmd: list[str], *, cwd: Path, env: dict[str, str], label: str
) -> None:
    """Run `cmd` and stream its merged stdout+stderr line-by-line to our
    own stdout. Plain `subprocess.run(check=True)` is supposed to inherit
    parent streams, but Colab/IPython's stdout capture layer silently
    swallows the child output -- when bfcl crashes during model load we
    end up with a bare `CalledProcessError: exit status 1` and no clue
    why. This wrapper makes vLLM/bfcl logs visible in real time.
    """
    import sys

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    print(f"[{label}] done (exit 0)", file=sys.stdout, flush=True)


def _venv_subprocess_env(bfcl_executable: str) -> dict[str, str]:
    """Build subprocess env so bfcl can find its sibling binaries on PATH.

    `bfcl generate --backend vllm` shells out to `vllm serve ...` via a bare
    `subprocess.Popen(["vllm", ...])` -- PATH-lookup, not absolute path. When
    we invoke bfcl from an isolated venv at `/content/bfcl-venv/bin/bfcl`,
    `vllm` lives next door at `/content/bfcl-venv/bin/vllm` but isn't on the
    parent kernel's PATH, so the inner Popen raises `FileNotFoundError: vllm`.
    Prepend the venv's bin dir to PATH so the inner Popen finds it.

    Only prepends when the caller passed a path (has a directory component);
    bare `"bfcl"` leaves PATH alone so we don't accidentally shadow system
    binaries with whatever happens to be in cwd.
    """
    env = os.environ.copy()
    if os.path.dirname(bfcl_executable):
        exec_dir = str(Path(bfcl_executable).resolve().parent)
        env["PATH"] = f"{exec_dir}{os.pathsep}{env.get('PATH', '')}"
    return env


def _build_evaluate_cmd(
    *,
    model: str,
    test_categories: list[str],
    bfcl_executable: str = "bfcl",
) -> list[str]:
    return [
        bfcl_executable,
        "evaluate",
        "--model",
        model,
        "--test-category",
        ",".join(test_categories),
    ]


def _parse_score_summary(score_file: Path) -> dict[str, Any]:
    """Read line 1 of a BFCL `*_score.json` JSONL. Subsequent lines are
    per-item miss diagnostics, which we don't need for the headline."""
    with score_file.open("r", encoding="utf-8") as fh:
        first = fh.readline()
    return json.loads(first)


def _collect_per_category(score_dir: Path) -> dict[str, dict[str, Any]]:
    """Walk `score/<model>/<section>/BFCL_v4_<cat>_score.json`. Returns a
    flat `{category: {accuracy, correct, total, section}}` dict. Missing
    directory => empty dict; the caller decides if that's fatal."""
    if not score_dir.exists():
        return {}
    per_cat: dict[str, dict[str, Any]] = {}
    for section_dir in sorted(p for p in score_dir.iterdir() if p.is_dir()):
        for f in sorted(section_dir.glob("BFCL_v*_*_score.json")):
            # filename pattern: BFCL_v<N>_<category>_score.json
            stem = f.stem  # e.g. "BFCL_v4_simple_python_score"
            # strip leading "BFCL_v<N>_" and trailing "_score"
            parts = stem.split("_")
            if len(parts) < 4 or parts[0] != "BFCL" or parts[-1] != "score":
                continue
            category = "_".join(parts[2:-1])
            summary = _parse_score_summary(f)
            per_cat[category] = {
                "accuracy": float(summary.get("accuracy", 0.0)),
                "correct": int(summary.get("correct_count", 0)),
                "total": int(summary.get("total_count", 0)),
                "section": section_dir.name,
            }
    return per_cat


def _aggregate_overall(per_category: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Micro-average: `sum(correct) / sum(total)` across all evaluated
    categories. BFCL's leaderboard uses a macro average over sections,
    but for a single-number headline the micro average treats every
    prompt equally and is the honest summary of "how many BFCL items
    this model got right." Per-section/macro numbers stay available
    on the `per_category` dict for callers who need them."""
    correct = sum(cat["correct"] for cat in per_category.values())
    total = sum(cat["total"] for cat in per_category.values())
    accuracy = correct / total if total else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def run_bfcl(
    *,
    model: str,
    mode: str = "fc",
    out_dir: str | Path,
    test_categories: list[str] | None = None,
    bfcl_cwd: str | Path | None = None,
    backend: str = "vllm",
    local_model_path: str | None = None,
    lora_modules: dict[str, str] | None = None,
    extra_generate_args: list[str] | None = None,
    skip_generate: bool = False,
    skip_evaluate: bool = False,
    bfcl_executable: str = "bfcl",
) -> dict[str, Any]:
    """Run `bfcl generate` + `bfcl evaluate` and collect a flat results dict.

    `bfcl_cwd` is the working dir the BFCL CLI runs in -- the canonical
    evaluator writes `result/<model>/...` and `score/<model>/...` relative
    to its cwd, so we anchor here to find them deterministically. Defaults
    to the current directory if not set.

    `skip_generate` / `skip_evaluate` let the notebook split heavy GPU work
    onto Colab and run the parsing-only collection step locally afterwards.

    Returns a dict with: `model` (resolved name), `mode`, `n_total`,
    `evaluated_at` (ISO date), `overall` ({accuracy, correct, total}),
    `per_category` ({cat: {...}}), `score_dir`. A copy is also written to
    `out_dir/results.json` so the comparison step can pick it up later.
    """
    resolved = _resolve_model_name(model, mode)
    categories = test_categories or DEFAULT_TEST_CATEGORIES
    cwd = Path(bfcl_cwd) if bfcl_cwd is not None else Path.cwd()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    subprocess_env = _venv_subprocess_env(bfcl_executable)
    if not skip_generate:
        gen_cmd = _build_generate_cmd(
            model=resolved,
            test_categories=categories,
            backend=backend,
            local_model_path=local_model_path,
            lora_modules=lora_modules,
            extra_args=extra_generate_args,
            bfcl_executable=bfcl_executable,
        )
        _run_streaming(gen_cmd, cwd=cwd, env=subprocess_env, label="bfcl generate")

    if not skip_evaluate:
        eval_cmd = _build_evaluate_cmd(
            model=resolved,
            test_categories=categories,
            bfcl_executable=bfcl_executable,
        )
        _run_streaming(eval_cmd, cwd=cwd, env=subprocess_env, label="bfcl evaluate")

    score_dir = cwd / "score" / resolved
    per_category = _collect_per_category(score_dir)
    if not per_category:
        raise RuntimeError(
            f"no BFCL score files found under {score_dir} -- "
            "did `bfcl evaluate` complete successfully?"
        )
    overall = _aggregate_overall(per_category)

    results: dict[str, Any] = {
        "model": resolved,
        "mode": mode,
        "n_total": overall["total"],
        "evaluated_at": date.today().isoformat(),
        "overall": overall,
        "per_category": per_category,
        "score_dir": str(score_dir),
    }

    (out / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    return results
