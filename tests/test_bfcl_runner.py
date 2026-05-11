"""Tests for the Phase 4 gorilla `bfcl` CLI wrapper.

The wrapper itself shells out to `bfcl generate` + `bfcl evaluate`, which
need vLLM + a GPU and aren't unit-testable. These tests cover the
import-light building blocks: model-name resolution, subprocess-command
construction, and parsing of the `score/MODEL/<section>/BFCL_v4_<cat>_score.json`
files that the canonical evaluator emits. The end-to-end orchestrator is
verified via `run_bfcl(skip_generate=True, skip_evaluate=True, ...)` over
a fixture score directory built on `tmp_path`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tooltuned_qwen.eval.bfcl_runner import (
    _aggregate_overall,
    _build_evaluate_cmd,
    _build_generate_cmd,
    _collect_per_category,
    _parse_score_summary,
    _resolve_model_name,
    run_bfcl,
)


def test_resolve_model_name_appends_fc_suffix() -> None:
    assert _resolve_model_name("tooltuned-qwen-3.5-4b", "fc") == "tooltuned-qwen-3.5-4b-FC"


def test_resolve_model_name_is_idempotent_for_fc() -> None:
    """Caller might already supply the `-FC` suffix; don't double-append."""
    assert _resolve_model_name("tooltuned-qwen-3.5-4b-FC", "fc") == "tooltuned-qwen-3.5-4b-FC"


def test_resolve_model_name_leaves_prompt_mode_alone() -> None:
    assert _resolve_model_name("tooltuned-qwen-3.5-4b", "prompt") == "tooltuned-qwen-3.5-4b"


def test_resolve_model_name_strips_fc_for_prompt_mode() -> None:
    """If the caller supplied `-FC` but asked for prompt mode, the prompt
    variant is the one without the suffix -- the runner is the right place
    to normalise this rather than silently evaluating the wrong model."""
    assert (
        _resolve_model_name("tooltuned-qwen-3.5-4b-FC", "prompt")
        == "tooltuned-qwen-3.5-4b"
    )


def test_resolve_model_name_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        _resolve_model_name("model", "thinking")


def test_build_generate_cmd_basic_shape() -> None:
    cmd = _build_generate_cmd(
        model="tooltuned-qwen-3.5-4b-FC",
        test_categories=["simple_python", "multiple"],
        backend="vllm",
        local_model_path=None,
        lora_modules=None,
        extra_args=None,
    )
    assert cmd[0] == "bfcl"
    assert cmd[1] == "generate"
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "tooltuned-qwen-3.5-4b-FC"
    assert "--test-category" in cmd
    # BFCL CLI takes a comma-joined list, not repeated flags -- the README
    # example is `--test-category simple_python,parallel,live_multiple`.
    assert cmd[cmd.index("--test-category") + 1] == "simple_python,multiple"
    assert "--backend" in cmd
    assert cmd[cmd.index("--backend") + 1] == "vllm"


def test_build_generate_cmd_includes_local_model_path() -> None:
    cmd = _build_generate_cmd(
        model="tooltuned-qwen-3.5-4b-FC",
        test_categories=["simple_python"],
        backend="vllm",
        local_model_path="/content/qwen3.5-4b-base",
        lora_modules=None,
        extra_args=None,
    )
    assert "--local-model-path" in cmd
    assert cmd[cmd.index("--local-model-path") + 1] == "/content/qwen3.5-4b-base"


def test_build_generate_cmd_includes_lora_modules() -> None:
    cmd = _build_generate_cmd(
        model="tooltuned-qwen-3.5-4b-FC",
        test_categories=["simple_python"],
        backend="vllm",
        local_model_path="/content/qwen3.5-4b-base",
        lora_modules={"tooltuned": "/content/adapter"},
        extra_args=None,
    )
    assert "--enable-lora" in cmd
    assert "--lora-modules" in cmd
    # Format from BFCL README: `--lora-modules NAME=PATH`.
    idx = cmd.index("--lora-modules")
    assert cmd[idx + 1] == "tooltuned=/content/adapter"


def test_build_generate_cmd_passes_extra_args_through() -> None:
    cmd = _build_generate_cmd(
        model="m",
        test_categories=["simple_python"],
        backend="vllm",
        local_model_path=None,
        lora_modules=None,
        extra_args=["--num-gpus", "1", "--gpu-memory-utilization", "0.9"],
    )
    assert "--num-gpus" in cmd
    assert cmd[cmd.index("--num-gpus") + 1] == "1"
    assert "--gpu-memory-utilization" in cmd
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.9"


def test_build_evaluate_cmd_basic_shape() -> None:
    cmd = _build_evaluate_cmd(
        model="tooltuned-qwen-3.5-4b-FC",
        test_categories=["simple_python", "multiple"],
    )
    assert cmd[:2] == ["bfcl", "evaluate"]
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "tooltuned-qwen-3.5-4b-FC"
    assert "--test-category" in cmd
    assert cmd[cmd.index("--test-category") + 1] == "simple_python,multiple"


def test_parse_score_summary_reads_first_jsonl_line(tmp_path: Path) -> None:
    """BFCL evaluators emit JSONL with the summary `{accuracy, correct_count,
    total_count}` on line 1 and per-item miss diagnostics on subsequent lines.
    The parser must read only line 1; the misses are bulky and irrelevant to
    the headline number."""
    score_file = tmp_path / "BFCL_v4_simple_python_score.json"
    score_file.write_text(
        '{"accuracy": 0.955, "correct_count": 382, "total_count": 400}\n'
        '{"id": "simple_python_13", "valid": false, "error": ["nested type"]}\n'
        '{"id": "simple_python_51", "valid": false, "error": ["value err"]}\n',
        encoding="utf-8",
    )
    summary = _parse_score_summary(score_file)
    assert summary == {"accuracy": 0.955, "correct_count": 382, "total_count": 400}


def test_parse_score_summary_handles_extra_fields(tmp_path: Path) -> None:
    """Future BFCL versions may add fields (latency, cost, etc.); the parser
    must surface what's there rather than enforcing a closed schema."""
    score_file = tmp_path / "BFCL_v4_simple_python_score.json"
    score_file.write_text(
        '{"accuracy": 0.5, "correct_count": 1, "total_count": 2, "latency_mean": 4.2}\n',
        encoding="utf-8",
    )
    summary = _parse_score_summary(score_file)
    assert summary["accuracy"] == 0.5
    assert summary["latency_mean"] == 4.2


def _make_score_fixture(
    root: Path,
    model_name: str,
    *,
    cats: dict[str, tuple[float, int, int]],
) -> Path:
    """Build a `score/<model>/{non_live,live}/BFCL_v4_<cat>_score.json` tree.

    `cats` maps `<section>/<category>` -> `(accuracy, correct, total)`. The
    section determines the subdirectory, the category determines the filename.
    """
    score_dir = root / "score" / model_name
    for cat_path, (acc, correct, total) in cats.items():
        section, category = cat_path.split("/", 1)
        sub = score_dir / section
        sub.mkdir(parents=True, exist_ok=True)
        (sub / f"BFCL_v4_{category}_score.json").write_text(
            json.dumps(
                {"accuracy": acc, "correct_count": correct, "total_count": total}
            )
            + "\n",
            encoding="utf-8",
        )
    return score_dir


def test_collect_per_category_walks_section_subdirs(tmp_path: Path) -> None:
    score_dir = _make_score_fixture(
        tmp_path,
        "tooltuned-qwen-FC",
        cats={
            "non_live/simple_python": (0.6, 60, 100),
            "non_live/multiple": (0.4, 40, 100),
            "live/live_simple": (0.5, 50, 100),
        },
    )
    per_cat = _collect_per_category(score_dir)
    assert set(per_cat.keys()) == {"simple_python", "multiple", "live_simple"}
    assert per_cat["simple_python"]["accuracy"] == 0.6
    assert per_cat["simple_python"]["section"] == "non_live"
    assert per_cat["live_simple"]["section"] == "live"
    assert per_cat["multiple"]["total"] == 100


def test_collect_per_category_returns_empty_on_missing_dir(tmp_path: Path) -> None:
    """A model that hasn't been evaluated yet -- the runner shouldn't blow up
    on an absent directory; an empty dict lets the caller decide whether
    that's a fatal condition."""
    missing = tmp_path / "score" / "no-such-model"
    assert _collect_per_category(missing) == {}


def test_aggregate_overall_is_micro_average() -> None:
    """Overall accuracy is `sum(correct) / sum(total)` -- not the mean of
    per-category accuracies. BFCL's data_overall.csv computes the macro average
    of *sections*; for our scalar headline a micro average over the categories
    we actually evaluated is the honest number (treating every BFCL prompt
    equally regardless of which category bucket it falls into)."""
    per_cat = {
        "simple_python": {"accuracy": 0.9, "correct": 90, "total": 100, "section": "non_live"},
        "multiple": {"accuracy": 0.5, "correct": 50, "total": 100, "section": "non_live"},
        "live_simple": {"accuracy": 0.7, "correct": 35, "total": 50, "section": "live"},
    }
    overall = _aggregate_overall(per_cat)
    # 90 + 50 + 35 = 175 correct, 100 + 100 + 50 = 250 total => 0.70
    assert overall["accuracy"] == pytest.approx(0.70, abs=1e-9)
    assert overall["correct"] == 175
    assert overall["total"] == 250


def test_aggregate_overall_handles_empty_dict() -> None:
    overall = _aggregate_overall({})
    assert overall == {"accuracy": 0.0, "correct": 0, "total": 0}


def test_run_bfcl_skip_subprocess_just_parses_existing_results(tmp_path: Path) -> None:
    """The two-phase pattern (`bfcl generate` then `bfcl evaluate` on Colab,
    then `run_bfcl(skip_generate=True, skip_evaluate=True)` on the laptop to
    fold the score files into a model-card-compatible dict) is the path the
    Phase 4.3 notebook actually takes. Lock the parse-only behaviour now."""
    _make_score_fixture(
        tmp_path,
        "tooltuned-qwen-3.5-4b-FC",
        cats={
            "non_live/simple_python": (0.6, 60, 100),
            "live/live_simple": (0.5, 50, 100),
        },
    )
    out_dir = tmp_path / "out"
    results = run_bfcl(
        model="tooltuned-qwen-3.5-4b",
        mode="fc",
        out_dir=str(out_dir),
        bfcl_cwd=str(tmp_path),
        skip_generate=True,
        skip_evaluate=True,
    )

    assert results["model"] == "tooltuned-qwen-3.5-4b-FC"
    assert results["mode"] == "fc"
    assert results["n_total"] == 200
    assert results["overall"]["accuracy"] == pytest.approx(0.55, abs=1e-9)
    assert set(results["per_category"].keys()) == {"simple_python", "live_simple"}
    assert "evaluated_at" in results  # ISO date string from the parser

    # Result JSON should also be persisted under out_dir so the comparison
    # step can read it back without re-running.
    persisted = out_dir / "results.json"
    assert persisted.exists()
    on_disk = json.loads(persisted.read_text(encoding="utf-8"))
    assert on_disk["model"] == "tooltuned-qwen-3.5-4b-FC"
    assert on_disk["n_total"] == 200


def test_run_bfcl_raises_when_score_dir_is_empty(tmp_path: Path) -> None:
    """An empty score dir means generate or evaluate silently failed --
    returning a 0/0 result would let a broken run flow into the gate check
    as a false negative. Surface it loudly instead."""
    (tmp_path / "score" / "tooltuned-qwen-3.5-4b-FC").mkdir(parents=True)
    with pytest.raises(RuntimeError, match=r"no .* score files"):
        run_bfcl(
            model="tooltuned-qwen-3.5-4b",
            mode="fc",
            out_dir=str(tmp_path / "out"),
            bfcl_cwd=str(tmp_path),
            skip_generate=True,
            skip_evaluate=True,
        )


def test_run_bfcl_test_categories_default_to_known_set(tmp_path: Path) -> None:
    """When the caller doesn't pass `test_categories`, the runner uses a
    documented default set so reproductions don't accidentally diverge on
    coverage. The default lives in module scope as `DEFAULT_TEST_CATEGORIES`
    so a reviewer can audit it in one place."""
    from tooltuned_qwen.eval.bfcl_runner import DEFAULT_TEST_CATEGORIES

    assert isinstance(DEFAULT_TEST_CATEGORIES, list)
    assert len(DEFAULT_TEST_CATEGORIES) > 0
    # Every category must be a non-empty string -- a stray `None` here would
    # produce a malformed `--test-category ,foo` arg downstream.
    assert all(isinstance(c, str) and c for c in DEFAULT_TEST_CATEGORIES)


def _assert_subprocess_recorded(
    calls: list[list[str]], subcommand: str, model: str
) -> None:
    matched: list[list[str]] = []
    for cmd in calls:
        if cmd[:2] == ["bfcl", subcommand]:
            matched.append(cmd)
    assert matched, f"no bfcl {subcommand} call recorded for {model}"
    assert any(model in cmd for cmd in matched), (
        f"bfcl {subcommand} called but never with model={model}: {matched}"
    )


def test_run_bfcl_invokes_generate_and_evaluate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end orchestrator: when `skip_generate`/`skip_evaluate` are
    False, both subprocess calls fire with the resolved model name. Mock
    `subprocess.run` so the test stays GPU-free; verify the recorded calls
    instead of letting BFCL actually shell out."""
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        calls.append(cmd)
        # When `bfcl evaluate` "runs", fabricate the score files the parser
        # will look for so the orchestrator's post-step parse succeeds.
        if cmd[:2] == ["bfcl", "evaluate"]:
            _make_score_fixture(
                tmp_path,
                "tooltuned-qwen-3.5-4b-FC",
                cats={"non_live/simple_python": (1.0, 10, 10)},
            )

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Result()

    import tooltuned_qwen.eval.bfcl_runner as runner_mod

    monkeypatch.setattr(runner_mod.subprocess, "run", fake_run)

    out_dir = tmp_path / "out"
    results = run_bfcl(
        model="tooltuned-qwen-3.5-4b",
        mode="fc",
        out_dir=str(out_dir),
        bfcl_cwd=str(tmp_path),
        test_categories=["simple_python"],
        local_model_path="/content/qwen3.5-4b-base",
        lora_modules={"tooltuned": "/content/adapter"},
    )

    _assert_subprocess_recorded(calls, "generate", "tooltuned-qwen-3.5-4b-FC")
    _assert_subprocess_recorded(calls, "evaluate", "tooltuned-qwen-3.5-4b-FC")
    assert results["overall"]["accuracy"] == 1.0
