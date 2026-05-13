"""Tests for the in-tree BFCL parser + scorer + multi-category aggregator.

The heavy `run_bfcl_holdout` / `run_bfcl_full` need Unsloth + a GPU; these
tests cover the import-light pieces -- parsing, AST scoring, decision-to-call
scoring, the HF-mirror loader (HTTP mocked), and the result-shape helper that
hands a dict to `eval/compare.build_comparison`. Catching a regression here
matters because Phase 4's headline is only as trustworthy as the scorer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tooltuned_qwen.eval.bfcl_holdout import (
    BFCL_CATEGORIES,
    DEFAULT_CATEGORIES,
    load_bfcl_categories,
    parse_tool_call,
    parse_tool_calls,
    score_item,
    score_must_call,
    score_must_not_call,
    score_prediction,
    score_prediction_multi,
    shape_results,
)


def test_parse_tool_call_extracts_wrapped_block() -> None:
    completion = (
        'I will call the function.\n<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "Tokyo"}}\n</tool_call>'
    )
    parsed = parse_tool_call(completion)
    assert parsed == {"name": "get_weather", "arguments": {"city": "Tokyo"}}


def test_parse_tool_call_handles_string_arguments() -> None:
    """Some checkpoints emit `arguments` as a JSON-encoded string -- decode it
    so downstream scoring sees a dict, matching the dict-not-string convention
    we adopted in the formatter for Qwen 3.5's chat template."""
    completion = (
        '<tool_call>{"name": "f", "arguments": "{\\"x\\": 1}"}</tool_call>'
    )
    parsed = parse_tool_call(completion)
    assert parsed == {"name": "f", "arguments": {"x": 1}}


def test_parse_tool_call_falls_back_to_bare_json() -> None:
    completion = '{"name": "math.factorial", "arguments": {"number": 5}}'
    parsed = parse_tool_call(completion)
    assert parsed is not None
    assert parsed["name"] == "math.factorial"
    assert parsed["arguments"] == {"number": 5}


def test_parse_tool_call_returns_none_on_garbage() -> None:
    assert parse_tool_call("the model just talks") is None
    assert parse_tool_call("<tool_call>{bad json</tool_call>") is None


def test_parse_tool_call_extracts_xml_tag_form() -> None:
    """Our Qwen 3.5 4B fine-tune emits XML-tag tool calls
    (`<function=NAME>...<parameter=K>V</parameter></function>`) rather than
    the JSON-payload form the chat template's `<tool_call>` block expects.
    Confirmed during the Phase 2 BFCL holdout run -- see ADR 0003."""
    completion = (
        "<tool_call>\n"
        "<function=calculate_triangle_area>\n"
        "<parameter=base>\n10\n</parameter>\n"
        "<parameter=height>\n5\n</parameter>\n"
        "<parameter=unit>\nunits\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    parsed = parse_tool_call(completion)
    assert parsed is not None
    assert parsed["name"] == "calculate_triangle_area"
    # Numeric values are JSON-decoded; unquoted strings stay as strings.
    assert parsed["arguments"] == {"base": 10, "height": 5, "unit": "units"}


def test_parse_tool_call_xml_form_without_tool_call_wrapper() -> None:
    """Some completions skip the outer `<tool_call>` tag and emit the
    `<function=...>...</function>` block directly -- still parseable."""
    completion = (
        "<function=math.factorial>\n"
        "<parameter=number>\n5\n</parameter>\n"
        "</function>"
    )
    parsed = parse_tool_call(completion)
    assert parsed == {"name": "math.factorial", "arguments": {"number": 5}}


def test_parse_tool_call_xml_form_handles_dotted_function_names() -> None:
    completion = (
        "<function=math.hypot>"
        "<parameter=x>4</parameter>"
        "<parameter=y>5</parameter>"
        "</function>"
    )
    parsed = parse_tool_call(completion)
    assert parsed is not None
    assert parsed["name"] == "math.hypot"
    assert parsed["arguments"] == {"x": 4, "y": 5}


def test_score_prediction_exact_match() -> None:
    gt = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
    pred = {
        "name": "calculate_triangle_area",
        "arguments": {"base": 10, "height": 5, "unit": "units"},
    }
    assert score_prediction(pred, gt)


def test_score_prediction_optional_arg_can_be_omitted() -> None:
    gt = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
    # `unit` is optional (empty-string sentinel in accepted list); a model that
    # omits it must still score as correct.
    pred = {"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}
    assert score_prediction(pred, gt)


def test_score_prediction_wrong_function_name_misses() -> None:
    gt = [{"math.factorial": {"number": [5]}}]
    pred = {"name": "math.fact", "arguments": {"number": 5}}
    assert not score_prediction(pred, gt)


def test_score_prediction_wrong_value_misses() -> None:
    gt = [{"math.factorial": {"number": [5]}}]
    pred = {"name": "math.factorial", "arguments": {"number": 7}}
    assert not score_prediction(pred, gt)


def test_score_prediction_string_coerced_value_matches() -> None:
    """BFCL's simple scorer tolerates `5` vs `"5"` slack between numeric forms."""
    gt = [{"math.factorial": {"number": [5]}}]
    pred = {"name": "math.factorial", "arguments": {"number": "5"}}
    assert score_prediction(pred, gt)


def test_score_prediction_missing_required_arg_misses() -> None:
    gt = [{"calculate_triangle_area": {"base": [10], "height": [5]}}]
    pred = {"name": "calculate_triangle_area", "arguments": {"base": 10}}
    assert not score_prediction(pred, gt)


def test_score_prediction_extra_unrequested_arg_is_tolerated() -> None:
    gt = [{"math.factorial": {"number": [5]}}]
    pred = {"name": "math.factorial", "arguments": {"number": 5, "verbose": True}}
    assert score_prediction(pred, gt)


def test_score_prediction_none_is_a_miss() -> None:
    gt = [{"math.factorial": {"number": [5]}}]
    assert not score_prediction(None, gt)


def test_score_prediction_alternative_acceptable_values() -> None:
    """Per-arg acceptance is a list -- any one of the alternatives must match."""
    gt = [{"f": {"unit": ["m", "meter", "metre"]}}]
    for unit in ("m", "meter", "metre"):
        assert score_prediction({"name": "f", "arguments": {"unit": unit}}, gt)
    assert not score_prediction({"name": "f", "arguments": {"unit": "feet"}}, gt)


# --------------------------------------------------------------------------
# Multi-call parsing + scoring (S7: parallel / live_* / relevance coverage)
# --------------------------------------------------------------------------


def test_parse_tool_calls_extracts_two_blocks() -> None:
    """`parallel` category prompts make the model emit 2+ `<tool_call>`
    blocks; the multi-call parser must surface them all so set-overlap
    scoring can see every emitted call."""
    completion = (
        '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>\n'
        '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
    )
    calls = parse_tool_calls(completion)
    assert calls == [
        {"name": "a", "arguments": {"x": 1}},
        {"name": "b", "arguments": {"y": 2}},
    ]


def test_parse_tool_calls_extracts_multiple_xml_blocks() -> None:
    """The XML-tag form our fine-tune emits also goes multi-call: two
    `<function=...>...</function>` blocks inside one or more `<tool_call>`
    wrappers, or bare. Both should land in the list."""
    completion = (
        "<function=alpha><parameter=x>1</parameter></function>\n"
        "<function=beta><parameter=y>2</parameter></function>"
    )
    calls = parse_tool_calls(completion)
    assert calls == [
        {"name": "alpha", "arguments": {"x": 1}},
        {"name": "beta", "arguments": {"y": 2}},
    ]


def test_parse_tool_calls_empty_on_plain_prose() -> None:
    """`must_not_call` categories need this case to score correctly:
    a model that just talks (declines the tool call) yields zero calls."""
    assert parse_tool_calls("I cannot help with that using the offered tools.") == []


def test_parse_tool_call_singular_preserves_legacy_contract() -> None:
    """The singular form is what Phase 2's `run_bfcl_holdout` calls; it
    must keep returning the FIRST call (or None), unchanged after the
    plural refactor."""
    completion = (
        '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>\n'
        '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
    )
    assert parse_tool_call(completion) == {"name": "a", "arguments": {"x": 1}}


def test_score_prediction_multi_returns_true_when_any_call_matches() -> None:
    gt = [{"f": {"x": [1]}}]
    calls = [
        {"name": "g", "arguments": {"y": 2}},  # wrong
        {"name": "f", "arguments": {"x": 1}},  # right
    ]
    assert score_prediction_multi(calls, gt)


def test_score_prediction_multi_misses_when_no_calls_match() -> None:
    gt = [{"f": {"x": [1]}}]
    calls = [{"name": "g", "arguments": {"y": 2}}]
    assert not score_prediction_multi(calls, gt)


def test_score_prediction_multi_misses_on_empty_list() -> None:
    """`parallel*` items with zero predicted calls are unambiguous misses
    -- guard against an empty-list `any(...)` returning the wrong default."""
    assert not score_prediction_multi([], [{"f": {"x": [1]}}])


def test_score_must_call_and_must_not_call() -> None:
    """Decision-to-call categories: `live_relevance` wants any call,
    `irrelevance`/`live_irrelevance` want none."""
    assert score_must_call([{"name": "f", "arguments": {}}])
    assert not score_must_call([])
    assert score_must_not_call([])
    assert not score_must_not_call([{"name": "f", "arguments": {}}])


def test_score_item_dispatches_by_category_mode() -> None:
    """`score_item` reads the category off the row and dispatches to the
    right scorer; the test pins the mapping so a future BFCL_CATEGORIES
    typo can't silently route, say, `irrelevance` through AST scoring."""
    ast_item = {
        "id": "simple_0",
        "category": "simple",
        "ground_truth": [{"f": {"x": [1]}}],
    }
    assert score_item(ast_item, [{"name": "f", "arguments": {"x": 1}}])
    assert not score_item(ast_item, [{"name": "g", "arguments": {}}])

    rel_item = {"id": "live_relevance_0", "category": "live_relevance"}
    assert score_item(rel_item, [{"name": "anything", "arguments": {}}])
    assert not score_item(rel_item, [])

    irrel_item = {"id": "irrelevance_0", "category": "irrelevance"}
    assert score_item(irrel_item, [])
    assert not score_item(irrel_item, [{"name": "anything", "arguments": {}}])


def test_score_item_unknown_category_raises() -> None:
    with pytest.raises(ValueError, match="unknown BFCL category"):
        score_item({"id": "x", "category": "made_up"}, [])


# --------------------------------------------------------------------------
# Multi-category loader + result-shape helper
# --------------------------------------------------------------------------


def test_default_categories_match_table() -> None:
    """`DEFAULT_CATEGORIES` is the public list run_bfcl_full uses; the
    BFCL_CATEGORIES table is the single source of truth. If either drifts
    from the other a notebook will silently skip categories."""
    assert [cat for cat, _, _ in BFCL_CATEGORIES] == DEFAULT_CATEGORIES
    assert len(DEFAULT_CATEGORIES) == 11
    modes = {mode for _, mode, _ in BFCL_CATEGORIES}
    assert modes == {"ast_match", "must_call", "must_not_call"}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def test_load_bfcl_categories_tags_rows_and_truncates(tmp_path: Path) -> None:
    """The loader: (a) tags each row with its category, (b) honors
    `n_per_cat`, (c) leaves ground_truth as None for the relevance
    categories (no possible_answer file on HF for those)."""
    cache = tmp_path / "cache"

    def fake_download(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "BFCL_v3_simple.json" in url and "possible_answer" not in url:
            _write_jsonl(
                dest,
                [
                    {"id": "simple_0", "question": [[]], "function": []},
                    {"id": "simple_1", "question": [[]], "function": []},
                    {"id": "simple_2", "question": [[]], "function": []},
                ],
            )
        elif "possible_answer/BFCL_v3_simple.json" in url:
            _write_jsonl(
                dest,
                [
                    {"id": "simple_0", "ground_truth": [{"f": {"x": [1]}}]},
                    {"id": "simple_1", "ground_truth": [{"f": {"x": [2]}}]},
                    # simple_2 deliberately missing -- the loader should skip it.
                ],
            )
        elif "BFCL_v3_irrelevance.json" in url:
            _write_jsonl(
                dest,
                [
                    {"id": "irrelevance_0", "question": [[]], "function": []},
                    {"id": "irrelevance_1", "question": [[]], "function": []},
                ],
            )
        else:
            raise AssertionError(f"unexpected URL: {url}")

    with patch("tooltuned_qwen.eval.bfcl_holdout._download", side_effect=fake_download):
        rows = load_bfcl_categories(
            ["simple", "irrelevance"], n_per_cat=2, cache_dir=cache
        )

    # simple_2 dropped (no GT). simple capped at 2. irrelevance capped at 2.
    assert [r["id"] for r in rows] == [
        "simple_0",
        "simple_1",
        "irrelevance_0",
        "irrelevance_1",
    ]
    assert [r["category"] for r in rows] == [
        "simple",
        "simple",
        "irrelevance",
        "irrelevance",
    ]
    # AST rows carry ground_truth; relevance rows do not.
    assert rows[0]["ground_truth"] == [{"f": {"x": [1]}}]
    assert rows[2]["ground_truth"] is None


def test_shape_results_matches_runner_dict_shape() -> None:
    """The shape MUST match `bfcl_runner.run_bfcl`'s output dict so
    `eval/compare.build_comparison` consumes it unchanged. Schema:
    `{model, mode, n_total, evaluated_at, overall, per_category}`."""
    per_item = [
        {"id": "simple_0", "category": "simple", "correct": True},
        {"id": "simple_1", "category": "simple", "correct": False},
        {"id": "irrelevance_0", "category": "irrelevance", "correct": True},
        {"id": "live_relevance_0", "category": "live_relevance", "correct": True},
        {"id": "live_relevance_1", "category": "live_relevance", "correct": False},
    ]
    out = shape_results(
        model_name="my-adapter", per_item=per_item, evaluated_at="2026-05-13"
    )
    assert out["model"] == "my-adapter"
    assert out["mode"] == "intree"
    assert out["evaluated_at"] == "2026-05-13"
    assert out["n_total"] == 5
    assert out["overall"] == {"accuracy": 3 / 5, "correct": 3, "total": 5}
    assert out["per_category"]["simple"] == {
        "accuracy": 0.5,
        "correct": 1,
        "total": 2,
        "section": "non_live",
    }
    assert out["per_category"]["irrelevance"]["section"] == "non_live"
    assert out["per_category"]["live_relevance"]["section"] == "live"


def test_shape_results_feeds_compare_unchanged() -> None:
    """End-to-end: two shape_results outputs flow into build_comparison
    without translation, and the model-card-schema dict comes out the
    other side. Locks the producer/consumer contract across modules."""
    from tooltuned_qwen.eval.compare import build_comparison

    base = shape_results(
        model_name="base",
        per_item=[
            {"id": "simple_0", "category": "simple", "correct": False},
            {"id": "simple_1", "category": "simple", "correct": True},
            {"id": "irrelevance_0", "category": "irrelevance", "correct": True},
        ],
        evaluated_at="2026-05-13",
    )
    tuned = shape_results(
        model_name="tuned",
        per_item=[
            {"id": "simple_0", "category": "simple", "correct": True},
            {"id": "simple_1", "category": "simple", "correct": True},
            {"id": "irrelevance_0", "category": "irrelevance", "correct": True},
        ],
        evaluated_at="2026-05-13",
    )

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        results = build_comparison(
            base_results=base, tuned_results=tuned, out_dir=td
        )
    # Model-card schema fields all populated. Base got 2/3 (simple_1 +
    # irrelevance_0), tuned got 3/3.
    assert results["overall_base"] == pytest.approx(2 / 3)
    assert results["overall_tuned"] == pytest.approx(1.0)
    assert results["delta"] == pytest.approx(1 / 3)
    assert results["n_total"] == 3
    assert {r["category"] for r in results["rows"]} == {"simple", "irrelevance"}
