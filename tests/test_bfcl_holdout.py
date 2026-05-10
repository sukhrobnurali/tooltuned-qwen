"""Tests for the Phase 2 BFCL-holdout parser + scorer.

The full `run_bfcl_holdout` needs Unsloth + a GPU; these tests cover only
the import-light building blocks (parsing model output, scoring against
BFCL's `[{<name>: {<arg>: [accepted...]}}]` ground-truth shape). Catching a
parser regression cheaply matters because the ablation result is only as
trustworthy as the scorer.
"""

from __future__ import annotations

from tooltuned_qwen.eval.bfcl_holdout import parse_tool_call, score_prediction


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
