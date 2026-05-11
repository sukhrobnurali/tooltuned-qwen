"""Tests for the Phase 4.3 base-vs-tuned comparison module.

`build_comparison` consumes two `run_bfcl()` output dicts (one from the
base model, one from the adapter) and emits the model-card-compatible
results dict + a markdown table + a PNG bar chart. The schema it returns
is the same one `hub.model_card.generate_card(bfcl_results=...)` expects,
locked at the top of `hub/model_card.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tooltuned_qwen.eval.compare import build_comparison


def _fake_run_results(
    *,
    model_name: str,
    per_category: dict[str, tuple[float, int, int]],
    evaluated_at: str = "2026-05-15",
    mode: str = "fc",
) -> dict[str, object]:
    """Shape returned by `bfcl_runner.run_bfcl`. Keep this synced with that
    schema -- if the runner changes, this fixture (and these tests) update too."""
    correct_total = sum(c for _, c, _ in per_category.values())
    total_total = sum(t for _, _, t in per_category.values())
    return {
        "model": model_name,
        "mode": mode,
        "n_total": total_total,
        "evaluated_at": evaluated_at,
        "overall": {
            "accuracy": correct_total / total_total if total_total else 0.0,
            "correct": correct_total,
            "total": total_total,
        },
        "per_category": {
            cat: {"accuracy": acc, "correct": c, "total": t, "section": "non_live"}
            for cat, (acc, c, t) in per_category.items()
        },
        "score_dir": f"/tmp/fake/score/{model_name}",
    }


def test_build_comparison_returns_model_card_schema(tmp_path: Path) -> None:
    """The returned dict must match `hub/model_card.py`'s documented schema
    so `generate_card(bfcl_results=...)` consumes it verbatim. The schema
    is: overall_base, overall_tuned, delta, n_total, evaluated_at, rows
    (+ optional base_model)."""
    base = _fake_run_results(
        model_name="Qwen_Qwen3.5-4B",
        per_category={"simple": (0.50, 50, 100), "multiple": (0.40, 40, 100)},
    )
    tuned = _fake_run_results(
        model_name="tooltuned-qwen-3.5-4b-FC",
        per_category={"simple": (0.60, 60, 100), "multiple": (0.50, 50, 100)},
    )

    results = build_comparison(
        base_results=base,
        tuned_results=tuned,
        out_dir=str(tmp_path),
        base_model_name="Qwen/Qwen3.5-4B",
    )

    assert results["overall_base"] == pytest.approx(0.45)
    assert results["overall_tuned"] == pytest.approx(0.55)
    assert results["delta"] == pytest.approx(0.10)
    assert results["n_total"] == 200
    assert results["evaluated_at"] == "2026-05-15"
    assert results["base_model"] == "Qwen/Qwen3.5-4B"
    assert isinstance(results["rows"], list)


def test_build_comparison_rows_sorted_alphabetically(tmp_path: Path) -> None:
    """Per-category rows must come out in a stable, predictable order so the
    rendered table doesn't change shape between runs that evaluated the same
    categories. Alphabetic is the simplest stable ordering."""
    base = _fake_run_results(
        model_name="base",
        per_category={"parallel": (0.4, 40, 100), "simple": (0.5, 50, 100)},
    )
    tuned = _fake_run_results(
        model_name="tuned",
        per_category={"parallel": (0.5, 50, 100), "simple": (0.6, 60, 100)},
    )
    results = build_comparison(
        base_results=base,
        tuned_results=tuned,
        out_dir=str(tmp_path),
    )
    categories = [r["category"] for r in results["rows"]]
    assert categories == sorted(categories)


def test_build_comparison_rows_carry_per_category_pair(tmp_path: Path) -> None:
    base = _fake_run_results(
        model_name="base",
        per_category={"simple": (0.5, 50, 100), "multiple": (0.4, 40, 100)},
    )
    tuned = _fake_run_results(
        model_name="tuned",
        per_category={"simple": (0.6, 60, 100), "multiple": (0.5, 50, 100)},
    )
    results = build_comparison(
        base_results=base, tuned_results=tuned, out_dir=str(tmp_path)
    )
    rows_by_cat = {r["category"]: r for r in results["rows"]}
    assert rows_by_cat["simple"]["base"] == pytest.approx(0.5)
    assert rows_by_cat["simple"]["tuned"] == pytest.approx(0.6)
    assert rows_by_cat["multiple"]["base"] == pytest.approx(0.4)
    assert rows_by_cat["multiple"]["tuned"] == pytest.approx(0.5)


def test_build_comparison_drops_categories_missing_from_either_arm(
    tmp_path: Path,
) -> None:
    """If one model has a category the other doesn't, the pair is incomparable.
    Surface that as a dropped row rather than silently scoring zero on the
    missing side -- which would let an asymmetric run quietly understate the
    base or overstate the tuned."""
    base = _fake_run_results(
        model_name="base",
        per_category={"simple": (0.5, 50, 100), "multiple": (0.4, 40, 100)},
    )
    tuned = _fake_run_results(
        model_name="tuned",
        per_category={"simple": (0.6, 60, 100)},  # no multiple
    )
    results = build_comparison(
        base_results=base, tuned_results=tuned, out_dir=str(tmp_path)
    )
    categories = {r["category"] for r in results["rows"]}
    assert categories == {"simple"}


def test_build_comparison_writes_markdown_table(tmp_path: Path) -> None:
    base = _fake_run_results(
        model_name="base", per_category={"simple": (0.50, 50, 100)}
    )
    tuned = _fake_run_results(
        model_name="tuned", per_category={"simple": (0.55, 55, 100)}
    )
    build_comparison(base_results=base, tuned_results=tuned, out_dir=str(tmp_path))

    md = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "| Category |" in md
    assert "simple" in md
    assert "50.0%" in md  # base
    assert "55.0%" in md  # tuned
    assert "+5.0pp" in md  # delta with explicit sign


def test_build_comparison_markdown_uses_minus_sign_for_regression(
    tmp_path: Path,
) -> None:
    """A regression must visibly read as negative -- a sign flip here would
    make a worse model look like a better one in the published artifact."""
    base = _fake_run_results(
        model_name="base", per_category={"simple": (0.60, 60, 100)}
    )
    tuned = _fake_run_results(
        model_name="tuned", per_category={"simple": (0.55, 55, 100)}
    )
    build_comparison(base_results=base, tuned_results=tuned, out_dir=str(tmp_path))
    md = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "-5.0pp" in md
    assert "+5.0pp" not in md


def test_build_comparison_writes_results_json(tmp_path: Path) -> None:
    """JSON sidecar so the comparison can be re-rendered into the model card
    later without rerunning anything."""
    base = _fake_run_results(
        model_name="base", per_category={"simple": (0.5, 50, 100)}
    )
    tuned = _fake_run_results(
        model_name="tuned", per_category={"simple": (0.6, 60, 100)}
    )
    returned = build_comparison(
        base_results=base, tuned_results=tuned, out_dir=str(tmp_path)
    )
    on_disk = json.loads(
        (tmp_path / "bfcl_results.json").read_text(encoding="utf-8")
    )
    assert on_disk["overall_base"] == returned["overall_base"]
    assert on_disk["overall_tuned"] == returned["overall_tuned"]
    assert on_disk["delta"] == returned["delta"]


def test_build_comparison_writes_bar_chart_png(tmp_path: Path) -> None:
    """The chart is the visual hero of the model card (brief 11 q13); a
    broken PNG would silently degrade the artifact. Check the magic bytes
    so we know matplotlib actually rendered something valid, not an empty
    or corrupted file."""
    base = _fake_run_results(
        model_name="base", per_category={"simple": (0.5, 50, 100)}
    )
    tuned = _fake_run_results(
        model_name="tuned", per_category={"simple": (0.6, 60, 100)}
    )
    build_comparison(base_results=base, tuned_results=tuned, out_dir=str(tmp_path))
    png = tmp_path / "bfcl_comparison.png"
    assert png.exists()
    header = png.read_bytes()[:8]
    # PNG magic: 89 50 4E 47 0D 0A 1A 0A
    assert header == b"\x89PNG\r\n\x1a\n"


def test_build_comparison_dict_round_trips_through_model_card(tmp_path: Path) -> None:
    """End-to-end sanity: the dict `build_comparison` returns must flow into
    `generate_card(bfcl_results=...)` cleanly, producing a card whose
    headline matches the comparison's delta -- this is the contract the
    Phase 4.3 -> 4.4 -> push pipeline depends on."""
    from tooltuned_qwen.hub.model_card import generate_card

    repo_root = Path(__file__).resolve().parents[1]
    default_yaml = repo_root / "configs" / "default.yaml"

    base = _fake_run_results(
        model_name="Qwen/Qwen3.5-4B",
        per_category={"simple": (0.503, 503, 1000), "multiple": (0.40, 400, 1000)},
        evaluated_at="2026-05-12",
    )
    tuned = _fake_run_results(
        model_name="tooltuned-qwen-3.5-4b-FC",
        per_category={"simple": (0.547, 547, 1000), "multiple": (0.48, 480, 1000)},
        evaluated_at="2026-05-12",
    )
    results = build_comparison(
        base_results=base,
        tuned_results=tuned,
        out_dir=str(tmp_path),
        base_model_name="Qwen/Qwen3.5-4B",
    )
    card_path = tmp_path / "MODEL_CARD.md"
    generate_card(
        bfcl_results=results,
        training_config_path=str(default_yaml),
        out_path=str(card_path),
    )
    body = card_path.read_text(encoding="utf-8")
    # Tuned headline: (547 + 480) / 2000 = 51.35% -> rounded "51.3%" or "51.4%"
    # depending on matplotlib-agnostic rounding; assert the integer part.
    assert "51." in body
    assert "TBD (Phase 4)" not in body
    # Per-category rows should populate the breakdown table.
    assert "simple" in body and "multiple" in body
