"""Build the base-vs-tuned comparison table, bar chart, and results JSON.

Consumes two `run_bfcl()` output dicts (base + adapter) and emits the
flat results dict `hub/model_card.py` reads (schema documented at the
top of that file). Side-effect outputs land under `out_dir`:

  - `bfcl_results.json` -- the dict that flows into the model card.
  - `comparison.md`     -- human-readable markdown table.
  - `bfcl_comparison.png` -- bar chart (hero image, brief 11 q13).

Categories that aren't present in BOTH arms are dropped. Silently scoring
them as 0 on the missing side would let an asymmetric eval flow into the
published artifact -- e.g. if the base run skipped `live_relevance` due
to an API hiccup but the tuned run included it, the delta would inflate
just because the base "got 0" on the missing rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


def build_comparison(
    *,
    base_results: dict[str, Any],
    tuned_results: dict[str, Any],
    out_dir: str | Path,
    base_model_name: str | None = None,
    title: str = "BFCL V4 — base vs. fine-tuned Qwen 3.5 4B",
) -> dict[str, Any]:
    """Render the comparison artifacts and return the model-card dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_per_cat = cast(dict[str, dict[str, Any]], base_results.get("per_category", {}))
    tuned_per_cat = cast(
        dict[str, dict[str, Any]], tuned_results.get("per_category", {})
    )
    common = sorted(set(base_per_cat) & set(tuned_per_cat))

    rows: list[dict[str, Any]] = []
    base_correct = base_total = 0
    tuned_correct = tuned_total = 0
    for cat in common:
        b = base_per_cat[cat]
        t = tuned_per_cat[cat]
        rows.append(
            {
                "category": cat,
                "base": float(b["accuracy"]),
                "tuned": float(t["accuracy"]),
            }
        )
        base_correct += int(b["correct"])
        base_total += int(b["total"])
        tuned_correct += int(t["correct"])
        tuned_total += int(t["total"])

    overall_base = base_correct / base_total if base_total else 0.0
    overall_tuned = tuned_correct / tuned_total if tuned_total else 0.0
    delta = overall_tuned - overall_base

    # `evaluated_at` -- prefer the tuned date; the card is "about" the adapter
    # and that's the more recent run in practice. Falls back to base if the
    # tuned dict didn't carry one.
    evaluated_at = tuned_results.get("evaluated_at") or base_results.get("evaluated_at")

    results: dict[str, Any] = {
        "overall_base": overall_base,
        "overall_tuned": overall_tuned,
        "delta": delta,
        "n_total": tuned_total,
        "evaluated_at": evaluated_at,
        "rows": rows,
        "base_model": base_model_name or base_results.get("model"),
    }

    (out / "bfcl_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "comparison.md").write_text(
        _render_markdown(results), encoding="utf-8"
    )
    _render_chart(
        rows=rows,
        overall_base=overall_base,
        overall_tuned=overall_tuned,
        out_path=out / "bfcl_comparison.png",
        title=title,
    )
    return results


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fmt_delta(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x * 100:.1f}pp"


def _render_markdown(results: dict[str, Any]) -> str:
    base_model = results.get("base_model") or "base"
    lines: list[str] = [
        "# BFCL V4 comparison\n",
        f"_Evaluated {results.get('evaluated_at') or 'n/a'}, "
        f"n={results['n_total']}_\n",
        "| Model | Overall accuracy |",
        "| --- | --- |",
        f"| Base ({base_model}) | {_fmt_pct(results['overall_base'])} |",
        f"| **Fine-tuned adapter** | **{_fmt_pct(results['overall_tuned'])}** |",
        f"| Delta | **{_fmt_delta(results['delta'])}** |",
        "",
        "## Per-category breakdown\n",
        "| Category | Base | Tuned | Delta |",
        "| --- | --- | --- | --- |",
    ]
    for r in results["rows"]:
        d = r["tuned"] - r["base"]
        lines.append(
            f"| {r['category']} | {_fmt_pct(r['base'])} | "
            f"{_fmt_pct(r['tuned'])} | {_fmt_delta(d)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_chart(
    *,
    rows: list[dict[str, Any]],
    overall_base: float,
    overall_tuned: float,
    out_path: Path,
    title: str,
) -> None:
    """Bar chart: per-category base vs tuned, plus an "overall" pair at the
    end. Lazy matplotlib import so test collection (and non-chart paths)
    don't pay the matplotlib startup cost."""
    # `Agg` backend = headless rendering, no display required (Colab + CI safe).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = [r["category"] for r in rows] + ["overall"]
    base_vals = [r["base"] * 100 for r in rows] + [overall_base * 100]
    tuned_vals = [r["tuned"] * 100 for r in rows] + [overall_tuned * 100]

    x = list(range(len(categories)))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(6.4, 0.8 * len(categories) + 2), 4.5))
    ax.bar([i - width / 2 for i in x], base_vals, width, label="Base", color="#888888")
    ax.bar(
        [i + width / 2 for i in x],
        tuned_vals,
        width,
        label="Fine-tuned",
        color="#3b7dd8",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
