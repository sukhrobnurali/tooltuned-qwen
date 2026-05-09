"""Build the base-vs-tuned comparison table and bar chart."""

from typing import Any


def build_comparison(
    *,
    base_results: dict[str, Any],
    tuned_results: dict[str, Any],
    out_dir: str,
) -> dict[str, Any]:
    raise NotImplementedError("Phase 4.3")
