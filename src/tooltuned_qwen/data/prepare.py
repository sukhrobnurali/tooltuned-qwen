"""High-level dataset prep helpers used by the Colab launcher notebook.

Anything more elaborate than a one-liner over `load_xlam` / `load_hermes`
goes here so the notebook stays "no logic in cells" (brief section 6.2 item 3).
"""

from __future__ import annotations

import json
from typing import Any


def smoke_dataset(n: int = 8, *, source: str = "synthetic") -> Any:
    """Tiny xlam-shaped slice for the Phase 1.3 end-to-end smoke run.

    Defaults to a self-contained synthetic slice so the smoke gate doesn't
    block on per-user HF dataset access (xLAM is gated behind a one-click
    Salesforce form). xLAM access is still required for the Phase 2 dataset
    ablation -- pass `source="xlam"` once you have it.
    """
    if source == "synthetic":
        return _synthetic_xlam(n)
    if source == "xlam":
        from .load_xlam import load_xlam

        return load_xlam().select(range(n))
    if source == "hermes":
        from .load_hermes import load_hermes

        return load_hermes().select(range(n))
    raise ValueError(f"unknown smoke source: {source}")


def _synthetic_xlam(n: int) -> Any:
    """Hand-rolled xlam-shaped rows -- exercises the same format path as real xLAM."""
    from datasets import Dataset

    cities = ["Tokyo", "Lagos", "Quito", "Helsinki", "Lima", "Hanoi", "Riga", "Kyiv"]
    rows = [
        {
            "query": f"What's the weather like in {cities[i % len(cities)]} right now?",
            "tools": json.dumps(
                [
                    {
                        "name": "get_weather",
                        "description": "Get current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ]
            ),
            "answers": json.dumps(
                [{"name": "get_weather", "arguments": {"city": cities[i % len(cities)]}}]
            ),
        }
        for i in range(n)
    ]
    return Dataset.from_list(rows)
