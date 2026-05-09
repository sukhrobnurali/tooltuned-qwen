"""Generate MODEL_CARD.md from BFCL results + training config + version pins."""

from typing import Any


def generate_card(
    *,
    bfcl_results: dict[str, Any],
    training_config_path: str,
    out_path: str,
) -> str:
    raise NotImplementedError("Phase 5.1")
