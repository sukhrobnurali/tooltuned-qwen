"""Wrap the gorilla `bfcl` CLI for BFCL V4 evaluation."""

from typing import Any


def run_bfcl(
    *,
    model: str,
    mode: str = "fc",
    out_dir: str,
) -> dict[str, Any]:
    raise NotImplementedError("Phase 4.1")
