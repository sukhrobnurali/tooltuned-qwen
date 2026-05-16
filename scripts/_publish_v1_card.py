"""One-shot v1.0 publish: regenerate MODEL_CARD.md + upload to HF main repo.

Loads the committed comparison artifact, renders the model card with the
gate-disclosure callout active (delta < +3pp gate), and uploads card +
chart to https://huggingface.co/sukhrobnurali/tooltuned-qwen-3.5-4b.

Run once locally with `HF_TOKEN` set:

    HF_TOKEN=hf_... uv run python scripts/_publish_v1_card.py

Idempotent -- re-runs overwrite cleanly. Uses `HfApi(token=...)` per the
Phase 1.3 finding #7 pattern (sticky HF token cache): no global state.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

from tooltuned_qwen.hub.model_card import generate_card

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = REPO_ROOT / "results" / "bfcl_intree" / "comparison" / "bfcl_results.json"
COMPARISON_PNG = REPO_ROOT / "results" / "bfcl_intree" / "comparison" / "bfcl_comparison.png"
CONFIG_YAML = REPO_ROOT / "configs" / "default.yaml"
CARD_OUT = REPO_ROOT / "MODEL_CARD.md"

HF_REPO = "sukhrobnurali/tooltuned-qwen-3.5-4b"


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("error: HF_TOKEN env var is required", file=sys.stderr)
        return 2

    bfcl_results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))

    generate_card(
        bfcl_results=bfcl_results,
        training_config_path=str(CONFIG_YAML),
        out_path=str(CARD_OUT),
        hf_repo=HF_REPO,
    )
    print(f"wrote {CARD_OUT.relative_to(REPO_ROOT)}")

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(CARD_OUT),
        path_in_repo="README.md",
        repo_id=HF_REPO,
        repo_type="model",
        commit_message="v1.0: model card with BFCL results + gate disclosure",
    )
    api.upload_file(
        path_or_fileobj=str(COMPARISON_PNG),
        path_in_repo="bfcl_comparison.png",
        repo_id=HF_REPO,
        repo_type="model",
        commit_message="v1.0: BFCL per-category comparison chart",
    )
    print(f"published: https://huggingface.co/{HF_REPO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
