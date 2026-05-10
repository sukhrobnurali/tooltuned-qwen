"""Push the LoRA adapter folder to the HF hub.

Used by Phase 1.3 (private smoke repo) and Phase 5.2 (public artifacts).
"""

from __future__ import annotations

import os


def push(
    adapter_path: str,
    repo: str,
    *,
    private: bool = False,
    token: str | None = None,
) -> str:
    """Upload an adapter folder to `repo` on the HF hub. Returns the HF URL.

    `token` defaults to the `HF_TOKEN` env var when not given. Pass it
    explicitly when running in environments that may have stale tokens
    cached at ~/.cache/huggingface (e.g. Colab notebooks where someone
    else logged in earlier).
    """
    from huggingface_hub import HfApi

    auth = token if token is not None else os.environ.get("HF_TOKEN")
    api = HfApi(token=auth)

    api.create_repo(repo_id=repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=adapter_path, repo_id=repo, repo_type="model")
    return f"https://huggingface.co/{repo}"
