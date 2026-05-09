"""Push the LoRA adapter folder to the HF hub.

Used by Phase 1.3 (private smoke repo) and Phase 5.2 (public artifacts).
HF token comes from `HF_TOKEN` in the environment or `huggingface-cli login`.
"""

from __future__ import annotations


def push(adapter_path: str, repo: str, *, private: bool = False) -> str:
    """Upload an adapter folder to `repo` on the HF hub. Returns the HF URL."""
    from huggingface_hub import HfApi, create_repo

    create_repo(repo, repo_type="model", private=private, exist_ok=True)
    HfApi().upload_folder(
        folder_path=adapter_path,
        repo_id=repo,
        repo_type="model",
    )
    return f"https://huggingface.co/{repo}"
