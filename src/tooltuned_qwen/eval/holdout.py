"""Cheap in-the-loop eval -- the Phase 1 smoke check.

Loads the saved adapter on top of the base model, runs one short generation,
and asserts the completion is non-empty. The full BFCL holdout slice lives
in eval/bfcl_runner.py (Phase 4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_PROMPT = "What's the weather in Tokyo? Use the available tools."


def quick_eval(
    adapter_path: str, *, prompt: str = DEFAULT_PROMPT, max_new_tokens: int = 64
) -> dict[str, Any]:
    """Load `adapter_path` on the base it was trained from, generate once, return the output.

    Smoke-only. Fails loud if generation is empty -- the whole point of
    Phase 1.3 is to catch a silent template/loading break before real training.
    """
    from unsloth import FastLanguageModel

    base_model = _read_base_model(adapter_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=1024,
        load_in_4bit=False,
        dtype=None,
    )
    model.load_adapter(adapter_path, adapter_name="default")
    FastLanguageModel.for_inference(model)

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    completion = tokenizer.decode(
        output_ids[0][inputs.shape[-1] :], skip_special_tokens=True
    )

    if not completion.strip():
        raise RuntimeError(
            f"Empty completion from adapter at {adapter_path}; "
            "smoke run failed -- inspect tokenizer / chat template / training step."
        )
    return {"prompt": prompt, "completion": completion}


def _read_base_model(adapter_path: str) -> str:
    """Recover the base model name from the adapter's PEFT config."""
    import json

    cfg_path = Path(adapter_path) / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise RuntimeError(f"adapter_config.json at {cfg_path} has no base_model_name_or_path")
    return base
