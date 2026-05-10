"""Cheap in-the-loop eval -- the Phase 1 smoke check.

Loads the saved adapter on top of the base model, runs one short generation,
and asserts the completion is non-empty. The full BFCL holdout slice lives
in eval/bfcl_runner.py (Phase 4).
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_PROMPT = "What's the weather in Tokyo? Use the available tools."


def quick_eval(
    adapter_path: str, *, prompt: str = DEFAULT_PROMPT, max_new_tokens: int = 64
) -> dict[str, Any]:
    """Load the adapter, generate once, return the output.

    Smoke-only. Fails loud if generation is empty -- the whole point of
    Phase 1.3 is to catch a silent template/loading break before real training.
    """
    import unsloth  # noqa: F401  load before transformers/peft per Unsloth's load-order docs.
    from unsloth import FastLanguageModel

    # Single-GPU L4: Accelerate's device-map guard would still fire here even
    # though we're not using a Trainer. Matches the bypass set in train.py.
    os.environ.setdefault("ACCELERATE_BYPASS_DEVICE_MAP", "true")

    # Pass the adapter path directly -- Unsloth resolves the base model from
    # adapter_config.json and applies LoRA weights in one shot. Avoids the
    # layer-name prefix mismatch (`model.layers.*` vs `model.language_model.layers.*`)
    # we hit when calling `load_adapter` separately on Qwen 3.5's
    # ConditionalGeneration architecture.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=1024,
        load_in_4bit=False,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)

    # Qwen 3.5 ships as a multimodal Processor; its `tokenize=True` path
    # iterates content as a list of typed parts and barfs on plain strings.
    # Render to text first, then tokenize -- same path SFTTrainer used at train.
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    inputs = text_tokenizer(text, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )
    input_len = inputs["input_ids"].shape[-1]
    completion = text_tokenizer.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    )

    if not completion.strip():
        raise RuntimeError(
            f"Empty completion from adapter at {adapter_path}; "
            "smoke run failed -- inspect tokenizer / chat template / training step."
        )
    return {"prompt": prompt, "completion": completion}
