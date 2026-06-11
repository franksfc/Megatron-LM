#!/usr/bin/env python3
"""Inspect the ssm9 HF checkpoint and emit the settings Megatron launchers must copy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import safe_open


DEFAULT_CHECKPOINT = Path(
    "/data/PonderLM/fcsong/LLaMA-Factory-adaptive/"
    "trained-model-pile/icml/orinnew-410m-25b-3iteration412m-ssm9/checkpoint-50000"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    checkpoint = args.checkpoint
    config_path = checkpoint / "config.json"
    training_args_path = checkpoint / "training_args.bin"
    safetensors_path = checkpoint / "model.safetensors"

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    training_args = torch.load(training_args_path, map_location="cpu", weights_only=False)
    train_keys = [
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "max_steps",
        "num_train_epochs",
        "warmup_ratio",
        "lr_scheduler_type",
        "lr_scheduler_kwargs",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "max_grad_norm",
        "bf16",
        "fp16",
        "logging_steps",
        "save_steps",
        "eval_steps",
        "dataloader_num_workers",
        "seed",
        "deepspeed",
    ]

    loop_keys = []
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
        all_keys = list(handle.keys())
        for key in all_keys:
            if key.startswith("loop_axis_ssm."):
                tensor = handle.get_tensor(key)
                loop_keys.append({"key": key, "shape": list(tensor.shape), "dtype": str(tensor.dtype)})

    summary = {
        "checkpoint": str(checkpoint),
        "model": {
            "hidden_size": config.get("hidden_size"),
            "intermediate_size": config.get("intermediate_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "vocab_size": config.get("vocab_size"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "recurrent_model": config.get("recurrent_model"),
            "more_iterations": config.get("more_iterations"),
            "memory_size": config.get("memory_size"),
            "scale_embeds": config.get("scale_embeds"),
        },
        "training_args": {key: str(getattr(training_args, key, None)) for key in train_keys},
        "num_tensors": len(all_keys),
        "loop_axis_ssm_tensors": loop_keys,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
