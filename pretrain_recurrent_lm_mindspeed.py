#!/usr/bin/env python3
# isort: skip_file
"""MindSpeed-LLM pretrain entry for recurrent LM model implementations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "FALSE")

import torch
import torch.nn as nn
import torch_npu  # noqa: F401
from torch.utils.data import Dataset
from torch_npu.contrib import transfer_to_npu  # noqa: F401
from llama_config.megatron_export import (
    attach_llamafactory_initialization,
    attach_mindspeed_runtime_config,
    build_llama_config_for_megatron,
)
from modeling.registry import build_model
from runtime.mindspeed_runtime import (
    install_mindspeed_cross_entropy_patches,
    load_mindspeed_runtime,
)


mindspeed_get_batch_on_this_cp_rank, pretrain = load_mindspeed_runtime()

# MindSpeed patches torch.compile, Transformer Engine, Apex, and NPU helpers.
# Import Megatron only after those patches are installed.
from megatron.core import mpu, tensor_parallel
from megatron.core.datasets.utils import Split
from megatron.core.enums import ModelType
from megatron.training import get_args, get_timers, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import average_losses_across_data_parallel_group


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_CONFIG = REPO_ROOT / "llama_config/410m"
DEFAULT_TOKENIZED_PATH = Path("/data/PonderLM/uint16smallpile")


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value if value else default


def _path_arg(value: str) -> Path:
    return Path(value).expanduser().resolve()


def extra_args_provider(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("recurrent_lm")
    group.add_argument("--model-config", type=_path_arg, default=DEFAULT_BASE_CONFIG)
    group.add_argument("--tokenized-path", type=_path_arg, default=DEFAULT_TOKENIZED_PATH)
    group.add_argument("--train-split-name", default="train")
    group.add_argument("--valid-tokenized-path", type=_path_arg, default=None)
    group.add_argument("--valid-split-name", default="validation")
    group.add_argument("--attn-implementation", default="flash_attention_2")
    group.add_argument("--more-iterations", type=int, default=3)
    group.add_argument("--more-eval-iterations", type=int, default=0)
    group.add_argument("--memory-size", type=int, default=1024)
    group.add_argument("--pause-token-id", type=int, default=50288)
    group.add_argument("--softmax-temperature", type=float, default=1.0)
    group.add_argument("--interpolation", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--interpolation-use-topk", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--interpolation-topk", type=int, default=100)
    group.add_argument("--scale-embeds", "--scale_embeds", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument(
        "--residual-interpolated-embeds",
        "--residual_interpolated_embeds",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    group.add_argument("--vary-refine-steps", "--vary_refine_steps", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--training-refinement-steps", type=int, default=5)
    group.add_argument("--eval-refinement-steps", type=int, default=10)
    group.add_argument("--consistency-weight", type=float, default=0.0)
    group.add_argument("--min-weight-penalty-lambda-start", type=float, default=0.0)
    group.add_argument("--min-weight-penalty-lambda-max", type=float, default=0.0)
    group.add_argument("--min-weight-penalty-warmup-steps", type=int, default=1000)
    group.add_argument("--min-weight-penalty-peak-steps", type=int, default=4000)
    group.add_argument(
        "--min-weight-penalty-method",
        choices=("accuracy", "delta_loss", "ce_loss"),
        default="accuracy",
    )
    group.add_argument("--damping-alpha", type=float, default=0.1)
    group.add_argument("--last-n-steps-update-w", type=int, default=1)
    group.add_argument(
        "--loop-mamba-variant",
        choices=("legacy", "mamba2", "mamba2_fast"),
        default="mamba2_fast",
    )
    group.add_argument("--loop-mamba-n-groups", type=int, default=8)
    group.add_argument("--model-max-position-embeddings", type=int, default=4096)
    group.add_argument("--pad-token-id", type=int, default=1)
    group.add_argument(
        "--sampler-seed-mode",
        choices=("megatron", "llamafactory"),
        default=_env_or_default("SAMPLER_SEED_MODE", "llamafactory").lower(),
        help="Dataset sampler seed semantics. Use 'llamafactory' to mirror HF Trainer/Accelerate shuffling.",
    )
    group.add_argument(
        "--sampler-data-seed",
        type=int,
        default=None,
        help=(
            "Optional sampler seed. In llamafactory mode, defaults to --seed, "
            "matching TrainingArguments.data_seed=None."
        ),
    )
    group.add_argument("--model-bf16-autocast", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--mcore-native", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--experiment-output-dir", type=_path_arg, default=None)
    group.add_argument("--model-impl", default="llama_orin_ssm")
    return parser


class TokenizedDataset(Dataset):
    """Expose a HF tokenized split in Megatron pretraining batch format."""

    def __init__(
        self,
        tokenized_path: Path,
        split_name: str,
        seq_len: int,
        split: Split,
        pad_token_id: int,
    ) -> None:
        self.dataset = load_tokenized_split(tokenized_path, split_name)
        self.seq_len = seq_len
        self.split = split
        self.tokenized_path = tokenized_path
        self.split_name = split_name
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.dataset[int(idx) % len(self.dataset)]
        input_ids = list(example["input_ids"][: self.seq_len])
        attention_mask = list(example.get("attention_mask", [1] * len(input_ids))[: self.seq_len])
        if len(input_ids) < self.seq_len:
            pad = self.seq_len - len(input_ids)
            input_ids.extend([self.pad_token_id] * pad)
            attention_mask.extend([0] * pad)

        tokens = torch.tensor(input_ids, dtype=torch.long)
        mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = tokens.clone()
        labels = labels.masked_fill(mask == 0, -100)
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": mask.to(torch.float32),
            "attention_mask": mask,
            "position_ids": torch.arange(self.seq_len, dtype=torch.long),
        }


def load_tokenized_split(tokenized_path: Path, split_name: str) -> Any:
    """Load one split from a HF dataset saved with ``datasets.save_to_disk``."""

    from datasets import load_from_disk

    tokenized = load_from_disk(str(tokenized_path))
    if hasattr(tokenized, "keys"):
        available_splits = list(tokenized.keys())
        if split_name not in available_splits:
            raise ValueError(
                f"Split '{split_name}' not found in {tokenized_path}. "
                f"Available splits: {', '.join(available_splits)}"
            )
        return tokenized[split_name]
    return tokenized


def model_provider(pre_process: bool = True, post_process: bool = True, **_: Any) -> nn.Module:
    del pre_process, post_process
    args = get_args()
    if args.pipeline_model_parallel_size != 1:
        raise ValueError("The recurrent MindSpeed backend currently supports PP=1 only.")
    if args.context_parallel_size > 1 and getattr(args, "context_parallel_algo", None) != "mamba_cp_algo":
        raise ValueError("recurrent CP requires --context-parallel-algo mamba_cp_algo.")
    megatron_config = core_transformer_config_from_args(args)
    attach_mindspeed_runtime_config(megatron_config, args)
    if not args.mcore_native:
        raise ValueError("The recurrent MindSpeed backend only supports the MCore native model path.")
    model_config = build_llama_config_for_megatron(args)
    attach_llamafactory_initialization(megatron_config, model_config)
    vocab_size = getattr(args, "padded_vocab_size", None) or getattr(model_config, "vocab_size")
    return build_model(
        args.model_impl,
        config=megatron_config,
        model_config=model_config,
        vocab_size=vocab_size,
        max_sequence_length=args.model_max_position_embeddings,
        pre_process=True,
        post_process=True,
        parallel_output=True,
        use_transformer_engine_spec=args.transformer_impl == "transformer_engine",
    )


def _slice_attention_mask_for_mamba_cp(batch: dict[str, torch.Tensor | None]) -> None:
    args = get_args()
    if (
        getattr(args, "context_parallel_size", 1) <= 1
        or getattr(args, "context_parallel_algo", None) != "mamba_cp_algo"
    ):
        return
    attention_mask = batch.get("attention_mask")
    tokens = batch.get("tokens")
    if attention_mask is None or tokens is None or attention_mask.shape[1] == tokens.shape[1]:
        return
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    batch["attention_mask"] = attention_mask.chunk(cp_size, dim=1)[cp_rank].contiguous()


def _is_pause_model_impl(model_impl: str) -> bool:
    return model_impl in {"llama_pause", "modeling_llama_pause"}


def _expand_pause_inputs_before_cp(batch: dict[str, torch.Tensor | None]) -> None:
    """Expand pause tokens before CP sequence sharding while labels stay unexpanded."""

    args = get_args()
    if not _is_pause_model_impl(getattr(args, "model_impl", "")):
        return
    if int(getattr(args, "context_parallel_size", 1) or 1) <= 1:
        return

    tokens = batch.get("tokens")
    if tokens is None:
        return
    num_pause_tokens = int(getattr(args, "more_iterations", 0) or 0)
    if num_pause_tokens <= 0:
        return

    batch_size, seq_len = tokens.shape
    group = num_pause_tokens + 1
    expanded_len = seq_len * group
    pause_token_id = int(getattr(args, "pause_token_id", 50288))

    expanded_tokens = torch.full(
        (batch_size, expanded_len),
        pause_token_id,
        dtype=tokens.dtype,
        device=tokens.device,
    )
    original_positions = torch.arange(seq_len, device=tokens.device) * group
    expanded_tokens[:, original_positions] = tokens
    batch["tokens"] = expanded_tokens

    position_ids = batch.get("position_ids")
    if position_ids is None:
        position_ids = torch.arange(seq_len, dtype=torch.long, device=tokens.device).unsqueeze(0).expand_as(tokens)
    batch["position_ids"] = position_ids.repeat_interleave(group, dim=1).contiguous()

    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        batch["attention_mask"] = attention_mask.repeat_interleave(group, dim=1).contiguous()


def get_batch(data_iterator: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = next(data_iterator) if mpu.get_tensor_model_parallel_rank() == 0 else None
    int_batch = tensor_parallel.broadcast_data(
        ["tokens", "labels", "attention_mask", "position_ids"],
        data,
        torch.int64,
    )
    float_batch = tensor_parallel.broadcast_data(["loss_mask"], data, torch.float32)
    batch: dict[str, torch.Tensor | None] = {
        "tokens": int_batch["tokens"],
        "labels": int_batch["labels"],
        "loss_mask": float_batch["loss_mask"],
        "attention_mask": int_batch["attention_mask"],
        "position_ids": int_batch["position_ids"],
    }
    if getattr(get_args(), "context_parallel_size", 1) > 1:
        _expand_pause_inputs_before_cp(batch)
        batch = mindspeed_get_batch_on_this_cp_rank(batch)
        _slice_attention_mask_for_mamba_cp(batch)
    return (
        batch["tokens"],
        batch["labels"],
        batch["loss_mask"],
        batch["attention_mask"],
        batch["position_ids"],
    )


def loss_func(loss: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = loss.float()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    logged_loss = averaged_loss[0]
    if mpu.get_context_parallel_world_size() > 1:
        logged_loss = logged_loss.clone()
        torch.distributed.all_reduce(logged_loss, group=mpu.get_context_parallel_group())
    if os.getenv("RECURRENT_DEBUG_LOSS_LOG", "0") == "1":
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if rank in (0, world_size - 1):
            iteration = int(getattr(get_args(), "curr_iteration", -1)) + 1
            print(
                "[RECURRENT_DEBUG_LOSS] "
                f"rank={rank} iteration={iteration} "
                f"raw={loss.detach().item():.8e} "
                f"logged={logged_loss.detach().item():.8e} "
                f"raw_finite={bool(torch.isfinite(loss.detach()).item())} "
                f"logged_finite={bool(torch.isfinite(logged_loss.detach()).item())}",
                flush=True,
            )
    return loss, {"lm loss": logged_loss}


def forward_step(data_iterator: Any, model: nn.Module) -> tuple[torch.Tensor, Any]:
    args = get_args()
    timers = get_timers()
    timers("batch-generator", log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers("batch-generator").stop()
    del loss_mask
    global_step_offset = int(os.getenv("WANDB_GLOBAL_STEP_OFFSET", os.getenv("GLOBAL_STEP_OFFSET", "0")))
    global_step = int(getattr(args, "curr_iteration", 0)) + global_step_offset
    loss = model(
        tokens,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
        global_step=global_step,
    )
    return loss, loss_func


def train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int],
) -> tuple[Dataset, Dataset | None, Dataset | None]:
    args = get_args()
    print_rank_0("> building HF-tokenized datasets for recurrent LM MindSpeed training ...")
    train_ds = TokenizedDataset(
        args.tokenized_path,
        args.train_split_name,
        args.seq_length,
        Split.train,
        args.pad_token_id,
    )
    valid_ds = None
    test_ds = None
    if args.eval_iters and args.eval_iters > 0:
        valid_path = args.valid_tokenized_path or args.tokenized_path
        valid_ds = TokenizedDataset(
            valid_path,
            args.valid_split_name,
            args.seq_length,
            Split.valid,
            args.pad_token_id,
        )
    args.dataset_train_len = len(train_ds)
    args.dataset_valid_len = len(valid_ds) if valid_ds is not None else 0
    print_rank_0(
        json.dumps(
            {
                "dataset_train_len": args.dataset_train_len,
                "dataset_train_path": str(args.tokenized_path),
                "dataset_train_split": args.train_split_name,
                "dataset_requested_train": train_val_test_num_samples[0],
                "dataset_valid_enabled": valid_ds is not None,
                "dataset_valid_len": args.dataset_valid_len,
                "dataset_valid_path": str(args.valid_tokenized_path or args.tokenized_path),
                "dataset_valid_split": args.valid_split_name,
                "dataset_requested_valid": train_val_test_num_samples[1],
                "sampler_data_seed": args.sampler_data_seed if args.sampler_data_seed is not None else args.seed,
                "sampler_seed_mode": args.sampler_seed_mode,
            },
            sort_keys=True,
        )
    )
    return train_ds, valid_ds, test_ds


def main() -> None:
    install_mindspeed_cross_entropy_patches()
    train_valid_test_datasets_provider.is_distributed = False
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=extra_args_provider,
        args_defaults={
            "tokenizer_type": "NullTokenizer",
            "dataloader_type": "cyclic",
        },
    )


if __name__ == "__main__":
    main()
