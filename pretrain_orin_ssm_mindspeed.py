#!/usr/bin/env python3
# isort: skip_file
"""MindSpeed-LLM pretrain entry for the Orin/SSM MCore-native model."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "FALSE")

import torch
import torch.nn as nn
import torch_npu  # noqa: F401
from torch.utils.data import Dataset
from torch_npu.contrib import transfer_to_npu  # noqa: F401
from transformers.models.llama.configuration_llama import LlamaConfig

from megatron.orin_ssm.mindspeed_runtime_patches import (
    install_llamafactory_wandb_training_log,
    install_orin_mtp_feature_guard,
)


def _load_mindspeed_runtime() -> tuple[Any, Any]:
    original_backend = os.environ.get("TRAINING_BACKEND")
    backend = os.environ.get("TRAINING_BACKEND", "mcore")
    if backend.lower() == "mcore":
        os.environ["TRAINING_BACKEND"] = "orin_manual_adaptor"
    try:
        install_orin_mtp_feature_guard()
    finally:
        if original_backend is None:
            os.environ.pop("TRAINING_BACKEND", None)
        else:
            os.environ["TRAINING_BACKEND"] = original_backend

    importlib.import_module("mindspeed_llm.tasks.megatron_adaptor_v2")
    get_batch_utils = importlib.import_module("mindspeed_llm.core.context_parallel.get_batch_utils")
    training_module = importlib.import_module("mindspeed_llm.training.training")
    install_llamafactory_wandb_training_log(training_module)
    return get_batch_utils.get_batch_on_this_cp_rank, training_module.pretrain


mindspeed_get_batch_on_this_cp_rank, pretrain = _load_mindspeed_runtime()

# MindSpeed patches torch.compile, Transformer Engine, Apex, and NPU helpers.
# Import Megatron only after those patches are installed.
from megatron.core import mpu, tensor_parallel
from megatron.core.datasets.utils import Split
from megatron.core.enums import ModelType
from megatron.training import get_args, get_timers, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import average_losses_across_data_parallel_group


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_CONFIG = Path("/data/PonderLM/fcsong/LLaMA-Factory-adaptive/Llama_config/410m")
DEFAULT_TOKENIZED_PATH = Path("/data/PonderLM/uint16smallpile")


def _install_mindspeed_cross_entropy_patches() -> None:
    """Install MindSpeed-LLM's NPU-friendly vocab-parallel CE helpers."""

    from megatron.core.tensor_parallel.cross_entropy import VocabParallelCrossEntropy
    from mindspeed_llm.core.tensor_parallel.cross_entropy import (
        calculate_logits_max,
        calculate_predicted_logits,
    )

    VocabParallelCrossEntropy.calculate_logits_max = staticmethod(calculate_logits_max)
    VocabParallelCrossEntropy.calculate_predicted_logits = staticmethod(calculate_predicted_logits)


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value if value else default


def _add_repo_to_path() -> None:
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _path_arg(value: str) -> Path:
    return Path(value).expanduser().resolve()


def extra_args_provider(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("orin_ssm")
    group.add_argument("--orin-config", type=_path_arg, default=DEFAULT_BASE_CONFIG)
    group.add_argument("--orin-tokenized-path", type=_path_arg, default=DEFAULT_TOKENIZED_PATH)
    group.add_argument("--orin-train-split-name", default="train")
    group.add_argument("--orin-valid-tokenized-path", type=_path_arg, default=None)
    group.add_argument("--orin-valid-split-name", default="validation")
    group.add_argument("--orin-attn-implementation", default="flash_attention_2")
    group.add_argument("--orin-more-iterations", type=int, default=3)
    group.add_argument("--orin-memory-size", type=int, default=1024)
    group.add_argument(
        "--orin-loop-mamba-variant",
        choices=("legacy", "mamba2", "orin_mamba2", "orin_mamba2_fast"),
        default="orin_mamba2_fast",
    )
    group.add_argument("--orin-loop-mamba-n-groups", type=int, default=8)
    group.add_argument("--orin-max-position-embeddings", type=int, default=4096)
    group.add_argument("--orin-pad-token-id", type=int, default=1)
    group.add_argument(
        "--orin-seed-mode",
        choices=("megatron", "llamafactory"),
        default=_env_or_default("ORIN_SEED_MODE", "llamafactory").lower(),
        help="Dataset sampler seed semantics. Use 'llamafactory' to mirror HF Trainer/Accelerate shuffling.",
    )
    group.add_argument(
        "--orin-data-seed",
        type=int,
        default=None,
        help=(
            "Optional sampler seed. In llamafactory mode, defaults to --seed, "
            "matching TrainingArguments.data_seed=None."
        ),
    )
    group.add_argument("--orin-bf16-autocast", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--orin-mcore-native", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--orin-output-dir", type=_path_arg, default=None)
    return parser


def apply_llamafactory_orin_config(config: LlamaConfig, args: argparse.Namespace) -> None:
    """Mirror the LLaMA-Factory model_args fields used by the Orin/SSM run."""

    llama_factory_defaults = {
        "output_hidden_states": False,
        "ponder_size": 48,
        "is_normalize_hidden_states": False,
        "normalize_topk_sample": False,
        "hidden_layer_num": -2,
        "more_eval_iterations": 0,
        "vary_position": False,
        "add_loss_for_ponderer": False,
        "replace_embeddings": False,
        "softmax_temperature": 1.0,
        "add_ponderer_token": False,
        "memory_output_gate": False,
        "memory_mix_gate": False,
        "add_adapter": False,
        "recurrent_interval": 0,
        "recurrent_layer": False,
        "high_memory_mode": False,
        "add_gate": False,
        "top_k_num": 100,
        "back_iterations": 3,
        "checkpoint_num_layers": 100,
        "uniform_real_time": False,
        "use_all_logits": False,
        "interpolation_use_topk": False,
        "stage_router_update_w": False,
        "training_refinement_steps": 5,
        "eval_refinement_steps": 5,
        "interpolation": False,
        "use_anderson": True,
        "anderson_depth": 2,
        "anderson_beta": 1.0,
        "anderson_regularization": 0.001,
        "anderson_convex_only": True,
        "anderson_residual_increase_thr": 1.05,
        "anderson_reset_interval": 0,
        "consistency_weight": 0.0,
        "ponder_ent_lambda_start": 0.0,
        "ponder_ent_lambda_max": 0.2,
        "ponder_ent_warmup_steps": 1000,
        "ponder_ent_peak_steps": 4000,
        "ponder_cost_lambda_start": 0.0,
        "ponder_cost_lambda_max": 0.1,
        "ponder_cost_warmup_steps": 1000,
        "ponder_cost_peak_steps": 4000,
        "diverse_lambda_start": 0.0,
        "diverse_lambda_max": 0.1,
        "diverse_warmup_steps": 1000,
        "diverse_peak_steps": 4000,
        "weight_dist_lambda_start": 0.0,
        "weight_dist_lambda_max": 0.1,
        "weight_dist_warmup_steps": 1000,
        "weight_dist_peak_steps": 4000,
        "min_weight_penalty_lambda_start": 0.0,
        "min_weight_penalty_lambda_max": 0.0,
        "min_weight_penalty_warmup_steps": 1000,
        "min_weight_penalty_peak_steps": 4000,
        "min_weight_penalty_method": "accuracy",
        "delta_method": "neg",
        "sigma_slope": 50.0,
        "last_n_steps_update_w": 1,
        "damping_alpha": 1.0,
        "anderson_ridge": 1e-5,
        "sigma_2": False,
    }
    for key, value in llama_factory_defaults.items():
        setattr(config, key, value)

    config.more_iterations = args.orin_more_iterations
    config.recurrent_model = True
    config.memory_size = args.orin_memory_size
    config.loop_mamba_variant = args.orin_loop_mamba_variant
    config.loop_mamba_n_groups = args.orin_loop_mamba_n_groups
    config.scale_embeds = True
    config.residual_interpolated_embeds = True
    config.max_position_embeddings = args.orin_max_position_embeddings
    config.vary_refine_steps = True
    config.classifier_dropout = 0


class OrinTokenizedDataset(Dataset):
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


def build_orin_config(args: argparse.Namespace) -> LlamaConfig:
    _add_repo_to_path()

    config = LlamaConfig.from_pretrained(str(args.orin_config))
    apply_llamafactory_orin_config(config, args)
    config._attn_implementation = args.orin_attn_implementation
    if args.orin_attn_implementation == "flash_attention_2":
        setattr(config, "_attn_implementation_autoset", True)
    return config


def attach_mindspeed_runtime_config(config: Any, args: argparse.Namespace) -> None:
    """Attach MindSpeed runtime fields that are not Megatron Core dataclass fields."""

    runtime_fields = {
        "transformer_impl": args.transformer_impl,
        "use_flash_attn": bool(getattr(args, "use_flash_attn", False)),
        "attention_mask_type": getattr(args, "attention_mask_type", "causal"),
        "seq_length": args.seq_length,
        "micro_batch_size": args.micro_batch_size,
        "pre_tockens": getattr(args, "pre_tockens", 1048576),
        "next_tockens": getattr(args, "next_tockens", 0),
        "sparse_mode": getattr(args, "sparse_mode", 0),
        "shape_order": getattr(args, "shape_order", "SBH"),
        "context_parallel_algo": getattr(args, "context_parallel_algo", "megatron_cp_algo"),
    }
    for key, value in runtime_fields.items():
        setattr(config, key, value)


def _llamafactory_trunc_normal(std: float) -> Any:
    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-3 * std, b=3 * std)

    return init_


def attach_llamafactory_initialization(config: Any, orin_config: LlamaConfig) -> None:
    """Mirror the Orin mamba-fast initialization used by LLaMA-Factory."""

    hidden_size = int(getattr(orin_config, "hidden_size"))
    num_layers = int(getattr(orin_config, "num_hidden_layers"))
    train_iterations = int(getattr(orin_config, "more_iterations", 0) or 0)
    eval_iterations = int(getattr(orin_config, "more_eval_iterations", 0) or 0)
    recurrent_depth = max(1, train_iterations + 1, eval_iterations + 1)
    base_std = math.sqrt(2.0 / (5 * hidden_size))
    output_std = base_std / math.sqrt(2.0 * num_layers * recurrent_depth)

    config.init_method_std = base_std
    config.init_method = _llamafactory_trunc_normal(base_std)
    config.output_layer_init_method = _llamafactory_trunc_normal(output_std)
    config.orin_more_iterations = train_iterations
    config.orin_more_eval_iterations = eval_iterations


def model_provider(pre_process: bool = True, post_process: bool = True, **_: Any) -> nn.Module:
    del pre_process, post_process
    args = get_args()
    if args.pipeline_model_parallel_size != 1:
        raise ValueError("The Orin MindSpeed backend currently supports PP=1 only.")
    if args.context_parallel_size > 1 and getattr(args, "context_parallel_algo", None) != "mamba_cp_algo":
        raise ValueError("Orin recurrent CP requires --context-parallel-algo mamba_cp_algo.")
    megatron_config = core_transformer_config_from_args(args)
    attach_mindspeed_runtime_config(megatron_config, args)
    if not args.orin_mcore_native:
        raise ValueError("The Orin MindSpeed backend only supports the MCore native model path.")
    from megatron.orin_ssm.mcore_orin_model import OrinMCoreModel

    orin_config = build_orin_config(args)
    attach_llamafactory_initialization(megatron_config, orin_config)
    vocab_size = getattr(args, "padded_vocab_size", None) or getattr(orin_config, "vocab_size")
    return OrinMCoreModel(
        config=megatron_config,
        orin_config=orin_config,
        vocab_size=vocab_size,
        max_sequence_length=args.orin_max_position_embeddings,
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
        logged_loss = logged_loss / mpu.get_context_parallel_world_size()
    if os.getenv("ORIN_DEBUG_LOSS_LOG", "0") == "1":
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if rank in (0, world_size - 1):
            iteration = int(getattr(get_args(), "curr_iteration", -1)) + 1
            print(
                "[ORIN_DEBUG_LOSS] "
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
    print_rank_0("> building Orin HF-tokenized datasets for MindSpeed ...")
    train_ds = OrinTokenizedDataset(
        args.orin_tokenized_path,
        args.orin_train_split_name,
        args.seq_length,
        Split.train,
        args.orin_pad_token_id,
    )
    valid_ds = None
    test_ds = None
    if args.eval_iters and args.eval_iters > 0:
        valid_path = args.orin_valid_tokenized_path or args.orin_tokenized_path
        valid_ds = OrinTokenizedDataset(
            valid_path,
            args.orin_valid_split_name,
            args.seq_length,
            Split.valid,
            args.orin_pad_token_id,
        )
    args.orin_dataset_train_len = len(train_ds)
    args.orin_dataset_valid_len = len(valid_ds) if valid_ds is not None else 0
    print_rank_0(
        json.dumps(
            {
                "orin_dataset_train_len": args.orin_dataset_train_len,
                "orin_dataset_train_path": str(args.orin_tokenized_path),
                "orin_dataset_train_split": args.orin_train_split_name,
                "orin_dataset_requested_train": train_val_test_num_samples[0],
                "orin_dataset_valid_enabled": valid_ds is not None,
                "orin_dataset_valid_len": args.orin_dataset_valid_len,
                "orin_dataset_valid_path": str(args.orin_valid_tokenized_path or args.orin_tokenized_path),
                "orin_dataset_valid_split": args.orin_valid_split_name,
                "orin_dataset_requested_valid": train_val_test_num_samples[1],
                "orin_data_seed": args.orin_data_seed if args.orin_data_seed is not None else args.seed,
                "orin_seed_mode": args.orin_seed_mode,
            },
            sort_keys=True,
        )
    )
    return train_ds, valid_ds, test_ds


def main() -> None:
    _install_mindspeed_cross_entropy_patches()
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
