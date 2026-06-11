"""Export LLaMA config directories into Megatron-compatible runtime settings."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers.models.llama.configuration_llama import LlamaConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def add_repo_to_path() -> None:
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def apply_llamafactory_model_args(config: LlamaConfig, args: argparse.Namespace) -> None:
    """Mirror the LLaMA-Factory model_args fields required by this Megatron backend."""

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

    config.more_iterations = args.more_iterations
    config.recurrent_model = True
    config.memory_size = args.memory_size
    config.loop_mamba_variant = args.loop_mamba_variant
    config.loop_mamba_n_groups = args.loop_mamba_n_groups
    config.scale_embeds = True
    config.residual_interpolated_embeds = True
    config.max_position_embeddings = args.model_max_position_embeddings
    config.vary_refine_steps = True
    config.classifier_dropout = 0


def build_llama_config_for_megatron(args: argparse.Namespace) -> LlamaConfig:
    add_repo_to_path()

    config = LlamaConfig.from_pretrained(str(args.model_config))
    apply_llamafactory_model_args(config, args)
    config._attn_implementation = args.attn_implementation
    if args.attn_implementation == "flash_attention_2":
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


def attach_llamafactory_initialization(config: Any, model_config: LlamaConfig) -> None:
    """Mirror the recurrent Mamba-fast initialization used by LLaMA-Factory."""

    hidden_size = int(getattr(model_config, "hidden_size"))
    num_layers = int(getattr(model_config, "num_hidden_layers"))
    train_iterations = int(getattr(model_config, "more_iterations", 0) or 0)
    eval_iterations = int(getattr(model_config, "more_eval_iterations", 0) or 0)
    recurrent_depth = max(1, train_iterations + 1, eval_iterations + 1)
    base_std = math.sqrt(2.0 / (5 * hidden_size))
    output_std = base_std / math.sqrt(2.0 * num_layers * recurrent_depth)

    config.init_method_std = base_std
    config.init_method = _llamafactory_trunc_normal(base_std)
    config.output_layer_init_method = _llamafactory_trunc_normal(output_std)
    config.more_iterations = train_iterations
    config.more_eval_iterations = eval_iterations
