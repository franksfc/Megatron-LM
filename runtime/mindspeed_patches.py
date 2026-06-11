"""Runtime patches for the MindSpeed-LLM backend."""

from __future__ import annotations

import os
from types import ModuleType
from typing import Any, Callable

import torch


def install_mtp_feature_guard() -> None:
    """Disable MindSpeed-LLM MTP patch registration unless MTP is enabled."""

    from mindspeed_llm.features_manager.transformer.mtp import MultiTokenPredictionFeature

    original_register_patches = MultiTokenPredictionFeature.register_patches
    if getattr(original_register_patches, "_recurrent_lm_mtp_guard", False):
        return

    def register_patches(self: Any, patch_manager: Any, args: Any) -> Any:
        if not getattr(args, "mtp_num_layers", None):
            return None
        return original_register_patches(self, patch_manager, args)

    register_patches._recurrent_lm_mtp_guard = True  # type: ignore[attr-defined]
    MultiTokenPredictionFeature.register_patches = register_patches


def _use_llamafactory_wandb(training_module: ModuleType, args: Any, wandb_writer: Any) -> bool:
    if wandb_writer is None or not training_module.is_last_rank():
        return False
    style = os.getenv("RECURRENT_WANDB_LOG_STYLE", "").strip().lower()
    if style in ("llamafactory", "hf", "trainer"):
        return True
    if style in ("", "0", "false", "off", "none"):
        return False
    return bool(getattr(args, "tokenized_path", None))


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().float().item()
    return float(value)


def _wandb_global_step(iteration: int) -> int:
    offset = int(os.getenv("WANDB_GLOBAL_STEP_OFFSET", os.getenv("GLOBAL_STEP_OFFSET", "0")) or "0")
    return int(iteration) + offset


def _recurrent_epoch(args: Any) -> float:
    train_samples = int(
        getattr(args, "dataset_train_len", 0)
        or os.getenv("RECURRENT_TRAIN_SAMPLES", "0")
        or "0"
    )
    if train_samples <= 0:
        train_samples = 16_000_000
    return float(args.consumed_train_samples) / float(max(1, train_samples))


def _preview_llamafactory_loss(
    loss_dict: dict[str, Any],
    total_loss_dict: dict[str, Any],
    skipped_iter: int,
) -> float | None:
    advanced_iters_key = "advanced iterations"
    skipped_iters_key = "skipped iterations"
    nan_iters_key = "nan iterations"
    shadow = dict(total_loss_dict)
    if not skipped_iter:
        shadow[advanced_iters_key] = shadow.get(advanced_iters_key, 0) + 1
    elif advanced_iters_key not in shadow:
        shadow[advanced_iters_key] = 0
    shadow[skipped_iters_key] = shadow.get(skipped_iters_key, 0) + skipped_iter

    got_nan = False
    for key, value in loss_dict.items():
        if not skipped_iter:
            shadow[key] = shadow.get(
                key,
                torch.tensor([0.0], dtype=torch.float, device=value.device),
            ) + value
        else:
            float_value = value.float().sum().item()
            is_nan = (
                float_value == float("inf")
                or float_value == -float("inf")
                or float_value != float_value
            )
            got_nan = got_nan or is_nan
    shadow[nan_iters_key] = shadow.get(nan_iters_key, 0) + int(got_nan)

    loss = None
    denominator = float(max(1, shadow[advanced_iters_key]))
    for key, value in shadow.items():
        if key in (advanced_iters_key, skipped_iters_key, nan_iters_key):
            continue
        avg = value.item() / denominator
        if key == "lm loss" or loss is None:
            loss = avg
    return loss


def _log_llamafactory_train_wandb(
    wandb_writer: Any,
    args: Any,
    iteration: int,
    loss: Any,
    grad_norm: Any,
    learning_rate: Any,
) -> None:
    metrics = {
        "train/epoch": _recurrent_epoch(args),
        "train/global_step": _wandb_global_step(iteration),
    }
    loss_value = _scalar(loss)
    if loss_value is not None:
        metrics["train/loss"] = loss_value
    grad_norm_value = _scalar(grad_norm)
    if grad_norm_value is not None:
        metrics["train/grad_norm"] = grad_norm_value
    learning_rate_value = _scalar(learning_rate)
    if learning_rate_value is not None:
        metrics["train/learning_rate"] = learning_rate_value
    wandb_writer.log(metrics)


def _log_llamafactory_eval_wandb(
    wandb_writer: Any,
    args: Any,
    iteration: int,
    loss: float,
) -> None:
    metrics = {
        "eval/epoch": _recurrent_epoch(args),
        "eval/global_step": _wandb_global_step(iteration),
        "eval/loss": loss,
    }
    wandb_writer.log(metrics)


def install_llamafactory_wandb_training_log(training_module: ModuleType) -> None:
    """Make MindSpeed-LLM trainer emit LLaMA-Factory-shaped train W&B metrics."""

    original_training_log = training_module.training_log
    if getattr(original_training_log, "_recurrent_lm_llamafactory_wandb", False):
        return

    def training_log(
        loss_dict: dict[str, Any],
        total_loss_dict: dict[str, Any],
        learning_rate: Any,
        decoupled_learning_rate: Any,
        iteration: int,
        loss_scale: float,
        report_memory_flag: bool,
        skipped_iter: int,
        grad_norm: Any,
        params_norm: Any,
        num_zeros_in_grad: Any,
    ) -> bool:
        args = training_module.get_args()
        raw_wandb_writer = training_module.get_wandb_writer()
        llamafactory_wandb_writer = (
            raw_wandb_writer
            if _use_llamafactory_wandb(training_module, args, raw_wandb_writer)
            else None
        )
        llamafactory_loss = None
        if llamafactory_wandb_writer and iteration % args.log_interval == 0:
            llamafactory_loss = _preview_llamafactory_loss(loss_dict, total_loss_dict, skipped_iter)

        if not llamafactory_wandb_writer:
            return original_training_log(
                loss_dict,
                total_loss_dict,
                learning_rate,
                decoupled_learning_rate,
                iteration,
                loss_scale,
                report_memory_flag,
                skipped_iter,
                grad_norm,
                params_norm,
                num_zeros_in_grad,
            )

        original_get_wandb_writer: Callable[[], Any] = training_module.get_wandb_writer
        training_module.get_wandb_writer = lambda: None
        try:
            result = original_training_log(
                loss_dict,
                total_loss_dict,
                learning_rate,
                decoupled_learning_rate,
                iteration,
                loss_scale,
                report_memory_flag,
                skipped_iter,
                grad_norm,
                params_norm,
                num_zeros_in_grad,
            )
        finally:
            training_module.get_wandb_writer = original_get_wandb_writer

        if iteration % args.log_interval == 0:
            _log_llamafactory_train_wandb(
                llamafactory_wandb_writer,
                args,
                iteration,
                llamafactory_loss,
                grad_norm,
                learning_rate,
            )
        return result

    training_log._recurrent_lm_llamafactory_wandb = True  # type: ignore[attr-defined]
    training_module.training_log = training_log


def install_llamafactory_wandb_eval_log(training_module: ModuleType) -> None:
    """Make MindSpeed-LLM eval emit LLaMA-Factory-shaped W&B metrics."""

    original_eval_print = training_module.evaluate_and_print_results
    if getattr(original_eval_print, "_recurrent_lm_llamafactory_eval_wandb", False):
        return

    globals_ = original_eval_print.__globals__
    evaluate = globals_["evaluate"]
    get_args = globals_["get_args"]
    get_tensorboard_writer = globals_["get_tensorboard_writer"]
    get_wandb_writer = globals_["get_wandb_writer"]
    is_last_rank = globals_["is_last_rank"]
    math_module = globals_["math"]
    print_rank_last = globals_["print_rank_last"]

    def evaluate_and_print_results(
        prefix: str,
        forward_step_func: Any,
        data_iterator: Any,
        model: Any,
        iteration: int,
        process_non_loss_data_func: Any,
        config: Any,
        verbose: bool = False,
        write_to_tensorboard: bool = True,
        non_loss_data_func: Any = None,
    ) -> None:
        args = get_args()
        writer = get_tensorboard_writer() if write_to_tensorboard else None
        raw_wandb_writer = get_wandb_writer()
        llamafactory_wandb_writer = (
            raw_wandb_writer
            if _use_llamafactory_wandb(training_module, args, raw_wandb_writer)
            else None
        )

        total_loss_dict, collected_non_loss_data, timelimit = evaluate(
            forward_step_func,
            data_iterator,
            model,
            process_non_loss_data_func,
            config,
            verbose,
            non_loss_data_func,
        )
        if timelimit:
            return

        string = f" validation loss at {prefix} | "
        eval_loss = None
        for key in total_loss_dict:
            loss_value = total_loss_dict[key].item()
            string += "{} value: {:.6E} | ".format(key, loss_value)
            ppl = math_module.exp(min(20, loss_value))
            string += "{} PPL: {:.6E} | ".format(key, ppl)
            if key == "lm loss" or eval_loss is None:
                eval_loss = loss_value
            if writer:
                writer.add_scalar("{} validation".format(key), loss_value, iteration)
                writer.add_scalar("{} validation vs samples".format(key), loss_value, args.consumed_train_samples)
                if args.log_validation_ppl_to_tensorboard:
                    writer.add_scalar("{} validation ppl".format(key), ppl, iteration)
                    writer.add_scalar("{} validation ppl vs samples".format(key), ppl, args.consumed_train_samples)

        if (
            llamafactory_wandb_writer
            and eval_loss is not None
            and is_last_rank()
        ):
            _log_llamafactory_eval_wandb(
                llamafactory_wandb_writer,
                args,
                iteration,
                eval_loss,
            )

        if process_non_loss_data_func is not None and writer and is_last_rank():
            process_non_loss_data_func(collected_non_loss_data, iteration, writer)

        length = len(string) + 1
        print_rank_last("-" * length)
        print_rank_last(string)
        print_rank_last("-" * length)

    evaluate_and_print_results._recurrent_lm_llamafactory_eval_wandb = True  # type: ignore[attr-defined]
    training_module.evaluate_and_print_results = evaluate_and_print_results
