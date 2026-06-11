"""MindSpeed runtime loading and patches for recurrent LM training."""

from __future__ import annotations

import importlib
import os
from typing import Any

from runtime.mindspeed_patches import (
    install_mtp_feature_guard,
    install_llamafactory_wandb_eval_log,
    install_llamafactory_wandb_training_log,
)


def load_mindspeed_runtime() -> tuple[Any, Any]:
    """Install runtime guards and return MindSpeed CP batch helper + pretrain."""

    os.environ.setdefault("TRAINING_BACKEND", "mindspeed")
    install_mtp_feature_guard()

    importlib.import_module("mindspeed_llm.tasks.megatron_adaptor_v2")
    get_batch_utils = importlib.import_module("mindspeed_llm.core.context_parallel.get_batch_utils")
    training_module = importlib.import_module("mindspeed_llm.training.training")
    install_llamafactory_wandb_training_log(training_module)
    install_llamafactory_wandb_eval_log(training_module)
    return get_batch_utils.get_batch_on_this_cp_rank, training_module.pretrain


def install_mindspeed_cross_entropy_patches() -> None:
    """Install MindSpeed-LLM's NPU-friendly vocab-parallel CE helpers."""

    from megatron.core.tensor_parallel.cross_entropy import VocabParallelCrossEntropy
    from mindspeed_llm.core.tensor_parallel.cross_entropy import (
        calculate_logits_max,
        calculate_predicted_logits,
    )

    VocabParallelCrossEntropy.calculate_logits_max = staticmethod(calculate_logits_max)
    VocabParallelCrossEntropy.calculate_predicted_logits = staticmethod(calculate_predicted_logits)
