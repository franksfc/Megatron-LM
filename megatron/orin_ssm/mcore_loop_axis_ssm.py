"""Megatron Core native Orin recurrent SSM block."""

from __future__ import annotations

import copy
import math
import os
import sys
import types
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from megatron.core import parallel_state
from megatron.core.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    gather_from_tensor_model_parallel_region,
    reduce_scatter_last_dim_to_tensor_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from megatron.core.transformer.transformer_config import TransformerConfig


_MINDSPEED_SSD_MODULE = None
_MINDSPEED_CONTEXT_MODULE = None
_NPU_RMS_NORM = None
_NPU_RMS_NORM_CHECKED = False
_ORIN_TP_INIT_COUNTER = 0


def _get_npu_rms_norm() -> Any:
    global _NPU_RMS_NORM, _NPU_RMS_NORM_CHECKED
    if not _NPU_RMS_NORM_CHECKED:
        import torch_npu

        _NPU_RMS_NORM = getattr(torch_npu, "npu_rms_norm", None)
        _NPU_RMS_NORM_CHECKED = True
    if _NPU_RMS_NORM is None:
        raise RuntimeError("torch_npu.npu_rms_norm is required for the Orin MindSpeed NPU RMSNorm path.")
    return _NPU_RMS_NORM


def _mindspeed_ssm_dir() -> Path:
    llm_root = Path(
        os.getenv(
            "ORIN_MINDSPEED_LLM_ROOT",
            str(Path(__file__).resolve().parents[2] / "third_party/orin_mindspeed/MindSpeed-LLM"),
        )
    )
    return llm_root / "mindspeed_llm/tasks/models/ssm"


def _ensure_mindspeed_ssm_packages() -> None:
    package_names = [
        "mindspeed_llm",
        "mindspeed_llm.tasks",
        "mindspeed_llm.tasks.models",
        "mindspeed_llm.tasks.models.ssm",
    ]
    for package_name in package_names:
        sys.modules.setdefault(package_name, types.ModuleType(package_name))


def _load_mindspeed_context_module() -> Any:
    global _MINDSPEED_CONTEXT_MODULE
    if _MINDSPEED_CONTEXT_MODULE is not None:
        return _MINDSPEED_CONTEXT_MODULE

    ssm_dir = _mindspeed_ssm_dir()
    context_path = ssm_dir / "state_space_context_parallel.py"
    if not context_path.exists():
        raise ImportError(f"MindSpeed-LLM SSM sources were not found under {ssm_dir}.")

    _ensure_mindspeed_ssm_packages()
    context_name = "mindspeed_llm.tasks.models.ssm.state_space_context_parallel"
    if context_name not in sys.modules:
        context_spec = importlib_util.spec_from_file_location(context_name, context_path)
        if context_spec is None or context_spec.loader is None:
            raise ImportError(f"Unable to load {context_path}.")
        context_module = importlib_util.module_from_spec(context_spec)
        sys.modules[context_name] = context_module
        context_spec.loader.exec_module(context_module)
    _MINDSPEED_CONTEXT_MODULE = sys.modules[context_name]
    return _MINDSPEED_CONTEXT_MODULE


def _load_mindspeed_ssd_module() -> Any:
    global _MINDSPEED_SSD_MODULE
    if _MINDSPEED_SSD_MODULE is not None:
        return _MINDSPEED_SSD_MODULE

    ssm_dir = _mindspeed_ssm_dir()
    duality_path = ssm_dir / "state_space_duality.py"
    if not duality_path.exists():
        raise ImportError(f"MindSpeed-LLM SSM sources were not found under {ssm_dir}.")

    _load_mindspeed_context_module()

    duality_name = "_orin_mcore_mindspeed_state_space_duality"
    duality_spec = importlib_util.spec_from_file_location(duality_name, duality_path)
    if duality_spec is None or duality_spec.loader is None:
        raise ImportError(f"Unable to load {duality_path}.")
    duality_module = importlib_util.module_from_spec(duality_spec)
    sys.modules[duality_name] = duality_module
    duality_spec.loader.exec_module(duality_module)
    _MINDSPEED_SSD_MODULE = duality_module
    return duality_module


def _is_autocast_enabled(device_type: str) -> bool:
    try:
        return torch.is_autocast_enabled(device_type)
    except (TypeError, RuntimeError):
        if device_type == "cuda":
            return torch.is_autocast_enabled()
        npu_module = getattr(torch, "npu", None)
        if device_type == "npu" and npu_module is not None and hasattr(npu_module, "is_autocast_enabled"):
            return npu_module.is_autocast_enabled()
    return False


def _get_autocast_dtype(device_type: str) -> torch.dtype | None:
    try:
        return torch.get_autocast_dtype(device_type)
    except (TypeError, RuntimeError):
        if device_type == "cuda":
            return torch.get_autocast_gpu_dtype()
        npu_module = getattr(torch, "npu", None)
        if device_type == "npu" and npu_module is not None and hasattr(npu_module, "get_autocast_dtype"):
            return npu_module.get_autocast_dtype()
    return None


def _add_bias(output: Tensor, bias: Tensor | None) -> Tensor:
    return output if bias is None else output + bias


def _column_linear(layer: ColumnParallelLinear, x: Tensor, gather_output: bool | None = None) -> Tensor:
    output, bias = layer(x, runtime_gather_output=gather_output)
    return _add_bias(output, bias)


def _row_linear(layer: RowParallelLinear, x: Tensor) -> Tensor:
    output, bias = layer(x)
    return _add_bias(output, bias)


def _row_linear_reduce_scatter_last_dim(layer: RowParallelLinear, x: Tensor) -> Tensor:
    if parallel_state.get_tensor_model_parallel_world_size() == 1:
        return _row_linear(layer, x)
    if not layer.input_is_parallel:
        x = scatter_to_tensor_model_parallel_region(x)
    output_parallel = F.linear(x, layer.weight, None)
    output = reduce_scatter_last_dim_to_tensor_parallel_region(output_parallel)
    if layer.bias is not None:
        rank = parallel_state.get_tensor_model_parallel_rank()
        local_size = layer.output_size // parallel_state.get_tensor_model_parallel_world_size()
        start = rank * local_size
        output = output + layer.bias[start : start + local_size]
    return output


def _linear(layer: nn.Module, x: Tensor, gather_output: bool | None = None) -> Tensor:
    if isinstance(layer, ColumnParallelLinear):
        return _column_linear(layer, x, gather_output=gather_output)
    if isinstance(layer, RowParallelLinear):
        return _row_linear(layer, x)
    return layer(x)


def _broadcast_tensor_model_parallel_tensor(tensor: Tensor) -> Tensor:
    if parallel_state.get_tensor_model_parallel_world_size() == 1:
        return tensor
    if tensor.device.type == "cpu":
        npu_module = getattr(torch, "npu", None)
        if npu_module is not None and hasattr(npu_module, "current_device"):
            comm_device = torch.device("npu", npu_module.current_device())
        elif torch.cuda.is_available():
            comm_device = torch.device("cuda", torch.cuda.current_device())
        else:
            raise RuntimeError("Tensor-parallel initialization broadcast requires an accelerator backend.")
        comm_tensor = tensor.to(comm_device)
        torch.distributed.broadcast(
            comm_tensor,
            src=parallel_state.get_tensor_model_parallel_src_rank(),
            group=parallel_state.get_tensor_model_parallel_group(),
        )
        tensor.copy_(comm_tensor.cpu())
        return tensor
    torch.distributed.broadcast(
        tensor,
        src=parallel_state.get_tensor_model_parallel_src_rank(),
        group=parallel_state.get_tensor_model_parallel_group(),
    )
    return tensor


def _init_full_weight(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    distribution: str,
    std: float | None = None,
    a: float | None = None,
) -> Tensor:
    global _ORIN_TP_INIT_COUNTER
    seed = int(os.getenv("ORIN_TP_INIT_SEED", os.getenv("SEED", "42"))) + _ORIN_TP_INIT_COUNTER
    _ORIN_TP_INIT_COUNTER += 1
    tensor = torch.empty(shape, device="cpu", dtype=torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if distribution == "trunc_normal":
            if std is None:
                raise ValueError("`std` is required for trunc_normal initialization.")
            nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-3 * std, b=3 * std)
        elif distribution == "normal":
            if std is None:
                raise ValueError("`std` is required for normal initialization.")
            nn.init.normal_(tensor, mean=0.0, std=std)
        elif distribution == "kaiming_uniform":
            nn.init.kaiming_uniform_(tensor, a=math.sqrt(5) if a is None else a)
        elif distribution == "uniform":
            if a is None:
                raise ValueError("`a` is required for uniform initialization.")
            nn.init.uniform_(tensor, -a, a)
        else:
            raise ValueError(f"Unsupported initialization distribution: {distribution}.")
    return tensor.to(device=device, dtype=dtype)


def _copy_tp_linear_weight(
    module: nn.Module,
    *,
    distribution: str,
    std: float | None = None,
) -> None:
    weight = module.weight
    with torch.no_grad():
        if isinstance(module, ColumnParallelLinear):
            full_weight = _init_full_weight(
                (module.output_size, module.input_size),
                device=weight.device,
                dtype=weight.dtype,
                distribution=distribution,
                std=std,
            )
            rank = parallel_state.get_tensor_model_parallel_rank()
            start = rank * module.output_size_per_partition
            end = start + module.output_size_per_partition
            weight.copy_(full_weight[start:end, :])
        elif isinstance(module, RowParallelLinear):
            full_weight = _init_full_weight(
                (module.output_size, module.input_size),
                device=weight.device,
                dtype=weight.dtype,
                distribution=distribution,
                std=std,
            )
            rank = parallel_state.get_tensor_model_parallel_rank()
            start = rank * module.input_size_per_partition
            end = start + module.input_size_per_partition
            weight.copy_(full_weight[:, start:end])
        else:
            if distribution == "trunc_normal":
                if std is None:
                    raise ValueError("`std` is required for trunc_normal initialization.")
                nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
            elif distribution == "normal":
                if std is None:
                    raise ValueError("`std` is required for normal initialization.")
                nn.init.normal_(weight, mean=0.0, std=std)
            else:
                raise ValueError(f"Unsupported dense linear initialization distribution: {distribution}.")


def _init_linear_like_llamafactory(module: nn.Module, hidden_size: int, num_layers: int, recurrent_depth: int) -> None:
    if not isinstance(module, (nn.Linear, ColumnParallelLinear, RowParallelLinear)):
        return
    std = math.sqrt(2.0 / (5 * hidden_size))
    if hasattr(module, "_is_attention_output") or hasattr(module, "_is_mlp_output"):
        std = std / math.sqrt(2.0 * num_layers * recurrent_depth)
    with torch.no_grad():
        _copy_tp_linear_weight(module, distribution="trunc_normal", std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


def _zero_linear_bias(module: nn.Module) -> None:
    bias = getattr(module, "bias", None)
    if bias is not None:
        nn.init.zeros_(bias)


def _mark_sequence_parallel_param(param: nn.Parameter, tp_size: int) -> None:
    if tp_size > 1:
        param.sequence_parallel = True


class _TensorParallelAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, tensor: Tensor) -> Tensor:
        output = tensor.clone()
        torch.distributed.all_reduce(output, group=parallel_state.get_tensor_model_parallel_group())
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, group=parallel_state.get_tensor_model_parallel_group())
        return grad_input


def _tensor_parallel_all_reduce_with_grad(tensor: Tensor) -> Tensor:
    if parallel_state.get_tensor_model_parallel_world_size() == 1:
        return tensor
    return _TensorParallelAllReduce.apply(tensor)


class MCoreRMSNorm(nn.Module):
    """RMSNorm with optional tensor-parallel statistics for sharded inputs."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        *,
        partitioned: bool = False,
        global_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.partitioned = partitioned
        self.global_hidden_size = global_hidden_size or hidden_size

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        if (
            not self.partitioned
            and hidden_states.device.type == "npu"
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
            and self.weight.dtype == hidden_states.dtype
        ):
            npu_rms_norm = _get_npu_rms_norm()
            if npu_rms_norm is not None:
                return npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]
        hidden_states_float = hidden_states.float()
        if self.partitioned and parallel_state.get_tensor_model_parallel_world_size() > 1:
            variance = hidden_states_float.pow(2).sum(dim=-1, keepdim=True)
            variance = _tensor_parallel_all_reduce_with_grad(variance)
            variance = variance / float(self.global_hidden_size)
        else:
            variance = hidden_states_float.pow(2).mean(dim=-1, keepdim=True)
        hidden_states_float = hidden_states_float * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states_float.to(input_dtype)


class MCoreGatedRMSNorm(nn.Module):
    """Gated RMSNorm used by the Orin token Mamba mixer."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        *,
        partitioned: bool = False,
        global_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.partitioned = partitioned
        self.global_hidden_size = global_hidden_size or hidden_size

    def forward(self, hidden_states: Tensor, gate: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        if (
            not self.partitioned
            and hidden_states.device.type == "npu"
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
            and gate.dtype == hidden_states.dtype
            and self.weight.dtype == hidden_states.dtype
        ):
            npu_rms_norm = _get_npu_rms_norm()
            if npu_rms_norm is not None:
                gated_states = hidden_states * F.silu(gate)
                return npu_rms_norm(gated_states, self.weight, epsilon=self.variance_epsilon)[0]
        hidden_states_float = hidden_states.float() * F.silu(gate.float())
        if self.partitioned and parallel_state.get_tensor_model_parallel_world_size() > 1:
            variance = hidden_states_float.pow(2).sum(dim=-1, keepdim=True)
            variance = _tensor_parallel_all_reduce_with_grad(variance)
            variance = variance / float(self.global_hidden_size)
        else:
            variance = hidden_states_float.pow(2).mean(dim=-1, keepdim=True)
        hidden_states_float = hidden_states_float * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states_float.to(input_dtype)


class MCoreMamba2TokenBlock(nn.Module):
    """TP-aware Mamba2 token mixer used inside the Orin recurrent state."""

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        rms_norm_eps: float,
        expand: float = 2.0,
        ssm_state_size: int = 16,
        conv_kernel: int = 4,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        chunk_size: int = 32,
        head_dim: int = 64,
        n_groups: int = 1,
        clamp_dt: bool = False,
        bias: bool = False,
        conv_bias: bool = True,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if n_groups <= 0:
            raise ValueError(f"`n_groups` must be positive, got {n_groups}.")
        if ssm_state_size <= 0:
            raise ValueError(f"`ssm_state_size` must be positive, got {ssm_state_size}.")
        if conv_kernel <= 0:
            raise ValueError(f"`conv_kernel` must be positive, got {conv_kernel}.")
        if dt_min <= 0 or dt_max <= dt_min:
            raise ValueError(f"`dt_min`/`dt_max` must satisfy 0 < dt_min < dt_max, got {dt_min}/{dt_max}.")

        self.config = config
        self.linear_config = copy.copy(config)
        self.linear_config.sequence_parallel = False
        self.hidden_size = hidden_size
        self.inner_size = int(expand * hidden_size)
        self.tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()
        if self.inner_size % self.tensor_model_parallel_size != 0:
            raise ValueError(
                f"MCoreMamba2TokenBlock inner_size={self.inner_size} must be divisible by "
                f"TP={self.tensor_model_parallel_size}."
            )
        head_dim = min(int(head_dim), self.inner_size)
        while self.inner_size % head_dim != 0:
            head_dim //= 2
        if head_dim <= 0:
            raise ValueError(f"`head_dim` must have a positive divisor for inner size {self.inner_size}.")
        self.head_dim = head_dim
        self.num_heads = self.inner_size // self.head_dim
        if self.num_heads % self.tensor_model_parallel_size != 0:
            raise ValueError(
                f"num_heads={self.num_heads} must be divisible by TP={self.tensor_model_parallel_size}."
            )
        self.num_heads_local = self.num_heads // self.tensor_model_parallel_size
        self.inner_size_local = self.num_heads_local * self.head_dim
        if n_groups % self.tensor_model_parallel_size != 0:
            raise ValueError(
                f"n_groups={n_groups} must be divisible by TP={self.tensor_model_parallel_size} "
                "for the Orin MCore Mamba2 path."
            )
        self.n_groups = n_groups
        self.n_groups_local = self.n_groups // self.tensor_model_parallel_size
        if self.num_heads_local % self.n_groups_local != 0:
            raise ValueError(
                f"local num_heads={self.num_heads_local} must be divisible by local "
                f"n_groups={self.n_groups_local}."
            )

        self.ssm_state_size = ssm_state_size
        self.conv_kernel = conv_kernel
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.chunk_size = chunk_size
        self.clamp_dt = clamp_dt
        self.residual_scale = residual_scale
        self._segment_mask_cache: dict[tuple[int, str, int | None], tuple[Tensor, Tensor]] = {}
        self.conv_dim_local = self.inner_size_local + 2 * self.n_groups_local * self.ssm_state_size
        self.use_local_fast_path = self.tensor_model_parallel_size == 1
        self.use_unified_in_proj = self.tensor_model_parallel_size == 1 and not self.use_local_fast_path

        self.norm = MCoreRMSNorm(hidden_size, eps=rms_norm_eps)
        _mark_sequence_parallel_param(self.norm.weight, self.tensor_model_parallel_size)
        self.out_norm = MCoreGatedRMSNorm(
            self.inner_size_local,
            eps=rms_norm_eps,
            partitioned=self.tensor_model_parallel_size > 1,
            global_hidden_size=self.inner_size,
        )
        if self.use_local_fast_path:
            self.in_proj = nn.Linear(
                hidden_size,
                self.inner_size + self.inner_size + 2 * self.n_groups * self.ssm_state_size + self.num_heads,
                bias=bias,
            )
        elif self.use_unified_in_proj:
            self.in_proj = ColumnParallelLinear(
                hidden_size,
                self.inner_size + self.inner_size + 2 * self.n_groups * self.ssm_state_size + self.num_heads,
                config=self.linear_config,
                init_method=config.init_method,
                bias=bias,
                gather_output=False,
                skip_bias_add=False,
            )
        else:
            self.gate_proj = ColumnParallelLinear(
                hidden_size,
                self.inner_size,
                config=self.linear_config,
                init_method=config.init_method,
                bias=bias,
                gather_output=False,
                skip_bias_add=False,
            )
            self.x_proj = ColumnParallelLinear(
                hidden_size,
                self.inner_size,
                config=self.linear_config,
                init_method=config.init_method,
                bias=bias,
                gather_output=False,
                skip_bias_add=False,
            )
            self.b_proj = ColumnParallelLinear(
                hidden_size,
                self.n_groups * self.ssm_state_size,
                config=self.linear_config,
                init_method=config.init_method,
                bias=bias,
                gather_output=False,
                skip_bias_add=False,
            )
            self.c_proj = ColumnParallelLinear(
                hidden_size,
                self.n_groups * self.ssm_state_size,
                config=self.linear_config,
                init_method=config.init_method,
                bias=bias,
                gather_output=False,
                skip_bias_add=False,
            )
            self.dt_proj = ColumnParallelLinear(
                hidden_size,
                self.num_heads,
                config=self.linear_config,
                init_method=config.init_method,
                bias=False,
                gather_output=False,
                skip_bias_add=False,
            )
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim_local,
            out_channels=self.conv_dim_local,
            kernel_size=conv_kernel,
            groups=self.conv_dim_local,
            padding=conv_kernel - 1,
            bias=conv_bias,
        )
        if self.use_local_fast_path:
            self.out_proj = nn.Linear(self.inner_size, hidden_size, bias=bias)
        else:
            self.out_proj = RowParallelLinear(
                self.inner_size,
                hidden_size,
                config=self.linear_config,
                init_method=config.init_method,
                bias=bias,
                input_is_parallel=True,
                skip_bias_add=False,
            )
        self.out_proj._is_mlp_output = True
        self.act = nn.SiLU()

        rank = parallel_state.get_tensor_model_parallel_rank()
        head_start = rank * self.num_heads_local
        A = torch.arange(head_start + 1, head_start + self.num_heads_local + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.num_heads_local))
        self.D._no_weight_decay = True
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads_local))

    def _reset_in_proj_from_full_weight(self) -> None:
        std = math.sqrt(2.0 / (5 * self.hidden_size))
        if self.use_local_fast_path:
            return
        full_conv_dim = self.inner_size + 2 * self.n_groups * self.ssm_state_size
        full_out = self.inner_size + full_conv_dim + self.num_heads
        full_weight = _init_full_weight(
            (full_out, self.hidden_size),
            device=self.dt_bias.device,
            dtype=self.dt_bias.dtype,
            distribution="trunc_normal",
            std=std,
        )
        rank = parallel_state.get_tensor_model_parallel_rank()
        inner_start = rank * self.inner_size_local
        inner_end = inner_start + self.inner_size_local
        bc_local = self.n_groups_local * self.ssm_state_size
        b_start = self.inner_size + self.inner_size + rank * bc_local
        b_end = b_start + bc_local
        c_start = self.inner_size + self.inner_size + self.n_groups * self.ssm_state_size + rank * bc_local
        c_end = c_start + bc_local
        dt_start = self.inner_size + full_conv_dim + rank * self.num_heads_local
        dt_end = dt_start + self.num_heads_local
        if self.use_unified_in_proj:
            start = rank * self.in_proj.output_size_per_partition
            end = start + self.in_proj.output_size_per_partition
            self.in_proj.weight.copy_(full_weight[start:end, :])
            _zero_linear_bias(self.in_proj)
            return
        self.gate_proj.weight.copy_(full_weight[inner_start:inner_end, :])
        self.x_proj.weight.copy_(full_weight[self.inner_size + inner_start : self.inner_size + inner_end, :])
        self.b_proj.weight.copy_(full_weight[b_start:b_end, :])
        self.c_proj.weight.copy_(full_weight[c_start:c_end, :])
        self.dt_proj.weight.copy_(full_weight[dt_start:dt_end, :])
        for module in (self.gate_proj, self.x_proj, self.b_proj, self.c_proj, self.dt_proj):
            _zero_linear_bias(module)

    def reset_mamba_parameters(self) -> None:
        with torch.no_grad():
            self._reset_in_proj_from_full_weight()
            full_conv_dim = self.inner_size + 2 * self.n_groups * self.ssm_state_size
            rank = parallel_state.get_tensor_model_parallel_rank()
            x_start = rank * self.inner_size_local
            x_end = x_start + self.inner_size_local
            bc_local = self.n_groups_local * self.ssm_state_size
            b_start = self.inner_size + rank * bc_local
            b_end = b_start + bc_local
            c_start = self.inner_size + self.n_groups * self.ssm_state_size + rank * bc_local
            c_end = c_start + bc_local
            full_conv_weight = _init_full_weight(
                (full_conv_dim, 1, self.conv_kernel),
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
                distribution="kaiming_uniform",
            )
            self.conv1d.weight.copy_(
                torch.cat(
                    (
                        full_conv_weight[x_start:x_end],
                        full_conv_weight[b_start:b_end],
                        full_conv_weight[c_start:c_end],
                    ),
                    dim=0,
                )
            )
            if self.conv1d.bias is not None:
                fan_in = self.conv1d.in_channels * self.conv1d.kernel_size[0] / self.conv1d.groups
                bound = 1 / math.sqrt(fan_in)
                full_conv_bias = _init_full_weight(
                    (full_conv_dim,),
                    device=self.conv1d.bias.device,
                    dtype=self.conv1d.bias.dtype,
                    distribution="uniform",
                    a=bound,
                )
                self.conv1d.bias.copy_(
                    torch.cat(
                        (
                            full_conv_bias[x_start:x_end],
                            full_conv_bias[b_start:b_end],
                            full_conv_bias[c_start:c_end],
                        ),
                        dim=0,
                    )
                )

            full_dt = _init_full_weight(
                (self.num_heads,),
                device=self.dt_bias.device,
                dtype=self.dt_bias.dtype,
                distribution="uniform",
                a=1.0,
            )
            full_dt = torch.exp(
                math.log(self.dt_min) + ((full_dt + 1.0) * 0.5) * (math.log(self.dt_max) - math.log(self.dt_min))
            ).clamp(min=1e-4)
            head_start = rank * self.num_heads_local
            head_end = head_start + self.num_heads_local
            inv_dt = full_dt[head_start:head_end] + torch.log(-torch.expm1(-full_dt[head_start:head_end]))
            self.dt_bias.copy_(inv_dt)
            _copy_tp_linear_weight(self.out_proj, distribution="normal", std=5e-3)
            if self.out_proj.bias is not None:
                self.out_proj.bias.zero_()

    def _pad_sequence(self, x: Tensor, pad_size: int) -> Tensor:
        if pad_size == 0:
            return x
        if x.dim() == 3:
            return F.pad(x, (0, 0, 0, pad_size, 0, 0))
        if x.dim() == 4:
            return F.pad(x, (0, 0, 0, 0, 0, pad_size, 0, 0))
        raise ValueError(f"Unsupported Mamba SSD tensor rank: {x.dim()}.")

    def _get_segment_masks(self, chunk_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        key = (chunk_size, device.type, device.index)
        masks = self._segment_mask_cache.get(key)
        if masks is None:
            lower_mask = torch.tril(
                torch.ones(chunk_size, chunk_size, device=device, dtype=torch.bool),
                diagonal=-1,
            )
            lower_equal_mask = torch.tril(
                torch.ones(chunk_size, chunk_size, device=device, dtype=torch.bool),
                diagonal=0,
            )
            masks = (lower_mask, lower_equal_mask)
            self._segment_mask_cache[key] = masks
        return masks

    def _segmented_sum(self, x: Tensor) -> Tensor:
        chunk_size = x.size(-1)
        lower_mask, lower_equal_mask = self._get_segment_masks(chunk_size, x.device)
        x = x.unsqueeze(-1).expand(*x.shape, chunk_size)
        x = x.masked_fill(~lower_mask, 0)
        x = torch.cumsum(x, dim=-2)
        return x.masked_fill(~lower_equal_mask, -torch.inf)

    def _expand_group_params(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.n_groups_local, self.ssm_state_size)
        if self.n_groups_local == self.num_heads_local:
            return x
        repeat_factor = self.num_heads_local // self.n_groups_local
        return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, self.n_groups_local, repeat_factor, self.ssm_state_size)
            .reshape(batch_size, seq_len, self.num_heads_local, self.ssm_state_size)
        )

    def _selective_scan_mindspeed_fast(self, x: Tensor, dt_logits_without_bias: Tensor, B: Tensor, C: Tensor) -> Tensor:
        """MindSpeed-LLM SSD specialized for context-parallel size 1."""
        input_dtype = x.dtype
        batch_size, seq_len, _ = x.shape
        chunk_size = min(max(1, self.chunk_size), seq_len)
        pad_size = (chunk_size - (seq_len % chunk_size)) % chunk_size
        matmul_dtype = (
            input_dtype
            if x.device.type != "cpu" and input_dtype in (torch.float16, torch.bfloat16)
            else torch.float32
        )

        x = rearrange(x.float(), "b l (h p) -> b l h p", h=self.num_heads_local, p=self.head_dim)
        x_residual = x
        dt = F.softplus(dt_logits_without_bias.float() + self.dt_bias.float().view(1, 1, self.num_heads_local))
        if self.clamp_dt:
            dt = dt.clamp(min=self.dt_min, max=self.dt_max)

        A = -torch.exp(self.A_log.float())
        A = A.view(1, 1, self.num_heads_local) * dt
        x = x * dt.unsqueeze(-1)
        B = self._expand_group_params(B.float())
        C = self._expand_group_params(C.float())

        x = rearrange(self._pad_sequence(x, pad_size), "b (c l) h p -> b c l h p", l=chunk_size)
        A = rearrange(self._pad_sequence(A, pad_size), "b (c l) h -> b c l h", l=chunk_size)
        B = rearrange(self._pad_sequence(B, pad_size), "b (c l) h n -> b c l h n", l=chunk_size)
        C = rearrange(self._pad_sequence(C, pad_size), "b (c l) h n -> b c l h n", l=chunk_size)

        A_hcl = rearrange(A, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_hcl, dim=-1)
        decay_within_chunk = torch.exp(self._segmented_sum(A_hcl)).to(matmul_dtype)

        C_r = C.permute(0, 3, 1, 2, 4)
        B_r = B.permute(0, 3, 1, 2, 4)
        x_r = x.permute(0, 3, 1, 2, 4)
        C_b = C_r.reshape(-1, chunk_size, self.ssm_state_size).to(matmul_dtype)
        B_b = B_r.reshape(-1, chunk_size, self.ssm_state_size).transpose(1, 2).to(matmul_dtype)
        x_b = x_r.reshape(-1, chunk_size, self.head_dim).to(matmul_dtype)
        cb = torch.bmm(C_b, B_b)
        y_diag = torch.bmm(cb * decay_within_chunk.reshape(-1, chunk_size, chunk_size), x_b)
        y_diag = y_diag.reshape(x_r.shape).permute(0, 2, 3, 1, 4).contiguous()

        state_decay = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum).to(matmul_dtype)
        states = torch.bmm(
            (B_r.to(matmul_dtype) * state_decay.unsqueeze(-1))
            .reshape(-1, chunk_size, self.ssm_state_size)
            .transpose(1, 2),
            x_b,
        )
        states = states.transpose(1, 2).reshape(
            batch_size,
            self.num_heads_local,
            -1,
            self.head_dim,
            self.ssm_state_size,
        )
        states = states.permute(0, 2, 1, 3, 4).contiguous()

        initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        chunk_decay = torch.exp(self._segmented_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))).to(matmul_dtype)
        new_states = torch.einsum("bhzc,bchpn->bzhpn", chunk_decay, states.to(matmul_dtype))
        states = new_states[:, :-1]

        state_decay_out = torch.exp(A_cumsum).to(matmul_dtype)
        states_b = states.permute(0, 2, 1, 3, 4).reshape(-1, self.head_dim, self.ssm_state_size)
        states_b = states_b.transpose(-1, -2)
        cs = torch.bmm(C_b, states_b)
        cs = cs.reshape(C_r.shape[0], C_r.shape[1], C_r.shape[2], C_r.shape[3], self.head_dim)
        y_off = (cs * state_decay_out.unsqueeze(-1)).permute(0, 2, 3, 1, 4).contiguous()

        D = self.D.float().view(1, 1, self.num_heads_local, 1)
        x_padded = self._pad_sequence(x_residual, pad_size)
        y = rearrange(y_diag + y_off, "b c l h p -> b (c l) h p")
        y = y + D.to(y.dtype) * x_padded.to(y.dtype)
        y = y[:, :seq_len]
        return rearrange(y, "b l h p -> b l (h p)").to(input_dtype)

    def _selective_scan_mindspeed(self, x: Tensor, dt_logits_without_bias: Tensor, B: Tensor, C: Tensor) -> Tensor:
        ssd_module = _load_mindspeed_ssd_module()
        input_dtype = x.dtype
        dt_min = self.dt_min if self.clamp_dt else 0.0
        dt_max = self.dt_max if self.clamp_dt else torch.finfo(torch.float32).max
        config = {
            "nheads_local": self.num_heads_local,
            "ngroups_local": self.n_groups_local,
            "dt_min": dt_min,
            "dt_max": dt_max,
            "dt_bias": self.dt_bias,
            "headdim": self.head_dim,
            "d_state": self.ssm_state_size,
            "chunk_size": min(max(1, self.chunk_size), x.shape[1]),
            "D_has_hdim": False,
        }
        inputs = ssd_module.ProcessInputs(
            x=x,
            dt=dt_logits_without_bias,
            A=-torch.exp(self.A_log.float()),
            B=B,
            C=C,
            D=self.D,
        )
        processor = ssd_module.StateSpaceProcessor(config=config)
        y = processor.process(inputs, ssd_module.StateOptions())
        return rearrange(y, "b l h p -> b l (h p)").to(input_dtype)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        output_is_tensor_parallel: bool = False,
    ) -> Tensor:
        residual = hidden_states
        input_dtype = hidden_states.dtype
        hidden_states = self.norm(hidden_states)
        if attention_mask is not None and attention_mask.dim() == 2:
            hidden_states = hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)

        if self.use_local_fast_path:
            projected_states = self.in_proj(hidden_states)
            gate, hidden_states_B_C, dt = torch.split(
                projected_states,
                [self.inner_size_local, self.conv_dim_local, self.num_heads_local],
                dim=-1,
            )
        elif self.use_unified_in_proj:
            projected_states = _column_linear(self.in_proj, hidden_states)
            gate, hidden_states_B_C, dt = torch.split(
                projected_states,
                [self.inner_size_local, self.conv_dim_local, self.num_heads_local],
                dim=-1,
            )
        else:
            gate = _column_linear(self.gate_proj, hidden_states)
            x = _column_linear(self.x_proj, hidden_states)
            B = _column_linear(self.b_proj, hidden_states)
            C = _column_linear(self.c_proj, hidden_states)
            dt = _column_linear(self.dt_proj, hidden_states)
            hidden_states_B_C = torch.cat((x, B, C), dim=-1)
        seq_len = hidden_states_B_C.shape[1]
        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size > 1:
            context_module = _load_mindspeed_context_module()
            hidden_states_B_C, dt = context_module.SequenceParallelConvFunction.apply(
                hidden_states_B_C,
                dt,
                self.conv1d.weight,
                self.conv1d.bias,
                self.dt_bias,
                parallel_state.get_context_parallel_group(),
                cp_size,
                parallel_state.get_context_parallel_rank(),
                self.conv_kernel,
                self.num_heads_local,
                self.inner_size_local,
                self.ssm_state_size,
                self.n_groups_local,
            )
        else:
            hidden_states_B_C = self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2)
            hidden_states_B_C = self.act(hidden_states_B_C)
        x, B, C = torch.split(
            hidden_states_B_C,
            [
                self.inner_size_local,
                self.n_groups_local * self.ssm_state_size,
                self.n_groups_local * self.ssm_state_size,
            ],
            dim=-1,
        )
        if attention_mask is not None and attention_mask.dim() == 2:
            x = x * attention_mask[:, :, None].to(x.dtype)

        if cp_size == 1:
            scan_output = self._selective_scan_mindspeed_fast(x, dt, B, C)
        else:
            scan_output = self._selective_scan_mindspeed(x, dt, B, C)
        scan_output = self.out_norm(scan_output, gate).to(input_dtype)
        if output_is_tensor_parallel:
            if parallel_state.get_tensor_model_parallel_world_size() == 1:
                output = residual + self.residual_scale * self.out_proj(scan_output)
                if attention_mask is not None and attention_mask.dim() == 2:
                    output = output * attention_mask[:, :, None].to(output.dtype)
                return output
            if not isinstance(self.out_proj, RowParallelLinear):
                raise RuntimeError("Tensor-parallel Mamba output requires a RowParallelLinear out_proj.")
            residual = scatter_to_tensor_model_parallel_region(residual)
            output = residual + self.residual_scale * _row_linear_reduce_scatter_last_dim(
                self.out_proj,
                scan_output,
            )
        else:
            output = residual + self.residual_scale * _linear(self.out_proj, scan_output)
        if attention_mask is not None and attention_mask.dim() == 2:
            output = output * attention_mask[:, :, None].to(output.dtype)
        return output

class MCoreLoopAxisSSM(nn.Module):
    """TP-aware MCore implementation of the Orin recurrent SSM loop."""

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        state_size: int,
        rms_norm_eps: float,
        lambda_min: float = 0.01,
        lambda_max: float = 4.0,
        beta: float = 0.8,
        out_scale: float = 0.3,
        eta0: float = 0.3,
        token_mamba_expand: float = 2.0,
        token_mamba_state_size: int = 16,
        token_mamba_conv_kernel: int = 4,
        token_mamba_dt_rank: int | str = "auto",
        token_mamba_dt_min: float = 0.001,
        token_mamba_dt_max: float = 0.1,
        token_mamba_chunk_size: int = 32,
        token_mamba_head_dim: int = 64,
        token_mamba_variant: str = "mamba2",
        token_mamba_n_groups: int = 1,
        token_mamba_clamp_dt: bool = False,
        token_mamba_bias: bool = False,
        token_mamba_conv_bias: bool = True,
        token_mamba_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        del token_mamba_dt_rank
        token_mamba_variant = str(token_mamba_variant).strip().lower()
        if token_mamba_variant in ("orin_mamba2", "orin_mamba2_fast"):
            token_mamba_variant = "mamba2"
        if token_mamba_variant != "mamba2":
            raise ValueError("MCoreLoopAxisSSM currently supports the Orin Mamba2 recurrent path only.")

        self.config = config
        self.linear_config = copy.copy(config)
        self.linear_config.sequence_parallel = False
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.beta = beta
        self.out_scale = out_scale
        self.eta0 = eta0
        self.token_mamba_variant = token_mamba_variant
        self.tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()
        self.use_local_fast_path = self.tensor_model_parallel_size == 1

        self.drive_norm = MCoreRMSNorm(hidden_size, eps=rms_norm_eps)
        self.state_norm = MCoreRMSNorm(state_size, eps=rms_norm_eps)
        _mark_sequence_parallel_param(self.drive_norm.weight, self.tensor_model_parallel_size)
        _mark_sequence_parallel_param(self.state_norm.weight, self.tensor_model_parallel_size)
        if self.use_local_fast_path:
            self.u_proj = nn.Linear(hidden_size, state_size, bias=False)
            self.delta_proj = nn.Linear(hidden_size, state_size, bias=True)
            self.state_to_z = nn.Linear(state_size, hidden_size, bias=False)
            self.u_to_z = nn.Linear(hidden_size, hidden_size, bias=False)
            self.gate_proj = nn.Linear(hidden_size + state_size, hidden_size, bias=True)
            self.state_to_out = nn.Linear(state_size, hidden_size, bias=False)
        else:
            self.u_proj = ColumnParallelLinear(
                hidden_size,
                state_size,
                config=self.linear_config,
                init_method=config.init_method,
                bias=False,
                gather_output=True,
                skip_bias_add=False,
            )
            self.delta_proj = ColumnParallelLinear(
                hidden_size,
                state_size,
                config=self.linear_config,
                init_method=config.init_method,
                bias=True,
                gather_output=True,
                skip_bias_add=False,
            )
            self.state_to_z = RowParallelLinear(
                state_size,
                hidden_size,
                config=self.linear_config,
                init_method=config.init_method,
                bias=False,
                input_is_parallel=False,
                skip_bias_add=False,
            )
            self.u_to_z = ColumnParallelLinear(
                hidden_size,
                hidden_size,
                config=self.linear_config,
                init_method=config.init_method,
                bias=False,
                gather_output=True,
                skip_bias_add=False,
            )
            self.gate_proj = ColumnParallelLinear(
                hidden_size + state_size,
                hidden_size,
                config=self.linear_config,
                init_method=config.init_method,
                bias=True,
                gather_output=True,
                skip_bias_add=False,
            )
            self.state_to_out = RowParallelLinear(
                state_size,
                hidden_size,
                config=self.linear_config,
                init_method=config.init_method,
                bias=False,
                input_is_parallel=False,
                skip_bias_add=False,
                )

        self.state_correction_logit = nn.Parameter(torch.full((state_size,), -2.0))
        self.state_delta_logit = nn.Parameter(torch.full((state_size,), -2.0))
        self.z_correction_logit = nn.Parameter(torch.full((hidden_size,), -2.0))
        self.state_correction_logit._no_weight_decay = True
        self.state_delta_logit._no_weight_decay = True
        self.z_correction_logit._no_weight_decay = True
        _mark_sequence_parallel_param(self.state_correction_logit, self.tensor_model_parallel_size)
        _mark_sequence_parallel_param(self.state_delta_logit, self.tensor_model_parallel_size)
        _mark_sequence_parallel_param(self.z_correction_logit, self.tensor_model_parallel_size)
        self.token_mamba_block = MCoreMamba2TokenBlock(
            config=config,
            hidden_size=state_size,
            rms_norm_eps=rms_norm_eps,
            expand=token_mamba_expand,
            ssm_state_size=token_mamba_state_size,
            conv_kernel=token_mamba_conv_kernel,
            dt_min=token_mamba_dt_min,
            dt_max=token_mamba_dt_max,
            chunk_size=token_mamba_chunk_size,
            head_dim=token_mamba_head_dim,
            n_groups=token_mamba_n_groups,
            clamp_dt=token_mamba_clamp_dt,
            bias=token_mamba_bias,
            conv_bias=token_mamba_conv_bias,
            residual_scale=token_mamba_residual_scale,
        )

        lambdas = torch.logspace(math.log10(lambda_min), math.log10(lambda_max), state_size)
        self.A_log = nn.Parameter(torch.log(lambdas))
        self.A_log._no_weight_decay = True
        recurrent_depth = max(
            1,
            int(getattr(config, "orin_more_iterations", 0) or 0) + 1,
            int(getattr(config, "orin_more_eval_iterations", 0) or 0) + 1,
        )
        num_layers = int(getattr(config, "num_layers", 1) or 1)
        self.apply(
            lambda module: _init_linear_like_llamafactory(
                module,
                hidden_size=hidden_size,
                num_layers=num_layers,
                recurrent_depth=recurrent_depth,
            )
        )
        self.reset_small_init()

    def reset_small_init(self) -> None:
        eta0 = max(float(self.eta0), 1e-4)
        inv_eta0 = eta0 + math.log(-math.expm1(-eta0))
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.constant_(self.delta_proj.bias, inv_eta0)
        _copy_tp_linear_weight(self.state_to_z, distribution="normal", std=1e-2)
        _copy_tp_linear_weight(self.u_to_z, distribution="normal", std=1e-2)
        _copy_tp_linear_weight(self.gate_proj, distribution="normal", std=1e-3)
        nn.init.constant_(self.gate_proj.bias, 1.0)
        _copy_tp_linear_weight(self.state_to_out, distribution="normal", std=5e-3)
        nn.init.constant_(self.state_correction_logit, -2.0)
        nn.init.constant_(self.state_delta_logit, -2.0)
        nn.init.constant_(self.z_correction_logit, -2.0)
        self.token_mamba_block.reset_mamba_parameters()

    def _get_compute_dtype(self, tensor: Tensor) -> torch.dtype:
        if tensor.device.type in ("cuda", "npu") and _is_autocast_enabled(tensor.device.type):
            autocast_dtype = _get_autocast_dtype(tensor.device.type)
            if autocast_dtype is not None:
                return autocast_dtype
        return tensor.dtype

    def init_state(self, embeds: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, seq_len, _ = embeds.shape
        state = embeds.new_zeros(batch_size, seq_len, self.state_size)
        z = torch.zeros_like(embeds)
        return state, z

    def get_decay_rate(self) -> Tensor:
        return torch.exp(self.A_log.float()).view(1, 1, -1)

    def forward(
        self,
        state: Tensor,
        z: Tensor,
        backbone_out: Tensor,
        loop_idx: int = 0,
        attention_mask: Tensor | None = None,
        decay_rate: Tensor | None = None,
        compute_next_z: bool = True,
    ) -> tuple[Tensor, Tensor]:
        next_state, next_z, _ = self.forward_with_state_norm_cache(
            state=state,
            z=z,
            backbone_out=backbone_out,
            loop_idx=loop_idx,
            attention_mask=attention_mask,
            decay_rate=decay_rate,
            compute_next_z=compute_next_z,
            prev_state_norm=None,
        )
        return next_state, next_z

    def forward_with_state_norm_cache(
        self,
        state: Tensor,
        z: Tensor,
        backbone_out: Tensor,
        loop_idx: int = 0,
        attention_mask: Tensor | None = None,
        decay_rate: Tensor | None = None,
        compute_next_z: bool = True,
        prev_state_norm: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        del loop_idx
        compute_dtype = self._get_compute_dtype(backbone_out)
        state = state.to(compute_dtype)
        z = z.to(compute_dtype)
        u_norm = self.drive_norm(backbone_out).to(compute_dtype)
        drive = _linear(self.u_proj, u_norm, gather_output=True)
        delta = F.softplus(_linear(self.delta_proj, u_norm, gather_output=True).float())
        if decay_rate is None:
            decay_rate = self.get_decay_rate()
        decay = torch.exp(-decay_rate * delta).to(dtype=compute_dtype)
        eta = 1.0 - decay
        base_state_loop = state + eta * (drive - state)
        state_correction = torch.sigmoid(self.state_correction_logit.float()).view(1, 1, -1).to(compute_dtype)
        next_state_loop = base_state_loop + state_correction * eta * (drive - base_state_loop)

        next_state = self.token_mamba_block(next_state_loop, attention_mask=attention_mask)
        if not compute_next_z:
            if attention_mask is not None and attention_mask.dim() == 2:
                mask = attention_mask[:, :, None]
                next_state = next_state * mask.to(next_state.dtype)
            return next_state, z, None

        if prev_state_norm is None:
            prev_state_norm = self.state_norm(state).to(compute_dtype)
        else:
            prev_state_norm = prev_state_norm.to(compute_dtype)
        state_norm = self.state_norm(next_state).to(compute_dtype)
        state_delta = state_norm - prev_state_norm
        state_delta_gain = torch.sigmoid(self.state_delta_logit.float()).view(1, 1, -1).to(compute_dtype)
        target_z = _linear(self.u_to_z, u_norm, gather_output=True) + _linear(
            self.state_to_z,
            state_norm + state_delta_gain * state_delta,
        )
        alpha = torch.sigmoid(_linear(self.gate_proj, torch.cat((u_norm, state_norm), dim=-1), gather_output=True))
        base_z = z + alpha * (target_z - z)
        z_correction = torch.sigmoid(self.z_correction_logit.float()).view(1, 1, -1).to(compute_dtype)
        next_z = base_z + z_correction * alpha * (target_z - base_z)
        if attention_mask is not None and attention_mask.dim() == 2:
            mask = attention_mask[:, :, None]
            next_state = next_state * mask.to(next_state.dtype)
            next_z = next_z * mask.to(next_z.dtype)
            state_norm = state_norm * mask.to(state_norm.dtype)
        return next_state, next_z, state_norm

    def output_readout(self, state: Tensor) -> Tensor:
        return self.out_scale * _linear(self.state_to_out, state)


class MCoreLoopAxisSBHTPSSM(nn.Module):
    """SBH-layout Orin recurrent loop with TP-sharded recurrent state.

    The recurrent state is partitioned across tensor-parallel ranks. Operations
    that depend on full-state semantics, including token Mamba, gate projection,
    and RMSNorm statistics, gather or all-reduce as needed to match the
    LLaMA-Factory Orin Mamba2-fast architecture.
    """

    uses_sbh_layout = True

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        state_size: int,
        rms_norm_eps: float,
        lambda_min: float = 0.01,
        lambda_max: float = 4.0,
        beta: float = 0.8,
        out_scale: float = 0.3,
        eta0: float = 0.3,
        token_mamba_expand: float = 2.0,
        token_mamba_state_size: int = 16,
        token_mamba_conv_kernel: int = 4,
        token_mamba_dt_rank: int | str = "auto",
        token_mamba_dt_min: float = 0.001,
        token_mamba_dt_max: float = 0.1,
        token_mamba_chunk_size: int = 32,
        token_mamba_head_dim: int = 64,
        token_mamba_variant: str = "mamba2",
        token_mamba_n_groups: int = 1,
        token_mamba_clamp_dt: bool = False,
        token_mamba_bias: bool = False,
        token_mamba_conv_bias: bool = True,
        token_mamba_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        del token_mamba_dt_rank
        token_mamba_variant = str(token_mamba_variant).strip().lower()
        if token_mamba_variant in ("orin_mamba2", "orin_mamba2_fast"):
            token_mamba_variant = "mamba2"
        if token_mamba_variant != "mamba2":
            raise ValueError("MCoreLoopAxisSBHTPSSM supports the Orin Mamba2 recurrent path only.")

        self.config = config
        self.linear_config = copy.copy(config)
        self.linear_config.sequence_parallel = False
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.beta = beta
        self.out_scale = out_scale
        self.eta0 = eta0
        self.token_mamba_variant = token_mamba_variant
        self.tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()
        if state_size % self.tensor_model_parallel_size != 0:
            raise ValueError(f"state_size={state_size} must be divisible by TP={self.tensor_model_parallel_size}.")
        if hidden_size % self.tensor_model_parallel_size != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by TP={self.tensor_model_parallel_size}.")
        self.state_size_local = state_size // self.tensor_model_parallel_size
        self.hidden_size_local = hidden_size // self.tensor_model_parallel_size

        self.drive_norm = MCoreRMSNorm(hidden_size, eps=rms_norm_eps)
        _mark_sequence_parallel_param(self.drive_norm.weight, self.tensor_model_parallel_size)
        self.state_norm = MCoreRMSNorm(
            self.state_size_local,
            eps=rms_norm_eps,
            partitioned=self.tensor_model_parallel_size > 1,
            global_hidden_size=state_size,
        )
        self.u_proj = ColumnParallelLinear(
            hidden_size,
            state_size,
            config=self.linear_config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
        )
        self.delta_proj = ColumnParallelLinear(
            hidden_size,
            state_size,
            config=self.linear_config,
            init_method=config.init_method,
            bias=True,
            gather_output=False,
            skip_bias_add=False,
        )
        self.state_to_z = RowParallelLinear(
            state_size,
            hidden_size,
            config=self.linear_config,
            init_method=config.init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
        )
        self.u_to_z = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            config=self.linear_config,
            init_method=config.init_method,
            bias=False,
            gather_output=True,
            skip_bias_add=False,
        )
        self.gate_proj = ColumnParallelLinear(
            hidden_size + state_size,
            hidden_size,
            config=self.linear_config,
            init_method=config.init_method,
            bias=True,
            gather_output=True,
            skip_bias_add=False,
        )
        self.state_to_out = RowParallelLinear(
            state_size,
            hidden_size,
            config=self.linear_config,
            init_method=config.init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
        )

        rank = parallel_state.get_tensor_model_parallel_rank()
        start = rank * self.state_size_local
        end = start + self.state_size_local
        lambdas = torch.logspace(math.log10(lambda_min), math.log10(lambda_max), state_size)
        self.A_log = nn.Parameter(torch.log(lambdas[start:end].contiguous()))
        self.A_log._no_weight_decay = True
        self.state_correction_logit = nn.Parameter(torch.full((self.state_size_local,), -2.0))
        self.state_delta_logit = nn.Parameter(torch.full((self.state_size_local,), -2.0))
        self.z_correction_logit = nn.Parameter(torch.full((hidden_size,), -2.0))
        self.state_correction_logit._no_weight_decay = True
        self.state_delta_logit._no_weight_decay = True
        self.z_correction_logit._no_weight_decay = True
        _mark_sequence_parallel_param(self.z_correction_logit, self.tensor_model_parallel_size)
        self.token_mamba_block = MCoreMamba2TokenBlock(
            config=config,
            hidden_size=state_size,
            rms_norm_eps=rms_norm_eps,
            expand=token_mamba_expand,
            ssm_state_size=token_mamba_state_size,
            conv_kernel=token_mamba_conv_kernel,
            dt_min=token_mamba_dt_min,
            dt_max=token_mamba_dt_max,
            chunk_size=token_mamba_chunk_size,
            head_dim=token_mamba_head_dim,
            n_groups=token_mamba_n_groups,
            clamp_dt=token_mamba_clamp_dt,
            bias=token_mamba_bias,
            conv_bias=token_mamba_conv_bias,
            residual_scale=token_mamba_residual_scale,
        )

        recurrent_depth = max(
            1,
            int(getattr(config, "orin_more_iterations", 0) or 0) + 1,
            int(getattr(config, "orin_more_eval_iterations", 0) or 0) + 1,
        )
        num_layers = int(getattr(config, "num_layers", 1) or 1)
        self.apply(
            lambda module: _init_linear_like_llamafactory(
                module,
                hidden_size=hidden_size,
                num_layers=num_layers,
                recurrent_depth=recurrent_depth,
            )
        )
        self.reset_small_init()

    def reset_small_init(self) -> None:
        eta0 = max(float(self.eta0), 1e-4)
        inv_eta0 = eta0 + math.log(-math.expm1(-eta0))
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.constant_(self.delta_proj.bias, inv_eta0)
        _copy_tp_linear_weight(self.state_to_z, distribution="normal", std=1e-2)
        _copy_tp_linear_weight(self.u_to_z, distribution="normal", std=1e-2)
        _copy_tp_linear_weight(self.gate_proj, distribution="normal", std=1e-3)
        nn.init.constant_(self.gate_proj.bias, 1.0)
        _copy_tp_linear_weight(self.state_to_out, distribution="normal", std=5e-3)
        nn.init.constant_(self.state_correction_logit, -2.0)
        nn.init.constant_(self.state_delta_logit, -2.0)
        nn.init.constant_(self.z_correction_logit, -2.0)
        self.token_mamba_block.reset_mamba_parameters()

    def _get_compute_dtype(self, tensor: Tensor) -> torch.dtype:
        if tensor.device.type in ("cuda", "npu") and _is_autocast_enabled(tensor.device.type):
            autocast_dtype = _get_autocast_dtype(tensor.device.type)
            if autocast_dtype is not None:
                return autocast_dtype
        return tensor.dtype

    def init_state(self, embeds: Tensor) -> tuple[Tensor, Tensor]:
        seq_len, batch_size, _ = embeds.shape
        state = embeds.new_zeros(seq_len, batch_size, self.state_size_local)
        z = embeds.new_zeros(seq_len, batch_size, self.hidden_size)
        return state, z

    def z_for_backbone(self, z: Tensor) -> Tensor:
        return z

    def get_decay_rate(self) -> Tensor:
        return torch.exp(self.A_log.float()).view(1, 1, -1)

    def _mask_sbh(self, tensor: Tensor, attention_mask: Tensor | None) -> Tensor:
        if attention_mask is None or attention_mask.dim() != 2:
            return tensor
        mask = attention_mask.transpose(0, 1).contiguous()[:, :, None]
        return tensor * mask.to(tensor.dtype)

    def _gather_state(self, state: Tensor) -> Tensor:
        if self.tensor_model_parallel_size == 1:
            return state
        return gather_from_tensor_model_parallel_region(state)

    def forward(
        self,
        state: Tensor,
        z: Tensor,
        backbone_out: Tensor,
        loop_idx: int = 0,
        attention_mask: Tensor | None = None,
        decay_rate: Tensor | None = None,
        compute_next_z: bool = True,
    ) -> tuple[Tensor, Tensor]:
        next_state, next_z, _ = self.forward_with_state_norm_cache(
            state=state,
            z=z,
            backbone_out=backbone_out,
            loop_idx=loop_idx,
            attention_mask=attention_mask,
            decay_rate=decay_rate,
            compute_next_z=compute_next_z,
            prev_state_norm=None,
        )
        return next_state, next_z

    def forward_with_state_norm_cache(
        self,
        state: Tensor,
        z: Tensor,
        backbone_out: Tensor,
        loop_idx: int = 0,
        attention_mask: Tensor | None = None,
        decay_rate: Tensor | None = None,
        compute_next_z: bool = True,
        prev_state_norm: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        del loop_idx
        compute_dtype = self._get_compute_dtype(backbone_out)
        state = state.to(compute_dtype)
        z = z.to(compute_dtype)
        u_norm = self.drive_norm(backbone_out).to(compute_dtype)
        drive = _linear(self.u_proj, u_norm, gather_output=False)
        delta = F.softplus(_linear(self.delta_proj, u_norm, gather_output=False).float())
        if decay_rate is None:
            decay_rate = self.get_decay_rate()
        decay = torch.exp(-decay_rate * delta).to(dtype=compute_dtype)
        eta = 1.0 - decay
        base_state_loop = state + eta * (drive - state)
        state_correction = torch.sigmoid(self.state_correction_logit.float()).view(1, 1, -1).to(compute_dtype)
        next_state_loop = base_state_loop + state_correction * eta * (drive - base_state_loop)

        next_state_loop_full = self._gather_state(next_state_loop)
        next_state = self.token_mamba_block(
            next_state_loop_full.transpose(0, 1).contiguous(),
            attention_mask=attention_mask,
            output_is_tensor_parallel=True,
        ).transpose(0, 1).contiguous()
        if not compute_next_z:
            return self._mask_sbh(next_state, attention_mask), z, None

        if prev_state_norm is None:
            prev_state_norm = self.state_norm(state).to(compute_dtype)
        else:
            prev_state_norm = prev_state_norm.to(compute_dtype)
        state_norm = self.state_norm(next_state).to(compute_dtype)
        state_delta = state_norm - prev_state_norm
        gate_state = self._gather_state(state_norm)
        state_delta_gain = torch.sigmoid(self.state_delta_logit.float()).view(1, 1, -1).to(compute_dtype)
        state_z_input = state_norm + state_delta_gain * state_delta
        target_z = _linear(self.u_to_z, u_norm, gather_output=True) + _linear(
            self.state_to_z,
            state_z_input,
        )
        alpha = torch.sigmoid(_linear(self.gate_proj, torch.cat((u_norm, gate_state), dim=-1), gather_output=True))
        base_z = z + alpha * (target_z - z)
        z_correction = torch.sigmoid(self.z_correction_logit.float()).view(1, 1, -1).to(compute_dtype)
        next_z = base_z + z_correction * alpha * (target_z - base_z)
        next_state = self._mask_sbh(next_state, attention_mask)
        next_z = self._mask_sbh(next_z, attention_mask)
        state_norm_for_next_loop = self._mask_sbh(state_norm, attention_mask)
        return next_state, next_z, state_norm_for_next_loop

    def output_readout(self, state: Tensor) -> Tensor:
        return self.out_scale * _linear(self.state_to_out, state)
