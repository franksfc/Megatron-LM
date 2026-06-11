# Orin/SSM MindSpeed Megatron Audit

Date: 2026-06-09

This worktree is the MindSpeed/Megatron migration copy:

```text
/data/PonderLM/fcsong/megatron_mindspeed_core_v0.12_orin
```

The original Megatron-LM checkout is not used as the active training path.
MindSpeed and MindSpeed-LLM are vendored under:

```text
third_party/orin_mindspeed/MindSpeed
third_party/orin_mindspeed/MindSpeed-LLM
```

## Maintained Runtime Path

Current launcher:

```text
examples/orin_ssm/train_ssm9_mindspeed_backend_2node.sh
```

Current training entry:

```text
pretrain_orin_ssm_mindspeed.py
```

Current model files:

```text
megatron/orin_ssm/mcore_orin_model.py
megatron/orin_ssm/mcore_loop_axis_ssm.py
```

The training entry imports `torch_npu`, `transfer_to_npu`,
`mindspeed_llm.megatron_adaptor`, and runs MindSpeed-LLM
`training.pretrain`. The model provider requires `--orin-mcore-native`.

## Removed Legacy Paths

The previous compatibility model/trainer files, dated modeling snapshots, old
launchers, W&B sidecar, and local scan comparison tools were deleted. The
worktree now has one maintained Orin training path: the MindSpeed/Megatron
MCore-native entry.

## Scan Backend

The default and maintained scan backend is:

```text
ORIN_TOKEN_MAMBA_SCAN_IMPL=mindspeed
```

`MCoreMamba2TokenBlock` now supports one maintained scan backend:

```text
mindspeed: MindSpeed-LLM state_space_duality StateSpaceProcessor
```

There is no `torch` scan branch and no CUDA/Triton scan branch in
`mcore_loop_axis_ssm.py`.

## Parallelism Status

Current full run is TP=1, PP=1, CP=1. That is a launcher setting; the model
code remains on the MindSpeed/Megatron MCore-native path.

Tensor parallel is partially implemented in the recurrent block through MCore
`ColumnParallelLinear` and `RowParallelLinear`. The launcher exposes:

```text
TENSOR_MODEL_PARALLEL_SIZE
SEQUENCE_PARALLEL
```

Pipeline parallel and context parallel are still blocked in `model_provider`.
Context parallel should not be enabled until recurrent state and scan state
propagation across context shards is implemented and loss-equivalence tested.

## Validation So Far

Static checks:

```text
python -m py_compile megatron/orin_ssm/mcore_loop_axis_ssm.py \
  megatron/orin_ssm/mcore_orin_model.py \
  pretrain_orin_ssm_mindspeed.py

bash -n examples/orin_ssm/train_ssm9_mindspeed_backend_2node.sh
```

30-step 2-node comparison before removing the local scan baseline:

```text
mindspeed_scan_30step_20260609_075534:
  avg iter > 5 = 2025.808 ms
  final loss = 8.662897

torch_scan_30step_20260609_075840:
  avg iter > 5 = 2035.636 ms
  final loss = 8.663040
```

Checkpoint smoke:

```text
mindspeed_scan_save_smoke2_fix_20260609_081547
STATUS=0
latest_checkpointed_iteration.txt = 2
```

Current full run:

```text
run id: mindspeed_scan_mcore_50k_20260609_081935
nodes: 10.119.6.252,10.119.7.0
scan: ORIN_TOKEN_MAMBA_SCAN_IMPL=mindspeed
train iters: 50000
eval interval: 5000
save interval: 5000
wandb run: 41aukqpp
```

The first full-run eval/checkpoint proof point remains step 5000.
