# Orin/SSM MindSpeed Megatron Path

This directory now keeps only the Megatron/MindSpeed launcher for the Orin/SSM
MCore-native training path.

Maintained launcher:

```bash
NODE_IPS=10.119.6.252,10.119.7.0 \
MASTER_ADDR=10.119.6.252 \
NNODES=2 NPROC_PER_NODE=16 \
bash examples/orin_ssm/train_ssm9_mindspeed_backend_2node.sh
```

Important defaults:

- `ORIN_MCORE_NATIVE=1`
- `TRAINING_BACKEND=mcore`
- `LEARNING_RATE=1e-3`
- `MIN_LR=1e-4`
- `MAX_STEPS=50000`
- `GLOBAL_BATCH_SIZE=256`
- `PER_DEVICE_TRAIN_BATCH_SIZE=8` for TP=1, `16` for TP>1
- `TENSOR_MODEL_PARALLEL_SIZE=1`
- `PIPELINE_MODEL_PARALLEL_SIZE=1`
- `CONTEXT_PARALLEL_SIZE=2`
- `CONTEXT_PARALLEL_ALGO=mamba_cp_algo`
- `LOOP_MAMBA_VARIANT=orin_mamba2_fast`
- `LOOP_MAMBA_N_GROUPS=8`
- `EVAL_INTERVAL=5000`
- `SAVE_INTERVAL=100000` unless overridden

Useful smoke checks:

```bash
DRY_RUN=1 bash examples/orin_ssm/train_ssm9_mindspeed_backend_2node.sh
```

```bash
MAX_STEPS=30 SAVE_INTERVAL=100000 EVAL_ITERS=0 \
bash examples/orin_ssm/train_ssm9_mindspeed_backend_2node.sh
```

Tensor parallel and sequence parallel can be tried by setting
`TENSOR_MODEL_PARALLEL_SIZE` and `SEQUENCE_PARALLEL=1`. The launcher switches
the recurrent loop to the SBH TP layout whenever TP or SP is enabled.
