#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_REPO_ROOT=${SCRIPT_DIR}
REPO_ROOT=${REPO_ROOT:-${DEFAULT_REPO_ROOT}}
CONDA_SH=${CONDA_SH:-/data/PonderLM/fcsong/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-ponderinglm}
ASCEND_ENV_SH=${ASCEND_ENV_SH:-/usr/local/Ascend/ascend-toolkit/set_env.sh}

NODE_IPS=${NODE_IPS:-10.119.6.252,10.119.7.0}
MASTER_ADDR=${MASTER_ADDR:-10.119.6.252}
MASTER_PORT=${MASTER_PORT:-29941}
NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
ORCHESTRATE=${ORCHESTRATE:-1}
DRY_RUN=${DRY_RUN:-0}
SKIP_BUSY_CHECK=${SKIP_BUSY_CHECK:-0}
EXTRA_MEGATRON_ARGS=${EXTRA_MEGATRON_ARGS:-}

RUN_ID=${RUN_ID:-recurrent_lm_backend_$(date +%Y%m%d_%H%M%S)}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/recurrent_lm_${RUN_ID}}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/outputs/recurrent_lm_${RUN_ID}}
SAVE_PATH=${SAVE_PATH:-${OUTPUT_DIR}/checkpoints}

MODEL_IMPL=${MODEL_IMPL:-llama_orin_ssm}
MODEL_CONFIG=${MODEL_CONFIG:-${REPO_ROOT}/llama_config/410m}
TOKENIZED_PATH=${TOKENIZED_PATH:-/data/PonderLM/uint16smallpile}
TRAIN_SPLIT_NAME=${TRAIN_SPLIT_NAME:-train}
VALID_TOKENIZED_PATH=${VALID_TOKENIZED_PATH:-${TOKENIZED_PATH}}
VALID_SPLIT_NAME=${VALID_SPLIT_NAME:-validation}
PAD_TOKEN_ID=${PAD_TOKEN_ID:-1}

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-1}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
CONTEXT_PARALLEL_ALGO=${CONTEXT_PARALLEL_ALGO:-mamba_cp_algo}
SEQUENCE_PARALLEL=${SEQUENCE_PARALLEL:-0}
if [ -z "${PER_DEVICE_TRAIN_BATCH_SIZE+x}" ]; then
    if [ "${TENSOR_MODEL_PARALLEL_SIZE}" -gt 1 ]; then
        PER_DEVICE_TRAIN_BATCH_SIZE=16
    else
        PER_DEVICE_TRAIN_BATCH_SIZE=4
    fi
fi
MAX_STEPS=${MAX_STEPS:-50000}
SEED=${SEED:-42}
SAMPLER_SEED_MODE=${SAMPLER_SEED_MODE:-llamafactory}
SAMPLER_DATA_SEED=${SAMPLER_DATA_SEED:-${SEED}}
LR_SCHEDULER_STEPS=${LR_SCHEDULER_STEPS:-50000}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
MIN_LR=${MIN_LR:-1e-4}
LOGGING_STEPS=${LOGGING_STEPS:-1}
EVAL_INTERVAL=${EVAL_INTERVAL:-5000}
EVAL_ITERS=${EVAL_ITERS:-}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
DISABLE_SAVE=${DISABLE_SAVE:-0}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
USE_DISTRIBUTED_OPTIMIZER=${USE_DISTRIBUTED_OPTIMIZER:-1}
OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-1}
USE_FUSED_RMSNORM=${USE_FUSED_RMSNORM:-1}
USE_FUSED_SWIGLU=${USE_FUSED_SWIGLU:-1}
USE_FUSED_ROPE=${USE_FUSED_ROPE:-1}
WANDB_PROJECT=${WANDB_PROJECT:-Ponder2-adaptive}
WANDB_EXP_NAME=${WANDB_EXP_NAME:-${RUN_ID}}
WANDB_RUN_ID=${WANDB_RUN_ID:-${WANDB_EXP_NAME}}
WANDB_SAVE_DIR=${WANDB_SAVE_DIR:-${OUTPUT_DIR}/wandb}
WANDB_GLOBAL_STEP_OFFSET=${WANDB_GLOBAL_STEP_OFFSET:-${GLOBAL_STEP_OFFSET:-0}}
USE_MEGATRON_WANDB=${USE_MEGATRON_WANDB:-1}
RECURRENT_WANDB_LOG_STYLE=${RECURRENT_WANDB_LOG_STYLE:-llamafactory}
# Megatron emits the human-readable train line from one logging rank. The loss
# function now all-reduces lm loss across the data-parallel group before
# returning the metric dict, matching HF Trainer's _nested_gather(...).mean().
export WANDB_API_KEY=${WANDB_API_KEY:-0c5dd50169e7ffe87db052c62e857039b3c282fc}
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.bandw.top}
export WANDB_ENTITY=${WANDB_ENTITY:-franksfc-lumia-group}
export WANDB_RUN_ID
MODEL_BF16_AUTOCAST=${MODEL_BF16_AUTOCAST:-0}
USE_MEGATRON_BF16=${USE_MEGATRON_BF16:-1}
MCORE_NATIVE=1
DISABLE_GMM_FP8=${DISABLE_GMM_FP8:-0}
DISABLE_NAN_CHECK=${DISABLE_NAN_CHECK:-1}

TRANSFORMER_IMPL=${TRANSFORMER_IMPL:-transformer_engine}
USE_FLASH_ATTN=${USE_FLASH_ATTN:-1}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flash}
export MODEL_USE_NULL_ATTENTION_MASK=${MODEL_USE_NULL_ATTENTION_MASK:-1}
export CHUNKED_LM_LOSS_TOKENS=${CHUNKED_LM_LOSS_TOKENS:-0}
export LM_LOSS_UPCAST=${LM_LOSS_UPCAST:-0}
export DEBUG_LOSS_SHAPES=${DEBUG_LOSS_SHAPES:-0}
if [ "${TENSOR_MODEL_PARALLEL_SIZE}" -gt 1 ] || [ "${CONTEXT_PARALLEL_SIZE}" -gt 1 ]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
fi

ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
MORE_ITERATIONS=${MORE_ITERATIONS:-3}
MEMORY_SIZE=${MEMORY_SIZE:-1024}
LOOP_MAMBA_VARIANT=${LOOP_MAMBA_VARIANT:-mamba2_fast}
LOOP_MAMBA_N_GROUPS=${LOOP_MAMBA_N_GROUPS:-8}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-4096}

MINDSPEED_ROOT=${MINDSPEED_ROOT:-${REPO_ROOT}/third_party/orin_mindspeed/MindSpeed}
MINDSPEED_LLM_ROOT=${MINDSPEED_LLM_ROOT:-${REPO_ROOT}/third_party/orin_mindspeed/MindSpeed-LLM}

export NODE_IPS MASTER_ADDR MASTER_PORT NNODES NPROC_PER_NODE LOG_DIR OUTPUT_DIR SAVE_PATH
export MODEL_IMPL
export TOKENIZED_PATH TRAIN_SPLIT_NAME VALID_TOKENIZED_PATH VALID_SPLIT_NAME PAD_TOKEN_ID
export SAMPLER_SEED_MODE SAMPLER_DATA_SEED
export GLOBAL_BATCH_SIZE
export TENSOR_MODEL_PARALLEL_SIZE PIPELINE_MODEL_PARALLEL_SIZE CONTEXT_PARALLEL_SIZE CONTEXT_PARALLEL_ALGO SEQUENCE_PARALLEL
export DISABLE_SAVE
if [ -z "${MODEL_LOOP_LAYOUT+x}" ]; then
    if [ "${TENSOR_MODEL_PARALLEL_SIZE}" -gt 1 ] || [ "${SEQUENCE_PARALLEL}" = "1" ]; then
        MODEL_LOOP_LAYOUT=sbh_tp
    else
        MODEL_LOOP_LAYOUT=bsh
    fi
fi
export MODEL_LOOP_LAYOUT
export CHUNKED_LM_LOSS_CHECKPOINT=${CHUNKED_LM_LOSS_CHECKPOINT:-0}
export TRANSFORMER_IMPL USE_FLASH_ATTN ATTENTION_BACKEND MODEL_USE_NULL_ATTENTION_MASK
export RECURRENT_WANDB_LOG_STYLE
export TRAINING_BACKEND=${TRAINING_BACKEND:-mindspeed}
export PYTHONPATH="${MINDSPEED_ROOT}:${MINDSPEED_LLM_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [ "${ORCHESTRATE}" = "1" ]; then
    mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${SAVE_PATH}"
    done_file="${LOG_DIR}/.training_done"
    rm -f "${done_file}"
    IFS=',' read -r -a nodes <<< "${NODE_IPS}"
    pids=()

    if [ "${SKIP_BUSY_CHECK}" != "1" ]; then
        for host in "${nodes[@]}"; do
            busy_cmd="ps -eo pid,stat,cmd | grep -E 'torchrun|torch.distributed.run|llamafactory.launcher|deepspeed|pretrain_recurrent_lm' | grep -v grep | grep -v '${RUN_ID}' | grep -v '${OUTPUT_DIR}' | head -20"
            busy_output=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "${host}" "${busy_cmd}" || true)
            if [ -n "${busy_output}" ]; then
                echo "Refusing to launch because ${host} already has training-like processes:" >&2
                echo "${busy_output}" >&2
                exit 2
            fi
        done
    fi

    cleanup_remote_ranks() {
        local pattern="${RUN_ID}|${LOG_DIR}|${OUTPUT_DIR}"
        local host
        for host in "${nodes[@]}"; do
            ssh -o BatchMode=yes -o ConnectTimeout=5 "${host}" \
                "pids=\$(pgrep -f '${pattern}' || true); if [ -n \"\$pids\" ]; then echo \"\$pids\" | xargs -r kill -9; fi" \
                >/dev/null 2>&1 || true
        done
    }

    set +e
    status=0
    for idx in "${!nodes[@]}"; do
        host="${nodes[$idx]}"
        ssh -o BatchMode=yes -o ConnectTimeout=8 "${host}" \
            "RUN_ID='${RUN_ID}' REPO_ROOT='${REPO_ROOT}' MODEL_IMPL='${MODEL_IMPL}' NODE_IPS='${NODE_IPS}' MASTER_ADDR='${MASTER_ADDR}' MASTER_PORT='${MASTER_PORT}' NNODES='${NNODES}' NODE_RANK='${idx}' NPROC_PER_NODE='${NPROC_PER_NODE}' LOG_DIR='${LOG_DIR}' OUTPUT_DIR='${OUTPUT_DIR}' SAVE_PATH='${SAVE_PATH}' MODEL_CONFIG='${MODEL_CONFIG}' TOKENIZED_PATH='${TOKENIZED_PATH}' TRAIN_SPLIT_NAME='${TRAIN_SPLIT_NAME}' VALID_TOKENIZED_PATH='${VALID_TOKENIZED_PATH}' VALID_SPLIT_NAME='${VALID_SPLIT_NAME}' PAD_TOKEN_ID='${PAD_TOKEN_ID}' SAMPLER_SEED_MODE='${SAMPLER_SEED_MODE}' SAMPLER_DATA_SEED='${SAMPLER_DATA_SEED}' GLOBAL_BATCH_SIZE='${GLOBAL_BATCH_SIZE}' PER_DEVICE_TRAIN_BATCH_SIZE='${PER_DEVICE_TRAIN_BATCH_SIZE}' TENSOR_MODEL_PARALLEL_SIZE='${TENSOR_MODEL_PARALLEL_SIZE}' PIPELINE_MODEL_PARALLEL_SIZE='${PIPELINE_MODEL_PARALLEL_SIZE}' CONTEXT_PARALLEL_SIZE='${CONTEXT_PARALLEL_SIZE}' CONTEXT_PARALLEL_ALGO='${CONTEXT_PARALLEL_ALGO}' SEQUENCE_PARALLEL='${SEQUENCE_PARALLEL}' MAX_STEPS='${MAX_STEPS}' SEED='${SEED}' LR_SCHEDULER_STEPS='${LR_SCHEDULER_STEPS}' LEARNING_RATE='${LEARNING_RATE}' MIN_LR='${MIN_LR}' LOGGING_STEPS='${LOGGING_STEPS}' EVAL_INTERVAL='${EVAL_INTERVAL}' EVAL_ITERS='${EVAL_ITERS}' SAVE_INTERVAL='${SAVE_INTERVAL}' DISABLE_SAVE='${DISABLE_SAVE}' DATALOADER_NUM_WORKERS='${DATALOADER_NUM_WORKERS}' USE_DISTRIBUTED_OPTIMIZER='${USE_DISTRIBUTED_OPTIMIZER}' OVERLAP_GRAD_REDUCE='${OVERLAP_GRAD_REDUCE}' USE_FUSED_RMSNORM='${USE_FUSED_RMSNORM}' USE_FUSED_SWIGLU='${USE_FUSED_SWIGLU}' USE_FUSED_ROPE='${USE_FUSED_ROPE}' WANDB_PROJECT='${WANDB_PROJECT}' WANDB_EXP_NAME='${WANDB_EXP_NAME}' WANDB_RUN_ID='${WANDB_RUN_ID}' WANDB_SAVE_DIR='${WANDB_SAVE_DIR}' WANDB_GLOBAL_STEP_OFFSET='${WANDB_GLOBAL_STEP_OFFSET}' WANDB_API_KEY='${WANDB_API_KEY}' WANDB_BASE_URL='${WANDB_BASE_URL}' WANDB_ENTITY='${WANDB_ENTITY}' USE_MEGATRON_WANDB='${USE_MEGATRON_WANDB}' RECURRENT_WANDB_LOG_STYLE='${RECURRENT_WANDB_LOG_STYLE}' USE_MEGATRON_BF16='${USE_MEGATRON_BF16}' MCORE_NATIVE='${MCORE_NATIVE}' TRANSFORMER_IMPL='${TRANSFORMER_IMPL}' USE_FLASH_ATTN='${USE_FLASH_ATTN}' ATTENTION_BACKEND='${ATTENTION_BACKEND}' MODEL_USE_NULL_ATTENTION_MASK='${MODEL_USE_NULL_ATTENTION_MASK}' DISABLE_GMM_FP8='${DISABLE_GMM_FP8}' DISABLE_NAN_CHECK='${DISABLE_NAN_CHECK}' MODEL_BF16_AUTOCAST='${MODEL_BF16_AUTOCAST}' ATTN_IMPLEMENTATION='${ATTN_IMPLEMENTATION}' MORE_ITERATIONS='${MORE_ITERATIONS}' MEMORY_SIZE='${MEMORY_SIZE}' LOOP_MAMBA_VARIANT='${LOOP_MAMBA_VARIANT}' LOOP_MAMBA_N_GROUPS='${LOOP_MAMBA_N_GROUPS}' MAX_POSITION_EMBEDDINGS='${MAX_POSITION_EMBEDDINGS}' MINDSPEED_ROOT='${MINDSPEED_ROOT}' MINDSPEED_LLM_ROOT='${MINDSPEED_LLM_ROOT}' MODEL_LOOP_LAYOUT='${MODEL_LOOP_LAYOUT}' CHUNKED_LM_LOSS_TOKENS='${CHUNKED_LM_LOSS_TOKENS}' CHUNKED_LM_LOSS_CHECKPOINT='${CHUNKED_LM_LOSS_CHECKPOINT}' LM_LOSS_UPCAST='${LM_LOSS_UPCAST}' DEBUG_LOSS_SHAPES='${DEBUG_LOSS_SHAPES}' EXTRA_MEGATRON_ARGS='${EXTRA_MEGATRON_ARGS}' TRAINING_BACKEND='${TRAINING_BACKEND}' PYTHONPATH='${PYTHONPATH}' DRY_RUN='${DRY_RUN}' ORCHESTRATE=0 bash '${REPO_ROOT}/train_recurrent_lm_mindspeed_2node.sh'" \
            >"${LOG_DIR}/ssh_rank${idx}.log" 2>&1 &
        pids+=("$!")
    done

    for pid in "${pids[@]}"; do
        wait "${pid}" || status=$?
    done
    touch "${done_file}"
    if [ "${status}" -ne 0 ]; then
        cleanup_remote_ranks
    fi
    set -e
    echo "RUN_ID=${RUN_ID}"
    echo "MODEL_IMPL=${MODEL_IMPL}"
    echo "MODEL_CONFIG=${MODEL_CONFIG}"
    echo "LOG_DIR=${LOG_DIR}"
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "STATUS=${status}"
    exit "${status}"
fi

if [ -f "${CONDA_SH}" ]; then
    source "${CONDA_SH}"
fi
conda activate "${CONDA_ENV_NAME}"
if [ -f "${ASCEND_ENV_SH}" ]; then
    source "${ASCEND_ENV_SH}"
fi

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONNOUSERSITE=1
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset http_proxy https_proxy no_proxy all_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY ALL_PROXY

make_device_range() {
    local count="${1:-0}"
    local devices=""
    local i
    for ((i = 0; i < count; ++i)); do
        if [ -n "${devices}" ]; then
            devices+=","
        fi
        devices+="${i}"
    done
    printf '%s\n' "${devices}"
}

get_local_ips() {
    python - <<'PY'
import os
import socket

targets = os.environ.get("NODE_IPS", "").split(",") + [os.environ.get("MASTER_ADDR", ""), "8.8.8.8"]
seen = set()
for target in targets:
    target = target.strip()
    if not target:
        continue
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((target, 80))
        ip = sock.getsockname()[0]
    except OSError:
        ip = ""
    finally:
        sock.close()
    if ip and ip not in seen:
        seen.add(ip)
        print(ip)
PY
}

infer_node_rank() {
    if [ -n "${NODE_RANK}" ]; then
        echo "${NODE_RANK}"
        return
    fi
    local local_ip
    local node_ip
    local idx=0
    for node_ip in ${NODE_IPS//,/ }; do
        for local_ip in $(get_local_ips); do
            if [ "${local_ip}" = "${node_ip}" ]; then
                echo "${idx}"
                return
            fi
        done
        idx=$((idx + 1))
    done
}

infer_socket_ifname() {
    python - <<'PY'
import array
import os
import socket
import struct

target = os.environ.get("MASTER_ADDR", "10.119.6.252")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    sock.connect((target, 80))
    local_ip = sock.getsockname()[0]
finally:
    sock.close()

bytes_out = 128 * 32
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
names = array.array("B", b"\0" * bytes_out)
try:
    import fcntl
    outbytes = struct.unpack(
        "iL",
        fcntl.ioctl(s.fileno(), 0x8912, struct.pack("iL", bytes_out, names.buffer_info()[0])),
    )[0]
except Exception:
    print("")
    raise SystemExit
namestr = names.tobytes()
for i in range(0, outbytes, 40):
    name = namestr[i:i+16].split(b"\0", 1)[0].decode()
    ip = socket.inet_ntoa(namestr[i+20:i+24])
    if ip == local_ip:
        print(name)
        break
PY
}

if [ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
    ASCEND_RT_VISIBLE_DEVICES=$(make_device_range "${NPROC_PER_NODE}")
    export ASCEND_RT_VISIBLE_DEVICES
fi

NODE_RANK=$(infer_node_rank)
export NODE_RANK
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-$(infer_socket_ifname)}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${HCCL_SOCKET_IFNAME}}

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${SAVE_PATH}"

if [ -z "${EVAL_ITERS}" ]; then
    EVAL_ITERS=$(python - <<'PY'
import math
import os
from datasets import load_from_disk

path = os.environ["VALID_TOKENIZED_PATH"]
split_name = os.environ["VALID_SPLIT_NAME"]
global_batch_size = max(1, int(os.environ["GLOBAL_BATCH_SIZE"]))
tokenized = load_from_disk(path)
if hasattr(tokenized, "keys"):
    if split_name not in tokenized:
        raise SystemExit(
            f"Split '{split_name}' not found in {path}. Available splits: {list(tokenized.keys())}"
        )
    valid_len = len(tokenized[split_name])
else:
    valid_len = len(tokenized)
print(max(1, math.ceil(valid_len / global_batch_size)))
PY
)
fi
export EVAL_ITERS

launcher_cmd=()
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/torchrun" ]; then
    launcher_cmd=("${CONDA_PREFIX}/bin/torchrun")
else
    launcher_cmd=("${CONDA_PREFIX}/bin/python" -m torch.distributed.run)
fi

train_cmd=(
    "${launcher_cmd[@]}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --nproc_per_node "${NPROC_PER_NODE}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
    --rdzv_backend static
    "${REPO_ROOT}/pretrain_recurrent_lm_mindspeed.py"
    --num-layers 24
    --hidden-size 1024
    --ffn-hidden-size 2752
    --num-attention-heads 16
    --seq-length 2048
    --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
    --swiglu
    --normalization RMSNorm
    --use-rotary-position-embeddings
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --micro-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --train-iters "${MAX_STEPS}"
    --seed "${SEED}"
    --lr "${LEARNING_RATE}"
    --min-lr "${MIN_LR}"
    --lr-decay-style cosine
    --lr-decay-iters "${LR_SCHEDULER_STEPS}"
    --lr-warmup-fraction 0.01
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --clip-grad 1.0
    --log-interval "${LOGGING_STEPS}"
    --eval-interval "${EVAL_INTERVAL}"
    --eval-iters "${EVAL_ITERS}"
    --save-interval "${SAVE_INTERVAL}"
    --num-workers "${DATALOADER_NUM_WORKERS}"
    --dataloader-type cyclic
    --tokenizer-type NullTokenizer
    --vocab-size 50304
    --make-vocab-size-divisible-by 128
    --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
    --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
    --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
    --context-parallel-algo "${CONTEXT_PARALLEL_ALGO}"
    --transformer-impl "${TRANSFORMER_IMPL}"
    --attention-backend "${ATTENTION_BACKEND}"
    --no-load-optim
    --no-load-rng
    --no-save-optim
    --no-save-rng
    --model-impl "${MODEL_IMPL}"
    --model-config "${MODEL_CONFIG}"
    --tokenized-path "${TOKENIZED_PATH}"
    --train-split-name "${TRAIN_SPLIT_NAME}"
    --valid-tokenized-path "${VALID_TOKENIZED_PATH}"
    --valid-split-name "${VALID_SPLIT_NAME}"
    --pad-token-id "${PAD_TOKEN_ID}"
    --sampler-seed-mode "${SAMPLER_SEED_MODE}"
    --sampler-data-seed "${SAMPLER_DATA_SEED}"
    --attn-implementation "${ATTN_IMPLEMENTATION}"
    --more-iterations "${MORE_ITERATIONS}"
    --memory-size "${MEMORY_SIZE}"
    --loop-mamba-variant "${LOOP_MAMBA_VARIANT}"
    --loop-mamba-n-groups "${LOOP_MAMBA_N_GROUPS}"
    --model-max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
    --experiment-output-dir "${OUTPUT_DIR}"
)

if [ "${DISABLE_SAVE}" != "1" ]; then
    train_cmd+=(--save "${SAVE_PATH}")
fi

if [ "${USE_DISTRIBUTED_OPTIMIZER}" = "1" ]; then
    train_cmd+=(--use-distributed-optimizer)
fi
if [ "${OVERLAP_GRAD_REDUCE}" = "1" ]; then
    train_cmd+=(--overlap-grad-reduce)
fi
if [ "${DISABLE_GMM_FP8}" = "1" ]; then
    train_cmd+=(--no-use-gmm-fp8)
fi
if [ "${DISABLE_NAN_CHECK}" = "1" ]; then
    train_cmd+=(--no-check-for-nan-in-loss-and-grad)
fi
if [ "${USE_MEGATRON_BF16}" = "1" ]; then
    train_cmd+=(--bf16)
fi
if [ "${USE_FLASH_ATTN}" = "1" ]; then
    train_cmd+=(--use-flash-attn)
fi
if [ "${USE_FUSED_RMSNORM}" = "1" ]; then
    train_cmd+=(--use-fused-rmsnorm)
fi
if [ "${USE_FUSED_SWIGLU}" = "1" ]; then
    train_cmd+=(--use-fused-swiglu)
fi
if [ "${USE_FUSED_ROPE}" = "1" ]; then
    train_cmd+=(--use-fused-rotary-pos-emb)
fi
if [ "${USE_MEGATRON_WANDB}" = "1" ] && [ -n "${WANDB_PROJECT}" ]; then
    train_cmd+=(--use-wandb)
    train_cmd+=(--wandb-project "${WANDB_PROJECT}")
    train_cmd+=(--wandb-exp-name "${WANDB_EXP_NAME}")
    train_cmd+=(--wandb-save-dir "${WANDB_SAVE_DIR}")
fi
if [ "${SEQUENCE_PARALLEL}" = "1" ]; then
    train_cmd+=(--sequence-parallel)
fi
if [ "${MODEL_BF16_AUTOCAST}" = "1" ]; then
    train_cmd+=(--model-bf16-autocast)
else
    train_cmd+=(--no-model-bf16-autocast)
fi
train_cmd+=(--mcore-native)
if [ -n "${EXTRA_MEGATRON_ARGS}" ]; then
    read -r -a extra_megatron_args <<< "${EXTRA_MEGATRON_ARGS}"
    train_cmd+=("${extra_megatron_args[@]}")
fi

{
    echo "RUN_ID=${RUN_ID}"
    echo "MODEL_IMPL=${MODEL_IMPL}"
    echo "MODEL_CONFIG=${MODEL_CONFIG}"
    echo "MASTER_ADDR=${MASTER_ADDR}"
    echo "MASTER_PORT=${MASTER_PORT}"
    echo "NODE_IPS=${NODE_IPS}"
    echo "NODE_RANK=${NODE_RANK}"
    echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
    echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
    echo "HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME}"
    echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
    echo "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
    echo "SEED=${SEED}"
    echo "PAD_TOKEN_ID=${PAD_TOKEN_ID}"
    echo "SAMPLER_SEED_MODE=${SAMPLER_SEED_MODE}"
    echo "SAMPLER_DATA_SEED=${SAMPLER_DATA_SEED}"
    echo "TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE}"
    echo "PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE}"
    echo "CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE}"
    echo "CONTEXT_PARALLEL_ALGO=${CONTEXT_PARALLEL_ALGO}"
    echo "SEQUENCE_PARALLEL=${SEQUENCE_PARALLEL}"
    echo "MAX_STEPS=${MAX_STEPS}"
    echo "EVAL_INTERVAL=${EVAL_INTERVAL}"
    echo "EVAL_ITERS=${EVAL_ITERS}"
    echo "TRAIN_SPLIT_NAME=${TRAIN_SPLIT_NAME}"
    echo "VALID_TOKENIZED_PATH=${VALID_TOKENIZED_PATH}"
    echo "VALID_SPLIT_NAME=${VALID_SPLIT_NAME}"
    echo "USE_DISTRIBUTED_OPTIMIZER=${USE_DISTRIBUTED_OPTIMIZER}"
    echo "OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE}"
    echo "USE_FUSED_RMSNORM=${USE_FUSED_RMSNORM}"
    echo "USE_FUSED_SWIGLU=${USE_FUSED_SWIGLU}"
    echo "USE_FUSED_ROPE=${USE_FUSED_ROPE}"
    echo "WANDB_PROJECT=${WANDB_PROJECT}"
    echo "WANDB_EXP_NAME=${WANDB_EXP_NAME}"
    echo "WANDB_RUN_ID=${WANDB_RUN_ID}"
    echo "WANDB_SAVE_DIR=${WANDB_SAVE_DIR}"
    echo "WANDB_GLOBAL_STEP_OFFSET=${WANDB_GLOBAL_STEP_OFFSET}"
    echo "USE_MEGATRON_WANDB=${USE_MEGATRON_WANDB}"
    echo "RECURRENT_WANDB_LOG_STYLE=${RECURRENT_WANDB_LOG_STYLE}"
    echo "USE_MEGATRON_BF16=${USE_MEGATRON_BF16}"
    echo "MCORE_NATIVE=${MCORE_NATIVE}"
    echo "MODEL_LOOP_LAYOUT=${MODEL_LOOP_LAYOUT}"
    echo "TRANSFORMER_IMPL=${TRANSFORMER_IMPL}"
    echo "USE_FLASH_ATTN=${USE_FLASH_ATTN}"
    echo "ATTENTION_BACKEND=${ATTENTION_BACKEND}"
    echo "MODEL_USE_NULL_ATTENTION_MASK=${MODEL_USE_NULL_ATTENTION_MASK}"
    echo "CHUNKED_LM_LOSS_TOKENS=${CHUNKED_LM_LOSS_TOKENS}"
    echo "CHUNKED_LM_LOSS_CHECKPOINT=${CHUNKED_LM_LOSS_CHECKPOINT}"
    echo "LM_LOSS_UPCAST=${LM_LOSS_UPCAST}"
    echo "EXTRA_MEGATRON_ARGS=${EXTRA_MEGATRON_ARGS}"
    echo "DISABLE_GMM_FP8=${DISABLE_GMM_FP8}"
    echo "DISABLE_NAN_CHECK=${DISABLE_NAN_CHECK}"
    echo "MODEL_BF16_AUTOCAST=${MODEL_BF16_AUTOCAST}"
    echo "TRAINING_BACKEND=${TRAINING_BACKEND}"
    echo "LOOP_MAMBA_VARIANT=${LOOP_MAMBA_VARIANT}"
    echo "LOOP_MAMBA_N_GROUPS=${LOOP_MAMBA_N_GROUPS}"
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "SAVE_PATH=${SAVE_PATH}"
    printf 'Launch command:\n'
    printf ' %q' "${train_cmd[@]}"
    printf '\n'
} | tee "${LOG_DIR}/rank${NODE_RANK}.mindspeed_backend.launcher.log"

if [ "${DRY_RUN}" = "1" ]; then
    exit 0
fi

"${train_cmd[@]}" >"${LOG_DIR}/rank${NODE_RANK}.log" 2>&1
rank_status="$?"
echo "STATUS=${rank_status}" >"${LOG_DIR}/rank${NODE_RANK}.done"
exit "${rank_status}"
