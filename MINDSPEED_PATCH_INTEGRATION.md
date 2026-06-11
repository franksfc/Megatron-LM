# MindSpeed 接入说明

本文说明这个仓库里是如何把 MindSpeed / MindSpeed-LLM 接进 Megatron Core 的。这里的「patch 进来」不是把 Megatron 原有训练主路径整体替换掉，而是在 `core_v0.12.1` 基线上新增一条独立的 recurrent LM + Ascend NPU 训练路径：MindSpeed 代码放在 `third_party/`，启动时通过 `PYTHONPATH`、import 顺序和少量 monkey patch 激活。

## 基线和改动层次

当前分支基于 Megatron-LM `core_v0.12.1` 附近的 `a845aa7`，主要通过两个提交完成接入：

1. `79439db Add Orin MindSpeed NPU training path`
2. `52ca3ca Modularize recurrent MindSpeed training`

最终形态可以按五层理解：

1. `third_party/orin_mindspeed/`：内嵌 MindSpeed 和 MindSpeed-LLM 源码。
2. `runtime/`：负责加载 MindSpeed-LLM、安装运行时 guard 和日志 patch。
3. `modeling/`：提供 Megatron Core 版 Orin recurrent LM 模型实现。
4. `pretrain_recurrent_lm_mindspeed.py`：新的 MindSpeed 后端训练入口。
5. `train_recurrent_lm_mindspeed_2node.sh`：多节点 Ascend NPU 启动脚本。

这样做的核心目标是：尽量保留上游 Megatron Core 的主体结构，只在新入口和少量兼容点上接入 MindSpeed。

## 第三方源码放置

MindSpeed 相关源码被 vendor 到：

- `third_party/orin_mindspeed/MindSpeed`
- `third_party/orin_mindspeed/MindSpeed-LLM`

启动脚本会把这两个目录加到 `PYTHONPATH`：

```bash
MINDSPEED_ROOT=${REPO_ROOT}/third_party/orin_mindspeed/MindSpeed
MINDSPEED_LLM_ROOT=${REPO_ROOT}/third_party/orin_mindspeed/MindSpeed-LLM
export PYTHONPATH="${MINDSPEED_ROOT}:${MINDSPEED_LLM_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
```

因此 Python import 解析顺序是：

1. MindSpeed
2. MindSpeed-LLM
3. 当前 Megatron 仓库

这一步只是让源码可 import，真正的 patch 激活发生在训练入口中。

## 运行时 patch 顺序

入口文件 `pretrain_recurrent_lm_mindspeed.py` 的关键顺序是：

1. 先 import `torch_npu` 和 `torch_npu.contrib.transfer_to_npu`，让 PyTorch/NPU 侧的设备迁移能力生效。
2. 调用 `runtime.mindspeed_runtime.load_mindspeed_runtime()`。
3. `load_mindspeed_runtime()` 内部先设置 `TRAINING_BACKEND=mindspeed`。
4. 安装 MTP guard，避免 MindSpeed-LLM 默认 MTP patch 在没有启用 MTP 时误注册。
5. import `mindspeed_llm.tasks.megatron_adaptor_v2`，触发 MindSpeed-LLM 对 Megatron 的适配注册。
6. import `mindspeed_llm.core.context_parallel.get_batch_utils`，拿到 CP batch 切分 helper。
7. import `mindspeed_llm.training.training`，拿到 MindSpeed-LLM 的 `pretrain`。
8. 对 MindSpeed-LLM trainer 安装 LLaMA-Factory 风格的 W&B train/eval 日志 patch。
9. 之后才 import Megatron Core / Megatron training 模块。

这个顺序很重要，因为 MindSpeed 会 patch torch.compile、Transformer Engine、Apex 和 NPU helper。Megatron 相关模块必须在这些 patch 安装之后再 import，否则有些替换点不会生效。

## runtime 模块做了什么

`runtime/mindspeed_runtime.py` 是总开关：

- `load_mindspeed_runtime()`：加载 MindSpeed-LLM adaptor、CP batch helper 和 MindSpeed-LLM 的 `pretrain`。
- `install_mindspeed_cross_entropy_patches()`：把 Megatron Core 的 `VocabParallelCrossEntropy` 两个静态方法替换成 MindSpeed-LLM 的 NPU 友好实现。

`runtime/mindspeed_patches.py` 做的是兼容层：

- `install_mtp_feature_guard()`：没有 `mtp_num_layers` 时跳过 MTP patch 注册。
- `install_llamafactory_wandb_training_log()`：保留 MindSpeed-LLM 原训练日志，同时额外输出 `train/loss`、`train/grad_norm`、`train/learning_rate`、`train/epoch`、`train/global_step`。
- `install_llamafactory_wandb_eval_log()`：把 eval 指标输出成 `eval/loss`、`eval/epoch`、`eval/global_step`。

这些 patch 都是运行时安装，不直接改 MindSpeed-LLM vendored 源码，后面同步上游时相对容易隔离。

## 模型侧怎么接 MindSpeed

模型实现集中在 `modeling/llama_orin_ssm.py`，注册入口是 `modeling/registry.py`：

```python
_MODEL_MODULES = {
    "llama_orin_ssm": "modeling.llama_orin_ssm",
}
```

训练时 `--model-impl llama_orin_ssm` 会走这个模型。

模型侧的 MindSpeed 接入主要有三类：

1. 使用 Megatron Core 的 TP/CP parallel state 管理张量并行和上下文并行。
2. 动态加载 MindSpeed-LLM 的 SSM 源文件：
   - `mindspeed_llm/tasks/models/ssm/state_space_context_parallel.py`
   - `mindspeed_llm/tasks/models/ssm/state_space_duality.py`
3. 使用 NPU 侧能力：
   - `torch_npu.npu_rms_norm`
   - MindSpeed-LLM 的 context-parallel convolution
   - MindSpeed-LLM 的 SSD / state-space duality 处理逻辑

这里仍然保留本仓库自己的 Orin recurrent LM 结构，MindSpeed-LLM 被用作后端能力来源，而不是直接把 MindSpeed-LLM 的完整模型训练脚本搬过来跑。

## 训练入口怎么接 Megatron 和 MindSpeed

`pretrain_recurrent_lm_mindspeed.py` 是 glue layer：

- `extra_args_provider()` 增加 Orin/recurrent 相关参数，比如 `--model-config`、`--tokenized-path`、`--loop-mamba-variant`、`--sampler-seed-mode`、`--mcore-native`。
- `TokenizedDataset` 直接读取 Hugging Face `datasets.save_to_disk()` 产物，把样本包装成 Megatron pretrain batch 格式。
- `model_provider()` 构造 Megatron Core config，然后调用 `build_model()` 构建 `llama_orin_ssm`。
- `get_batch()` 使用 Megatron TP broadcast 数据；当 `context_parallel_size > 1` 时，再调用 MindSpeed-LLM 的 `get_batch_on_this_cp_rank()`。
- `loss_func()` 对 loss 做 DP/CP 聚合，返回 Megatron trainer 期望的 `{"lm loss": ...}`。
- `main()` 安装 MindSpeed cross entropy patch，然后调用 MindSpeed-LLM 的 `pretrain()`。

因此训练循环本身用的是 MindSpeed-LLM trainer，但模型、数据集和 forward step 是本仓库提供的。

## LLaMA-Factory 语义适配

为了让 Megatron/MindSpeed 路径尽量对齐原来的 LLaMA-Factory recurrent 实验，新增了两块适配：

1. `llama_config/`
   - 保存 70m、125m、160m、410m、834m、1.4b、2.8b 等 LLaMA config/tokenizer 文件。
   - `llama_config/megatron_export.py` 把 LLaMA config 转成 Megatron runtime 需要的字段。
   - `attach_llamafactory_initialization()` 复刻 recurrent Mamba-fast 初始化尺度。

2. `megatron/legacy/data/data_samplers.py`
   - 增加 `sampler_seed_mode=llamafactory`。
   - 复刻 HF Trainer / Accelerate 的 batch shard 语义。
   - shuffle seed 使用 `seed + epoch`，并支持 `sampler_data_seed`。

这部分是为了保证训练数据顺序和初始化不要因为迁移到 Megatron/MindSpeed 后悄悄漂移。

## W&B 适配

W&B 有两层处理：

1. `megatron/training/global_vars.py`
   - 使用 `WANDB_RUN_ID` 作为 run id。
   - 设置 `resume="allow"`。
   - 支持 `WANDB_ENTITY`。
   - 定义 `train/global_step` 作为 W&B step metric。

2. `runtime/mindspeed_patches.py`
   - 把 MindSpeed-LLM trainer 的训练和评估日志补成 LLaMA-Factory/HF Trainer 风格 metric key。

启动脚本默认：

```bash
USE_MEGATRON_WANDB=1
RECURRENT_WANDB_LOG_STYLE=llamafactory
WANDB_BASE_URL=https://api.bandw.top
WANDB_RUN_ID=${WANDB_EXP_NAME}
```

也就是说，当前推荐路径是训练进程内直接写 W&B，而不是额外 sidecar replay。

## 启动脚本做了什么

`train_recurrent_lm_mindspeed_2node.sh` 是目前完整启动面：

- 配置 conda 环境和 Ascend runtime。
- 设置多节点参数：`NODE_IPS`、`MASTER_ADDR`、`NNODES`、`NPROC_PER_NODE`。
- 设置 NPU/HCCL/GLOO 环境。
- 设置 `MINDSPEED_ROOT`、`MINDSPEED_LLM_ROOT` 和 `PYTHONPATH`。
- 远程 orchestrate 两台机器。
- 用 `torchrun` 启动 `pretrain_recurrent_lm_mindspeed.py`。
- 开启 MindSpeed/Megatron 相关参数：
  - `--use-distributed-optimizer`
  - `--overlap-grad-reduce`
  - `--use-flash-attn`
  - `--use-fused-rmsnorm`
  - `--use-fused-swiglu`
  - `--use-fused-rotary-pos-emb`
  - `--context-parallel-algo mamba_cp_algo`

它同时把每次 launch 的关键环境和最终命令写入 `logs/.../rank*.mindspeed_backend.launcher.log`，方便之后复盘。

## 为什么不是直接改 Megatron 主入口

这次接入刻意没有直接改 `pretrain_gpt.py` 或 Megatron 默认 GPT path，原因是：

1. Orin recurrent LM 不是标准 GPT transformer。
2. MindSpeed-LLM 的 trainer/adaptor 有自己的 patch 生命周期。
3. Ascend NPU 环境、context parallel、SSM selective scan 都需要额外 runtime 条件。
4. 把入口独立出来，可以让上游 Megatron Core 变动和 Orin/MindSpeed 实验变动分开管理。

所以当前结构更像是「Megatron Core + MindSpeed runtime + Orin recurrent model」的一条专用训练后端。

## 后续维护边界

如果要改 MindSpeed 接入，优先看这些文件：

- `train_recurrent_lm_mindspeed_2node.sh`：环境、集群、NPU、W&B、Megatron CLI 参数。
- `pretrain_recurrent_lm_mindspeed.py`：训练入口、数据集、batch、loss、forward step。
- `runtime/mindspeed_runtime.py`：MindSpeed-LLM import 和关键 patch 开关。
- `runtime/mindspeed_patches.py`：MTP guard 和 W&B 日志兼容。
- `modeling/llama_orin_ssm.py`：Orin recurrent model、MindSpeed SSM/CP 调用、NPU RMSNorm。
- `llama_config/megatron_export.py`：LLaMA-Factory 配置和初始化语义。
- `megatron/legacy/data/data_samplers.py`：数据顺序和 sampler 对齐。
- `megatron/training/global_vars.py`：W&B run/resume/global step 对齐。

如果要同步新的 MindSpeed / MindSpeed-LLM 上游源码，建议先只替换 `third_party/orin_mindspeed/`，再用 dry run 检查：

```bash
DRY_RUN=1 bash train_recurrent_lm_mindspeed_2node.sh
```

然后至少验证三件事：

1. `pretrain_recurrent_lm_mindspeed.py` 能正常 import MindSpeed-LLM adaptor。
2. `runtime/mindspeed_patches.py` 的 monkey patch 目标函数名没有变。
3. `modeling/llama_orin_ssm.py` 动态加载的 SSM 文件路径和 API 仍然存在。

## 一句话总结

这次 MindSpeed patch 的方式是：把 MindSpeed/MindSpeed-LLM 作为 vendored backend 放进 `third_party`，通过启动脚本注入 `PYTHONPATH`，在新训练入口最早阶段加载 MindSpeed adaptor 和 runtime patch，再把本仓库的 Orin recurrent LM、HF tokenized dataset、LLaMA-Factory 初始化/采样/W&B 语义接到 MindSpeed-LLM 的 Megatron trainer 上。
