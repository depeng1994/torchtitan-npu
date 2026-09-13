# DeepSeek-V4 training examples

本目录只保留少量稳定入口，模型规模、序列长度、并行策略和量化方式优先通过 CLI 覆盖，不为每一种组合新增独立脚本。

## 入口关系

Flash 多机训练以 `deepseek_v4_flash_cpt_4k_a3.sh` 为公共基线：

```text
deepseek_v4_flash_cpt_4k_a3.sh
└── deepseek_v4_flash_cpt_4k_a5.sh
    └── deepseek_v4_flash_cpt_1024k_a5.sh
```

- `deepseek_v4_flash_cpt_4k_a3.sh`：公共训练参数、override、checkpoint、optimizer 等基线配置，默认启用 `COMPILE_BACKEND=aot_eager`。
- `deepseek_v4_flash_cpt_4k_a5.sh`：只增加 A5 运行环境和默认 block-FP8 量化，其他配置复用 A3 基线。
- `deepseek_v4_flash_cpt_1024k_a5.sh`：继续复用 A5 4K 入口，只通过 CLI 覆盖 1M 所需的 CP、DP、序列长度和 global batch size，量化默认随 A5 4K 入口继承。
- 用户传入的 `"$@"` 始终位于最后，因此可以覆盖脚本中的默认 CLI 参数，包括使用 `--extension.quantization.no-enable-quantized-training` 切换到 BF16。

## 常用环境变量

运行前至少根据环境设置以下路径和节点信息：

```sh
export NODE_IPS="192.168.1.10,192.168.1.11,..."
export HF_ASSETS_PATH=/path/to/DeepSeekV4_tokenizer
export CKPT_SAVE_LOAD_PATH=/path/to/save_or_resume_ckpt
export CKPT_INIT_LOAD_PATH=/path/to/init_hf_ckpt
```

A5 默认每节点使用 8 张 NPU，A3 默认每节点使用 16 张 NPU；需要改变时设置 `NGPU`。

## 裁剪 Debug 模型

Debug/performance 验证优先使用等价裁剪模型，避免为了快速验证直接启动完整规模模型。这里统一放置不同规模的裁剪 Debug 入口，卡数不是这一节的分类依据。

### 1P mini 模型

```sh
bash examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh

USE_GOLDEN=1 \
bash examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh
```

### 8P Flash 16-expert Debug

```sh
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh \
  --training.steps 10
```

需要采集 profile 时显式追加：

```sh
--profiler.enable-profiling
```

### 8P Flash 40 层 16 expert + ViT（V4.1 Golden）

`debug/deepseek_v41_flash_8p_cpt_4k_a3.sh` 是 V4.1（V4 基座 + ViT 多模态）的单机 8 卡入口。
当前支持范围固定为 **CP1 / PP1 / eager / reference-golden**；V4.1 配置会拒绝 `torch.compile`，launcher 也会拒绝 `USE_GOLDEN=0`。

```sh
bash examples/deepseek_v4/debug/deepseek_v41_flash_8p_cpt_4k_a3.sh
# 可覆盖训练步数：
bash examples/deepseek_v4/debug/deepseek_v41_flash_8p_cpt_4k_a3.sh --training.steps 5
```

V4.1 当前仅支持 reference/golden operator 路径；`USE_GOLDEN=0`（AscendC）会在 launcher 阶段直接拒绝。
AscendC 支持将在 ratio-1 shared global KV kernel contract 完成后开放。

该入口默认 `--checkpoint.no-enable`（不保存也不加载）；需要 checkpoint 时用 CLI 显式打开。

#### V4.1 支持矩阵

| 路径 | 状态 | 说明 |
|---|---|---|
| reference/golden (`USE_GOLDEN=1`) | ✅ **实现支持 / 待最终轨迹复验** | ratio-1 已按真实 shared/global-KV 语义 materialize；该 correctness refactor 改变了旧冻结轨迹，需在最终 HEAD 上重新跑 8P 100-step 后迁移 Golden |
| AscendC 融合算子 (`USE_GOLDEN=0`) | ❌ **暂不支持**（fail-fast） | V4.1 的 CSA2 ratio-1 shared global KV 尚未在 AscendC 稀疏注意力核中实现 |
| `torch.compile` / GraphTrainer | ❌ **暂不支持**（fail-fast） | 当前 CSA2 cross-layer state 仍是 eager-only contract |
| CP > 1 / PP > 1 | ❌ **暂不支持**（fail-fast） | 当前 landing scope 为 CP1 / PP1 |

`USE_GOLDEN=1` 时 launcher 装入三个 operator overrides：
`torchtitan_npu.override.common.rope.workaround`、`torchtitan_npu.override.deepseek_v4.sparse_attn.golden`、
`torchtitan_npu.override.deepseek_v41.golden_moe.golden`，外加 virtual optimizer override。
模型侧统一通过 `torchtitan_npu.models.deepseek_v4.golden.golden_enabled()` 读取开关。

模型宽度、层数、专家数和视觉层数由 config 决定；V4.1 的 compressor/indexer ownership、CSA2 source/reuse/reindex policy 与 ratio-1 materialization 由 `deepseek_v41` 侧显式组装，V4 基座只保留版本中性的 primitive / extension seams。

### 32P Pro 32-expert Debug

Pro 32-expert 是独立的裁剪模型 debug/performance 入口，不与 8P Flash launcher 强制复用：

```sh
NODE_IPS="${NODE_IPS}" \
COMPILE_BACKEND=aot_eager \
bash examples/deepseek_v4/debug/deepseek_v4_pro_32p_cpt_4k_a5.sh \
  --debug.moe-force-load-balance \
  --training.steps 20
```

128K 场景不新增 launcher，只通过 CLI 改变并行和训练配置：

```sh
NODE_IPS="${NODE_IPS}" \
COMPILE_BACKEND=aot_eager \
bash examples/deepseek_v4/debug/deepseek_v4_pro_32p_cpt_4k_a5.sh \
  --parallelism.context-parallel-degree 32 \
  --parallelism.data-parallel-shard-degree 1 \
  --parallelism.data-parallel-replicate-degree 1 \
  --training.seq-len 131072 \
  --training.global-batch-size 8 \
  --debug.moe-force-load-balance \
  --training.steps 20
```

A5 脚本默认启用 block-FP8；如需 BF16，请在命令末尾追加后文的禁用选项。

### 导出和加载 Debug Hugging Face checkpoint

导出 checkpoint：

```sh
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh \
  --checkpoint.folder ./export_ckpt \
  --checkpoint.enable \
  --checkpoint.no-load-only \
  --checkpoint.last-save-in-hf \
  --training.steps 1 \
  --debug.seed 42 \
  --debug.deterministic
```

从导出的 HF checkpoint 冷启动时，使用新的或空的 `checkpoint.folder`，避免已有 step checkpoint 优先触发 resume：

```sh
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh \
  --checkpoint.folder ./load_ckpt \
  --checkpoint.enable \
  --checkpoint.initial-load-path ./export_ckpt/checkpoint/step-1 \
  --checkpoint.initial-load-in-hf \
  --training.steps 1 \
  --debug.seed 42 \
  --debug.deterministic
```

## 多机完整模型训练

### A3 4K

```sh
NODE_IPS="${NODE_IPS}" \
bash examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a3.sh \
  --training.steps 500
```

### A5 4K

```sh
NODE_IPS="${NODE_IPS}" \
COMPILE_BACKEND=aot_eager \
bash examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a5.sh \
  --debug.moe-force-load-balance \
  --training.steps 20
```

### A5 64K

64K 不需要新增 launcher，直接在 A5 4K 入口上覆盖训练和并行参数：

```sh
NODE_IPS="${NODE_IPS}" \
COMPILE_BACKEND=aot_eager \
bash examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a5.sh \
  --parallelism.context-parallel-degree 8 \
  --parallelism.data-parallel-shard-degree 16 \
  --parallelism.data-parallel-replicate-degree 1 \
  --training.seq-len 65536 \
  --training.global-batch-size 128 \
  --debug.moe-force-load-balance \
  --training.steps 20
```

### A5 1M

1M wrapper 只提供该场景的默认 CLI 覆盖；仍可在命令末尾继续覆盖：

```sh
NODE_IPS="${NODE_IPS}" \
COMPILE_BACKEND=aot_eager \
bash examples/deepseek_v4/deepseek_v4_flash_cpt_1024k_a5.sh \
  --debug.moe-force-load-balance \
  --training.steps 20
```

等价的主要差异参数为：

```sh
--parallelism.context-parallel-degree 128 \
--parallelism.data-parallel-shard-degree 1 \
--parallelism.data-parallel-replicate-degree 1 \
--training.seq-len 1048576 \
--training.global-batch-size 8
```

## A5 量化与 BF16

所有 A5 脚本（`*_a5.sh`）默认启用 block-FP8 量化训练，等价于自动追加：

```sh
--extension.quantization.enable-quantized-training \
--extension.quantization.recipe all_block_fp8
```

例如 A5 64K、1M 和 Pro 32P 命令均默认使用该量化配置。需要保留 BF16 训练方案时，在命令末尾追加以下选项，覆盖脚本默认值：

```sh
--extension.quantization.no-enable-quantized-training
```

A3 脚本不默认启用量化；如需 A3 量化训练，请显式传入上面的 enable 和 recipe 参数。

## Profiling

需要采集 profile 时显式开启：

```sh
--profiler.enable-profiling
```

公共 A3 基线提供如下 schedule，A5 Flash 直接继承，用户仍可通过 CLI 覆盖：

```sh
--profiler.profile-freq 1 \
--profiler.profiler-warmup 0 \
--profiler.profiler-active 1 \
--profiler.profiler-repeat 1 \
--profiler.profiler-skip-first 4
```

如需不依赖 recipe 固定单次采样窗口，请在命令末尾完整追加：

```sh
--profiler.enable-profiling \
--profiler.profile-freq 10 \
--profiler.profiler-warmup 3 \
--profiler.profiler-active 1 \
--profiler.profiler-repeat 1 \
--profiler.profiler-skip-first 0
```

## Checkpoint 说明

Flash 多机 CPT 基线默认启用 `--checkpoint.load-only`，即只负责加载，不会保存新的训练 checkpoint。需要保存时必须显式覆盖：

```sh
--checkpoint.no-load-only
```

当 `checkpoint.folder` 已存在有效的 `step-*` checkpoint 时，TorchTitan 会优先 resume，该情况下 `checkpoint.initial-load-path` 不作为冷启动来源；要从 `initial-load-path` 冷启动，请使用新的或空的 `checkpoint.folder`。
