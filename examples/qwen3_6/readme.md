# Qwen3.5/Qwen3.6 NPU training examples

本目录提供 Qwen3.5/Qwen3.6 的 NPU 训练入口：

- `run_train_qwen3_5.sh`：Qwen3.5 多模态训练 launcher，默认使用 `qwen35_debugmodel`
  配置与 cc12m-test 数据集。
- `run_qwen3_6_4k.sh`：Qwen3.5 27B 4K SFT 入口，需设置 `HF_ASSETS_PATH` 与
  `DATA_FILES`，默认 16 卡并从 HF checkpoint 初始化。

## Qwen3.5 multimodal dependencies

The Qwen3.5 adapter reuses TorchTitan's upstream image and video preprocessing.
Install the torchvision nightly that declares compatibility with the container's
Torch build, then install this repository without resolving the container's
custom Torch/torch_npu pair:

```bash
python -m pip install av einops pillow
python -m pip install --no-deps \
  --index-url https://download.pytorch.org/whl/nightly/cpu \
  torchvision==0.29.0.dev20260719+cpu
python -m pip install --no-deps -e ../torchtitan
python -m pip install --no-deps -e .
```

This pairing targets `torch==2.14.0.dev20260719+cpu` on Python 3.12/aarch64.
`--no-deps` is intentional because the custom torch_npu wheel declares the
stable `torch==2.14.0` version even though the validated runtime uses a nightly
Torch build.

Fresh environments that install `requirements.txt` instead resolve
`torchvision==0.29.0.dev20260720`: that nightly build declares
`torch==2.14.0.dev20260719` exactly, so plain pip resolution accepts it beside
the pinned Torch. Both builds belong to the validated 0.29.0.dev line.

The NPU eager FlexAttention workaround prefers the `return_aux=AuxRequest(...)`
protocol and falls back to the pinned build's legacy `return_lse` argument when
that protocol is unavailable. Results are normalized to `AuxOutput` so callers
can use one contract across both Torch API generations.

## Qwen3.5 NPU override 开关

`examples/qwen3_6/run_train_qwen3_5.sh` 默认显式启用 Triton GDN override。配置工厂本身
保持与上游完全一致，不会静默注入 NPU 实现；视觉 mask、视频 MRoPE 和 FLA
兼容补丁由 `torchtitan_npu.models.qwen3_5` 导入时按上游接口自动安装。

MoE token dispatcher 是性能选项（`ENABLE_NPU_MOE_DISPATCHER` 缺省值为 `0`），默认关闭。可通过以下任一方式显式启用：

```bash
ENABLE_NPU_MOE_DISPATCHER=1 ./examples/qwen3_6/run_train_qwen3_5.sh

OVERRIDE_IMPORTS="torchtitan_npu.override.qwen3_5.gated_delta.npu,torchtitan_npu.override.common.token_dispatcher.asc" \
  ./examples/qwen3_6/run_train_qwen3_5.sh
```

用户设置的 `OVERRIDE_IMPORTS` 始终优先；显式设置为空字符串可禁用全部默认
override，用于消融或定位问题。

### 启动参数与路径

`run_train_qwen3_5.sh` 与仓库通用的 `run_train.sh` 使用相同的参数透传方式：脚本
只负责准备 Ascend 环境、默认 NPU 运行时变量和 Qwen 多模态数据参数，所有训练
参数继续交给 `torchtitan.train`。上游目录按以下优先级解析：

1. `TORCHTITAN_REPO`；
2. `TORCHTITAN_DIR`；
3. launcher 所在仓库的同级目录 `../torchtitan`。

脚本将容器中已安装的 TorchTitan 作为唯一代码来源，并把适配仓库加入
`PYTHONPATH`；`TORCHTITAN_REPO` 仅用于 tokenizer 和数据集资产路径。启动前会
打印 TorchTitan 与 NPU adapter 的实际导入路径，避免源码 checkout 与已安装包
版本不匹配。数据和日志路径也可以覆盖：`HF_ASSETS_PATH` 默认为
`${TORCHTITAN_REPO}/tests/assets/tokenizer`，`DATASET_PATH` 默认为
`${TORCHTITAN_REPO}/tests/assets/cc12m_test`，`LOG_DIR` 默认为仓库同级的
`../log`。如需使用 torch.compile，可设置 `COMPILE_BACKEND`，脚本会追加
`--compile.enable --compile.components model --compile.backend` 参数。varlen GDN
卷积的 CPU `cu_seqlens` 元数据循环保持在 eager 边界，编译图会在该边界处
graph-break；这是动态文档切分检查的兼容设计。

示例（单机 8 卡 MoE）：

```bash
TORCHTITAN_DIR=/path/to/torchtitan \
NGPU=8 CONFIG=qwen35_debugmodel_moe LOG_RANK=5 \
  bash examples/qwen3_6/run_train_qwen3_5.sh --training.steps 2
```

### Qwen3.5-VL 适配链路

运行时调用关系如下，NPU 适配只替换必要边界，上游 recipe 和模型主体仍由
TorchTitan 提供：

```text
run_train_qwen3_5.sh
  ├─ CANN set_env + NPU 默认变量 + TorchTitan 导入路径诊断
  ├─ 组装 tokenizer、cc12m-test、override.imports 和用户参数
  └─ torchrun -m torchtitan.train
       ├─ torchtitan_npu.models.qwen3_5 导入时安装视觉/MRoPE/FLA patch
       ├─ 上游 config_registry 构造 qwen35_debugmodel(_moe)
       ├─ GDN override 替换 delta kernel；可选 dispatcher 替换 MoE dispatch
       ├─ parallelize_qwen3_5_npu 建立 TP/EP/FSDP；CP 使用序列元数据与 mesh resolver
       ├─ upstream multimodal collator 读取文本、图像/视频并生成 MRoPE
       ├─ vision encoder + language decoder 前向，执行 GDN/MoE/CP 数据交换
       ├─ backward、优化器更新与 TensorBoard/JSONL 日志
       └─ 进程组销毁
```

`spmd_types` 使用 TorchTitan v0.3 的 `resolve_fsdp_mesh`/
`resolve_sparse_fsdp_mesh`，而旧 backend 保留 `fsdp/efsdp` 名称；因此同一套
parallelizer 可兼容当前主线和旧 NPU 运行环境。当前 Qwen3.5-VL 多模态视觉 mask
尚未完成 CP 序列元数据适配；VL 配置启用 CP 时会在并行化入口明确失败，避免
运行到错误的 dict-mask 路径。纯文本长上下文配置仍可使用专用 CP recipe。
VL CP2 条目在集成列表中保留但默认禁用，CPU 边界用例位于
`tests/unit_tests/models/qwen3_5/`。

集成测试通过以下命令运行 Qwen3.5 的 1-rank 冒烟用例；多模态 CP2 条目因
上述明确的 unsupported guard 默认禁用。测试使用与上述脚本相同的 cc12m-test
与 tokenizer 默认值：

```bash
python -m tests.integration_tests.run_tests /tmp/qwen3_5-tests \
  --test_suite qwen3_5 --test_name all --ngpu 2
```

## Qwen3.5 视觉 mask 边界

当前 NPU eager Flex 兼容层只处理 Qwen3.5 vision patch 显式标记的、等长且
各 attention head 共享的 self-attention mask；其他模型和其他 Flex 语义继续走
原生路径。dense bool mask 在一次 vision forward 内缓存并由所有 vision block
复用。

为避免合法的大图或长视频直接触发 NPU OOM，物化上限为 `16,777,216` 个 bool
元素（单样本等长输入约为 4096 个 merged vision token）。超过上限会在算子下发
前 fail-fast；请通过数据配置降低图片最大像素、视频帧数或视觉序列长度。
