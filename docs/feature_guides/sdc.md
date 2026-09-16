# 静默数据损坏检测（SDC）

SDC（Silent Data Corruption，静默数据损坏）用于检测 Ascend NPU 编译训练中的 gradient、
HCCL 和 matmul checksum 异常，支持标准 NPU Trainer 的 Inductor 编译路径，以及
GraphTrainer 的 `aot_fx_trace` 路径。该能力默认关闭，通过 `sdc` 对应的 CLI 参数按需启用。

## 启动入口

所有模式都通过 `torchrun -m torchtitan_npu.train` 启动。该入口会在上游配置加载前安装
NPU 配置转换，使标准配置构造
`TrainerEx`，将上游 GraphTrainer 配置转换为 `GraphTrainerEx.Config`。直接使用
`torchrun -m torchtitan.train` 可能构造上游 Trainer，不支持作为
NPU SDC 入口。

`scripts/run_train.sh` 和 `scripts/run_train_multinodes.sh` 默认使用
`torchtitan_npu.train`，无需把 `scripts/` 加入 `PYTHONPATH`。

## 环境变量

### Eager

Eager 模式使用 torch-npu 原生控制：

```bash
NPU_ASD_CONFIG='enable:true,with_checksum:false' \
torchrun --nproc_per_node=8 -m torchtitan_npu.train \
  --module torchtitan_npu.models.deepseek_v4 \
  --config deepseek_v4_flash_43layers_16experts
```

`NPU_ASD_CONFIG` 和 `NPU_ASD_ENABLE` 在 Python 进程导入 torch-npu 前生效，可能安装
torch-npu 的 `torch.nn.Module.__call__` wrapper。仅设置原生变量时，`TrainerEx` 不安装
compiled gradient、checksum 或 typed HCCL 控制。

这两个环境变量由 torch-npu 原生实现负责解释；compiled SDC 仅接受 typed CLI/config。

### Compile

编译模型使用 typed SDC 参数：

```bash
torchrun --nproc_per_node=8 -m torchtitan_npu.train \
  --module torchtitan_npu.models.deepseek_v4 \
  --config deepseek_v4_flash_43layers_16experts \
  --sdc.gradient-enabled \
  --sdc.with-checksum \
  --sdc.hccl-mode 2 \
  --compile.enable \
  --compile.components model \
  --compile.backend inductor
```

typed SDC 参数如下：

| CLI 参数 | 默认值 | 约束与作用 |
| --- | ---: | --- |
| `--sdc.gradient-enabled` | `false` | 启用 gradient detection。 |
| `--sdc.with-checksum` | `false` | 启用 matmul checksum，依赖 gradient detection。 |
| `--sdc.hccl-mode` | `0` | HCCL 检测模式，可取 `0`、`1`、`2`、`3`，`0` 表示关闭。 |
| `--sdc.cooldown` | `5` | 异常检测冷却时间，单位为分钟，必须大于 `0`。 |
| `--sdc.strikes-num` | `3` | 窗口内触发检测的异常次数，必须大于 `0`。 |
| `--sdc.strikes-window` | `480` | 异常计数窗口，单位为分钟，必须大于 `0`。 |
| `--sdc.checksum-cooldown` | `180` | checksum 冷却时间，单位为分钟，必须大于 `0`。 |
| `--sdc.upper-thresh1` | `1000000` | native gradient 检测阈值，必须大于或等于 `3`。 |
| `--sdc.upper-thresh2` | `100` | native gradient 检测阈值，必须大于或等于 `3`。 |
| `--sdc.grad-sample-interval` | `3` | 每隔多少个候选参数选择一个检测目标，必须大于 `0`。 |

有效开启条件是 `gradient_enabled or hccl_mode != 0`。全部使用默认值时，SDC 不导入
torch-npu，也不校验编译兼容性。任一 gradient tuning 参数偏离默认值时，必须启用
`--sdc.gradient-enabled`；上述 7 个 tuning 参数都按整数解析，非法值直接报错。

typed compiled SDC 开启时，允许未设置原生变量，或使用关闭配置
`NPU_ASD_CONFIG=enable:false`、`NPU_ASD_ENABLE=0`。`NPU_ASD_CONFIG` 的 `enable`
按原生语法解析，缺省为 `false`；仅检查开关，不重新校验其他原生调参。
任一原生开关启用或取值非法均报错，不因另一个开关关闭而忽略它。
CLI 指定非零 HCCL 模式时，在导入 torch-npu 后覆盖原生关闭值 `0`。
这不支持导入时已启用 Eager 后再修改环境来切换模式。typed SDC 关闭时不会读取或修改
这两个变量，原生 eager SDC 仍完全由 torch-npu 管理。

`SDC` 直接配置 torch-npu 原生 checker，并调用其 `_startup`、`_detect_grad` 和私有 HCCL
状态接口，不增加 Adapter 或仅转发调用的管理层。本仓库不重实现检测算法；checksum
pass/custom op 调用 `torch_npu.matmul_checksum`，把 native checksum 调用插入 compiled graph。

compiled SDC 要求：

- `compile.enable=true`，且 `compile.components` 包含 `model`。
- 标准 Trainer 要求 `compile.backend=inductor`；GraphTrainer 的 `aot_fx_trace` 路径
  通过自身图 pass 流水线编译，不用该 backend 字段判断是否采用 Inductor。
- `parallelism.pipeline_parallel_degree=1`，不支持流水线并行。
- HCCL 检测在模型构建完成后一次性启用，覆盖此后的 forward、backward 和训练步骤间
  collective，不覆盖 Trainer 构造期间的初始化通信。启用 HCCL SDC 可能带来额外性能损耗。

### GraphTrainer

使用已有、能够正常运行网络的 GraphTrainer 配置，通过上述 NPU 启动入口加载，并按需追加
`--sdc.gradient-enabled`、`--sdc.with-checksum` 和 `--sdc.hccl-mode`。
SDC 不修复模型本身的 tracing 或算子兼容性问题。

`compile.mode=aot_fx_trace` 且开启 checksum 时，还要求：

- `compile.enable_passes=true`。
- 不设置 `compile.precompile_artifact_dir`，不支持直接加载预编译图产物。
- 输入配置的 `compile.pass_pipeline` 保持 `default`，不支持自定义流水线。
- `compile.disable_passes` 不包含 `sdc_checksum_graph_pass`。

框架自动选择内部的 `npu_sdc_checksum` 流水线，用户无需手动填写这个名称。
该流水线复用上游默认 passes，只在区域或全图 Inductor 编译 pass 前插入 checksum pass。
因此，检测节点的插入不依赖 Inductor 内部 hook，但后续仍使用上游 Inductor 编译 pass。

## 生命周期

`TrainerEx` 按以下顺序编排 SDC：

```text
Trainer.__init__
-> distributed initialization and model wrapping
-> config.sdc.build(
       trainer_config=config,
       model_parts=self.model_parts,
       gradient_accumulation_steps=self.gradient_accumulation_steps,
   )
   -> inactive: return
   -> validate compile compatibility and native environment
   -> import torch-npu before setting the HCCL environment
   -> configure HCCL, install checksum pass and initialize gradient checker

Trainer.forward_backward_step()
-> SDC.finalize_sdc_step()
```

模型和梯度累积步数是 SDC 构造参数，不再另设后置初始化接口。`torch.compile` 包装本身不捕获
训练图，因此可在模型包装后、首次前向前安装 checksum pass；框架不能在构造 SDC 前提前执行
训练图。

HCCL native gate 读取并缓存 `NPU_ASD_ENABLE`。compiled SDC 在 torch-npu 导入完成后，
把 typed `hccl_mode` 写入该变量，并将 module state 设为 `train`、call state 设为 `backward`。
这两个状态在后续训练中保持不变。首次导入前设置原生变量可能启用 Eager 包装；native 检测路径
已经缓存 HCCL 模式后再修改变量则不会生效。标准 Trainer 构造期间不启用上述检测状态；其他框架
接入时也必须保证设置开关前没有提前触发 native 检测路径。更改模式需要重启 worker。
SDC 使用进程级 torch-npu 和 Inductor 状态，不提供运行期重配置或卸载接口。

标准 Trainer 中，`with_checksum=True` 时，`SDC.__init__()` 安装进程级 Inductor
post-grad pass。GraphTrainer 的 AOT 路径则在上游初始化前调用
`config.sdc.prepare_graph(config.compile)` 选择流水线，在前向、反向联合图上插入检测节点，
不再为该路径安装 Inductor post-grad hook。两种入口复用同一份 checksum 插入逻辑。

首次构建的训练图即包含由 `checker.checksum_enable` 控制的 checksum custom op。Gradient checker
达到阈值后只打开 native gate，后续 step 直接在原图执行 checksum，不调用
`torch.compiler.reset()`。

checksum pass 对训练图中的 BF16 NPU `aten.mm.default`、`aten.matmul.default` 和
`aten.bmm.default` 插入 checksum custom op。

`SDC.finalize_sdc_step()` 只在成功 step 后执行 gradient accumulation 边界提交。HCCL 状态不会在
step 结束时恢复或切换；失败 step 的异常直接向上传播，不执行后处理。

## 模块职责

| 路径 | 职责 |
| --- | --- |
| `torchtitan_npu/extensions/trainer.py` | 编排标准 Trainer 与 SDC 生命周期。 |
| `torchtitan_npu/extensions/graph_trainer.py` | 准备 SDC 图配置，复用上游训练流程和 NPU Trainer 生命周期。 |
| `torchtitan_npu/extensions/components/sdc/sdc.py` | 完成配置校验、图流水线注册、HCCL、checksum 和 gradient 初始化；在 accumulation 边界直接提交梯度。 |
| `torchtitan_npu/compile/sdc_checksum.py`、`torchtitan_npu/ops/misc/sdc_checksum.py` | 提供 Inductor 和联合图两种 pass 入口，复用检测节点插入逻辑并注册 custom op。 |

## 验证范围

`tests/unit_tests/test_sdc.py` 覆盖 config、gradient 和 HCCL 的 CPU 可观察契约，
通过 native checker 替身验证 typed tuning、参数筛选和累积提交，并检查 `TrainerEx`
在模型构建后一次初始化 SDC，失败 backward 不提交梯度。

`tests/unit_tests/extensions/test_graph_trainer.py` 覆盖 GraphTrainer 配置适配、checksum
流水线约束、pass 顺序和检测节点插入等契约。

`tests/smoke_tests/sdc/test_sdc.py` 为每个场景启动独立 Python worker，覆盖 Eager/compiled
控制与无效配置、BF16 gradient、checksum custom-op 输出修改、编译图 consumer、初始化通信后
启用 HCCL 的训练路径，以及 three-strikes 后不重新编译的 checksum 激活。该文件是专项 NPU
冒烟测试，并包含手工前向 FX 图的 checksum 插入及执行场景。该场景不覆盖 GraphTrainer
实际的前向、反向联合图构建和上游编译流水线，也不替代 `tests/integration_tests` 中的完整
模型训练验证；真实联合图链路仍需验证，不能据此宣称 DeepSeek-V4 GraphTrainer 全网络
端到端已经验证通过。
