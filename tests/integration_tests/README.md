# TorchTitan-NPU 集成测试指南

本目录遵循 TorchTitan 的 `tests/integration_tests` 架构，负责维护集成测试（Integration Tests）定义、执行入口、数值精度对齐校验以及云端 Nightly CI 调度规范。

集成测试分为两种核心运行形态：
1. **本地单机集成测试**：在具备 NPU 的开发机或单节点上直接执行，使用仓内集成的并行 Runner，支持 Deterministic 数值 Loss 精确比对与算子冒烟；
2. **云端集群 Nightly CI 测试**：通过 GitHub Actions 触发，由内网 [Lite Actions](https://github.com/depeng1994/lite-actions) 调度机在 A3 / A5 物理集群上排队执行。

---

## 1. 运行入口与基本命令

### 1.1 脚本入口
CI 流水线主要通过以下入口启动集成测试：

```bash
.ci/integration_test.sh
```

该脚本会自动准备 CANN 环境变量与运行解释器环境。

### 1.2 Python 原生 Runner 入口
在开发机上手动运行测试时，可直接调用 Python 模块：

```bash
python -m tests.integration_tests.run_tests \
  ./test_reports/integration \
  --test_suite models \
  --ngpu 4
```

> **参数说明**：
> - `./test_reports/integration`：**必填**测试输出目录，运行前必须确保为空；
> - `--test_suite`：指定要执行的测试套件（如 `models`、`deepseek_v4_checkpoint` 等）；
> - `--ngpu`：指定单次测试使用的总卡数；
> - `--test_name`：（可选）仅执行该套件内的单条特定用例；
> - `--no-parallel`：（可选）强制串行执行，默认会开启多用例并行。

---

## 2. 本地集成测试全景矩阵与核心机制

### 2.1 核心测试矩阵清单

| Case 名称 | 模型 | 并行配置 | 卡数 | Check Loss | 核心校验点与说明 |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`dsv4_golden_1rank`** | DeepSeek-V4 | 1 Rank 参考配置 | 1 | **是** | 严格 Deterministic 模式，精确逐值比对参考 Loss |
| **`dsv4_golden_ep2_fsdp2`** | DeepSeek-V4 | EP2 + FSDP2 | 2 | **是** | 分布式并行环境下的数值 Loss 精确比对 |
| **`dsv4_checkpoint_resume_ep2_fsdp2`** | DeepSeek-V4 | EP2 + FSDP2 + AdamW NovaSwap | 2 | **动态比对** | 两阶段连续训练 vs 断点恢复训练，逐步比对 Loss 与 Grad Norm |
| **`dsv4_lora_ep2_fsdp2`** | DeepSeek-V4 LoRA | EP2 + FSDP2，训练 LoRA A/B | 2 | 否 | 校验基座参数冻结、Adapter 权重更新及 PEFT 导出完整性 |
| **`dsv4_muon_swap_ep2_fsdp2`** | DeepSeek-V4 | NPU 融合算子 + DistMuon NovaSwap | 2 | 否 | NPU 融合算子（RMSNorm/RoPE/SparseAttention）与 Swap 路径冒烟 |
| **`dsv4_anticipatory_recovery`** | DeepSeek-V4 | 1 Rank 原生 Checkpoint 回滚 | 1 | 否 | 一次性扰动训练 Loss，断言 Override 机制与完整恢复周期 |
| **`dsv4_smla_1rank_aot_eager`** | DeepSeek-V4 | 1 Rank，`aot_eager` 编译 | 1 | 否 | SMLA 算子图在单卡下的编译构图与前反向冒烟 |
| **`dsv4_smla_ep2_fsdp2`** | DeepSeek-V4 | EP2 + FSDP2，`aot_eager` | 2 | 否 | SMLA 算子在 EP+FSDP 分布式环境下的执行与 MoE Dispatcher 验证 |
| **`dsv4_smla_cp2_ep2_fsdp2`** | DeepSeek-V4 | CP2 + EP2 + FSDP2 | 4 | 否 | 混合并行（Context Parallel + Expert Parallel）算子冒烟 |
| **`dsv4_mtp_smla_cp2_headtail`** | DeepSeek-V4 MTP | CP2 + headtail | 2 | 否 | 覆盖 MTP forward、chunked loss 及 backward 算子路径 |
| **`dsv3_2_dsa_1rank`** | DeepSeek-V3.2 | 1 Rank，DSA 算子 | 1 | **是** | 校验 DSA attention override 并在 1-rank 下做数值 Loss 比对 |
| **`dsv3_2_dsa_ep2_fsdp2`** | DeepSeek-V3.2 | DSA + EP2/FSDP2 | 2 | **是** | 校验分布式 DSA 算子 100-step 精度对齐 |
| **`dsv3_2_dsa_cp2`** | DeepSeek-V3.2 | DSA + CP2 | 2 | 否 | 校验 DSA 在 CP2 模式下的训练触发与通信 |
| **`dsv4_ema_ep2_fsdp2`** | DeepSeek-V4 | Golden + EP2/FSDP2 + EMA CPU Offload | 2 | 否 | 校验 DCP Checkpoint Metadata 完整包含 `ema_optimizer.*` |

---

### 2.2 精度对齐判定原则：`use_golden` 与 `check_loss`

集成测试严格区分**算子实现选择**与**数值精确对齐**两个独立维度：
* **`use_golden`**：决定模型使用 Golden 参考标准实现，还是使用 SMLA / NPU 专用算子优化 override；
* **`check_loss`**：决定是否启用确定性计算（`--debug.deterministic`）、固定随机种子，并严格读取参考基准 Loss 执行逐值比对。

**判定规则**：
1. 设置 `check_loss=True` 的用例（如 Golden 用例与 DSA 对齐用例），会自动比对 TensorBoard 中的标量 `loss_metrics/global_avg_loss`，**要求执行步骤集合和每一个浮点数值均严格相等（不允许浮点容差）**；
2. 设置 `check_loss=False` 的用例主要用于验证算子图在特定编译模式（如 `aot_eager`）或并行拓扑下的稳定性与执行路径，只要正常完成指定步数且主进程退出码为 0 即判定通过。

---

### 2.3 核心特色用例设计机制

#### 1. DCP 动态断点续训比对 (`dsv4_checkpoint_resume_ep2_fsdp2`)
该用例用于严格验证在分布式 AdamW NovaSwap 场景下的 Checkpoint 保存与恢复精度一致性，采用**两阶段动态自校验**设计：
* **阶段一（连续基准）**：固定 `seed=42` 开启确定性模式，一次性连续训练 4 步，并在 step 2 保存完整的 DCP Checkpoint；
* **阶段二（续训恢复）**：新进程通过 `--checkpoint.load-step=2` 加载 step 2 权重，继续训练第 3、4 步；
* **断言验证**：比对两次运行在 TensorBoard 中记录的 `global_avg_loss` 与 `grad_norm`。第 3、4 步的数值必须逐值精确相等，不舍入、不设容差，杜绝续训精度漂移。

运行该用例：
```bash
python -m tests.integration_tests.run_tests ./test_reports/lora-resume \
  --test_suite deepseek_v4_checkpoint \
  --test_name dsv4_checkpoint_resume_ep2_fsdp2 \
  --ngpu 2
```

#### 2. LoRA 训练与 PEFT 导出 (`dsv4_lora_ep2_fsdp2`)
在模型并行化后冻结 Base 基座权重，仅针对 LoRA A/B 结构训练 2 步：
* 检查冻结层参数与 routing bias 保持不变；
* 检查可训练 adapter 权重正常更新；
* 训练结束后通过真实 Checkpointer 导出 PEFT 权重，自动断言导出文件的 key、shape、数值以及 rank/alpha 配置。

#### 3. Block-FP8 量化场景 (`deepseek_v4_quantized`)
该用例需要在支持 Block-FP8 的 Ascend 950 芯片及对应 CANN 运行时下运行，依赖 `torchao==0.17.0` 与可选扩展包 `experiments/torchao-npu`：
```bash
# 安装扩展包
pip install ./experiments/torchao-npu

# 执行量化测试
python -m tests.integration_tests.run_tests ./test_reports/lora-fp8 \
  --test_suite deepseek_v4_quantized \
  --test_name dsv4_lora_block_fp8_ep2_fsdp2 \
  --ngpu 2
```

#### 4. Engram HF 专项验证（超节点手动）
在具备 Engram MXFP8 Host 通信环境的超节点上，验证 V4.1 Engram 参数在原生 DCP 与 FP32 HF 权重格式之间的双向回环保存与恢复一致性：
```bash
HF_ASSETS_PATH=/path/to/v41-tokenizer \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
python -m tests.integration_tests.run_tests /tmp/engram-hf-output \
  --test_suite deepseek_v4_1_engram_hf \
  --ngpu 4 --no-parallel
```
成功执行后控制台输出 `HF_ROUNDTRIP PASS`。

---

### 2.4 单机并发调度（`GPUPool`）与端口防冲突

当执行包含多个用例的套件时，`run_tests` 默认复用 TorchTitan 的 `GPUPool` 机制并发执行测试，自动按卡号切分：
* **设备池管理**：若环境中预设了 `ASCEND_RT_VISIBLE_DEVICES`，仅在指定集合内切分卡；否则自动通过 `torch.npu.device_count()` 获取总卡数。用例按所需卡数由大到小排序提交调度；
* **超时与安全回收**：支持为用例配置超时时间，超时后会自动向训练进程组发送 `SIGTERM`（限时后 `SIGKILL`），杜绝孤儿 `torchrun` 残留占卡；
* **并发诊断**：测试结束后输出 `[parallel] pool`（设备利用率）与 `[parallel] overlap`（并发重叠时长）；
* **同机多任务踩坑重点**：
  > [!IMPORTANT]
  > 如果在同一台机器上手动同时启动多个独立的测试进程或分布式 run，必须为各个运行分配**互不重叠的通信端口范围**：
  > ```bash
  > export HCCL_NPU_SOCKET_PORT_RANGE="60000-60020"  # 任务 A
  > export HCCL_NPU_SOCKET_PORT_RANGE="60021-60040"  # 任务 B
  > ```
  > 否则节点间 HCCL 端口绑定冲突会导致初始化死锁或通信超时。

---

## 3. 集群 Nightly CI 测试矩阵（A3 / A5 真实集群）

Nightly CI 用例专为真实物理集群设计，由内网 [Lite Actions](https://github.com/depeng1994/lite-actions) 调度机通过 SSH 远程拉起：

### GitHub Actions 入口与公共工作流

.github/workflows/lite_actions_common.yml 为内部共享的 workflow_call 实现，
统一负责组织仓/目标分支准入、SHA 绑定、Matrix 规划、不可变 Artifact 上传和云端 Waiter。
各入口只声明触发条件与差异参数：

- a3_8p_lite_actions.yml：自动 PR + 手动选择 Case，a3-8p，自动执行 Suite 中全部 Case；
- a3_16p_lite_actions.yml：自动 PR + 手动选择 Case，a3-16p，自动执行 Suite 中全部 Case；
- a5_64p_lite_actions.yml：预留手动入口，复用同一 Matrix 协议，仍显式禁用；
- lite_actions_dispatcher_smoke.yml：复用公共 Waiter，但仅等待 npu-smi，不生成训练 Matrix。

调用方式使用 ./.github/workflows/lite_actions_common.yml，共享工作流与 PR
入口来自同一个提交。A3 资源分配、真实训练和全局串行仍由内网 Dispatcher 负责。
deploy-docs.yml、sync-upstream.yml 是不同权限/用途的独立工作流，不进入此共用机制。


| Suite 名称 | Case ID | 硬件规格 | 优化器与训练配置 | 验证指标 |
| :--- | :--- | :--- | :--- | :--- |
| **`a3_8p_tests`** | `dsv4_flash_a3_8p_example` | A3 单机 8P | Muon，Eager，5 steps | Exit Code 0、TensorBoard 步数 |
| **`a3_8p_tests`** | `dsv4_flash_a3_8p_adamw` | A3 单机 8P | AdamW + Virtual Optimizer，4K seq，5 steps | Exit Code 0、TensorBoard 步数 |
| **`a3_16p_tests`** | `dsv4_flash_a3_16p_example` | A3 双机 16P | AdamW，EP16，Eager，双机各 8P，5 steps | Exit Code 0、跨机通信与步数 |
| **`a5_64p_tests`** | `dsv4_pro_a5_64p` | A5 八机 64P | DeepSeek-V4 Pro，八机各 8P | 待硬件环境验收后开放 |

---

## 4. 如何新增一个 Nightly 集成用例（开发者实操）

向 Nightly CI 系统添加新用例非常简单，**只需在模型仓修改 Python 代码，无需改动调度机配置**。

### 步骤一：找到用例定义文件
Nightly 用例统一存放于：
`tests/integration_tests/nightly_all_models_test/`

根据用例的卡数规格，选择对应文件编辑（或新建 `*_tests.py` 文件，系统会自动动态发现并加载）：
- 单机 8P 用例：`a3_8p_tests.py`
- 双机 16P 用例：`a3_16p_tests.py`
- 八机 64P 用例：`a5_64p_tests.py`

### 步骤二：编写 `OverrideDefinitions`
每个用例都是一个 `OverrideDefinitions` 对象。核心字段说明如下：

| 字段名称 | 类型 | 说明 |
| :--- | :--- | :--- |
| **`test_name`** | `str` | 用例唯一标识（不可重复，用于 GitHub Matrix 调度命名与结果报告） |
| **`nnodes`** | `int` | 所需节点数（单机写 `1`，双机写 `2`） |
| **`ngpu`** | `int` | 单节点卡数（通常为 `8`） |
| **`override_args`** | `dict` | 传给训练启动脚本的命令行参数（覆盖默认 toml 配置） |
| **`env_vars`** | `dict` | 运行时环境变量（CANN 脚本路径、模型权重路径、优化器覆盖等） |
| **`expected_steps`** | `int` | 预期训练步数（用于 TensorBoard 与日志断言） |

### 步骤三：代码编写模板
打开 `tests/integration_tests/nightly_all_models_test/a3_8p_tests.py`，在 `build_test_list()` 中添加新的用例定义：

```python
from torchtitan_npu.testing.framework import OverrideDefinitions

def build_test_list():
    tests = [
        # 现有用例...

        # 新增用例示例：
        OverrideDefinitions(
            test_name="dsv4_flash_a3_8p_my_feature",
            nnodes=1,
            ngpu=8,
            override_args={
                "--job.config_file": "torchtitan_npu/models/deepseek_v4/train_configs/dsv4_flash_8p.toml",
                "--training.steps": "5",
                "--training.batch_size": "1",
            },
            env_vars={
                # 必填：执行机上的 CANN 环境配置脚本
                "ASCEND_SET_ENV_PATH": "/mnt/share/Ascend/ascend-toolkit/latest/set_env.sh",
                # 必填：Tokenizer 与模型资产路径
                "HF_ASSETS_PATH": "/mnt/share/models/DeepSeek-V4-Flash-bf16",
                # 必填：权重 Checkpoint 路径
                "CKPT_INIT_LOAD_PATH": "/mnt/share/dsv4_ckpt_8rank",
                # 可选：如果使用 Virtual Optimizer 优化 AdamW 显存占用
                "OPTIMIZER_OVERRIDES": "torchtitan_npu.override.common.optimizer.virtual",
                # 可选：设置共享 Hugging Face 缓存，防止 Docker 满盘
                "HF_HOME": "/mnt/share/ci_tests/.lite-actions-hf-cache",
            },
            expected_steps=5,
        ),
    ]
    return tests
```

### 步骤四：本地静态语法与元数据解析自测（无需 NPU）
在提交代码前，在普通的 CPU 开发机环境下执行 `inspect` 子命令，验证用例语法与参数是否合法：

```bash
# 验证 8P suite 能够正确解析并包含你的新增用例
python3 -m tests.integration_tests.tools.lite_actions.entrypoint inspect --suite a3_8p_tests
```

输出将以 JSON 格式打印出所有用例的 `test_id`、`nnodes`、`ngpu` 及环境变量。确认你的 `test_name` 正确显示且无 Python 报错即可。

---

## 5. 组织仓 PR 自动 CI 与白名单准入

正式 Actions 部署于 [TorchTitan-NPU 组织仓](https://github.com/torchtitan-npu/torchtitan-npu)。为了保证内网物理 NPU 集群的执行安全，防止恶意 PR 注入未经审查的代码，CI 调度实行严格的受控准入机制：

### 5.1 组织仓 PR 自动 CI 触发
1. **自动触发**：所有面向 `torchtitan-npu/torchtitan-npu:master` 提交的 PR（含来自外部 Fork 仓库的 PR）会自动触发组织仓的 GitHub Actions 工作流；
2. **安全防护机制**：
   - 外部 Fork 的首次 PR 受 GitHub 官方的 Workflow 审批策略保护（需要 Maintainer 批准运行）；
   - 调度机根据 PR API 严格比对 PR 编号、目标分支以及精确的 Head Commit SHA，坚决拒绝过期或关联异常的 Run；
   - 调度机在拉取代码时，仅下载该特定 PR Head SHA 对应的代码，杜绝中间人篡改；
   - 组织仓 Repository Variable `LITE_CI_RESULT_ACTOR` 配置为 Dispatcher 的结果发布者身份，避免误认为结果评论必须由 PR 作者发布；
   - 执行机**完全不持有 GitHub Token**，仅内网调度机单点具有回写 Comment 权限。

### 5.2 手动触发（Maintainer 冒烟）
维护者亦可通过 GitHub Actions 网页界面或 GitHub CLI 手动发起调度：

```bash
# 执行 A3 8P 全部用例 (Muon + AdamW 独立 Matrix Job 串行执行)
gh workflow run a3_8p_lite_actions.yml -R torchtitan-npu/torchtitan-npu --ref master \
  -f 'test_cases=[{"suite":"a3_8p_tests"}]'

# 仅执行指定 16P 用例
gh workflow run a3_16p_lite_actions.yml -R torchtitan-npu/torchtitan-npu --ref master \
  -f 'test_cases=[{"test_id":"dsv4_flash_a3_16p_example"}]'
```

---

## 6. 测试断言与排障说明

### 6.1 成功判定条件
CI 判定用例 PASS 必须同时满足：
1. 训练主进程 Exit Code 为 0；
2. 实际完成步数达到 `expected_steps`；
3. TensorBoard 正确记录且无 NaN / Inf 异常；
4. 训练结束后 NPU 物理资源成功释放。

### 6.2 常见错误与排查建议
* **`ASCEND_SET_ENV_PATH` 或权重路径找不到**：
  检查 `env_vars` 中的路径是否在所有执行机（如 `/mnt/share/...` 挂载点）均真实存在。
* **显存 OOM**：
  若在反向传播或优化器更新时 OOM，确认是否需要开启 `OPTIMIZER_OVERRIDES`（如 `virtual` 模式）。
* **多机通信超时 (HCCL Timeout)**：
  确认两台机器的通信小网 IP 是否连通，以及 CANN 环境变量是否完整。
* **物理机器或调度故障**：
  若任务一直排队未执行或出现网络异常，属于调度机与物理机状态问题，请参阅 [Lite Actions 调度系统文档](https://github.com/depeng1994/lite-actions) 的日常运维与应急排障小节。
