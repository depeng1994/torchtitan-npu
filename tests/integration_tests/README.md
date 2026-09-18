# 集成测试基础设施

本目录遵循 Torchtitan 的 `tests/integration_tests` 布局，负责维护集成测试定义、测试入口以及可选的 loss 精确比较。基础架构代码由
torchtitan 迁移而来。

当前支持 DeepSeek-V4、DeepSeek-V4.1 与 DeepSeek-V3.2 模型。

## 测试矩阵

| Case 名称 | 模型 | 并行配置 | Rank 数 | 编译配置 | Check Loss | 不检查 Loss 原因 |
|---|---|---|---|---:|---|---|
| `dsv4_golden_1rank` | DeepSeek-V4 | 1 Rank 参考配置 | 1 | - | 是 | - |
| `dsv4_golden_ep2_fsdp2` | DeepSeek-V4 | EP2 + FSDP2 | 2 | - | 是 | - |
| `dsv4_muon_swap_ep2_fsdp2` | DeepSeek-V4 | NPU 融合算子 + DistMuon/AdamW NovaSwap、EP2 + FSDP2、2 steps | 2 | - | 否 | 两步训练 smoke；未生成 swap 数值 golden，也未单独断言 swap action |
| `dsv4_checkpoint_resume_ep2_fsdp2` | DeepSeek-V4 | EP2 + FSDP2，step 2 恢复到 step 4 | 2 | - | 是，含 grad_norm | 与本次连续训练的 step 3、4 精确比较 |
| `dsv4_smla_1rank_aot_eager` | DeepSeek-V4 | 1 Rank | 1 | `aot_eager` | 否 | SMLA 暂不支持 `--debug.deterministic` |
| `dsv4_smla_ep2_fsdp2` | DeepSeek-V4 | EP2 + FSDP2 | 2 | `aot_eager` | 否 | SMLA 暂不支持 `--debug.deterministic` |
| `dsv4_smla_cp2_ep2_fsdp2` | DeepSeek-V4 | CP2 + EP2 + FSDP2（复用同一对 rank） | 2 | `aot_eager` | 否 | SMLA 暂不支持 `--debug.deterministic` |
| `dsv4_mtp_smla_cp2_headtail` | DeepSeek-V4 MTP | CP2 + headtail | 2 | - | 否 | SMLA 暂不支持 `--debug.deterministic` |
| `dsv3_2_dsa_1rank` | DeepSeek-V3.2 | 1 Rank，DSA | 1 | - | 是 | - |
| `dsv3_2_dsa_ep2_fsdp2` | DeepSeek-V3.2 | DSA + EP2/FSDP2 | 2 | - | 是 | - |
| `dsv3_2_dsa_cp2` | DeepSeek-V3.2 | DSA + CP2 | 2 | - | 否 | ST 仅验证训练触发；CPU metadata oracle 单独覆盖，暂未生成 CP2 golden loss |
| `dsv4_ema_ep2_fsdp2` | DeepSeek-V4 | Golden + EP2/FSDP2 + EMA CPU offload | 2 | - | 否 | 校验完整 DCP metadata 包含 `ema_optimizer.*` |
| `dsv41_golden_2p_ep2_fsdp2` | DeepSeek-V4.1 | Golden 调试模型（40 层全结构、调试宽度）+ FSDP2 + EP2，50 步精确 loss | 2 | - | 是 | 多模态 golden 轨迹守护，锚定 `tests/assets/losses/dsv41_golden_2p_ep2_fsdp2.txt`；8 卡形状作手动 A/B 回归，锚不入库 |

V4.1 模型栈完全独立于 `deepseek_v4`（无继承、无 import、无跨模型 override，见 `tests/unit_tests/models/deepseek_v41/test_independence.py`）；golden 参考算子是 `V41SparseAttention`/`V41MoE` 的原生路径，套件仅需 RoPE workaround 与 virtual optimizer 两个通用 override。

`use_golden` 与 `check_loss` 是两个独立维度：`use_golden` 仅决定使用 Golden 参考算子
还是 SMLA/NPU override；`check_loss` 决定是否启用 deterministic、读取参考 loss 并执行
精确数值比较。

当前三个 Golden case（V4 两个、V4.1 一个） 设置 `check_loss=True`，使用固定随机种子和 deterministic 模式，
比较 TensorBoard 标量 `loss_metrics/global_avg_loss`，要求 step 集合和每个浮点值均精确相等。

两个 DeepSeek-V3.2 case 同样设置 `check_loss=True`，使用 RoPE workaround、Ascend DSA
metadata/attention override，并分别对 1-rank 和 EP2/FSDP2 的 100-step loss 做精确比较。

`dsv4_checkpoint_resume_ep2_fsdp2` 合并 checkpoint 保存、恢复和精度对齐验证，已注册到
门禁的 `models` suite。它使用两卡 EP2 + FSDP2 和 Golden 算子，设置 `check_resume=True`，
固定 seed=42 并开启 deterministic。第一阶段连续训练 4 步，保留 step 2 的完整 checkpoint；
第二阶段在新进程中通过 `--checkpoint.load-step=2` 恢复，再训练第 3、4 步。
两阶段均设置 `--training.steps=4`，确保学习率调度一致，共用 checkpoint 目录，
分别写入 `tb_phase_0` 和 `tb_phase_1`。检查 TensorBoard 的
`loss_metrics/global_avg_loss` 和 `grad_norm`：步骤集合必须分别为 `(1, 2, 3, 4)` 和 `(3, 4)`，
续训两步的两个标量必须与连续训练逐值精确相等，不舍入、不使用容差；缺失、重复步骤或
非有限值均失败。本次连续训练是动态基准，不读取或更新仓内 golden loss 文件。

单独执行此用例：

```bash
python -m tests.integration_tests.run_tests /tmp/checkpoint_resume_output \
  --test_suite models --test_name dsv4_checkpoint_resume_ep2_fsdp2 --ngpu 2
```

四个 SMLA case 都设置 `check_loss=False`，因此不会启用 `--debug.deterministic`，也不会
读取 golden loss。它们用于覆盖 SMLA/NPU override 在单卡、EP+FSDP、CP+EP+FSDP 以及
MTP+CP 场景下的实际构图、编译和训练执行路径；单卡、EP2 和 CP2+EP2 场景均使用
`aot_eager`，并默认覆盖 fused MoE token dispatcher。MTP+CP 用例固定使用
`deepseek_v4_debugmodel`、CP2 和 headtail，在 C4 packed sequence 上执行完整的
MTP forward、chunked loss 和 backward。

`dsv4_muon_swap_ep2_fsdp2` 使用 NPU 融合算子：Ascend RMSNorm、complex RoPE、sparse
attention、MHC 和 MoE token dispatcher；两卡 EP2/FSDP2，并追加 `--optimizer.name=Muon`
和 `swap_optimizer` override。它运行两步，覆盖 DistMuon 与 AdamW fallback 在融合训练路径中
的 swap smoke。该 case 不读取 golden loss，不启用 deterministic，也不单独断言 H2D/D2H
action；因此不声称与未 swap 或 AdamW 路径数值等价。

这里的 integration recipe 聚焦 sparse-attention / MHC 回归边界。端到端 example 脚本
额外启用 Virtual Optimizer；checkpoint 保存兼容由 extension `CheckpointManager` 提供。
这些 optimizer state/checkpoint 路径不属于当前 integration loss regression 的覆盖范围。

## 入口

CI 通过以下脚本启动测试：

```bash
.ci/smoke_test.sh
```

或直接运行 Python 入口：


```bash
python -m tests.integration_tests.run_tests \
  ./test_reports/integration \
  --test_suite models \
  --ngpu 4
```
其中`./test_reports/integration` 是必填的测试输出目录，运行前需要确保该目录为空。

直接运行上述 Python 命令仅执行 integration tests。`--test_suite models` 与 CI 的集成测试配置保持一致，覆盖 DeepSeek-V4 和 DeepSeek-V3.2。完整 CI 流程还会在此之前执行 `tests/smoke_tests`。

## 并行调度

runner 迁移自 torchtitan 的 GPUPool 机制：默认将用例并发打包到固定的 NPU 池上，
每个用例通过 `ASCEND_RT_VISIBLE_DEVICES` 绑定到互不相交的物理 NPU 子集，
任一时刻在用 NPU 数量不超过设备池大小。设备池从真实可见性构造：若运行环境已通过
`ASCEND_RT_VISIBLE_DEVICES` 限定可用 NPU 子集（如 CI 按任务分配设备），池从该
子集构造并对超出的 `--ngpu` 硬报错；否则用 `torch.npu.device_count()` 枚举运行时
实际暴露的物理 ID，`--ngpu` 超出实际设备数时告警并截断——绝不按 `range(--ngpu)`
伪造 ID（不存在的 ID 会让子进程在 CANN `GetVisibleDevices` 阶段即失败，torchtitan
设备探测退回 "cuda" 后以 `torch._C._cuda_setDevice` AttributeError 崩溃）。池小于
某用例需求时该用例被显式 skip 而非在 `acquire()` 中死锁。用例按 `ngpu` 从大到小
提交以减少队头阻塞；并行结束后 runner 会输出两行调度遥测：池利用率
（`[parallel] pool: window/utilization/busy histogram/allocations`）与
用例重叠（`[parallel] overlap: sequential vs window、节省时长、并发度直方图`），
统计窗口均为首次分配到最后一次释放，可直接用于核验 CI canary 的打包与重叠效果。
各用例的输出被整体缓存，结束后以带 `[case 名]` 前缀的连续块输出，避免多用例
日志交错。如需强制串行执行，传入 `--no-parallel`。用例可通过
`OverrideDefinitions.timeout` 设置超时；超时后 runner 会向子进程所在进程组先发
`SIGTERM`、宽限期后再 `SIGKILL`，确保 `torchrun` 及各 rank 子进程全部退出，不会
留下占用 NPU 的孤儿进程（超时按失败处理并输出已捕获日志）。

调度器本身不设独立单元测试：其正确性（设备不重叠、失败/超时释放、并发打包、
golden loss 等价）由集成测试自身的 canary 运行直接验证。
