# DeepSeek V4.1 Review / Validation Record

## 当前状态

分支：`rfc/deepseek-v4-1-training`

本文件同时保留已经完成的历史验证，以及最新 architecture/correctness refactor 之后必须重新执行的 validation gate。**历史 Golden 结果不自动外推到最新 HEAD。**

## Review 闭环

| # | 问题 | 当前实现 | 状态 |
|---|---|---|---|
| 1 | V4 mHC 被改成 Single-Pass | V4 保持 classic mHC；V41 block/model 独立实现 Single-Pass | ✅ 闭环 |
| 2 | V4 post-wq_b RMS 被破坏 | V4 `post_q_rms_norm=True`；V41=False | ✅ 闭环 |
| 3 | Indexer Hadamard 被 Golden 开关控制 | rotation 成为通用配置；V4=`hadamard`，V41=`none` | ✅ 闭环 |
| 4 | V41 支持面过度声明 | CP!=1、PP!=1、compile、AscendC 均 fail-fast | ✅ 闭环 |
| 5 | V4 → V41 反向依赖 / 语义泄漏 | V4 只保留 version-neutral primitive / extension seam；CSA2 policy、vision sharding/FSDP、VL bias mapping 均由 `deepseek_v41` 所有 | ✅ 闭环 |
| 6 | V41 Attention policy 混在 V4 forward | `DeepSeekV41Attention` 独立负责 source/reuse/reindex/candidate orchestration | ✅ 闭环 |
| 7 | ratio=1 被当成“无压缩” sentinel | reference metadata 增加显式 `materialized_ratios`；V4 默认不 materialize，V41 将 ratio=1 materialize 为真实 token-for-token global KV | ✅ 闭环 |
| 8 | compressor/indexer ownership 由 ratio 隐式推断 | builder 使用 version-neutral ownership/key-source policy；V41 registry 显式组装 source/reindex ownership | ✅ 闭环 |
| 9 | StateDict V41 namespace 污染 V4 | V4 adapter 只处理 V4；V41 adapter 自己注册 `bias_vl` + vision/marker ownership | ✅ 闭环 |
| 10 | StateDict round-trip UT 无效 | exact HF key-set equality + tensor equality | ✅ 闭环 |
| 11 | Golden override 依赖 `_v41_*` 动态属性 | Golden sparse attention 显式消费 `sparse_indices` + active `compress_ratio`；V4 ratio-4 无外部 Top-K 时保留 local selection | ✅ 闭环 |
| 12 | mutable `V41AttentionContext` | 当前仍为 per-model/per-forward mutable state | ➡️ Follow-up：扩展 PP/CP2+/compile/reentrant 前必须 functionalize |
| 13 | 384-expert released scale | 当前 landing 仍是 16-expert resource crop | ➡️ Follow-up |

## 当前实现边界

支持目标：
- 40-layer V4.1 topology（另有 30-layer validation crop）
- 16 routed experts resource crop / EP8
- Single-Pass mHC
- CSA2 source/reuse/reindex + layer-20 candidate hierarchy
- ratio-2 compressed KV 与 ratio-1 real shared/global KV
- vision tower + marker/scatter path
- CP1 / PP1 / eager
- reference/golden operator path
- FullAC（仅当前 CP1/PP1 eager recipe）

不声明支持：
- AscendC V4.1 sparse attention
- CP > 1
- PP > 1
- `torch.compile` / GraphTrainer
- released 384-expert production scale
- private indexer-training objective / exact VL bias update recipe
- DSpark / Engram / quantized-training parity

## 历史验证证据

### FSDP lifecycle bisect

旧执行路径直接调用 block sub-forward，绕过 `nn.Module.__call__`，8P FSDP+EP 下会报 mixed Tensor/DTensor。`3798b91` 改回正常 `layer(...)` 调用后触发 FSDP hooks，恢复可运行性。该 bisect 证明此前一次 Golden migration 的根因是 **FSDP lifecycle correctness fix**，而不是随机 runtime nondeterminism。

### 已完成的历史验证

- V4 unit suite：此前轮次已通过（最终新 refactor 后仍需重跑）。
- V41 StateDict suite：此前 5 tests passed；round-trip test 后续已修成 strict equality。
- V41 8P FSDP8+EP8：此前版本曾完成 deterministic 100-step trajectory，并据 FSDP lifecycle root cause 迁移过一次 Golden。
- FullAC：此前同 HEAD 重复 100-step 轨迹 deterministic；只证明当前 recipe 的稳定性，不等价于 AC-off 数学等价性。

## 最新 architecture/correctness refactor

最新一轮重构完成以下结构性变化：

1. V4 attention 拆为通用 `_project_q / _project_window_kv / _build_long_range_context / _apply_sparse_attention / _project_output` primitives。
2. V41 source/reuse/reindex/candidate 调度移入 `DeepSeekV41Attention`。
3. V4 builder 不再出现版本判断，改成 compressor/indexer ownership 与 key-source capability 参数。
4. V41 registry 显式声明 KV source、index source、source-key indexer、external-key reindexer。
5. ratio=1 reference metadata 由 V41 显式 materialize，成为真实 causal global-KV container；V4 ratio=1 默认语义不变。
6. V41-only vision AC/FSDP、image-marker sharding、`bias_vl` sharding/StateDict ownership 从 V4 移出。
7. Golden override 改成显式 `sparse_indices + active compress_ratio` contract，不再读取 `_v41_*` 动态属性。

### 为什么必须重新跑 Golden

这一轮并非纯代码搬家：**ratio=1 从“reference mask 中不 materialize”修正为 V4.1 的真实 shared/global-KV 语义**。因此旧冻结 loss trajectory 已被 correctness 修复主动 supersede，不能继续作为最新 HEAD 的 regression oracle，也不能通过手工修改 loss 文件来掩盖变化。

## 最终 merge validation gate（最新 HEAD）

在 NPU 开发环境中执行：

```bash
# 1. Unit / static gates
pytest tests/unit_tests/models/deepseek_v4/
pytest tests/unit_tests/models/deepseek_v41/
pre-commit run --all-files

# 2. V41 final trajectory
python -m tests.integration_tests.run_tests <empty_out_dir> \
    --test_suite deepseek_v41 --ngpu 8 \
    --test_name dsv41_golden_8p_ep8 --no-parallel

# 3. V4 regression
python -m tests.integration_tests.run_tests <empty_out_dir> \
    --test_suite deepseek_v4 --ngpu 1 \
    --test_name dsv4_golden_1rank --no-parallel
```

V41 第一次运行最新 correctness implementation 时，应先记录 deterministic trajectory；确认重复运行 bit-exact 后，再以**独立 baseline-migration commit** 更新 `tests/assets/losses/dsv41_golden_8p_ep8.txt`，随后在迁移后的最终 HEAD 上要求 100/100 exact pass。

## Merge 判定

- **Architecture:** 当前 refactor 已完成既定收口；不再新增架构 gate，除非出现真实新回归。
- **Validation:** 最新 ratio=1 correctness refactor 后的 8P Golden 与最终 V4 regression 尚需 NPU 环境重新执行。
- **Merge:** 在上述最终 validation gate 完成前保持 pending；完成后可进入最终 Approve。
