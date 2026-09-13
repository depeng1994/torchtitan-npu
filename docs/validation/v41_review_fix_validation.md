# DeepSeek V4.1 - GPT Review Fix Validation Checklist

## 分支信息
- 分支：`rfc/deepseek-v4-1-training`
- 基于：`depeng1994/torchtitan-npu` 的 master （commit a23bda7）
- Git PR 798 原始提交：`38595e0` (feat-dsv41-golden-baseline) + `43da3cb` (test: switch debug-model)
- 修复提交：
  - `3dfd4c4` - fix: P0 blockers + P1 items（mHC restore, Q RMS rescale, Indexer rotation, CP1 config）
  - `e6df959` - fix: P1 items（checkpoint adapter, directory rename）
  - `6b2f643` - fix: remaining blockers（Q RMS per-head fix, CP1 runtime guard, dependency direction）
  - `39cb5cd` - fix: sharding getattr guard（V4 无 vision config 不报错）
  - `c38e5a3` - fix: V41 Config update_from_config 显式 parent call（slots dataclass super 兼容）
  - `3798b91` - fix: V41 block FSDP 兼容（layer() 经 __call__ 触发 shard/unshard hooks）

## GPT Review 闭环清单

| # | P级 | 问题 | 修复方法 | 状态 |
|---|------|------|----------|------|
| 1 | P0 | V4 mHC 被改成 Single-Pass | 恢复 DeepSeekV4TransformerBlock.forward() 经典语义；_forward_main() 恢复 hc_head；V41Model 独立实现 Single-Pass loop | ✅ 闭环 |
| 2 | P0 | V4 Attention wq_b 后 RMS rescale 被删 | Attention.Config.post_q_rms_norm=True（V4）/ False（V4.1）；per-head RMS after view | ✅ 闭环 |
| 3 | P1 | Indexer Hadamard 由 golden_enabled() 控制 | Indexer.Config.rotation="hadamard"（V4）/ "none"（V4.1） | ✅ 闭环 |
| 4 | P1 | CP 配置允许 CP2 | CropConfig CP1-only + V41Model.Config.update_from_config runtime guard | ✅ 闭环 |
| 5 | P1 | V4 base 依赖 V41（反向 import） | V41 plan/context 移到 V41Model.__init__，V4 只留 generic seam | ✅ 闭环 |
| 6 | P1 | Checkpoint adapter 未组合 vision | DeepSeekV41StateDictAdapter(DeepSeekV4StateDictAdapter) 组合 vision mapping | ✅ 闭环 |
| 7 | P1 | 目录 deepseek_v4_1 | 改名为 deepseek_v41（models/override/tests/examples） | ✅ 闭环 |
| 8 | P1 | V41AttentionContext 显式化 | 已降级为 follow-up（需在启用 PP/CP2 前完成） | ➡️ Follow-up |
| 9 | P1 | ratio=1 metadata 语义统一 | 已降级为 follow-up（需在 production CSA2 CP2+ 前完成） | ➡️ Follow-up |
| 10 | P2 | 384 experts full-scale flavor | 后续补充 | ➡️ Follow-up |

## 验证清单

| 验证点 | 验证方法 | 输出件 | 结果 |
|--------|----------|--------|------|
| 语法检查 | `python -m py_compile` 全部修改文件 | 无错误 | ✅ 通过 |
| V4 单元测试 | `pytest tests/unit_tests/models/deepseek_v4/`（244 tests） | 244 passed, 1 skipped（环境问题） | ✅ 通过 |
| 目录改名完整性 | grep 确认无 deepseek_v4_1 引用残留 | 0 残留引用 | ✅ 通过 |
| V4 → V41 反向依赖 | grep 确认 V4 侧无 deepseek_v41 import | 无 import | ✅ 通过 |
| CP1 runtime guard | V41Model.Config.update_from_config 拒绝 CP != 1 | 已实现 | ✅ 通过 |
| Checkpoint adapter 注册 | model_registry 使用 DeepSeekV41StateDictAdapter | 已注册 | ✅ 通过 |
| Ruff 静态检查 | `ruff check` 修改文件 | 1 个 RUF005 预存问题 | ✅ 通过 |
| OAT 合规检查 | `pre-commit run oat-check` | Passed | ✅ 通过 |
| dsv41_golden_8p_ep8 | 8卡 NPU 100-step exact loss | 100 步跑通；step 1 精确匹配，step 2+ 偏差 0.003%~0.05%（不累积） | ⚠️ golden 待上游刷新（见下文） |
| dsv41 1P 冒烟 | 1卡 debugmodel 2-step（本地 a3 2NPU 环境） | loss=12.20, rc=0 | ✅ 通过 |
| dsv4_golden_1rank | 1卡 V4 golden test | 需空闲 NPU | ⚠️ NPU 被占用 |
| dsv4_golden_ep2_fsdp2 | 2卡 V4 golden test | 需空闲 NPU | ⚠️ NPU 被占用 |

## 8P Golden 验证详情（2026-09-13，a3-4-docker 16NPU 环境）

运行命令：
```bash
python -m tests.integration_tests.run_tests <empty_out_dir> \
    --test_suite deepseek_v41 --ngpu 8 \
    --test_name dsv41_golden_8p_ep8 --no-parallel
```

结果（`assert_losses_equal` 为 bit-exact、无容差比较）：
- **100 步全部跑通**，loss 轨迹与 golden 趋势一致（12.32 → 4.85），无崩溃、无 grad_norm 异常。
- **step 1 精确匹配**（12.31539249420166 == golden 同值），证明初始参数下 step-1 forward loss 与冻结基线一致。
- **step 2 起出现微小偏差**：每步 abs diff 0.0004~0.003（相对 0.003%~0.05%），且**不随训练累积发散**（steps 1-100 各区间 max rel diff 均 ≤ 0.06%），差异首次在第一次参数更新后可见。
- 偏差来源：golden 在 `43da3cb` 冻结后，本分支后续 7 个修复 commit（Q RMS per-head 实现、FSDP `layer()` 调用路径、Indexer rotation 配置等）虽保持语义一致，但浮点运算路径（算子融合/通信归约顺序）发生变化，导致 bit-exact 轨迹漂移。

### Determinism 验证（2026-09-13，a3-4-docker）

- **同一 HEAD（af79151）连续两次 8P 100-step 运行 bit-exact**：两次结果逐 step loss 完全一致。
  → 排除了 distributed runtime nondeterminism；偏差是 **deterministic implementation-path change**。

### Golden Bisect（2-step，2026-09-13）

| Commit | 结果 | 结论 |
|--------|------|------|
| `c38e5a3`（3798b91 之前） | **8P 直接崩溃**：`RuntimeError: aten.matmul.default got mixed torch.Tensor and DTensor` | 旧执行路径（`layer.forward_with_pre_mix()` 绕过 `nn.Module.__call__`）在 FSDP8+EP8 下**根本无法运行** |
| `3798b91`（FSDP lifecycle 修复） | 2-step 可运行，step1 与 golden 精确一致 | 修复是必要的：`layer(...)` 经 `__call__` 触发 FSDP shard/unshard hooks |

**Bisect 结论**：divergence 起点是 `3798b91` 的 FSDP `__call__` lifecycle 修复——它把旧 baseline 依赖的 **broken execution path** 修正为正确的 module 调用路径。这属于"修复已知错误执行路径"，而非随机浮点漂移。

**处理决定**：Golden refresh 是合理的（regression invariant 应保护 correct implementation，而非永久保护 known-broken FSDP call path）。但按 GPT maintainer 要求，golden 文件的刷新作为单独 baseline-migration commit 提交，并在 commit message 中说明迁移原因；validation 文档明确记录 migration reason（FSDP lifecycle correctness fix）。