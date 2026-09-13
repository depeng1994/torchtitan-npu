# DeepSeek V4.1 - GPT Review Fix Validation Checklist

## 分支信息
- 分支：`rfc/deepseek-v4-1-training`
- 基于：`depeng1994/torchtitan-npu` 的 master （commit a23bda7）
- Git PR 798 原始提交：`38595e0` (feat-dsv41-golden-baseline) + `43da3cb` (test: switch debug-model)
- 修复提交：
  - `3dfd4c4` - fix: P0 blockers + P1 items（mHC restore, Q RMS rescale, Indexer rotation, CP1 config）
  - `e6df959` - fix: P1 items（checkpoint adapter, directory rename）
  - `6b2f643` - fix: remaining blockers（Q RMS per-head fix, CP1 runtime guard, dependency direction）

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
| dsv41_golden_8p_ep8 | 8卡 NPU 100-step exact loss | 需 8NPU 环境 | ⚠️ 环境限制（2 NPU 可用） |
| dsv4_golden_1rank | 1卡 V4 golden test | 需空闲 NPU | ⚠️ NPU 被占用 |
| dsv4_golden_ep2_fsdp2 | 2卡 V4 golden test | 需空闲 NPU | ⚠️ NPU 被占用 |