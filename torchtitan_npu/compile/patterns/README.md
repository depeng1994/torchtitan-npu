# 已有图 Pattern

本目录保存可按需启用的图 pattern。新增 pattern 的接入方式参见
[片段融合算子接入](../../../docs/graph_pattern_fusion.md)。

Pattern 在训练启动时由 `torchtitan_npu.compile.pattern_manager` 自动发现并注册，
不需要手工设置 Python module path。DSV4 partial 融合（specific）先于
generic interleaved RoPE（fallback）注册。

## DeepSeek-V4 Inplace Partial RoPE

该 pattern 将 DeepSeek-V4 中的 interleaved RoPE 小算子片段替换为
`inplace_partial_rotary_mul`。

### 启用

默认自动注册。开启 Inductor 编译即可：

```bash
--compile.enable
--compile.components model
--compile.backend inductor
```

`TrainerEx` 在 `--compile.backend=inductor` 且 `--compile.components` 包含 `"model"`
时自动将 RoPE canonicalization 收敛为 `decomposed` 路径（若 `asc_complex` 存在则替换之），
因此无需手动指定 RoPE override。

快速调试时使用：

```text
TORCHINDUCTOR_NPU_EXT_DEBUG=allfallback
--compile.enable
--compile.components model
--compile.backend inductor
```

性能验证时不设置 `TORCHINDUCTOR_NPU_EXT_DEBUG=allfallback`。

### 约束

- 算子当前只在 A5 上可用；
- `--compile.extension.no-enable-patterns` 可禁用所有 pattern，保留 decomposed Torch graph；
- 当前不支持 DeepSeek-V4 golden attention，整网验证使用 `sparse_attn.asc`；
- 完整验证入口使用 `tests/integration_tests` 中的 `dsv4_smla_1rank_inductor_rope` 用例。