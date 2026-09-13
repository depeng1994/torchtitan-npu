# 片段融合算子接入

当融合算子只替换 `forward()` 中的一段连续计算时，使用 pre-AOT pattern 可以避免复制
整个 Module 或修改模型代码。

## 什么时候使用

适合：原始片段结构稳定、可被 `torch.compile` 捕获，融合算子位于 Module 内部。

不适合：需要替换完整 Module、包含数据依赖控制流，或融合算子没有 Fake/Meta 实现。训练
算子还必须支持 Autograd。

## 怎么接入

以 DeepSeek-V4 的 `split -> complex RoPE -> cat` 替换为
`inplace_partial_rotary_mul` 为例，核心结构如下。`original_rope_fragment` 和
`fused_partial_rope` 分别表示原始计算片段和融合算子调用：

```python
from torchtitan_npu.compile import PatternReplacement


def make_pattern(*, inverse):
    def search_fn(x, cos, sin):
        return original_rope_fragment(x, cos, sin, inverse=inverse)

    def replacement_fn(x, cos, sin):
        return fused_partial_rope(x, cos, sin, inverse=inverse)

    return PatternReplacement(
        search_fn=search_fn,
        replacement_fn=replacement_fn,
        ignore_literals=True,
    )


PATTERNS = {
    "dsv4_parent_rope_inverse": make_pattern(inverse=True),
    "dsv4_parent_rope_forward": make_pattern(inverse=False),
}
```

将接入代码放在独立模块中，并导出 `PATTERNS` 字典。`pattern_manager` 会在训练启动时
自动发现、过滤并注册所有内置模块的 pattern。不再需要手工设置 Python module path。

接入时只需注意：

- `search_fn` 必须与模型中的原始片段一致；
- 仅当字面量（如 `split` 尺寸）应作为通配符时设置 `ignore_literals=True`；
- `replacement_fn` 必须保持相同的输出、dtype、layout 和 alias 语义；
- 可复用的 cache 前处理应在 cache 初始化时完成，避免每步重复计算；
- 不希望 FX 展开的算子调用可以使用 `torch.fx.wrap` 包装；
- 算子已注册 Autograd 时直接调用，不要在 override 中重复实现正反向；
- 多个结构变体用一个 factory 生成多个 `PatternReplacement`，然后一次注册。

完整实现参考
`torchtitan_npu/compile/patterns/deepseek_v4/inplace_partial_rope.py`。

## 启用和验证

Inductor 默认自动注册所有 NPU pre-AOT patterns。如需禁用或黑名单，通过 CLI 控制：

```bash
# 禁用所有 pattern（保留 decomposed Torch graph）
--compile.extension.no-enable-patterns

# 黑名单指定 pattern（仅阻止目标 pattern；其余自动注册）
--compile.extension.pattern-blacklist dsv4_partial_rope_wo_squeeze_forward
```

验证时确认 pattern 匹配数量、正反向数值、loss/grad norm，并通过 profiling 检查融合算子
是否生效以及是否新增 clone、copy 或 TensorMove。性能验证时不设置
`TORCHINDUCTOR_NPU_EXT_DEBUG=allfallback`。

> 注：当前 DeepSeek-V4 golden attention 不支持启用 inplace partial RoPE pattern。
> `torchtitan_npu.compile.patterns.deepseek_v4.inplace_partial_rope` 的使能依赖
> `torchtitan_npu.override.common.rope.decomposed`，启用 pattern 时应一并注入该 override。
