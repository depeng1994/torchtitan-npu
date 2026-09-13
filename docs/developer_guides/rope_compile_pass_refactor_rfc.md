# RFC: RoPE compile-pass 重构

## 状态

- 状态：Implemented
- 基线分支：`refactor/rope-compile-pass`
- 基线提交起点：`master@a23bda73d38306081248edb3b4df3b275fce87b8`
- 范围：NPU RoPE override、pre-AOT pattern 注册/调度、CLI 开关、可观测性与验证

本文档描述目标架构和迁移顺序。重构必须以本 RFC 与新增的端到端 UT 为行为契约；除明确列出的行为变化外，不应顺带扩展模型逻辑、训练入口或 shell/config 数量。

## 1. 背景与当前问题

当前 DeepSeek-V4 partial RoPE 优化由两层机制共同完成：

1. `torchtitan_npu.override.common.rope.workaround` 将 upstream `ComplexRoPE` 转换为 real-valued interleaved cos/sin cache，并用普通 Torch 算子表达 RoPE；
2. `torchtitan_npu.compile.patterns.deepseek_v4.inplace_partial_rope` 在 pre-AOT 阶段匹配 `split -> RoPE -> cat`，替换为 `inplace_partial_rotary_mul`。

当前另有 `asc_complex` override，直接把 RoPE 算术替换为 `torch_npu.npu_rotary_mul`。它与 partial pattern 不能共存：`asc_complex` 在模型构建阶段已把普通 Torch RoPE 图折叠成 `npu_rotary_mul`，随后执行的 pre-AOT partial pattern 看不到 `split/rotate/mul/cat` 结构，因此无法命中。

当前 pattern 通过 `TORCHTITAN_NPU_PATTERN_IMPORTS` / `PATTERN_IMPORTS` 显式导入并在 module import 时产生注册副作用。这有以下问题：

- 训练入口需要知道 Python module path，属于编译器内部实现细节；
- 注册发生在 CLI config 解析前，无法自然承接 compile extension 的 enable/blacklist 策略；
- pattern module import 与全局 Inductor pass 安装耦合，测试和后续扩展难以统一管理；
- `workaround` 名称表达的是历史原因，不再准确描述其长期职责；
- 当前日志只在单个 pattern 命中时打印，没有清晰的已注册数量和 graph/cumulative replacement 统计。

## 2. 核心判断

### 2.1 保留中间 canonical representation

本重构不要求 compile pass 直接理解 upstream `ComplexRoPE` 的 complex cache / `view_as_complex` 图。

技术上可以为 complex graph 写 pattern，但这样会把 cache representation 转换、complex decomposition 稳定性和融合选择同时压进 compiler pattern，增加对 PyTorch complex lowering 细节的依赖。

当前 `workaround` 已经完成了一个有价值的 canonicalization：

```text
upstream ComplexRoPE
        ↓
real-valued interleaved cos/sin cache
        ↓
x.float() * cos + rotate_interleaved(x.float()) * sin
```

目标是保留这一层，但将其正式命名为 `decomposed`，明确职责是“生成 compiler-friendly canonical RoPE graph”，而不是临时 workaround。

### 2.2 Override 与 Compile Pass 的职责边界

目标职责如下：

```text
Override / canonicalization:
    ComplexRoPE -> DecomposedComplexRoPE

Compile optimization:
    canonical RoPE graph
        ├─ model-specific larger fusion -> inplace_partial_rotary_mul
        └─ remaining generic RoPE      -> npu_rotary_mul
```

`decomposed` 自身不得调用 NPU fused op。它只负责 cache/layout canonicalization 和普通 Torch 语义实现。

### 2.3 specific pattern 必须先于 generic fallback

Inductor pre-AOT pass 的选择顺序必须稳定：

```text
1. DeepSeek-V4 partial patterns
   split + decomposed RoPE + cat
       -> inplace_partial_rotary_mul

2. Generic interleaved RoPE pattern
   decomposed RoPE arithmetic
       -> npu_rotary_mul

3. 未匹配
       -> 保留 decomposed Torch graph
```

这样关闭某个 DSV4 specific pattern 时，该位置仍可被 generic RoPE pattern 接住，而不是直接退化为未融合 Torch graph；关闭全部 NPU patterns 时则保留纯 decomposed graph，便于 correctness/debug。

## 3. 目标用户体验

正常 Inductor 训练不再要求用户配置 pattern module path，也不应新增 model-specific launcher/config。

目标入口：

```bash
--compile.enable
--compile.backend inductor
```

NPU compile extension 只暴露策略开关：

```text
--compile.extension.no-enable-patterns
--compile.extension.pattern-blacklist <pattern-name>...
```

不再暴露：

```text
TORCHTITAN_NPU_PATTERN_IMPORTS
PATTERN_IMPORTS
--compile.extension.pattern-imports ...
```

不做 A3/A5 SoC 判断。pattern module/依赖存在即可参与注册；导入失败时跳过对应能力。发现逻辑采用简单的 `try: import ... except ImportError: ...`，不增加硬件探测分支。

## 4. 目标代码结构

建议结构：

```text
torchtitan_npu/compile/
    pattern_replacement.py
    pattern_manager.py
    patterns/
        common/
            interleaved_rope.py
        deepseek_v4/
            inplace_partial_rope.py

torchtitan_npu/override/common/rope.py
    DecomposedComplexRoPE
    decomposed
    AscComplexRoPE
    asc_complex
```

### 4.1 `decomposed`

将：

```text
WorkaroundComplexRoPE -> DecomposedComplexRoPE
workaround            -> decomposed
```

实现语义保持不变：

- 继续预展开 complex cache 为共享的 interleaved cos/sin cache；
- 继续支持 meta build 后在 `init_states` 延迟 materialize；
- 继续复用 cache pool；
- `apply_rotary_emb` 继续只使用普通 Torch 算子；
- 不在该层做 `npu_rotary_mul` / `inplace_partial_rotary_mul` 选择。

仓内所有 `workaround` 引用、测试与文档同步改为 `decomposed`。

### 4.2 `asc_complex`

本轮重构先保留 `asc_complex`，用于 eager 或显式旧路径。暂不实现“compile 模式下手动配置 `asc_complex` 时打印专门错误”的逻辑。

Inductor 新路径不应依赖 `asc_complex`。

### 4.3 Pattern module

pattern module 不再在 import 时直接调用 `register_pre_aot_patterns()`。模块只导出 pattern 定义，例如：

```python
PATTERNS = {
    "dsv4_partial_rope_wo_squeeze_inverse": ...,
    "dsv4_partial_rope_wo_squeeze_forward": ...,
    "dsv4_partial_rope_attention_kv_forward": ...,
    "dsv4_partial_rope_compressor_kv_forward": ...,
}
```

新增 generic pattern：

```text
npu_interleaved_rope
```

匹配 decomposed 的普通 interleaved RoPE arithmetic，并替换为 `torch_npu.npu_rotary_mul(..., rotary_mode="interleave")`。replacement 必须以数值等价为首要约束，不得为了复用 `asc_complex` 的历史 dtype cast 而改变 decomposed graph 的精度语义。

### 4.4 Pattern manager

增加统一 pattern manager，职责仅限：

- 按稳定顺序尝试导入内置 pattern modules；
- 应用 `enable_patterns`；
- 应用 `pattern_blacklist`；
- 一次性调用 shared pre-AOT pass 注册入口；
- 打印注册摘要。

导入顺序必须保证 model-specific large fusion 在 generic fallback 之前，例如：

```text
1. deepseek_v4.inplace_partial_rope
2. common.interleaved_rope
```

不根据模型名、A3/A5 型号或 runtime graph 做注册前预测。model-specific pattern 通过图结构决定是否命中。

### 4.5 Compile extension config

建议：

```python
@dataclass(kw_only=True, slots=True)
class CompileExtensionConfig:
    enable_patterns: bool = True
    pattern_blacklist: list[str] = field(default_factory=list)
```

挂在 NPU `CompileConfig.extension` 下，并在 `TrainerEx.Config` converter 中完整保留 upstream compile fields。

黑名单以稳定 pattern name 为单位，不以 Python module path 为用户接口。

## 5. 注册与生效时序

必须区分“注册 pass”和“pattern 真正命中”。

目标时序：

```text
CLI parse
   ↓
得到最终 CompileConfig
   ↓
NPU compile setup
   ├─ 配置 decomposed canonicalization（仅 Inductor NPU compile 路径）
   └─ discover/filter/register patterns
   ↓
upstream apply_overrides()
   ↓
build model
   ↓
parallelize / torch.compile
   ↓
第一次执行触发 Dynamo/AOT/Inductor
   ↓
pre_grad_custom_pass
   ↓
actual pattern matching / replacement
```

任何 Override 决策都不得依赖“pattern 是否实际命中”，因为真实匹配发生在 Override 之后。

`decomposed` 的选择只能依赖已知 compile policy（例如 Inductor compile path），不能依赖未来 graph hit count。

## 6. 可观测性

保留单 pattern 命中日志，并增加汇总统计：

```text
NPU compile patterns registered: 5
NPU pre-AOT graph summary: registered=5, matched_patterns=2, replacements=7
```

建议同时维护 process cumulative counter，方便一个模型分多次 lazy compile 时核对：

```text
NPU pre-AOT graph summary: replacements=7, cumulative_replacements=35
```

不要构造虚假的“whole model compile completed” hook。`torch.compile` 是 lazy 的，统计应以每次 pre-AOT graph invocation 为事实边界。

## 7. Baseline UT 契约

在重构前新增：

```text
tests/unit_tests/compile/patterns/deepseek_v4/test_rope_compile_pipeline.py
```

该测试不是直接调用 `replacement_fn`，而是验证完整局部 pipeline：

```text
WorkaroundComplexRoPE/未来 DecomposedComplexRoPE arithmetic
    ↓
真实 FX GraphModule
    ↓
shared _PreAOTPatternPass
    ↓
DSV4 partial pattern replacement
    ↓
fake inplace_partial_rotary_mul
    ↓
与 upstream ComplexRoPE reference 比较 BF16 输出
```

同时覆盖 forward 与 inverse。

重构完成后，这个测试的行为断言不得改变；只允许把旧类名/override 名更新为 `DecomposedComplexRoPE` / `decomposed`。

现有测试继续承担：

- cache expansion/pool/meta materialization；
- 单个 pattern shape/literal 匹配；
- replacement_fn 局部数值；
- compressor/KV 特殊 layout。

## 8. 新增 UT 矩阵

重构时必须补齐以下测试：

| 维度 | 场景 | 预期 |
|---|---|---|
| discovery | pattern module 可导入 | 自动注册 |
| discovery | optional module `ImportError` | 跳过且训练可继续 |
| config | default | patterns enabled，blacklist empty |
| config | disable-all | 不注册 NPU patterns |
| config | blacklist one | 仅目标 pattern 不注册 |
| ordering | specific + generic | specific 先消费，generic 只处理剩余 RoPE |
| fallback | specific blacklisted | 同位置由 generic `npu_rotary_mul` 接管 |
| fallback | all patterns disabled | 保留 decomposed Torch graph |
| idempotency | setup/import 重复执行 | pass/pattern 不重复安装 |
| stats | zero hit | replacement=0 可观测 |
| stats | multi hit | per-pattern 与 cumulative 数量准确 |
| precision | generic replacement | 与 decomposed/upstream reference 一致 |
| precision | partial replacement | baseline E2E UT 持续通过 |
| CLI | Tyro parse | enable/blacklist 能正确解析并经 converter 保留 |

## 9. ST / 上板验证计划

优先复用 `tests/integration_tests` 已登记的 CI case，不使用 `examples/` launcher 作为验证依据。

现有相关 case：

- `dsv4_golden_1rank`：1 Rank、100 steps、exact golden loss；用于验证 `workaround -> decomposed` rename/canonicalization 不改语义。
- `dsv4_golden_ep2_fsdp2`：EP2 + FSDP2、100 steps、exact golden loss；用于多 rank 精度回归。
- `dsv4_smla_1rank_aot_eager`：1 Rank、AOT eager；用于确认非-Inductor compile 旧路径未被重构意外破坏。
- `dsv4_smla_ep2_fsdp2`：EP2 + FSDP2、AOT eager；用于多 rank NPU override compile smoke。
- `dsv4_smla_cp2_ep2_fsdp2`：CP2 + EP2 + FSDP2、AOT eager；用于覆盖 CP/layout 相关训练路径。
- `dsv4_checkpoint_resume_ep2_fsdp2`：exact resumed loss + grad_norm；作为更广泛训练状态回归，可在最终全量门禁执行。

注意：上述现有 compile ST 使用 `aot_eager`，不会执行 Inductor `pre_grad_custom_pass`，因此不能证明本 RFC 的 pattern 真正命中。

重构必须在现有 `_build_case` 基础上新增最小 Inductor ST，而不是复制新的 runner/shell：

```text
dsv4_smla_1rank_inductor_rope
```

建议 1 Rank、1 step，复用 DeepSeek-V4 integration assets/override recipe，只切换到 `--compile.backend=inductor`，并走新的自动 decomposed + pattern registration。该 case 至少证明真实训练图可以编译和执行；如 runner 后续增加日志 oracle，应进一步断言 `inplace_partial_rope`/generic RoPE replacement 统计非零。

在 1 Rank Inductor 通过后，再基于已有 case 逐级验证 EP2/FSDP2 和 CP2/EP2/FSDP2，不新建独立脚本。

## 10. 文档与清理

重构完成时必须同步：

- 更新 `docs/graph_pattern_fusion.md`，删除环境变量 import 说明；
- 删除 `scripts/run_train_multinodes.sh` 中 `TORCHTITAN_NPU_PATTERN_IMPORTS` / `PATTERN_IMPORTS` 传播；
- 更新所有 `workaround` 文档和配置为 `decomposed`；
- 明确 `asc_complex` 是 eager/legacy optimization，不是 Inductor partial-RoPE pipeline 的前置条件；
- 不在 `torchtitan_npu/patches` 放置本重构代码；本功能是 NPU compiler integration，不属于待上游 TorchTitan patch；
- 不新增 model-specific shell/config 文件。

## 11. 不在本轮做的事项

- 暂不增加 `asc_complex + compile` 的定制报错；
- 暂不删除 `asc_complex`；
- 暂不让 compile pass 直接匹配 upstream complex RoPE graph；
- 暂不扩展 DeepSeek-V3 MLA fusion；V3 应按自身 Q/K layout 另行设计；
- 暂不增加 A3/A5 runtime 分支；
- 暂不修改 `torch_npu` decomposition 以展开/回收 `npu_rotary_mul`。

## 12. 完成标准

满足以下条件方可认为重构完成：

1. baseline E2E UT 持续通过；
2. 用户不再配置 pattern Python module path；
3. `workaround` 完整迁移为 `decomposed`；
4. Inductor 下 specific partial pattern 优先、generic `npu_rotary_mul` fallback 后置；
5. disable-all / blacklist / statistics 有 UT；
6. pattern 注册无 A3/A5/model-name 防御式分支；
7. 现有 DeepSeek-V4 golden ST 精度不回退；
8. 新增最小 Inductor ST 实际跑通，并确认 pattern replacement 生效；
9. `docs/graph_pattern_fusion.md` 与代码一致；
10. 不增加额外训练 launcher/config 分叉。
