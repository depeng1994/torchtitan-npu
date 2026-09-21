删减者review结论:

# PR #20 / GitCode !878 Maintainer Review

## 1. Review 对象与总览

| 项目 | 结论 |
|---|---|
| Review 对象 | GitHub PR #20 `[feat] compressor & query RMSNorm override support`，镜像自 GitCode PR !878 |
| Base / Head | `master@d529f01baff629324e5c274cf1f6e0ed65c54477` ← `pr_878@6c7dc086af323a04e9ddcb93ebea9634f303c067` |
| 变更规模 | 11 files，+357 / -7 |
| 总体结论 | **暂不建议按当前形态合入。** 需要先处理 R1（不具备上游依据的 TorchTitan patch）、R2（真实 CANN compressor/native autograd 未进入现有 NPU ST）、R3（新增 package-level override 副作用导入）。R4-R8 属于应同步完成的删减、文档与覆盖收口。 |
| 测试专项结论 | **补充测试后合入**。当前 CPU UT 能证明 override 组合、wrapper schema/切片/后处理及 fallback，但不能证明真实 `cann_ops_transformer.compressor` 的 forward/backward/compile；现有 integration case 未启用 compressor override。 |
| 测试执行 | **未执行（仅静态审查）**。按 `.agents/skills/developer-tests-review/SKILL.md` 的 review 流程，本次只读代码、测试入口和固定上游，不执行 testcase。 |
| 重点 KEEP | `CompressorImplementation` 作为独立 override seam 是有真实语义的：TorchTitan override 对 parent/child 同时 claim 会冲突，若直接替换 `Compressor.Config` 会与已有 `norm` / `rope` 子节点 override 形成祖先/后代冲突；拆出 execution node 是当前机制下最小且可组合的方案。 |

## 2. Reducer Findings

| ID | Severity 建议 | Item | Location | Why it exists now | Why unnecessary / problematic | Existing alternative / evidence | Recommended action | Concrete reduction | Compatibility | Verification |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | **P1 / 阻塞** | 非上游来源的 RoPE reshape 改动混入 `patches/torchtitan` | `torchtitan_npu/patches/torchtitan/models/common/rope.py:111,115` | 把 `reshape(..., -1, 2)` 改成显式 `shape[-1] // 2`，看起来是在规避 symbolic `-1` fold / compile guard。 | 本仓规则要求 `patches/torchtitan` 只存“已有上游依据、固定版本尚未包含”的临时 backport。该文件头部引用的 split-aware commit 与 upstream PR #3634 都不是这两行改动的来源；固定 `v0.3.0` 以及本次 review 时的 upstream `main` 仍使用 `-1`。因此这是**本地新增行为伪装成 upstream backport**，且与本 PR compressor/query RMSNorm 的独立产品语义没有建立必要性。 | `pytorch/torchtitan@v0.3.0: torchtitan/models/common/rope.py` 与 2026-09-21 的 upstream `main` 均仍为 `-1`；本仓 AGENTS/测试架构均要求 torchtitan patch 有上游 PR/commit 依据。 | **从本 PR 删除这两行 patch。** 若它是通用 TorchTitan compile 修复，先向 upstream 提交并在 patch 头记录对应 PR/commit；若仅 NPU compiler 需要，放到 NPU override/Extension 的实际 owner，不得继续藏在 torchtitan patch。 | 直接删除本 PR 对该文件的 4 行 diff；不新增 replacement abstraction。 | 删除的是数学等价的本地 workaround，不改变本 PR 定义的 compressor/query RMSNorm 语义。 | 删除后复用现有 DSV4 golden/aot_eager 路径确认没有回归；若确实复现 compile 问题，再以最小 repro 推 upstream。 |
| R2 | **P1 / 阻塞** | “CANN native autograd” 没有真实 NPU 集成证据 | `torchtitan_npu/override/deepseek_v4/compressor/ascendc.py:43-76`; `tests/unit_tests/override/deepseek_v4/test_compressor.py`; `tests/integration_tests/deepseek_v4.py:NPU_OVERRIDES` | PR 新增 `AscCompressor`，PR 描述明确说 backward 直接使用 CANN native autograd。 | CPU UT 把 `torch.ops.cann_ops_transformer.compressor.default` monkeypatch 成普通 PyTorch Python 函数，因此 backward 是 fake 函数自身的 eager autograd，不是 CANN 注册算子的 backward。更关键的是，现有 `NPU_OVERRIDES` **没有** `torchtitan_npu.override.deepseek_v4.compressor.asc`，所以当前 integration suite 没有任何 testcase 真正进入新增 fused compressor。 | 已有 `dsv4_smla_1rank_aot_eager` 正好是 1-NPU、真实 NPU overrides、`torch.compile(..., backend=aot_eager)`、完整训练 1 step 的 carrier；无需新 runner/新 shell。 | 在现有 `dsv4_smla_1rank_aot_eager` 上通过 `extra_override_imports` 增加 `torchtitan_npu.override.deepseek_v4.compressor.asc`，让真实模型同时经过 compressor + RMSNorm + RoPE overrides，并完成 forward/backward/optimizer。**不要**把 compressor 全局塞进 `NPU_OVERRIDES`，因为当前 `AscCompressor.forward` 明确拒绝 CP，而同一 tuple 还被 CP2 case 复用。对于真实 kernel 数值/梯度，优先引用 ops-transformer 对当前算子版本已有的 forward/backward 测试；若上游没有，再补最小 NPU parity test。 | 复用已有 ST case，只增加一个 override token；不新增脚本、不新增 suite。 | CP2 仍保持原 recipe，不会因新增 compressor override 被错误击穿。 | integration case 必须实际完成指定 step；日志/配置中可定位 compressor override 已启用。native autograd 的数值正确性需要来自真实 CANN 测试或独立 NPU oracle，不能由 CPU fake 宣称。 |
| R3 | **P1 / 架构** | 新增 package-level override 副作用导入 | `torchtitan_npu/override/deepseek_v4/__init__.py:3-7` | 为了“import package 即注册 compressor factory”，把 `compressor` 加进 `from . import compressor, sparse_attn`。 | `.agents/AGENTS.md` 明确要求 override 通过完整 `module.function` 的 `override.imports` 显式启用，**不要在 `__init__.py` 中批量导入具体 override**。而 Python 在导入 `torchtitan_npu.override.deepseek_v4.compressor.asc` 时本就会加载该 subpackage，不需要 parent package 额外注册。新增副作用扩大 registry/import 面，也让“没有选择 compressor”时仍注册其 factory。 | 已有 TorchTitan override loader 会按用户给出的 `module.function` import 对应 module；本 PR 自己的 `compressor/__init__.py` 已定义 factory。 | 删除本 PR 新增的 `compressor` package-root import，并把 docstring 恢复为不宣称 root import 注册 compressor。已有 `sparse_attn` 是历史代码，本 PR 不需要借此继续扩大 side effect。 | 删除 1 个 import target + 相应 docstring 变化。 | 正常 `--override.imports torchtitan_npu.override.deepseek_v4.compressor.asc` 不受影响。 | CPU override composition UT 继续从真实 `OverrideConfig(imports=[...])` 进入并确认 replacement。 |
| R4 | **P2 / 应精简** | `UnitScaleRMSNorm` 保存了 reference path 永远不用的 ones buffer | `torchtitan_npu/models/deepseek_v4/attention.py:28-47`; `sharding.py:157-164` | 为“fixed unit gamma / 无 checkpoint 参数”显式删 parameter 后注册 non-persistent `weight` buffer，并在 init 时 fill 1。 | `UnitScaleRMSNorm.forward` 完全不读取 `self.weight`；buffer 不进入 checkpoint，但会进入 module buffer 生命周期和 sharding/distribution。它唯一可能的 consumer 是 fused `AscRMSNorm`，但 override 后实例类型已经变成 `AscRMSNorm`，且 R7 所在实现已自行在 `elementwise_affine=False` 时创建 ones buffer。因此 baseline UnitScale 的 buffer 是第二份无用户语义 state。 | Upstream `RMSNorm.Config(elementwise_affine=False)` 已能表达“无 learnable gamma”；当前自定义 class 真正不可替代的部分只有：保持旧 inline 公式的 input-dtype rounding，以及作为 RMSNorm.Config subtype 被 override 命中。 | **保留 `UnitScaleRMSNorm.Config` + custom `forward`，删除其 `__init__` 和 `_init_self_buffers`。** 让 reference path 没有 weight state；只有 `AscRMSNorm` fused replacement 在 kernel 确实需要 tensor weight 时物化 non-persistent ones buffer。 | 删除一套 buffer 注册/初始化状态；不删除 class 本身。 | checkpoint FQN 不变（当前 buffer 本就 non-persistent）；数学输出不变。现有 `state_shardings["weight"]` 可保留给 override 后的 `AscRMSNorm`，Module 只会处理实际存在的 state。 | golden path应保持旧 loss；可用小 tensor 独立公式验证 `UnitScaleRMSNorm(x)`，无需测试私有 buffer。 |
| R5 | **P2 / 应精简** | 每层每 step 计算并传入冗余 `seqused=lengths` | `torchtitan_npu/override/deepseek_v4/compressor/ascendc.py:18-35, 56-71` | `_state_inputs` 从 `cu_seqlens` 做一次差分得到 `lengths`，用于 table batch 维和 `seqused`。 | 当前语义中每个 document 的全部 token 都参与压缩，`seqused` 与 `cu_seqlens[n+1]-cu_seqlens[n]` 完全相同。CANN compressor 官方文档明确规定 `seqused=None` 就表示使用各 Batch 的 Sequence Length，因此这一张量是可从已有状态推导的重复 state/计算。 | CANN/ops-transformer compressor 文档：`seqused` 可选，None 时等于每个 Batch 的 Sequence Length；当前 `cu_seqlens` 已完整提供边界。 | 调用算子时传 `seqused=None`；table 第一维用 `cu_seqlens.numel()-1` 推导，删除 `lengths = ...` 和 `_state_inputs` 的 `lengths` 返回值。若 helper 收缩后只剩一次调用且很短，可进一步 inline 到 `forward`。 | 删除一张逐层构造的 int32 tensor及相关测试断言；可顺带减少单用 helper。 | 与当前“全 document 有效”的行为等价；未来如果出现 partial `seqused` 产品语义，再显式传入。 | CPU wrapper test 应检查传入 `seqused is None`；真实 NPU ST 覆盖算子接受该默认语义。 |
| R6 | **P2 / 应精简** | 每个 `AscCompressor` 实例都调用 `importlib.import_module` | `torchtitan_npu/override/deepseek_v4/compressor/ascendc.py:10, 39-46` | 目的是直到 fused component 被 build 才注册可选 CANN op。 | `ascendc.py` 本身已经只在 `compressor.asc` factory 被选中后 lazy import；因此在 class `__init__` 里再做 per-instance import 没有额外的 optional-dependency 隔离价值。Python import cache 虽避免重新执行 module，但仍产生每层一次 import lookup/lock 路径，也额外需要一个无语义 `__init__`。 | 同仓 `override/deepseek_v4/sparse_attn/ascendc.py` 在其 ascendc module scope 导入 CANN API；外层 factory 保持 lazy。 | 在 `ascendc.py` module scope 注册/导入 `cann_ops_transformer.ops.compressor`，删除 `importlib` 和 `AscCompressor.__init__`。 | 删除一层 lifecycle hook 与每实例 import。 | 未选择 override 时仍不会 import `ascendc.py`；选择后行为一致。 | CPU test 仍可在导入 ascendc 前注入 stub module；真实 ST 证明 op registration。 |
| R7 | **P2 / 文档与入口** | PR 声称“已更新文档”，但仓库没有任何 docs/example 变更，现有 DSV4 recipe 也未暴露 compressor override | PR body Checklist；`examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a3.sh:NPU_OPS_OVERRIDES` | 功能目前只能从 PR 描述知道完整 override path。 | 对一个显式 opt-in override，长期用户语义至少要说明“如何启用”和“不支持 CP”。当前主训练脚本的 `NPU_OPS_OVERRIDES` 没有 compressor，PR 也没有文档文件 diff；Checklist 与实际 diff 不一致。另起新 config/shell 会进一步破坏单一入口。 | 现有脚本已经允许通过 `--override.imports`/CLI 组合 override，并允许 `"$@"` 继续覆盖。 | 不新增脚本。若 A3/CP1 已是默认支持路径，把 compressor target 合入**现有兼容 recipe**；否则就在现有 DeepSeek-V4 文档/示例旁明确给出 `--override.imports torchtitan_npu.override.deepseek_v4.compressor.asc` 以及“当前 CP 不支持”的限制。同步修正 Checklist。 | 只改现有 recipe/doc，不引入新的 config、env 或 launcher。 | 不影响现有用户；避免用户误把 CP2 与 fused compressor 组合。 | 静态检查现有入口能表达 target；NPU ST 使用同一 target。 |
| R8 | **P2 / 测试删减** | CPU “CANN outputs/gradients” testcase 内置了第二份 compressor 数学实现 | `tests/unit_tests/override/deepseek_v4/test_compressor.py:48-66, 102-161` | `_document_pool` 同时被 fake CANN 和 expected 使用，制造可微输出以检查 wrapper 的后处理/梯度。 | 这不是 CANN 数学或 native autograd 的独立 oracle：fake op 与 expected 共享同一 `_document_pool`。长期维护第二份 compressor pooling 公式只会增加实现耦合，且容易让测试名称/报告被误读成“已验证 CANN”。CPU 测试真正有价值的是 schema、valid-prefix slicing、norm/RoPE 后处理、fallback/CP reject 和 override composition。 | 测试规范允许 fake 隔离 NPU，但明确禁止把 fake 当成被声明验证的 target kernel；真实 kernel 由 R2 的 NPU ST/上游算子测试负责。 | 保留 wrapper-contract testcase，但把 fake CANN 缩成一个**最小、可微、可独立计算的 sentinel 输出**，expected 只验证 wrapper-owned 行为；删除/显著缩短 `_document_pool` 这份 kernel 语义复刻。测试名称明确为 wrapper contract，不宣称 CANN native autograd correctness。 | 预计可删除当前测试文件中约 30-50 行 duplicate math/reference scaffolding，同时证据边界更清楚。 | 不降低 wrapper 覆盖；真实 kernel 覆盖由 R2 补齐。 | CPU UT 只检查 wrapper；真实 NPU evidence 单独记录。 |

## 3. 逐文件 Review Coverage

| 变更文件 | 主要改动 | Reducer 结论 | Review 意见 / 修改建议 |
|---|---|---|---|
| `tests/unit_tests/models/deepseek_v4/test_cp_dispatch.py` | 现有 fixture 增加 `q_head_norm` config | **KEEP** | 属于生产 Config 新必填字段后的机械同步；本身不构成 q-head norm / TP 语义覆盖。不要再围绕字段存在新增表示型 UT。 |
| `tests/unit_tests/override/deepseek_v4/test_compressor.py` | override composition、fake CANN schema/output/grad、empty/CP | **SIMPLIFY** | composition/empty/CP 边界有独立价值；fake kernel 不能证明 native autograd，见 R2/R8。当前版本已经用 `torch.random.fork_rng()` 隔离 RNG，且 parameterize ids 已补齐，这两点无需再改。 |
| `torchtitan_npu/models/deepseek_v4/__init__.py` | attention config 增加 `UnitScaleRMSNorm.Config` | **KEEP** | 把原 inline query norm 提升为可 override component 是本 PR 的真实产品语义；保持单一 config source，不需要额外 CLI field。 |
| `torchtitan_npu/models/deepseek_v4/attention.py` | 新增 `UnitScaleRMSNorm`；forward 改为子模块 | **SIMPLIFY** | class/Config/forward 保留；reference-path ones buffer 与 init 删除，见 R4。 |
| `torchtitan_npu/models/deepseek_v4/compressor.py` | 新增 `CompressorImplementation` + Config/field；原 forward 下沉为 `_forward` | **KEEP** | 这是少数通过 existence test 的新增 abstraction：解决 parent Compressor override 与 child norm/RoPE override 的结构性冲突，且默认 implementation 精确保持原路径。不要再扩展 strategy/registry 层。 |
| `torchtitan_npu/models/deepseek_v4/sharding.py` | q-head norm 声明 TP head-sharded activation + local_map | **KEEP（证据待补）** | placement 与 `q:[B,S,H,D]` 按 head 维 `S(2)` 的语义一致，last-dim RMSNorm 不需要 gather。当前 ST 只有 TP=1；若项目宣称 DSV4 TP>1 已支持，应复用现有 integration 框架补真实 TP case，而不是单独给这个字段写 mock UT。 |
| `torchtitan_npu/override/common/rms_norm.py` | `AscRMSNorm` 支持 `elementwise_affine=False`，为 NPU op 物化 unit weight buffer | **KEEP** | 这是 fused kernel 需要 tensor weight、而模型语义要求无 learnable gamma 的必要适配；buffer 应只存在于这个 override，不应在 baseline UnitScale 再复制一份。 |
| `torchtitan_npu/override/deepseek_v4/__init__.py` | root package 自动 import compressor | **DELETE 本 PR 新增部分** | 违反显式 override 规则，见 R3。 |
| `torchtitan_npu/override/deepseek_v4/compressor/__init__.py` | `@override(target=CompressorImplementation.Config, exact=True)` factory | **KEEP** | target 足够窄，`exact=True` 避免误匹配；factory lazy import ascendc，符合解耦方向。 |
| `torchtitan_npu/override/deepseek_v4/compressor/ascendc.py` | CANN wrapper、state input、CP reject、empty fallback | **SIMPLIFY + 补 ST** | wrapper owner 合理；删除 per-instance import，`seqused=None`，不引入新的 cache/state；真实 NPU 路径见 R2。 |
| `torchtitan_npu/patches/torchtitan/models/common/rope.py` | explicit reshape dim | **DELETE / UPSTREAM FIRST** | 不属于已有上游 backport，见 R1。 |

## 4. 正向功能与 UT 覆盖

| 正向功能单元 | 生产路径 / 应观察结果 | PR / 现有测试的直接检查 | expected 来源 | 状态 | 合入前置条件 |
|---|---|---|---|---|---|
| Compressor implementation override 与 norm/RoPE child override 可组合 | `Compressor.Config.implementation` 被 `compressor.asc` 替换，同时 `norm`、`rope` 保持各自 override，且 import order 不影响结果 | `test_compressor_override_composes_with_norm_and_rope` 通过真实 `OverrideConfig/apply_overrides`，两种顺序检查三个 config replacement | override tree 的类型/位置契约 | **已覆盖** | 保留该薄测试；R3 删除 root side-effect import 后仍应通过。 |
| baseline query normalization 从 inline formula 变成 component，但数值语义不变 | `Attention.forward: q -> q_head_norm -> rope` | 现有 golden training 路径会继续使用同一公式；PR 没有新增专门数值 UT | 原 inline 数学公式；`UnitScaleRMSNorm.forward` 直接保持该公式 | **部分覆盖** | 不需要为 private state 增测试；R4 简化后用现有 golden loss/必要时一个最小公式 UT 保证 dtype rounding 不变。 |
| `RMSNorm.Config(elementwise_affine=False)` 被 `rms_norm.asc` 替换后仍可调用 NPU kernel | Query q-head norm 在 non-golden NPU recipe 中会被 common RMS override 命中 | 现有 `dsv4_smla_1rank_aot_eager` 的 `NPU_OVERRIDES` 已包含 `torchtitan_npu.override.common.rms_norm.asc`，真实训练会经过该 branch | 训练完成性；reference 数值来自 golden path | **部分覆盖** | 现有 ST 可复用；不把“完成 1 step”写成 NPU RMSNorm 精度证明。 |
| CANN compressor wrapper 正确传 schema、截掉 padded tail、再执行 norm/RoPE | `AscCompressor.forward` | CPU fake 检查 weight/ape/state/table/cu、输出 prefix、postprocess 和梯度 plumbing | fake 输出 + reference postprocess | **已覆盖（仅 wrapper boundary）** | R8 缩小 fake；保持边界清晰。 |
| **真实** CANN compressor forward/backward/native autograd/compile | `torch.ops.cann_ops_transformer.compressor.default` on NPU, real model | CPU fake 没有调用真实 op；现有 integration case 没启用 compressor target | 当前只有 PR 描述中的人工 loss/grad_norm 图 | **未覆盖** | **R2：复用 `dsv4_smla_1rank_aot_eager` 启用 compressor override；真实 kernel 数值/梯度优先引用 ops-transformer 已有测试，否则补最小 NPU parity。** |
| empty compression plan fallback | empty `gather_indices` 回落 `compressor._forward` 并保持空输出/零梯度 | `test_compressor_empty_plan_and_cp[empty]` | 原 baseline `_forward` | **已覆盖** | KEEP。 |
| fused compressor 明确拒绝 CP | metadata 有 `window` 或 plan exchange 时抛错 | `test_compressor_empty_plan_and_cp[cp]` | public limitation | **已覆盖（CPU contract）** | 同步写入现有 docs/example，避免用户从单一入口组合出不支持路径。 |
| q-head norm TP sharding | head 维保持 `S(2)`，RMSNorm last dim local compute | 仅看到 sharding config；当前 DSV4 ST 未提供 TP>1 直接证据 | placement contract | **部分覆盖** | 若 DSV4 support matrix 包含 TP>1，复用现有 integration runner 补最小 TP case；若当前固定 TP=1，则不要把该改动描述成“TP2 已验证”。 |

## 5. NPU ST 触发与现有 case

| 生产代码改动 | 目标训练路径 | 现有 integration testcase | 当前状态 | 需要的调整 |
|---|---|---|---|---|
| `AscCompressor` + native autograd | DSV4、CP=1、NPU compressor + existing norm/RoPE overrides、forward/backward、compile | `dsv4_smla_1rank_aot_eager` 当前**未**包含 compressor target | **未覆盖** | 在该现有 case 的 `extra_override_imports` 增加 `torchtitan_npu.override.deepseek_v4.compressor.asc`；不新建 shell/runner。 |
| query `UnitScaleRMSNorm` + `AscRMSNorm(elementwise_affine=False)` | DSV4 NPU SMLA 路径 | `dsv4_smla_1rank_aot_eager` 已包含 `rms_norm.asc` | **已覆盖训练完成路径** | 复用；与 golden reference 共同界定数值证据。 |
| query norm reference 重构 | DSV4 golden path，保持原公式/loss | `dsv4_golden_1rank`（30 steps, `check_loss=True`） | **已覆盖入口** | 复用既有 golden；本次 review 未执行。 |
| q-head TP sharding | TP>1 real DeviceMesh/DTensor | 当前列出的 DSV4 integration cases 未设置 tensor-parallel degree >1 | **部分覆盖 / 支持范围待事实约束** | 只有在项目当前宣称 TP>1 可用时才复用/调整一个现有 case；不要为 placement 字段本身复制组合。 |
| compressor + CP | 当前实现主动报错 | `dsv4_smla_cp2_ep2_fsdp2` / `dsv4_mtp_smla_cp2_headtail` | **不应启用 compressor** | 因此不能把 compressor target 直接加入共享 `NPU_OVERRIDES`。 |

### 现有相关 ST 事实表

| 测试 | 路径 | 并行 | 目标 overrides | compile | NPU 数 | 完成/精度检查 | 与本 PR 的关系 |
|---|---|---|---|---|---:|---|---|
| `dsv4_golden_1rank` | reference/golden | TP1/CP1 | workaround RoPE + sparse golden | default | 1 | 30 steps + loss golden | 能保护 query norm reference 重构，但不进入 fused RMS/compressor |
| `dsv4_smla_1rank_aot_eager` | NPU SMLA | TP1/CP1 | rms/rope/sparse/mhc/token-dispatcher；**无 compressor** | aot_eager | 1 | 1 step，`check_loss=False` | 最适合承载新增 compressor override |
| `dsv4_smla_ep2_fsdp2` | NPU SMLA | EP2/FSDP2, CP1 | 同上；**无 compressor** | aot_eager | 2 | 1 step | 可作为后续并行补充，不需要先于 1-rank case |
| `dsv4_smla_cp2_ep2_fsdp2` | NPU SMLA | CP2/EP2/FSDP2 | 同上 | aot_eager | 4 | 1 step | `AscCompressor` 当前显式不支持 CP，不能继承 compressor target |

## 6. Test Reduction Matrix

| Test / helper | Unique evidence | Replacement evidence | Action |
|---|---|---|---|
| `test_compressor_override_composes_with_norm_and_rope` | 证明真实 override loader 下 parent execution seam 与 child norm/RoPE replacement 可组合且顺序无关 | 无等价现有证据 | **KEEP** |
| `_document_pool` | 当前只为 fake CANN 构造输出，同时又用于 expected；不证明真实 kernel | R2 的真实 NPU / ops-transformer kernel tests；wrapper UT 可用更小 sentinel | **DELETE / 大幅简化** |
| `test_compressor_cann_schema_document_outputs_and_gradients` | schema、valid-prefix slicing、postprocess、gradient plumbing | 真实 kernel correctness 由 R2 负责 | **KEEP but narrow scope** |
| `test_compressor_empty_plan_and_cp[empty]` | empty plan fallback 与空梯度 | 无重复证据 | **KEEP** |
| `test_compressor_empty_plan_and_cp[cp]` | 当前 fused implementation 的明确“不支持 CP”边界 | docs + runtime guard，但 testcase 仍能防止 silent fallback | **KEEP** |
| `test_cp_dispatch.py` 中新增 `q_head_norm` fixture field | 仅让已有 CP 测试适配新必填 config | 不是独立产品语义 | **KEEP as fixture maintenance；不要据此声称 q-head TP 已覆盖** |

## 7. Reduction Inventory

| Item | Current form | Unique semantic? | Existing alternative | Action |
|---|---|---:|---|---|
| `CompressorImplementation` class | Configurable execution node | **是**：避开 parent/child override conflict | upstream override 的 exact/FQN 仍无法消除 ancestor conflict | **KEEP** |
| `CompressorImplementation.Config` | 空 Config | **是**：override target identity | 无更小的可 override Config 节点 | **KEEP** |
| `Compressor.Config.implementation` field | default base implementation | **是**：保留 baseline + 显式切换 | 无 | **KEEP** |
| `Compressor._forward` | 原 forward body 下沉 | **是**：baseline fallback / implementation delegate | 无需再抽 helper | **KEEP** |
| `UnitScaleRMSNorm` class + Config + forward | fixed-unit, non-affine norm component | **是**：保持旧 dtype rounding + 可被 RMS override 命中 | 直接 upstream RMSNorm 可能改变低精度 rounding | **KEEP** |
| `UnitScaleRMSNorm.__init__` + ones buffer + `_init_self_buffers` | nonpersistent state | **否** | baseline forward 不读；AscRMSNorm 自己会物化 weight | **DELETE** |
| `Attention.Config.q_head_norm` / build / call | explicit component | **是** | inline 公式不可 override | **KEEP** |
| q-head sharding config | TP head-sharded local normalization | **是**（placement contract） | Module/ShardingConfig 已是统一机制 | **KEEP；不新增自定义 parallel wrapper** |
| `AscRMSNorm` no-affine buffer support | fused kernel adaptation | **是** | NPU op 需要 weight tensor | **KEEP** |
| `override.deepseek_v4.__init__` 新增 compressor import | package registration side effect | **否** | `override.imports module.function` | **DELETE** |
| compressor override factory | exact execution replacement | **是** | 无 | **KEEP** |
| `AscCompressor` class/Config | NPU execution implementation | **是** | CANN op wrapper owner | **KEEP** |
| `AscCompressor.__init__` import hook | per-instance op registration | **否** | ascendc module-scope lazy import | **DELETE / MOVE import** |
| `_state_inputs.lengths` / `seqused=lengths` | derived tensor | **否（当前全序列语义）** | `seqused=None` | **DELETE** |
| `state_block_table` | zero table required by current CANN tiling contract per author comment | **目前接受** | schema 虽 optional，但没有足够证据证明当前 torch binding 可安全省略 | **KEEP，避免为了省 alloc 新增 cache state** |
| `state_cache` | mutable CANN input | **是** | 算子会写入，不能跨 forward 无脑复用 | **KEEP** |
| RoPE explicit reshape patch | local compile-style tweak | **否，至少本 PR 未证明** | upstream 原实现 / upstream-first | **DELETE / UPSTREAM** |
| CPU fake-CANN duplicate pooling helper | second implementation | **否** | minimal differentiable sentinel + real NPU evidence | **DELETE / SIMPLIFY** |
| 新 shell/config/env | 本 PR 未新增 | N/A | 继续用现有 CLI | **KEEP 为“不新增”** |

## 8. Historical Review Fix Reduction Matrix

| Historical review | Original semantic | Current fix | Current-head status | Overbuild? | Minimal alternative | Related finding |
|---|---|---|---|---|---|---|
| state inputs 每 forward 新建；可选参数是否应传 None | 避免不必要 alloc/memset，并解释必需 state | 已删除单独 `start_pos`；为 zero `state_block_table` 增加“CANN tiling requires”注释；仍计算 `lengths` 并传 `seqused` | **PARTIALLY_RESOLVED** | `seqused` 仍是可推导重复 tensor | `seqused=None`，batch 从 `cu_seqlens.numel()-1` 推导；保留 state_cache，table 暂按 tiling 需要保留 | R5 |
| “这个需要放在这里吗？每次 import” | op registration 不应发生在每 layer instance build | 仍在 `AscCompressor.__init__` 调用 `importlib.import_module` | **STILL_OPEN** | 是 | ascendc module scope import；outer factory 已提供 lazy boundary | R6 |
| `UnitScaleRMSNorm` 无 parameter，`param_init` 永不生效 | 不保留误导性的 parameter init | 当前 `q_head_norm=UnitScaleRMSNorm.Config(...)` 已删除 `param_init=_NORM_INIT` | **RESOLVED** | 修复本身没有新增 Config；但 UnitScale 又保留了无用 buffer state | 删除 baseline UnitScale buffer，仅 fused AscRMSNorm 物化 unit weight | R4 |
| deepseek_v4 override package docstring 与实际 import 不一致 | 文档与当前 import 同步 | docstring 已改为 compressor+sparse_attn | **RESOLVED（原问题）** | 新文案准确描述了当前代码，但当前新增 compressor side-effect import 本身不应存在 | 删除新增 root compressor import，并同步缩回 docstring | R3 |

## 9. Top Reduction Plan（强制 30% complexity question）

如果要求在**不损失本 PR 必要产品语义**的前提下显著降低新增复杂度，我会按以下顺序处理。重点不是机械 LOC，而是删除第二份实现、第二份 state 与非必要生命周期：

| Priority | Reduction | 预计收益 | 为什么不损失语义 |
|---|---|---|---|
| **P0 可直接删除** | 删除 R1 的 RoPE patch hunk | 去掉与本 PR 目标无关、且无 upstream provenance 的 patch debt | compressor/query RMSNorm 都不依赖这份本地 upstream 修改作为产品语义 |
| **P0 可直接删除** | 删除 `override/deepseek_v4/__init__.py` 新增 compressor side-effect import | 收紧 registry/import 面 | `override.imports ...compressor.asc` 已能直接导入 factory |
| **P0 可直接删除** | 删除 UnitScale baseline ones buffer + init | 去掉每层额外 buffer/state/sharding 生命周期 | custom forward 不读 weight；fused replacement 自己物化 weight |
| **P1 可复用/合并** | `seqused=None`，删除 `lengths` tensor；必要时 inline `_state_inputs` | 每层每 step 少一份 derived tensor和一层单用 helper | CANN 官方默认语义与当前全 sequence 压缩一致 |
| **P1 可复用/合并** | 把 CANN op import 移到 ascendc module scope，删除 `AscCompressor.__init__` | 去掉 per-instance import/lifecycle hook | outer override factory 已经是 lazy import boundary |
| **P1 可直接删减测试** | 用最小 differentiable sentinel 替代 `_document_pool` 第二份 compressor 数学实现 | 测试文件可明显收缩，避免 same-implementation oracle | wrapper UT 只验证 wrapper-owned contract；真实 kernel 由 ST/upstream test 负责 |
| **KEEP** | `CompressorImplementation` seam | 这是新增复杂度中最有必要的一项 | 直接 parent override 会与 child norm/RoPE override 冲突 |
| **KEEP** | `UnitScaleRMSNorm.Config + forward` | 保留低精度 rounding 与 override target | 直接换成 stock RMSNorm 可能改变旧计算的 dtype rounding |

以上几项合计能删除/折叠本 PR 很大一部分“第二份实现 + 第二份 state + side effect + 单用 helper”复杂度；尤其 test fake 不再复刻 compressor 算法后，新增测试代码可以从 182 行显著回落，而真实证据反而更强。

## 10. 关键 KEEP abstraction

| Abstraction | KEEP 理由 | 不继续泛化的边界 |
|---|---|---|
| `CompressorImplementation` | 当前 TorchTitan override conflict rule 下，execution-level Config 节点是让 compressor fused execution 与 norm/RoPE child overrides 可组合的最小稳定 seam。 | 不新增 implementation registry、strategy hierarchy、额外 CLI field；用户仍只通过 `override.imports` 选择实现。 |
| `UnitScaleRMSNorm`（仅 Config + forward） | 旧 query norm 是显式 input-dtype 公式；把它变成 RMSNorm subtype 后才能复用 common RMS override，同时保持 reference 数值语义。 | 不保留 baseline weight/buffer，不新增独立 `unit_scale` 用户配置。 |
| `AscRMSNorm` no-affine materialization | NPU fused RMS op 仍需要 weight tensor，而产品语义又要求无 learnable gamma。 | unit weight 只属于 fused implementation，不回灌 baseline module state。 |
| `AscCompressor` | CANN operator 的 device-specific execution 应放在 override，而不是 model 或 torchtitan patch。 | CP 未支持前明确 fail；不要在 model core 加 `if NPU` 分支。 |

## 11. 文档、上游与源码一致性核对

| 检查项 | 结果 |
|---|---|
| PR Checklist “已经更新相应文档” | **不一致**：11 个 changed files 中没有 docs/example 变更；现有 DSV4 主 recipe 也没有 compressor target。见 R7。 |
| `patches/torchtitan` upstream provenance | **不满足**：本 PR 新增的 explicit reshape hunk 不在文件头引用的 backport 来源内；固定 v0.3.0 与 upstream main 仍为 `-1`。见 R1。 |
| Override 激活方式 | compressor 自己的 factory/target 设计正确，但 root package 新增副作用 import 与仓内规则冲突。见 R3。 |
| 训练入口复杂度 | 本 PR 没有新增 shell/config，这是正确方向；后续 ST/docs 也应继续复用现有 CLI/runner，禁止为 compressor 再起专用脚本。 |
| CANN `seqused` 契约 | 官方 ops-transformer compressor 文档明确：`seqused=None` 表示每个 Batch 使用完整 Sequence Length；当前实现正是这个语义，因此可删 derived `lengths`。 |
| CANN padded output | 官方文档给 TH 输出 `[min(T, T//cmp_ratio+B), D]` 并包含 per-batch compressed tokens + pad；PR 按 plan valid block count 做 prefix slice 有明确必要性，**KEEP**。 |

参考基线：
- TorchTitan pinned: `v0.3.0`
- Upstream RoPE: `https://github.com/pytorch/torchtitan/blob/v0.3.0/torchtitan/models/common/rope.py`
- Upstream current RoPE (review date 2026-09-21): `https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/common/rope.py`
- CANN compressor contract: `https://gitcode.com/cann/ops-transformer/tree/master/experimental/attention/compressor`

## 12. 复审条件

| 条件 | 必须完成的修改 / 证据 |
|---|---|
| R1 | 删除本 PR 的 RoPE explicit reshape patch；或提供真实 upstream PR/commit 并证明它属于可删除的临时 backport。 |
| R2 | 现有 `dsv4_smla_1rank_aot_eager`（或等价已注册 case）真实启用 `compressor.asc` 并完成 NPU compile + forward + backward + optimizer；真实 CANN backward 数值由 upstream 算子测试或独立 NPU oracle 支撑。 |
| R3 | 删除 `override/deepseek_v4/__init__.py` 新增 compressor 自动导入，保持 `override.imports` 唯一显式激活面。 |
| R4-R6 | 删除 baseline UnitScale 无用 buffer；`seqused=None`；CANN op registration 移出 per-instance `__init__`。 |
| R7 | 在现有 DSV4 文档/recipe 中说明 compressor target 与 CP 限制，不新增 shell/config；修正 PR Checklist 与实际变更一致。 |
| R8 | 收窄 CPU fake test 的结论与实现，避免第二份 compressor 数学被误当 CANN oracle。 |
| TP sharding | 若当前 support matrix 对外承诺 TP>1，则补真实 TP integration evidence；否则在描述中限定为 placement 配置接入，不宣称多 rank 已验证。 |
