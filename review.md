# PR !816 / GitHub #16 Maintainer Review

## 1. Review 结论

| 项目 | 结论 |
|---|---|
| PR | `[feat] enable torch.compile support for DSV4.1`（`pr_816` -> `master`，GitHub #16 / GitCode !816） |
| 基线 | `master@ffa35664b2a7e7bda68d33f429d626fa8ae81e23` |
| 被审提交 | `pr_816@d6176c74335cb6c43c112d76352db6ea71c29a93` |
| 变更规模 | 11 个文件，约 `+631/-42`；核心新增集中在 V4.1 双数据流、vectorized DSA、compile 装配和 token-dispatcher workaround |
| 上游基线 | `requirements.txt` 固定 `torchtitan==0.3.0`；CI 也按 `v0.3.0` checkout |
| 总体结论 | **暂停并澄清** |
| 测试执行 | **未执行（仅静态审查）** |
| 主要阻断原因 | ① PR 描述中的 compile 复现命令没有通过仓库实际支持的 `--compile.*` CLI 打开 compile；② README 宣称 `inductor` 已支持，但 PR 自身说明 plain inductor 当前仍失败；③ compile 路径通过模块级全局变量、环境变量和全局 Dynamo config 改写状态，存在跨实例/跨测试污染；④ V4.1 的 Dynamo workaround 被塞入共享 `patches/torchtitan/token_dispatcher.py`，与该 patch 所引用的 upstream PR #4095 语义不一致，并扩散到其它模型；⑤ 仓内没有注册的 V4.1 compile ST，也没有对 vectorized DSA/shared-dataflow 的直接 UT；⑥ example rename 与已有维护者行间意见冲突，尚未完成命名方案对齐。 |

## 2. 架构/代码问题清单

| ID | 严重级别 | 代码位置 | 问题点 | 影响 | 建议修改方案 |
|---|---|---|---|---|---|
| R1 | **Blocker** | PR 描述中的 A3/A5 复现命令；`examples/deepseek_v41/debug/deepseek_v41_flash_8p_cpt_4k.sh` 的 `COMPILE_ARGS`; `scripts/run_train.sh` | PR 描述使用 `COMPILE_BACKEND=aot_eager/inductor bash ...` 作为 compile 复现入口，但仓内没有 `COMPILE_BACKEND` 的消费点；通用训练入口明确使用 `--compile.enable --compile.components model --compile.backend ...`，而当前 V4.1 example 仍显式传 `--compile.no-enable`。因此按 PR 描述原样执行，无法证明进入了本 PR 新增的 compile 分支。 | 目前 PR 最关键的手工验证证据无法由描述中的命令复现；A3/A5 loss/性能数字不能直接作为“compile 已验证”的仓内证据。 | **统一为已有 CLI 入口，不新增 env 入口。** 所有复现命令改成 `... --compile.enable --compile.components model --compile.backend aot_eager`（inductor 同理）。建议进一步删除 example 中冗余的 `COMPILE_ARGS=--compile.no-enable`，依赖 `CompileConfig` 默认关闭，让用户只通过附加 CLI 打开 compile。修正命令后重新给出验证结果。 |
| R2 | **Blocker** | `examples/deepseek_v41/readme.md`；PR 描述“inductor整图AF当前有问题…”及“aot_eager 即当前可用的整图生产后端” | README 写成“compile（aot_eager / inductor）已支持”，但 PR 描述同时明确 plain inductor 当前存在 AscendC bf16 broadcast / unbacked SymInt 问题，真正可用的是 `aot_eager`，`inductor allfallback` 仅是规避验证。README 同一句又写“当前只支持 ... eager/reference”，支持矩阵自相矛盾。 | 用户会把当前已知失败的 backend 当成受支持生产能力；文档、实现状态、PR 背景不一致。 | 要么本 PR 只声明并交付 **`aot_eager` compile support**，把 inductor 标成 experimental/known limitation；要么先修复 plain inductor 并补正式 ST 后再声明支持。README 给出一条真实可运行的 CLI compile 示例。 |
| R3 | **Blocker** | `torchtitan_npu/models/deepseek_v41/attention.py::_SHARED_INPUTS`; `sparse_attention.py::_VECTORIZED`; `parallelize.py::_apply_compile_v41` | compile 通过模块级 `_SHARED_INPUTS=True`、`_VECTORIZED=True` 切换数据流，同时又暴露 `TTNPU_V41_SHARED_INPUTS`、`TTNPU_DSA_VECTORIZED` 两个 import-time 环境开关。`_apply_compile_v41` 修改后不恢复。 | 同一 Python 进程里只要编译过一个 V4.1 model，之后新建的 eager V4.1 model 也会走 shared/vectorized 路径；测试顺序和多模型进程会互相污染。并且用户语义被拆成 `--compile.*` + 隐藏 env 两套入口，违背仓库训练入口单一化原则。 | 删除这两个生产级全局/env 开关。由 `compile_config` 派生**每个 model/module 实例**的内部状态：model 决定是否构造 `shared`；block/attention 直接以 `shared is not None` 决定 compile dataflow；`V41SparseAttention` 使用实例级 `vectorized` 标志。A/B 测试若需要 vectorized eager，直接测试 `_forward_packed` 或用测试 fixture，不要新增用户 env 语义。 |
| R4 | **Major** | `torchtitan_npu/models/deepseek_v41/parallelize.py::_apply_compile_v41` | 新 helper 基本复制了 pinned upstream `torchtitan.distributed.compile.apply_compile`：`capture_scalar_outputs`、checkpoint side-effect flag、backend 选择、逐 block `fullgraph=True` compile；但本 PR 改为直接 import 私有 `_maybe_regional_inductor_backend`。`parallel_dims` 参数在 helper 内完全未使用。 | 与上游编译装配重复，升级 TorchTitan 时更容易漂移；依赖私有函数比使用公共 `apply_compile` 更脆弱。 | model-specific 逻辑只保留“准备 V4.1 compile dataflow”，随后复用 upstream **public `apply_compile`**。若 `dynamic=False` 被证明是独立必须语义，应先确认 public helper 默认行为是否已满足；确需扩展时优先向 upstream 增加 hook/参数，而不是复制整套 compile 流程。若暂时保留 helper，至少删掉未使用的 `parallel_dims` 并避免 private import。 |
| R5 | **Blocker** | `parallelize.py::_apply_compile_v41` 对 `torch._dynamo.config.recompile_limit/cache_size_limit` 的写入 | PR 把进程全局 recompile/cache limit 提高到 `n_layers + 8`（40 层即至少 48），且不恢复；但旁边注释又声称实际只有“约 6 个 topology role graph”，`recompile_limit` 默认 8 已经大于声称的角色图数量，理论上足以覆盖。 | 这会掩盖本应暴露的 guard/recompile explosion，并污染同进程其它模型；也让“6 graph”这一关键设计结论无法被真实默认配置验证。 | 删除这两个全局 limit 改写，在默认限制下证明 graph 数量/重编译次数合理。若默认 8 仍会触发，应先定位多出来的 guards/role，而不是把 budget 按层数放大。若最终确属 PyTorch/TorchTitan 通用需求，应走 upstream，而不是 V4.1 私有全局配置。 |
| R6 | **Blocker** | `torchtitan_npu/patches/torchtitan/models/common/token_dispatcher.py::_stable_argsort_expert_ids` 及两个 `_local_reorder` override | `patches/torchtitan` 当前文件头引用的 upstream PR 是 `pytorch/torchtitan#4095`（router score pre-W2 absorption）。本 PR 新增的“Dynamo/spmd_types 下 stable argsort workaround”不属于 #4095 的 scope，却被混入同一个共享 patch；该 patch 会替换通用 Local/AllToAll dispatcher，因此 workaround 会扩散到 V4.1 之外的模型。 | 违反本仓 patches 目录“只临时承载可回 upstream 代码”的边界，也把 V4.1/NPU compile 问题扩大为全仓 dispatcher 行为变化。后续 #4095 合入/删除 patch 时还会把这个无关 workaround 一起丢失。 | 两条路径二选一：**(a)** 若这是 TorchTitan 通用 bug，单独提 upstream issue/PR，并用独立 patch 文件/明确 provenance 临时 backport；**(b)** 若只在 NPU/spmd_types/V4.1 触发，移到 `override`/模型扩展的显式 opt-in 路径，禁止放通用 upstream patch。不要继续挂靠 #4095。 |
| R7 | **Major** | 同上 `_stable_argsort_expert_ids` | 当前稳定排序为了替代 argsort 构造 `one_hot[N, E]`，再 cumsum/gather/sum；MoE 的 `N=T*K`，该 workaround 的额外内存/算量为 `O(N*E)`，并且被共享 patch 扩散到所有 patched standard dispatch。Local 和 AllToAll 又复制了完全相同的 `_local_reorder` 方法体。 | 大 token 数/专家数下可能造成明显额外显存/带宽/图规模，compile workaround 反而成为热路径成本；重复实现增加维护面。 | 若最终确需该 workaround，可用更小的稳定 key：`key = expert_id * N + original_position`，然后对 key 做全宽 `topk(largest=False, sorted=True)`，即可得到与 stable argsort 相同的 lexicographic 顺序，避免 `one_hot[N,E]`；并只保留一个共享 helper，不复制两份 `_local_reorder`。该方案仍需 NPU/compile UT 验证。 |
| R8 | **Major** | `sparse_attention.py::V41SparseAttention._forward_packed` 的 `causal_dense` 分支 | `for start in range(..., chunk)` 每个 chunk 内都重新构造完整 `[S,S]` 的 `attend`：`doc_id.unsqueeze(0)==doc_id.unsqueeze(1)` + causal grid。对于 S=4096、chunk=128，静态 unroll 会出现 32 次同尺寸全局 mask 构造（aot_eager 下不能指望编译器帮忙消掉）。同时 `cont_valid` 是新增未使用变量。 | 图规模和运行时工作量随 chunk 数重复放大；这与脚本名/目标中的 4K 场景尤其冲突。 | 把与 chunk 无关的结构移出循环，或更好地在每个 chunk 只构造 `[rows,S]` 的 chunk-local causal/doc mask，避免重复生成 `[S,S]`。删除 `cont_valid`。补 4K/多 document 规模的性能或至少 shape 级回归验证。 |
| R9 | **Major** | `attention.py::DeepSeekV41Attention.forward/_forward_shared`; `block.py::DeepSeekV41TransformerBlock.forward`; `model.py::forward` | 同一个 forward API 根据进程全局 flag 改变返回契约：attention 为 `Tensor` vs `(Tensor, published)`，block 为 2-tuple vs 3-tuple；同时新增一整套 `_build_long_range_context_shared` 与 legacy 实现并行维护。 | hook、AC wrapper、未来上游 Module/compile wrapper 都要隐式依赖全局模式；任何一侧修 bug 都可能漏改另一侧。PR 为一个 compile feature 引入了大面积永久双实现。 | 首先消除全局 flag，至少让行为由显式 `shared` 参数决定。进一步建议保持 canonical eager forward contract 不变：compile 路径用一个 V4.1 专用 adapter/private entry 明确返回 published state；公共 block forward 不应因进程状态改变 arity。把真正共享的 projection/index/select 逻辑下沉为纯 helper，避免复制整段长期逻辑。 |
| R10 | **Major** | `model.py::__init__/forward`; `attention.py::V41SharedInputs/V41AttentionContext` | 新增 `_layer_shared_spec: dict[int, tuple]`，每层缓存一个无类型 9 元组，其中大量内容（source role、ratio、candidate 参数）都能从已有 `compression_plan`/module 结构推导；`publish_layer_id` 只新增+赋值但没有读取；`V41AttentionContext.publish()` 只定义没有调用；`model.forward` 每层循环里重复执行 `from . import attention`。 | 新状态/tuple 协议/死字段并没有独立用户语义，是典型可删复杂度；9 元组靠位置契约，后续极易错位。 | 删除 `publish_layer_id` 和未使用 `publish()`；删除 hot-loop import。优先删除 `_layer_shared_spec`，由 model 在 compiled region 外使用已有 plan 解析动态 shared 输入；若性能证明必须预计算，使用最小 typed structure，且只存无法从 module/plan 直接得到的值。`V41SharedInputs` 也建议只保留真正跨层动态 tensor（kv/index_key/topk/candidates），静态 role 常量放在 layer/attention 实例。 |
| R11 | **Major / Tests** | `tests/unit_tests/models/deepseek_v41/*`; `tests/integration_tests/deepseek_v41.py` | 本 PR 的核心语义是“真实 compile + shared state + vectorized DSA + FullAC/EP/FSDP 可训练”，但现有 V4.1 UT 都走 eager；唯一注册 ST `dsv41_golden_2p_ep2_fsdp2` 也没有任何 `--compile.*`，只能证明 eager frozen trajectory 没被破坏。PR 描述的 8 卡手工命令不属于仓内 ST。 | 没有门禁能阻止 compile 分支以后直接坏掉；也没有独立 oracle 证明 `_forward_packed` 对 multi-doc、ratio1/ratio>1、candidate/topk 的 forward/backward 语义。 | 至少新增：① CPU UT：固定 multi-doc metadata，对 legacy DSA 与 `_forward_packed` 做 forward + backward 对齐（明确容差，覆盖 ratio1 shared-full、ratio>1 precomputed topk、短文档 tail）；② `Indexer.select` `.sort -> topk` 的独立 exact oracle；③ shared-dataflow 与 legacy 对同一小模型的输出/梯度比较；④ **注册 1 个 ≤4 NPU 的真实 ST**，建议复用 2 卡 debugmodel：`--compile.enable --compile.components model --compile.backend aot_eager`，跑完整 train step。若 compile 数值因 reduction grouping 不做 exact loss，应在 ST 表中明确 `check_loss=False` 原因，并由 UT 承担数值等价 oracle。 |
| R12 | **Major / Tests** | `tests/unit_tests/override/common/test_moe_score_absorption.py::test_standard_dispatchers_reuse_torchtitan_helpers` | 测试从“helper identity”改成一次 `patched vs upstream` 比较；其中 `expert_ids_TK` literal 已包含 `[0,0]`、`[3,3]`、`[2,2]` 等重复 expert id，stable-order 的重复值语义已有覆盖。但 oracle 仍只是 upstream 实现而不是显式 expected permutation；测试只直接调用 Local override，新加的 AllToAll `_local_reorder` 没有任何直接测试，也没有证明 workaround 在真实 compile/spmd_types 消费路径上成立。 | AllToAll override 回归、边界 expert-id、N=0/小 N 或 compile fake-eval 回归时，当前测试的证明力仍不足。 | 若 R6 后仍保留 workaround：基于现有确定性 expert-id literal 直接断言 expected permutation、score 对齐和 routed input；为 AllToAll 新 `_local_reorder` 增加直接覆盖，并增加真实 `torch.compile(..., backend="aot_eager", fullgraph=True)` CPU/NPU 最小验证，证明 spmd_types/compile 消费路径。 |
| R13 | **Minor** | `compressor.py::Indexer.select` | `.sort(...).values` 改成对已选 distinct indices 做全宽 `topk(largest=False, sorted=True)`，从数学语义看成立，但 PR 没有为这处 compile workaround 增加直接 UT。 | 如果以后 topk shape/mask/candidate contract 改动，这个 workaround 缺少局部回归锚。 | 加一个很小的 exact UT：包含 score ties、dense mask、candidate mask，断言最终 indices 的升序和值域；无需再引入新的 abstraction。 |
| R14 | **Minor / Docs** | `torchtitan_npu/models/deepseek_v41/parallelize.py` 顶部 module docstring；`sparse_attention.py` 顶部 docstring；`tests/integration_tests/README.md` | `parallelize.py` 仍写“compile branches removed / V4.1 rejects torch.compile”，已经被本 PR 事实推翻；`sparse_attention.py` 的 module docstring 后部已经描述 vectorized compile 路径和 `TTNPU_DSA_VECTORIZED` opt-in，但开头仍自称“The V4.1 numerical-reference DSA (eager, per-document)”，该定位语与当前 eager + vectorized compile 双路径现状不一致；integration README 的 V4.1 表格写“50 步”，实际 case 和当前 V4.1 README 都是 30 步，并且测试矩阵没有 compile case。 | 文档与代码/测试矩阵不一致，后续维护者会被旧约束误导。 | 本 PR 一并刷新这些说明；`sparse_attention.py` 只需修正开头定位语，使其与下文已有双路径描述一致；若新增 R11 的 compile ST，在 integration README 矩阵登记 backend、rank 数和 `check_loss` 语义。 |
| R15 | **Minor / PR metadata** | PR 描述第 3 点；rename diff | PR 描述称 rename 时“删除强制 `export COMPILE_BACKEND=""`”，但本次实际 rename patch 只有脚本路径注释变化，没有这条删除；当前脚本也不存在该 export。 | PR 背景包含已经过时/不属于本 commit 的叙述，降低 review 可追溯性。 | 按最终 diff 重写 PR 描述，只保留本次真实变更；把历史尝试放到“调试历史/已废弃方案”而不是当前 change list。 |
| R16 | **Major / Maintainer feedback** | `examples/deepseek_v41/debug/deepseek_v41_flash_8p_cpt_4k.sh` rename；镜像 PR 第 9 行已有 GitCode maintainer 行间意见 | 镜像 PR 带入了 `@panchao-gitcode` 的明确意见：“别改名吧，后续还有a5的，区分配置差异”。当前提交把 `deepseek_v41_flash_8p_cpt_4k_a3.sh` 改成无平台后缀的 `deepseek_v41_flash_8p_cpt_4k.sh`，与这条已有维护者意见冲突。 | 文件命名涉及后续 A3/A5 配置并存的入口组织方式；在关注维护者尚未认可前，不能把 rename 视为已达成一致。 | 作者需先与 `panchao-gitcode` 对齐：要么恢复原 `*_a3.sh` 文件名，以便后续 A5 入口并存；要么取得其对统一命名方案的明确同意。在该事项解决前，本 review 不认可本次 rename。 |

## 3. 代码删减视角（必须先证明不能删）

| 新增项 | 当前判断 | 可删/可合并方向 |
|---|---|---|
| `_SHARED_INPUTS` + `TTNPU_V41_SHARED_INPUTS` | **不接受**：无独立用户语义，完全由 compile config 推导 | 删除；model 实例内部派生 compile dataflow，block/attention 看显式 `shared` |
| `_VECTORIZED` + `TTNPU_DSA_VECTORIZED` | **不接受为生产入口**：主要服务 compile/A-B | 删除 env/global；改为 attention 实例状态，UT 直接调用 packed helper |
| `_apply_compile_v41` 整套 helper | **大部分可删** | model-specific setup 后调用 upstream `apply_compile`；不要复制 upstream Dynamo config/backend 流程 |
| `recompile_limit/cache_size_limit=n_layers+8` | **应直接删** | 默认 budget 下修 guards/recompile；只有 upstream 通用证据充分时再增加 |
| `_layer_shared_spec` 9 元组 dict | **应显著缩减** | 从已有 plan/module 推导；确需缓存则 typed + 只保留不可推导字段 |
| `publish_layer_id` | **死状态** | 删除 |
| `V41AttentionContext.publish()` | **死 wrapper** | 删除，保留 `absorb` 即可 |
| Local/AllToAll 两份 `_local_reorder` | **重复代码** | 一个 helper/共享实现；更重要的是先解决 R6 patch 归属 |
| `one_hot[N,E]` stable sort | **过重实现** | 若 workaround 必须存在，用 `(expert_id * N + pos).topk(...)` 等 O(N) 辅助状态方案并做实机 compile 验证 |
| `cont_valid` | **未使用** | 删除 |
| 每层 `from . import attention` | **无必要** | 删除；全局 flag 方案移除后自然消失 |
| 双份 long-range context 实现 | **当前复杂度过高** | eager 调用栈如需冻结，则保留 eager entry；compile 用明确 adapter，并抽共享纯算术 helper，避免两个长期完整实现 |

## 4. Tests Review（按仓内 developer-tests-review 规则）

| 语义单元 | 独立 oracle / 真实触发要求 | 当前仓内证据 | 结论 |
|---|---|---|---|
| Eager 默认路径必须继续逐步命中冻结 loss | 真实 NPU integration，固定 seed/deterministic，逐 step exact loss | `dsv41_golden_2p_ep2_fsdp2`：2 NPU、EP2/FSDP2、30 step、`check_loss=True`，但 **compile 未开启** | **已覆盖 eager 回归**；不能证明 compile |
| CLI 打开 V4.1 compile 后真实进入 compiled block | 必须从真实 compile 入口触发并执行结果；只检查 config/flag 不够 | 无 UT；无注册 ST；PR 手工命令还使用仓内未消费的 `COMPILE_BACKEND` env | **缺失，Blocker** |
| Vectorized DSA 与 legacy 语义等价 | multi-doc、ratio1/ratio>1、forward/backward 独立比较 | 无直接 UT；PR 仅给 5-step 手工 loss 近似 | **缺失** |
| Shared cross-layer state 在 AC recompute 下正确 | shared compile dataflow + FullAC，比较输出/梯度/发布状态 | `test_full_ac_preserves_image_routing` 只覆盖 eager/image routing，不进入 shared compile dataflow | **缺失** |
| stable argsort workaround 保持稳定顺序 | 明确 expected permutation，覆盖重复 expert ids，并进入真实 consumer/compile | 当前 Local helper 的 literal 已覆盖重复 expert id，但仅做 patched-vs-upstream 比较；没有显式 expected permutation、没有 AllToAll 新 override 直接测试，也没有真实 compile/spmd_types 消费验证 | **部分覆盖，oracle/消费路径仍弱** |
| `Indexer.select` sort workaround 等价 | exact indices oracle，覆盖 mask/tie/candidate | 无新增测试 | **缺失** |
| `inductor` 是受支持 backend | 真实 NPU ST 完整 train | 无；PR 描述反而明确 plain inductor 当前有问题 | **不应在 README 宣称已支持** |
| Patch `apply()` 后消费者确实使用预期 dispatcher | patch entry + real consumer | 现有 score-absorption 测试中有 `dispatcher_patch.apply()` 和 Local/AllToAll 消费验证，但这部分原本服务 #4095；没有证明本 PR 新 stable-sort 的 compile 问题 | **原 patch 契约有覆盖；新增 workaround 仍缺 compile oracle** |

## 5. ST 增补建议（最小化，不扩测试框架）

| 项目 | 建议 |
|---|---|
| 新 case | 在现有 `tests/integration_tests/deepseek_v41.py` **直接增加一个** `dsv41_compile_aot_eager_ep2_fsdp2`，不要新 runner / 新 shell / 新 suite |
| 卡数 | 2 NPU（沿用现有 V4.1 debugmodel 的 EP2/FSDP2），满足仓内 ST 的最小设备原则 |
| 必需 CLI | `--compile.enable --compile.components model --compile.backend aot_eager`；其余尽量复用现有 V4.1 case |
| 步数 | 以“完整触发 compile + forward + backward + optimizer step”为准，尽量少；无需复制 30-step eager golden |
| 数值 | 如果 vectorized reduction grouping 已明确不保证 bitwise frozen loss，则可 `check_loss=False`，但必须在 case/README 写明原因；数值等价由专门 CPU/NPU UT 用容差证明 |
| inductor | 当前不建议加“正式支持”ST，先修复 PR 描述中已知 plain inductor failure；若保留 experimental allfallback，仅作为手工诊断，不算生产支持 |

## 6. 逐文件覆盖结果

| 文件 | Review 结果 |
|---|---|
| `examples/deepseek_v41/debug/deepseek_v41_flash_8p_cpt_4k.sh` | **rename 尚未认可**：镜像 PR 已带入 `@panchao-gitcode` 的行间意见“别改名吧，后续还有a5的，区分配置差异”。作者需先恢复 `*_a3.sh` 或取得该维护者对统一命名方案的同意；此外当前仍显式 `--compile.no-enable`，与 PR 描述的 `COMPILE_BACKEND=...` 复现方式冲突。 |
| `examples/deepseek_v41/readme.md` | **需修改**：aot_eager/inductor 支持状态写错且同句自相矛盾；补真实 CLI compile 示例。 |
| `tests/unit_tests/models/deepseek_v41/test_independence.py` | 路径随 rename 更新本身与当前 diff 一致；但 rename 是否保留需按 R16 与维护者对齐，它也不是 compile 功能测试。 |
| `tests/unit_tests/override/common/test_moe_score_absorption.py` | **需修改**：重复 expert id 已被 literal 覆盖，但 stable-sort oracle 仍是 upstream 实现；AllToAll 新 override 和真实 compile/spmd_types 消费路径仍无覆盖，且该测试被迫为一个本不应混入 #4095 patch 的行为改动。 |
| `torchtitan_npu/models/deepseek_v41/attention.py` | **需重构**：模块全局数据流开关、隐藏 env、双实现、变长 return contract、9-field shared bundle、死 `publish_layer_id/publish()`。 |
| `torchtitan_npu/models/deepseek_v41/block.py` | **需重构**：依赖模块全局 flag，forward arity 随全局状态变化；改成显式 shared/compile adapter。 |
| `torchtitan_npu/models/deepseek_v41/compressor.py` | 实现思路可接受，但补最小 exact UT，避免 compile workaround 无局部 oracle。 |
| `torchtitan_npu/models/deepseek_v41/model.py` | 移除 compile 拒绝本身方向正确；新增 `_layer_shared_spec`/hot-loop import/dead assignment 应删减，数据流模式改成 per-model state。 |
| `torchtitan_npu/models/deepseek_v41/parallelize.py` | **需重构**：优先复用 upstream public `apply_compile`；删除 private helper 依赖和全局 recompile/cache 调参；修 stale docstring。 |
| `torchtitan_npu/models/deepseek_v41/sparse_attention.py` | vectorized formulation 是 compile 所需的核心新增，但必须补 forward/backward equivalence UT；修 chunk 内重复 `[S,S]` mask 和全局 env/state；删 `cont_valid`；module docstring 已有双路径说明，但需修正开头 `eager, per-document` 的定位语。 |
| `torchtitan_npu/patches/torchtitan/models/common/token_dispatcher.py` | **当前不可接受**：新增 workaround 与文件所引用 upstream PR #4095 无关，且全仓扩散；先按 R6 重新归属，再讨论实现。 |

## 7. Upstream / Patch 边界核对

| 检查项 | 事实 | Review 判断 |
|---|---|---|
| TorchTitan pinned compile 能力 | `torchtitan==0.3.0` 的 public `torchtitan.distributed.compile.apply_compile()` 已负责 async TP（如启用）、`capture_scalar_outputs`、checkpoint side-effect flag、regional backend 选择和逐 TransformerBlock `fullgraph=True` compile | 本 PR 不应复制大部分逻辑，也不应直接绑定私有 `_maybe_regional_inductor_backend` |
| 当前 token dispatcher patch provenance | 文件头写 `Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4095` | #4095 是 router-score pre-W2 absorption；本 PR 的 stable-sort/Dynamo workaround 不在其 scope，不能无来源地附着在该 patch |
| Extensions | 本 PR 未修改 Extensions | 无直接问题；若 workaround 是 NPU 特有增强，应优先考虑 override/extension，而不是扩大 upstream patch |
| PyTorch/torch_npu patch | 本 PR 未新增 | 无问题 |

## 8. 文档/描述一致性

| 位置 | 不一致 | 要求 |
|---|---|---|
| PR 描述复现命令 | `COMPILE_BACKEND=...` 与仓内真实 CLI 入口不一致 | 全部改为 `--compile.*`，重新确认验证结果确实来自 compiled path |
| PR 描述 vs README | PR 写 plain inductor 当前失败；README 写 inductor 已支持 | 缩窄支持范围或先修 backend，二者必须统一 |
| PR 描述 rename 说明 | 声称本 commit 删除 `export COMPILE_BACKEND=""`，实际 rename diff 没有这条改动 | 删除过时叙述 |
| example rename vs maintainer 行间意见 | `@panchao-gitcode` 已明确提出“别改名吧，后续还有a5的，区分配置差异”，当前 diff 仍去掉 `_a3` 后缀 | 恢复原文件名或先取得该维护者对统一命名方案的明确同意；未对齐前 rename 不视为认可 |
| `parallelize.py` module docstring | 仍写“compile branches removed / rejects torch.compile” | 更新 |
| `sparse_attention.py` module docstring | 下文已有 vectorized compile 双路径说明，但开头仍自称 `eager, per-document` | 只需修正开头定位语，使其与已有双路径描述一致 |
| `tests/integration_tests/README.md` | V4.1 行写 50 step，实际 case 是 30 step；也没有 compile case | 修正为 30，并登记新增 compile ST |

## 9. 合入前置条件

| 优先级 | 必须完成事项 |
|---|---|
| P0 | 修正 PR/README/复现命令，使实际验证通过官方 `--compile.*` CLI 进入 compile；重新确认并记录 aot_eager 验证。 |
| P0 | 明确 backend 支持边界：当前至少不要把已知失败的 plain inductor 写成“已支持”。 |
| P0 | 去掉 `_SHARED_INPUTS/_VECTORIZED` 的进程全局可变状态和隐藏 env 控制，改成 per-model/per-module、由 compile config 派生的内部状态。 |
| P0 | 删除/移出 `patches/torchtitan/token_dispatcher.py` 中与 upstream #4095 无关的 stable-sort workaround；若保留，必须有独立 upstream provenance 或显式 NPU override 归属。 |
| P0 | 增加一个仓内注册的 ≤4 NPU V4.1 `aot_eager` compile ST，真实覆盖 compile + forward/backward/optimizer。 |
| P1 | 与 `panchao-gitcode` 对齐 example 命名：恢复 `*_a3.sh` 或取得其对统一命名方案的明确同意。 |
| P1 | 增加 vectorized DSA/shared-dataflow 的独立 forward/backward UT，以及 `Indexer.select` 和 stable-order 的精确 oracle。 |
| P1 | 复用 upstream public `apply_compile`，删除 global recompile/cache budget 放大，收缩 `_layer_shared_spec`/死字段/重复 helper。 |
| P1 | 修复 `_forward_packed` causal-dense 分支在每个 chunk 重建完整 `[S,S]` mask 的图规模/算量问题。 |
| P2 | 刷新 module docstring、integration README 和 PR change list。 |

## 10. Maintainer 最终意见

| 结论 | 说明 |
|---|---|
| **暂停并澄清** | 目标“V4.1 支持 torch.compile”可以继续推进，但当前提交尚未满足本仓对**单一 CLI 入口、上游解耦、patch 边界、代码最小化、可复现验证和正式 ST**的要求。先完成 P0 项，再进入下一轮 review；不建议在现状下合入。 |
