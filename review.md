# PR !833 / GitHub PR #15 Maintainer Review

## 1. 结论

**合入建议：暂停并澄清。**

本 PR 的主方向——V4.1 与 V4 解耦、跨层状态改为显式穿参、移除 `USE_GOLDEN`、收敛到 Attention Gym / 公共 MoE 工厂——总体符合仓库希望减少模型间耦合和隐藏状态的方向；但当前实现仍有两项架构级阻塞问题：

1. V4.1 专属的视觉路由语义被加入自动生效的 `torchtitan_npu/patches/torchtitan/models/common/moe.py`，违反 `patches/torchtitan` 的 ownership 约束，并扩大了所有模型的 import-time blast radius；
2. V4.1 继续依赖 `patches/torchtitan/models/common/linear.py::BatchedLinear`，但该文件声明的 TorchTitan PR #3634 provenance 与实际 upstream 内容不一致，当前官方 TorchTitan 也不存在 `BatchedLinear`，因此它不是一个可随上游版本升级自然删除的有效临时 backport。

此外，训练入口仍有环境变量和 shell/config 双重 source-of-truth，模型配置静态依赖 override replacement 类型，NPU ST 在删除 loss anchor 后仅验证进程完成，checkpoint/state-dict 也缺少真实模型全量 key 的 round-trip 保护。这些问题需要在合入前一起收敛。

**测试执行：未执行（仅静态审查）。** PR 描述中记录的 CPU/NPU 执行结果仅作为作者自验证背景，不作为本 review 的“已执行”证据。

---

## 2. 审查基线与固定依赖

| 项目 | 事实 |
| --- | --- |
| PR | GitHub #15，镜像 GitCode !833 |
| base | `master` / `5ffd25ecc173eed7f92f8178f8796738bbdf8368` |
| head | `pr_833` / `7b1bf45c3d71594b40d39f1d57b44ab84058c0ac` |
| changed files | 49 |
| TorchTitan 固定版本 | `requirements.txt` 固定 `torchtitan==0.3.0`；CI 通过 `.ci/setup_torchtitan.sh` / `.ci/smoke_test.sh` checkout `v0.3.0` |
| Attention Gym | `attn-gym==0.0.9` |
| 测试 review 规则 | `.agents/skills/developer-tests-review` 的 `review测试` workflow：UT、NPU ST、格式独立审查；不得执行或修改测试 |

上游核对结果：

- `pytorch/torchtitan` 的 PR #3634 merge commit `0ff2464637d0947d9f30b7da0dee3973a08b8f1a` 中，`torchtitan/models/common/linear.py` 只有 `Linear` / `ScaledBiasRowwiseLinear`，不存在 `BatchedLinear`；
- 当前官方 `pytorch/torchtitan` 代码搜索同样找不到 `BatchedLinear`；
- `BatchedLinear` 可在 `sdmyzlp/torchtitan:br_dpsk_v4_1` 的 `models/common/linear.py` 中找到，因此当前 patch 的真实来源是模型开发分支，而不是它文件头声明的官方 upstream PR #3634；
- 当前官方 `pytorch/torchtitan` 默认分支也没有 `torchtitan/models/deepseek_v4_1` 目录，因此本 PR 所称“上游 V4.1”应明确写成具体开发分支/commit，而不能泛化为已进入官方 upstream 的能力。

---

## 3. Findings（按严重度排序）

| ID | 严重度 | 位置 | 问题 | 影响 | 修改建议 |
| --- | --- | --- | --- | --- | --- |
| R1 | **Blocker** | `torchtitan_npu/patches/torchtitan/models/common/moe.py` | 本 PR 在自动生效的 TorchTitan common patch 中新增 `vision_enabled`、`bias_vl`、`image_mask`、`sorted_topk`。文件注释本身明确写了“upstream V4.1 is text-only, neither has an upstream counterpart”，即这些语义没有 upstream provenance，并且是 V4.1 multimodal 专属行为。 | `torchtitan_npu/__init__.py` 会提前导入 `patches`，`patches/torchtitan/__init__.py` 又自动导入 common `moe`；因此只要 import `torchtitan_npu`，所有模型看到的 `TokenChoiceTopKRouter`/`MoE` 都已经被替换。即使默认 flag 为 false，也扩大了公共类型、forward signature、state namespace 和升级冲突面。违反本仓“`patches/torchtitan` 仅保存有上游依据的临时 patch；模型专属实现不得进入 patch”的硬约束。 | 把 V4.1 视觉 bias / sorted selection 放回 `torchtitan_npu/models/deepseek_v4_1` 的模型专属 Router/MoE，或使用模型 scope 的 Override/Extension seam；common patch 只保留已经提交 upstream、能忠实 backport 的通用部分。若希望视觉路由成为 TorchTitan 通用能力，先提交 upstream PR，再按该 PR 的真实实现做临时 backport。 |
| R2 | **Blocker** | `torchtitan_npu/patches/torchtitan/models/common/linear.py`；`torchtitan_npu/models/deepseek_v4_1/__init__.py` | `BatchedLinear` 文件头声明 `Pending upstream PR: pytorch/torchtitan#3634`，但 #3634 的实际 merge commit 中没有该类，当前官方 upstream 也没有该符号。本 PR 又新增 `n_batches` alias 并继续让 V4.1 直接依赖这个 patch。 | 该 patch 无法满足“升级到包含 upstream 改动的版本即可删除”的生命周期；未来升级时 maintainer 无法根据 PR provenance 判断删除条件。它实际来自 `sdmyzlp/torchtitan:br_dpsk_v4_1`，属于 V4.1 开发分支代码。 | 在真正 upstream 接纳前，把 `BatchedLinear` 放到 V4.1 模型目录或按目录职责放入 Extension；或者先创建/引用一个**实际包含 BatchedLinear** 的 TorchTitan upstream PR，并让 patch 与该 PR 保持可机械对照的实现。禁止继续用 #3634 作为不成立的 provenance。 |
| R3 | **High** | `torchtitan_npu/models/deepseek_v4_1/__init__.py`；`torchtitan_npu/override/common/rope.py`；example / integration `override.imports` | V4.1 模型配置直接 `from torchtitan_npu.override.common.rope import WorkaroundComplexRoPE`，并把 `WorkaroundComplexRoPE.Config` 写死为模型 config 类型；与此同时 launcher/ST 仍传 `torchtitan_npu.override.common.rope.workaround`。但 `workaround` 的 `@override` 是 `target=ComplexRoPE.Config, exact=True`。 | 模型已经静态选中了 replacement config，CLI 中的 `rope.workaround` 对这些节点不再承担“显式选择 workaround”的职责，入口语义与仓库 Override 文档不一致；模型层反向依赖 override 层，也阻碍未来通过同一 CLI 切换 `asc_partial` 等实现。 | 模型 registry 应持有稳定的上游/base RoPE config（或 V4.1 自己的模型级 split-RoPE config），再由 `override.imports` 选择 replacement。若 pinned TorchTitan 缺少 split config，则该结构能力应放模型/Extension；不要让模型直接 import 某个 workaround replacement。补一条从真实 `Trainer.Config.override.imports` 进入并断言 config replacement 的 UT。 |
| R4 | **High** | `torchtitan_npu/models/deepseek_v4_1/config_registry.py`；`examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh`；`examples/deepseek_v4_1/readme.md` | 删除 `USE_GOLDEN` 是正确收敛，但训练数据入口仍同时依赖 CLI、shell 变量和隐藏环境变量。`config_registry.py` 直接读取 `DSV41_TOKENIZER_PATH` / `DSV4_TOKENIZER_PATH` / `DSV41_VISION_TEXT`；example shell 又提供 `HF_ASSETS_PATH`、`DATASET_PATH` 等 CLI；`HF_ASSETS_PATH` 并不会自动成为该 custom dataloader 的 `tokenizer_path`。脚本名仍写 `4k`，实际 `SEQ_LEN=512`。 | 同一条训练命令无法仅从最终 `Trainer.Config` 判断实际输入文本/tokenizer；环境变量成为第二 source-of-truth。用户修改 `HF_ASSETS_PATH` 可能以为同时改变了 dataloader tokenizer，实际仍可能落入 synthetic arange token 路径。脚本名/实际 seq_len 也已发生文档漂移。 | 把 `tokenizer_path`、`text`、必要的 image assets 暴露为 dataloader/config CLI 字段；删除 `os.environ` 读取。example shell 只做薄封装或直接给一条 `python -m torchtitan_npu.train --module ... --config ...` 命令，不重复 optimizer/scheduler/data 的默认值。将 `4k` 文件名与 512 实际配置统一。 |
| R5 | **High** | `tests/integration_tests/deepseek_v4_1.py`；`tests/integration_tests/run_tests.py`；`tests/integration_tests/README.md` | 本 PR 明确包含有意计算变化：MoE 从旧 V4.1 golden 算术切到公共 grouped-GEMM/clamp/score-absorbed 路径，并新增 `IndexerKLLoss`；同时删除 V4.1 loss anchor。新 case `dsv41_debugmodel_2p_ep2_fsdp2` 设置 `check_loss=False`、没有 `expected_steps`。runner 对这种 case 的 TensorBoard 检查会直接返回，只剩进程 rc/训练完成性。 | 现有 ST 只能证明 2 卡 FSDP2/EP2 reference 路径“能跑完”，不能证明新的 MoE 数值、`indexer_kl_loss`、vision routing 或 attention-gym 路径在 NPU 上保持期望数值。README 中列举历史末步 loss 也不会被 CI 自动检查。 | 保留 2 卡最小 case，但为**新预期语义**建立可自动检查的 deterministic 数值基线，而不是要求与旧 golden 相等：固定 seed/deterministic、开启 `check_loss=True` 并生成新的短迭代 loss 基准；同时至少增加对 `indexer_kl_loss/mean` 存在且 finite/非零的 runner 检查，或扩展现有结果检查 hook。这样只需一个 2 卡 case，不需要扩矩阵。 |
| R6 | **High** | `tests/unit_tests/models/deepseek_v4_1/test_independence.py::test_v41_import_does_not_patch_v4` | 该测试先 `import torchtitan_npu.models.deepseek_v4` 再记录 `torchtitan.models.common.moe` identity；但导入任何 `torchtitan_npu.*` 前都会先执行 `torchtitan_npu/__init__.py`，从而自动应用 `patches/torchtitan/models/common/moe.py`。因此 `before` 基线本身已经被 common patch 污染。 | 测试名称/注释声称证明“V4.1 import does not patch V4”，实际只能证明“V4.1 子包 import 没有在**已经被 torchtitan_npu common patch 修改过的世界**里再次改变 identity”。它无法发现 R1 的全局污染。 | 若 R1 按要求移除模型专属 common patch，此测试可简化为 model package side-effect 检查；若仍保留 patch，必须在隔离 subprocess 中先 import pristine `torchtitan`、记录 upstream class/config，再 import `torchtitan_npu` 并明确断言哪些公共符号允许变化、哪些不允许变化。patch 行为测试应放到 `tests/unit_tests/patches/torchtitan/models/common/`，而不是借 V4.1 model test 兜底。 |
| R7 | **High** | `torchtitan_npu/models/deepseek_v4_1/state_dict_adapter.py`；`tests/unit_tests/models/deepseek_v4_1/test_state_dict_adapter.py`；integration config | 本 PR 改变了大量实际 parameter/state namespace：新 compressor/indexer ownership、`bias_vl`、vision namespace、MoE common stack 等；但 state-dict UT 使用手写的少量 HF/local tensor 字典做自定义 adapter round-trip，没有从**真实 registry 构建出来的 model.state_dict()** 出发。NPU integration 又显式 `--checkpoint.no-enable`。 | adapter 中遗漏一个真实参数 key、实际 shape 或 DTensor 分片规则时，当前手写 fixture 仍可能通过；也没有证据说明旧 `deepseek_v41` 本地 checkpoint 是否明确不兼容、可迁移还是需要 adapter。 | 新增一条 tiny registered model 的全量 `state_dict()` round-trip：构建真实 debug config → 初始化模型 → 获取完整 local state → `to_hf` → `from_hf` → 对全部可映射 key/value 做一致性检查，重点覆盖 `bias_vl`、compressor/indexer、vision markers、grouped experts。若旧 DCP 不保证兼容，在 README/PR 中明确写“breaking checkpoint namespace”，不要只说“对齐上游”。必要时再补一个最小 2 卡 save/load case，而不是扩大普通训练矩阵。 |
| R8 | **Medium** | `torchtitan_npu/models/deepseek_v4_1/model.py::apply_activation_checkpointing_extensions` | Vision AC 直接调用 pinned TorchTitan `ActivationCheckpointing._wrap_block()` 私有方法。代码注释也承认这是 private API。 | TorchTitan 升级只要调整 private method 名称、参数或 policy 结构，V4.1 即 break；这与 Extension 应尽量利用稳定 hook、避免 monkey/private coupling 的仓库目标冲突。 | 优先向 upstream 补一个公共“wrap extra block / apply to module list” hook；在本仓过渡期，把兼容逻辑放在与 upstream 目录镜像的 Extension 层并集中隔离版本差异，模型只声明额外 block 列表，不直接调用 private method。 |
| R9 | **Medium** | `tests/unit_tests/models/deepseek_v4_1/test_indexer_distill_loss.py`；测试目录组织 | PR 修改的是自动 patch 和公共 MoE，但新增/调整的大多数验证都放在 `tests/unit_tests/models/deepseek_v4_1`；`tests/unit_tests/patches/torchtitan/models/common/` 没有与本次 `moe.py`/`linear.py` 改动对应的 patch-level test。另外 `test_indexer_distill_loss.py` 中相邻 top-level test 之间缺少标准空行，属于简单格式问题。 | 测试 ownership 与生产 ownership 不一致，会使后续删除/升级 patch 时找不到对应保护；格式问题也说明本次测试文件未完全按仓库统一 test format 收口。 | R1/R2 解决后，按最终 ownership 迁移测试：模型专属行为留在 model UT；真正保留的自动 patch 必须在 `tests/unit_tests/patches/...` 从 package import/apply 入口验证最小行为。顺手修复 test 文件空行/ruff 格式。 |
| R10 | **Medium** | `torchtitan_npu/models/deepseek_v4_1/__init__.py`、`model.py`、`compressor.py`、`indexer.py` | 本轮重构已经把 cross-layer state 改为显式 tuple，这一点比旧 `attention_context` 更清晰；但配置构建层仍混入若干只针对固定内置 topology 的防御式校验（例如 source/candidate 组合、registered layer table 自校验），且同一限制在 registry/model/loader 多处重复。 | 固定 flavor 的内部不变量被重复 runtime validate，会增加噪声并产生多个错误源；真正用户可配置的限制与内部 builder invariant 混在一起。 | 保留面向用户 CLI 的 TP/CP/PP/compile/seq-alignment 校验；对由常量 builder 自己保证、用户不可修改的内部组合，优先用构造结构本身保证或测试保护，减少重复 runtime `ValueError`。把“用户输入校验”和“开发期 invariant”区分开。 |

---

## 4. 架构与 ownership 逐项结论

| 维度 | 结论 | 说明 |
| --- | --- | --- |
| V4 / V4.1 解耦 | **部分达成** | 新 `deepseek_v4_1` 不再 import V4 model/override，cross-layer state 显式穿参是正向改进；但 V4.1 行为反而下沉到了全局 common MoE patch，形成更大的隐式共享面。 |
| Upstream 复用 | **部分达成** | Attention Gym 0.0.9、TorchTitan common MoE factory 的复用合理；但 `BatchedLinear` 和 split/partial RoPE 的 upstream provenance/ownership 还没有收敛。 |
| `patches/torchtitan` 边界 | **不符合** | R1、R2 为阻塞。模型专属能力不能放 common patch；临时 patch 必须有真实 upstream PR/commit。 |
| Override 边界 | **不符合** | 模型直接依赖 `WorkaroundComplexRoPE.Config`，使 `override.imports` 不再是唯一激活面。 |
| Extension/升级隔离 | **有风险** | Vision AC 直接依赖 `_wrap_block` 私有 API，升级 TorchTitan 易 break。 |
| 训练入口单一性 | **不符合预期** | `config_registry` + hidden env + shell CLI 共同决定数据/训练行为，且 example 名称与 seq_len 已不一致。 |
| State lifecycle | **cross-layer runtime state 改善，checkpoint 证据不足** | 显式 tuple 解决 mutable attention context 污染；但 checkpoint 真实 key/DTensor/save-load 未完整保护。 |
| 文档一致性 | **需要修订** | README 的 smoke-only 描述基本与代码一致，但 example `4k` vs 512、tokenizer source-of-truth 不一致；“上游 V4.1”需要注明具体 branch/commit。 |

---

## 5. 语义变换与独立 oracle

| 语义变换 | 新实现 | 独立 oracle / 当前证据 | 结论 |
| --- | --- | --- | --- |
| 跨层 compressed KV/index selection/candidate state | 由 model/block/attention return tuple 显式逐层传递 | `test_attention_threading.py` 通过 hooks 检查 producer/consumer 对象 identity；这是直接的 CPU observable contract | **已覆盖** |
| Compressor / Indexer 每层实例、非 source 无参数 | `Compressor.is_source` / `Indexer.is_source` | `test_attention_threading.py` 检查非 source parameter 为空并原样传递共享 tensor | **已覆盖** |
| CSA2 sparse inner attention 切到 Attention Gym `selected_attention` | `CompressedSparseInnerAttention2` | PR 有 CPU policy tests，但本 review 未看到一个完全独立 dense/reference 同时覆盖 window + selected entries + sink + doc isolation 的 end-to-end forward/backward oracle；作者 A/B 记录不能替代 committed UT | **部分覆盖** |
| Indexer KL distillation | `IndexerKLLoss`，梯度 `Z*Y-p` | `test_indexer_distill_loss.py` 用手算小例子直接检查 student gradient、loss value、invalid slots、per-layer sum | **已覆盖**（CPU 数学契约） |
| V4.1 multimodal router bias | common patch `bias_vl/image_mask/sorted_topk` | model UT 只验证 opt-in/default-off 和 FullAC route；没有合法 ownership，且全局 patch baseline test 无效 | **部分覆盖，架构阻塞** |
| MoE 切公共 grouped-GEMM/clamp/score-absorbed | common factory + patch | 有局部 CPU routed-expert 行为检查；NPU ST 无数值 anchor | **部分覆盖** |
| HF/local state mapping | composed text + vision adapter | 手写子集 round-trip | **部分覆盖**（缺真实 model state 全量 key） |

---

## 6. UT 正向功能覆盖表

| 正向功能 | 生产代码和应观察结果 | PR 中的直接检查 | 状态 | 合入前置条件 |
| --- | --- | --- | --- | --- |
| Registry 构建独立 V4.1 模型并跑主要 forward/backward | `model_registry("deepseek_v4_1_debugmodel")` 构建 40 层 tiny model；无 V4 import；输出 finite、专家梯度存在 | `test_independence.py::test_v41_builds_and_runs_without_v4` 使用 import blocker + tiny widths +真实 registry build，并做 forward/backward | 已覆盖 | 无 |
| Cross-layer state 只显式传递，不存 module mutable context | layer source 输出必须原对象到达 consumer；source 替换 in-flight tensor | `test_attention_threading.py::test_cross_layer_state_is_threaded_from_source_to_consumer` hooks block/core 并检查 object identity | 已覆盖 | 无 |
| Candidate pool / selection / student logits 生命周期 | layer 20 建 pool；consumer 使用同一 pool；inference 不生成 logits | 同文件 `test_candidate_pool_reaches_only_its_window_indexers`、`test_student_logits_are_training_only` | 已覆盖 | 无 |
| Indexer KL 数学与 backward | loss value 与 `dI=Z*Y-p`；invalid slot 梯度为 0 | `test_indexer_distill_loss.py` 手算独立 expected | 已覆盖 | 无 |
| V4.1 vision bias 仅 V4.1 opt-in，其他 common router 不变 | V4.1 `bias_vl` 参与 image token routing；普通 router 不受影响 | `test_baseline_contract.py::test_v41_moe_rides_the_common_stack_with_an_opt_in_vision_bias` 直接构造 common factory | 部分覆盖 | 先解决 R1 ownership；若 patch 保留，增加 pristine upstream → package import 的 patch-level test，不能用已被 patch 的 baseline |
| V4.1 import 不污染 V4 | 导入 V4.1 前后 V4 类型/配置不应因 V4.1 专属行为变化 | `test_v41_import_does_not_patch_v4` 的 `before` 已在 `torchtitan_npu` package patch 之后 | 检查无效 | 按 R6 重写或随着 R1 删除全局行为而简化 |
| FullAC 保持 image routing 与梯度 | checkpointed / non-checkpointed forward、router mask/id、参数梯度一致 | `test_training_contract.py::test_full_ac_preserves_image_routing` | 已覆盖（CPU） | 私有 `_wrap_block` 升级风险另见 R8 |
| V4.1 state adapter 覆盖真实模型全部 state | 真实 model state 全量 to_hf/from_hf 不漏 key，关键 tensor 一致 | 当前只用手写 `_base_hf_dict` / `_vision_hf_dict` 子集 | 部分覆盖 | 按 R7 增加 tiny real model full-state round-trip |
| Training entry 单一可追溯 | 同一 CLI/config 唯一确定 tokenizer/text/image/seq | 未见对应测试；环境变量在 `config_registry.py` 直接读取 | 未覆盖 | 先按 R4 收敛入口，再增加 config/CLI parsing UT |

### UT review 子结论

**补充测试后合入**（仅测试维度）。但整体 PR 仍受 R1/R2 架构 Blocker 约束，不能仅靠补测试解决。

---

## 7. NPU ST 触发判断

| 生产代码变更 | 受影响训练场景 | 为什么需要 ST | 现有代表测试 | 判断 | 合入前置条件 |
| --- | --- | --- | --- | --- | --- |
| V4.1 attention 改 Attention Gym + 显式 state threading | V4.1 eager/reference，FSDP2+EP2 | 需要确认真实 NPU 前后向、EP/FSDP 生命周期能完成 | `dsv41_debugmodel_2p_ep2_fsdp2` | 已覆盖“完成性”，未覆盖数值 | 保留现 case；新增 deterministic metric/loss 自动检查 |
| 新 `IndexerKLLoss` | 同上，训练时 source indexer 产生 logits、consumer 注入 aux gradient | CPU 数学不能证明 NPU 训练图/分布式归约/metric 正确 | 同 case | 部分覆盖 | runner 检查 `indexer_kl_loss/mean` 存在、finite，并保留新预期 loss baseline |
| MoE 改公共 grouped-GEMM/clamp/score-absorbed | EP2 routed experts + shared expert | 这是明确的有意计算变化，必须有 NPU 数值 guard | 同 case | 部分覆盖 | 对新实现建立 deterministic 新 baseline；不要求与旧 golden 相等 |
| V4.1 vision bias routing | multimodal tokens 在 EP2 router 走 `bias_vl` | CPU FullAC 测试不能证明 NPU distributed route/combine | 同 case 使用 vision loader，但无 route activation assertion | 部分覆盖 | 最小化增加 metric/hook/assertion，确认 image token path 实际启用；无需新增第二个并行组合 |
| Checkpoint namespace / adapter | 保存、加载、恢复 optimizer/model state | 当前 ST 明确 `--checkpoint.no-enable` | 无 | 未覆盖 | 若声明 checkpoint 支持，补最小 save/load；若本 PR 不承诺旧 checkpoint 兼容，至少文档明确 breaking 边界并先完成 CPU full-state adapter test |

### 4 张 NPU ST 事实表

| 测试 | 模型/配置 | 并行数值 | 替换实现/融合算子 | 编译模式 | NPU 数 | 启用与完成检查 | golden | 执行阶段 |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| `dsv41_debugmodel_2p_ep2_fsdp2` | `deepseek_v4_1_debugmodel`，40 层 debug widths | FSDP2 / EP2 / TP1 / CP1 / PP1 | eager Attention Gym；common RoPE workaround；virtual optimizer | eager，compile off | 2 | runner 检查子进程成功；case 本身无 target metric/route activation assertion | 无，`check_loss=False` | CI `models` suite 静态包含 |

矩阵投影：

| 并行 | Reference eager | Reference compile | AscendC sparse eager | AscendC compile |
| --- | --- | --- | --- | --- |
| FSDP2+EP2 | `dsv41_debugmodel_2p_ep2_fsdp2`：完成性已覆盖，数值 guard 缺失 | 不支持，model config 明确拒绝 compile | 不支持，ratio-1 shared KV 未接入 | 不支持 |
| 8 卡 FSDP8+EP8 | 仅 example 手动入口，不属于 integration ST | 不支持 | 不支持 | 不支持 |

### ST 不能证明的内容

现有 `check_loss=False` smoke 即使实际执行成功，也不能证明：

- Attention Gym 与旧 sparse attention 数值等价；
- 新 MoE arithmetic 的训练精度符合预期；
- `IndexerKLLoss` 的 NPU gradient 与 CPU 手算完全一致；
- checkpoint save/load 或旧 checkpoint 兼容；
- 8 卡吞吐、性能或长序列能力；
- future TorchTitan 升级下 private AC / patch 仍兼容。

---

## 8. 测试格式审查

| 位置 | 类型 | 代码阅读结果 | 修改建议 |
| --- | --- | --- | --- |
| `tests/unit_tests/models/deepseek_v4_1/test_independence.py` | 测试结构 / 状态隔离 | subprocess 隔离思路正确，但“V4 不受污染”的 baseline 创建时 package patch 已经执行，测试层级与声明不一致 | baseline 改为 pristine upstream import，或在 R1 删除全局 patch 后重新定义该测试目标 |
| `tests/unit_tests/models/deepseek_v4_1/test_baseline_contract.py` | 文件位置 | 测试了 common patch 新增的 `vision_enabled/sorted_topk/bias_vl`，但文件位于 model UT；若生产 owner 仍是 `patches/common`，测试 owner 不匹配 | 最终 ownership 按 R1 重构后同步移动；不要让 model UT 代替 patch UT |
| `tests/unit_tests/models/deepseek_v4_1/test_indexer_distill_loss.py` | 测试结构 | `test_distill_loss_is_attached_only_where_a_selection_exists` 与下一顶层 test 之间缺少标准空行 | 运行 formatter/ruff 后修正 |
| `tests/integration_tests/deepseek_v4_1.py` | integration runner 收集 | 已通过 `run_tests.py::build_models_test_list` 和 suite `deepseek_v4_1` 注册，符合既有 runner 结构 | 入口可复用，不要新增一次性 CI shell；仅增强结果检查 |

---

## 9. Clean code / 文档 / 升级风险补充

1. `torchtitan_npu/models/deepseek_v4_1/__init__.py` module docstring 有明显残句：`The width sets and topology constants match the frozen the registered...`，应清理。
2. `examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh` 文件名/历史语义是 4K，但代码固定 `SEQ_LEN=512`；README 直接推荐该文件，属于入口文档与实际配置不一致。
3. `DEFAULT_VISION_IMAGE_PATHS = ("tests/assets/dsv4_vit_test.jpeg",)` 把测试 asset 作为 production recipe 默认输入。debug/smoke flavor 可以有测试 asset，但正式 `flash_40layers...` recipe 不应隐式绑定仓库测试资源；应通过 CLI/config 明确选择，或区分 debug 与真实训练 recipe。
4. 当前 V4.1 只支持 TP1/CP1/PP1、compile off。这些限制在 model config 中 fail fast 是合理的；不要为了“防御式”继续在 shell、registry、runner 重复验证同一内部常量。
5. `attn-gym==0.0.9` 同时写入 `pyproject.toml`、`requirements.txt` 且 CI 因 `--no-deps` 额外显式安装，当前原因可以成立；建议注释明确“CI image workaround”的删除条件，避免以后版本升级后永久保留第三份 pin。

---

## 10. 合入前置条件

### P0 — 必须先处理 ownership

1. 从 `patches/torchtitan/models/common/moe.py` 移除 V4.1-only `vision_enabled/bias_vl/image_mask/sorted_topk`；放到 V4.1 model scope 或合法 Override/Extension。
2. 更正 `BatchedLinear` ownership/provenance：在真正官方 upstream PR 接纳前，不得继续伪装成 #3634 backport。
3. 去掉 `deepseek_v4_1 -> override.common.WorkaroundComplexRoPE` 静态依赖，使 `override.imports` 恢复为实现选择唯一入口。

### P1 — 收敛训练入口与状态契约

4. 将 tokenizer/text/image 等训练输入全部变成可见 config/CLI 字段，移除 `config_registry.py` 中的隐藏环境变量读取；example shell 只做薄封装。
5. 明确 checkpoint compatibility：至少增加真实 tiny model 全量 state round-trip；若旧 `deepseek_v41` checkpoint 不支持，README/PR 显式写 breaking change。
6. 将 example `4k`/512 命名、README、实际 config 对齐。

### P1 — 补齐测试证据

7. 修正 `test_v41_import_does_not_patch_v4` 的无效 baseline；如果 common patch 最终仍有合法保留内容，新增 patch-level import/apply UT。
8. 2 卡 ST 继续复用 `dsv41_debugmodel_2p_ep2_fsdp2`，不要扩 NPU 数；为本 PR 的**新预期数值**恢复 deterministic 自动检查，并对 `indexer_kl_loss/mean` 增加结果检查。

完成上述 P0/P1 后再重新 review；在此之前不建议合入 `master`。

---

## 11. Self-check

- [x] 已按 PR base/head 而不是只看最终文件做审查；
- [x] 已核对固定 TorchTitan 版本 `v0.3.0`；
- [x] 已核对官方 upstream #3634 实际 `linear.py`，确认 `BatchedLinear` provenance 不成立；
- [x] 已检查 `patches/torchtitan`、Override、模型、训练入口、state/checkpoint、UT、ST、文档一致性；
- [x] tests 目录按 `.agents/skills/developer-tests-review` 的 `review测试` workflow 静态审查；
- [x] 未把 PR 描述里的作者执行记录当作本 reviewer 的执行结果；
- [x] 未修改生产代码、测试或 `master`；仅在 `pr_833` 新增本 `review.md`。
