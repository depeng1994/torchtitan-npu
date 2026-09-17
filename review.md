# PR !833 / GitHub PR #15 Maintainer Review

## 1. 结论

**合入建议：暂停并澄清。**

> Revision note：根据 maintainer 对 Common MoE ownership 的补充口径，本版不再把 V4.1 新增的 `vision_enabled` / `bias_vl` / `image_mask` / `sorted_topk` 进入 Common MoE 本身视为架构污染。对于差异较小、语义可通用、后续有跨模型复用价值且默认关闭不改变已有模型行为的能力，优先扩展 Common MoE 比复制一套 model-specific Router/MoE 更符合「复用优于重复」原则。

本 PR 的主方向——V4.1 与 V4 模型实现解耦、跨层状态改为显式穿参、移除 `USE_GOLDEN`、收敛到 Attention Gym / 公共 MoE 工厂——总体符合仓库架构方向。原 R1 已调整为**认可的 Common 抽象决策**，不再作为阻塞项。

当前仍有一个明确的架构级 Blocker：V4.1 继续依赖 `patches/torchtitan/models/common/linear.py::BatchedLinear`，但该文件声明的 TorchTitan PR #3634 provenance 与实际 upstream 内容不一致，当前官方 TorchTitan 也不存在该符号，因此 patch 的上游归属和删除条件不清晰。

此外，模型配置静态依赖 override replacement 类型、训练入口仍存在隐藏环境变量与 CLI/config 双 source-of-truth、NPU ST 删除 V4.1 loss anchor 后只保留完成性检查、checkpoint/state-dict 缺少真实模型全量 key 的 round-trip 保护。这些问题仍建议在合入前收敛。

**测试执行：未执行（仅静态审查）。** PR 描述中记录的 CPU/NPU 执行结果仅作为作者自验证背景，不作为本 review 的「已执行」证据。

---

## 2. 审查基线与固定依赖

| 项目 | 事实 |
| --- | --- |
| PR | GitHub #15，镜像 GitCode !833 |
| base | `master` / `5ffd25ecc173eed7f92f8178f8796738bbdf8368` |
| head（review 前） | `pr_833` / `7b1bf45c3d71594b40d39f1d57b44ab84058c0ac` |
| changed files | 49 |
| TorchTitan 固定版本 | `requirements.txt` 固定 `torchtitan==0.3.0`；CI checkout `v0.3.0` |
| Attention Gym | `attn-gym==0.0.9` |
| 测试 review 规则 | `.agents/skills/developer-tests-review` 的 `review测试` workflow：UT、NPU ST、格式独立审查；不得执行或修改测试 |

上游核对结果：

- `pytorch/torchtitan` 的 PR #3634 merge commit `0ff2464637d0947d9f30b7da0dee3973a08b8f1a` 中，`torchtitan/models/common/linear.py` 不存在 `BatchedLinear`；
- 当前官方 `pytorch/torchtitan` 代码中同样没有 `BatchedLinear`；
- `BatchedLinear` 可在 `sdmyzlp/torchtitan:br_dpsk_v4_1` 的 `models/common/linear.py` 中找到，因此当前 patch 的实际来源与文件头声明的 #3634 不一致；
- 当前官方 `pytorch/torchtitan` 默认分支没有 `torchtitan/models/deepseek_v4_1` 目录，因此文档中的「上游 V4.1」应尽量注明具体开发分支/commit，避免与官方主线已合入状态混淆。

---

## 3. Findings 与架构决策

| ID | 严重度 | 位置 | 问题 / 决策 | 影响 | 修改建议 |
| --- | --- | --- | --- | --- | --- |
| R2 | **Blocker** | `torchtitan_npu/patches/torchtitan/models/common/linear.py`；`torchtitan_npu/models/deepseek_v4_1/__init__.py` | `BatchedLinear` 文件头声明 `Pending upstream PR: pytorch/torchtitan#3634`，但 #3634 的实际 merge commit 中没有该类，当前官方 upstream 也没有该符号。本 PR 又新增 `n_batches` alias 并继续让 V4.1 直接依赖这个 patch。 | patch 无法根据当前 provenance 判断何时可删除，也无法机械对照真正 upstream 实现；升级 TorchTitan 时容易长期滞留。 | 更正 provenance。若该能力计划进入 TorchTitan Common Linear，则引用一个**实际包含 BatchedLinear** 的 upstream PR/commit，并保持 patch 与其可机械对照；在 upstream 接纳前也至少要明确真实来源、贡献目标和删除条件。 |
| R3 | **High** | `torchtitan_npu/models/deepseek_v4_1/__init__.py`；`torchtitan_npu/override/common/rope.py`；example / integration `override.imports` | V4.1 模型配置直接 import `WorkaroundComplexRoPE` 并把 `WorkaroundComplexRoPE.Config` 写死为模型 config 类型；与此同时入口仍传 `torchtitan_npu.override.common.rope.workaround`，而该 override `exact=True` target 是 `ComplexRoPE.Config`。 | replacement 已在模型构建期静态选定，CLI 中的 `rope.workaround` 不再承担真正的实现选择；模型层也反向依赖 override 层，削弱同一训练入口切换实现的能力。 | 模型 registry 持有稳定 base/split-RoPE config，再由 `override.imports` 选择 workaround / AscendC replacement。若 pinned TorchTitan 缺 split config，应把结构性 config 放 Common Extension/patch，而不是直接绑定某个 replacement。 |
| R4 | **High** | `torchtitan_npu/models/deepseek_v4_1/config_registry.py`；`examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh`；README | 删除 `USE_GOLDEN` 是正确收敛，但数据入口仍同时依赖 CLI、shell 变量和 `DSV41_TOKENIZER_PATH` / `DSV4_TOKENIZER_PATH` / `DSV41_VISION_TEXT` 环境变量。脚本名为 `4k`，实际 `SEQ_LEN=512`。 | 同一最终 `Trainer.Config` 不能完整描述真实输入；环境变量成为第二 source-of-truth，脚本命名也与运行配置漂移。 | 将 tokenizer/text/image 等输入暴露为 dataloader/config CLI 字段；example shell 只做薄封装。统一 `4k` 文件名与实际 seq_len。 |
| R5 | **High** | `tests/integration_tests/deepseek_v4_1.py`；`tests/integration_tests/run_tests.py`；README | 本 PR 有明确计算语义变化：MoE 切公共 grouped-GEMM/clamp/score-absorbed 路径并新增 `IndexerKLLoss`；但 V4.1 case 删除 loss anchor，`check_loss=False` 且没有 `expected_steps`。 | 现有 NPU ST 主要证明 2 卡 FSDP2/EP2 训练进程完成，不能自动守护新的 MoE 数值轨迹和 KL loss metric。 | 复用现有 2 卡 case，不扩矩阵；为**新预期实现**建立 deterministic 短迭代基线，并至少检查 `indexer_kl_loss` metric 存在且 finite。 |
| R7 | **High** | `torchtitan_npu/models/deepseek_v4_1/state_dict_adapter.py`；`tests/unit_tests/models/deepseek_v4_1/test_state_dict_adapter.py` | 本 PR 改变 compressor/indexer ownership、vision namespace、MoE stack、`bias_vl` 等实际 state namespace，但 UT 主要使用手写的小规模 HF tensor dict round-trip，integration 又关闭 checkpoint。 | 真实 registry model 新增/遗漏 key、shape 或 adapter mapping 时，手写子集仍可能通过。 | 增加 tiny registered model 的全量 `state_dict()` → `to_hf` → `from_hf` round-trip；重点覆盖 compressor/indexer、vision markers、common MoE `bias_vl`、grouped experts。旧 checkpoint 若不承诺兼容，应明确记录 breaking boundary。 |
| R1 | **Accepted** | `torchtitan_npu/patches/torchtitan/models/common/moe.py` | **修订结论：Common MoE generalization 本身合理。** `vision_enabled` / `bias_vl` / `image_mask` / `sorted_topk` 都是 Router/MoE 层能够表达的通用能力；当前以 opt-in 方式启用，plain factory 默认关闭。对于这种小差异，做宽 Common MoE 比保留一套 V4.1-only Router/MoE 更有利于后续模型复用。 | 不再将其视为「V4.1 污染 Common」或要求移回 model-specific。真正需要控制的是 Common API 是否保持通用、默认行为是否严格兼容，以及 patch 是否有明确 upstream 贡献/删除生命周期。 | **保留 Common 方向。** 建议把注释从「V4.1 extension」改成能力语义描述，避免 Common 层绑定单模型叙事；补 Common-level UT，分别证明 default-off 与 opt-in 行为；若计划进入 TorchTitan，记录对应 upstream issue/PR 或贡献计划，便于后续删除 patch。 |
| R6 | **Medium** | `tests/unit_tests/models/deepseek_v4_1/test_independence.py::test_v41_import_does_not_patch_v4` | 原测试通过 type identity 判断「V4.1 import 不污染 V4」，但 Common MoE patch 本来就是 package-wide 的共享能力，identity 是否变化不是正确的架构 oracle。 | 该测试会把「合法的 Common 能力扩展」和「V4.1 子包额外 monkey patch」混在一起。真正应保护的是 V4/common 默认配置的**行为兼容**。 | 重写测试目标：验证导入 V4.1 子包不会再次修改 Common 类型/registry；同时从真实 V4/common config 构建 router/MoE，证明 `vision_enabled=False`、`sorted_topk=False` 时行为与既有路径一致。Common patch 自身的 default-off/opt-in 契约放 patch-level UT。 |
| R8 | **Medium** | `torchtitan_npu/models/deepseek_v4_1/model.py::apply_activation_checkpointing_extensions` | Vision AC 直接调用 pinned TorchTitan `ActivationCheckpointing._wrap_block()` 私有方法。 | TorchTitan 升级只要调整 private method 结构就可能 break。 | 优先推动 upstream 公共 extra-block AC hook；过渡期把 private API 适配集中到 Extension/compatibility seam，模型只声明额外 blocks。 |
| R9 | **Medium** | Common MoE/Linear patch tests；`test_indexer_distill_loss.py` | Common MoE/Linear 的新增契约主要由 V4.1 model UT 间接覆盖，缺少与生产 ownership 对齐的 patch-level test；另有简单格式空行问题。 | 后续其它模型复用 Common MoE 或删除 patch 时，难以单独判断公共契约是否保持。 | 为保留在 Common 的 MoE 能力增加 patch-level default-off/opt-in UT；`BatchedLinear` 在最终 ownership 确定后也放对应层级测试；修复格式问题。 |
| R10 | **Medium** | `torchtitan_npu/models/deepseek_v4_1/__init__.py`、`model.py`、`compressor.py`、`indexer.py` | cross-layer state 显式 tuple 是明显改善，但固定内置 topology 的部分 invariant 在 builder/model/loader 多处重复 runtime 校验。 | 用户配置错误和内部 builder invariant 混在一起，增加噪声与维护面。 | 保留 TP/CP/PP/compile/seq alignment 等用户可见校验；由固定常量构造保证的内部组合优先交给结构和 UT 保护。 |

---

## 4. 架构与 ownership 逐项结论

| 维度 | 结论 | 说明 |
| --- | --- | --- |
| V4 / V4.1 解耦 | **达成方向正确** | V4.1 不再依赖 V4 model-specific 实现，cross-layer state 显式穿参是正向改进。复用 Common MoE 不属于 V4/V4.1 模型耦合。 |
| Common MoE ownership | **认可** | 小差异、可通用、默认关闭的 Router/MoE 能力应优先进入 Common，避免每个模型复制一套实现。需要保护的是通用接口与默认兼容性，而不是强制 model-specific。 |
| Upstream 复用 | **部分达成** | Attention Gym 与 TorchTitan common MoE factory 的复用合理；`BatchedLinear` provenance 仍需修正。 |
| `patches/torchtitan` 边界 | **部分符合** | R1 的 Common MoE 方向可接受，前提是它被视为拟 upstream 的通用能力并有清晰生命周期；R2 的 `BatchedLinear` provenance 当前不成立。 |
| Override 边界 | **不符合预期** | 模型直接依赖 `WorkaroundComplexRoPE.Config`，使 `override.imports` 失去实现选择的唯一性。 |
| Extension/升级隔离 | **有风险** | Vision AC 直接依赖 `_wrap_block` 私有 API。 |
| 训练入口单一性 | **不符合预期** | hidden env + registry + shell CLI 共同决定输入。 |
| State lifecycle | **runtime state 明显改善，checkpoint 证据不足** | 显式 tuple 解决 mutable attention context；真实 model state/save-load 尚未完整保护。 |
| 文档一致性 | **需要修订** | `4k` vs 512、tokenizer source-of-truth、“上游 V4.1”具体来源需要对齐。 |

---

## 5. 语义变换与独立 oracle

| 语义变换 | 新实现 | 独立 oracle / 当前证据 | 结论 |
| --- | --- | --- | --- |
| 跨层 compressed KV/index selection/candidate state | model/block/attention return tuple 显式逐层传递 | `test_attention_threading.py` hooks producer/consumer 并检查 object identity | **已覆盖** |
| Compressor / Indexer 每层实例、非 source 无参数 | `is_source` 控制参数 ownership | UT 检查非 source parameter 为空并原样传递共享 tensor | **已覆盖** |
| CSA2 切 Attention Gym `selected_attention` | `CompressedSparseInnerAttention2` | 有 CPU policy/contract tests；缺少完整独立 dense oracle 同时覆盖 window + selected entries + sink + doc isolation 的 forward/backward | **部分覆盖** |
| Indexer KL distillation | `IndexerKLLoss`，梯度 `Z*Y-p` | 手算小例子检查 gradient/loss/invalid slots/per-layer sum | **已覆盖**（CPU 数学契约） |
| Common multimodal Router 能力 | `bias_vl/image_mask/sorted_topk` opt-in | `test_baseline_contract.py` 已检查 V4.1 opt-in、plain router default-off；还缺 production ownership 对应的 patch-level 契约测试 | **部分覆盖，架构方向认可** |
| MoE 公共 grouped-GEMM/clamp/score-absorbed | Common factory + patch | 有局部 CPU 行为检查；NPU ST 无数值 anchor | **部分覆盖** |
| HF/local state mapping | text + vision composed adapter | 手写子集 round-trip | **部分覆盖** |

---

## 6. UT 正向功能覆盖表

| 正向功能 | 生产代码和应观察结果 | PR 中的直接检查 | 状态 | 合入前置条件 |
| --- | --- | --- | --- | --- |
| Registry 构建独立 V4.1 模型并跑主要 forward/backward | 真实 registry build，不依赖 V4 model-specific 代码 | `test_independence.py::test_v41_builds_and_runs_without_v4` | 已覆盖 | 无 |
| Cross-layer state 只显式传递 | source tensor 原对象到达 consumer | `test_attention_threading.py` hooks block/core | 已覆盖 | 无 |
| Candidate pool / selection / student logits 生命周期 | pool/selection 按 topology 复用，inference 不产生 training logits | 对应 threading tests | 已覆盖 | 无 |
| Indexer KL 数学与 backward | loss / gradient 与独立手算一致 | `test_indexer_distill_loss.py` | 已覆盖 | 无 |
| Common Router default-off + V4.1 opt-in | plain Common Router 不受 image mask 影响；V4.1 开启 `bias_vl/sorted_topk` | `test_baseline_contract.py::test_v41_moe_rides_the_common_stack_with_an_opt_in_vision_bias` | 部分覆盖 | 增加 Common patch-level test，并从真实 V4/common config 验证 default-off compatibility |
| V4.1 子包不产生额外全局副作用 | import V4.1 不应在 package Common patch 之外再次 mutate Common/V4 registry | 现有 identity test 的 oracle 需要重定义 | 部分覆盖 | 按 R6 调整测试目标，不再把合法 Common patch 当污染 |
| FullAC 保持 image routing 与梯度 | checkpointed/non-checkpointed 路径一致 | `test_training_contract.py::test_full_ac_preserves_image_routing` | 已覆盖（CPU） | private `_wrap_block` 风险另见 R8 |
| V4.1 state adapter 覆盖真实模型全部 state | full local state to_hf/from_hf 不漏 key | 当前手写子集 | 部分覆盖 | 增加 tiny real model full-state round-trip |
| Training entry 单一可追溯 | 最终 CLI/config 唯一确定 tokenizer/text/image/seq | 环境变量仍直接参与 | 未覆盖 | 先收敛入口，再加 config parsing UT |

### UT review 子结论

**补充测试后合入**（仅测试维度）。Common MoE generalization 不再构成架构 Blocker；整体 PR 仍受 R2 以及 R3/R4 等生产架构问题约束。

---

## 7. NPU ST 触发判断

| 生产代码变更 | 受影响训练场景 | 为什么需要 ST | 现有代表测试 | 判断 | 合入前置条件 |
| --- | --- | --- | --- | --- | --- |
| V4.1 attention 改 Attention Gym + 显式 state threading | V4.1 eager，FSDP2+EP2 | 确认真实 NPU 前后向与分布式生命周期 | `dsv41_debugmodel_2p_ep2_fsdp2` | 已覆盖完成性，未覆盖数值 | 复用现 case；增加 deterministic result guard |
| 新 `IndexerKLLoss` | indexer logits + consumer aux gradient | CPU 数学不能证明 NPU graph/reduce/metric | 同 case | 部分覆盖 | 检查 KL metric 存在、finite，并保留新预期 loss baseline |
| MoE 切公共 grouped-GEMM/clamp/score-absorbed | EP2 routed/shared experts | 明确计算变化，需要 NPU 数值 guard | 同 case | 部分覆盖 | 对新实现建立 deterministic baseline |
| Common multimodal Router opt-in | image token 在 EP2 router 走 `bias_vl` | 需要确认 distributed route/combine 的 opt-in 路径 | 同 case 使用 vision loader，但无 route activation assertion | 部分覆盖 | 增加最小 activation/metric 检查；无需新增第二组合 |
| Checkpoint namespace / adapter | save/load/restore | 当前 case `--checkpoint.no-enable` | 无 | 未覆盖 | 若承诺 checkpoint 支持，补最小 save/load；否则明确 breaking boundary 并先完成 CPU full-state test |

### 4 张 NPU ST 事实表

| 测试 | 模型/配置 | 并行数值 | 替换实现/融合算子 | 编译模式 | NPU 数 | 启用与完成检查 | golden | 执行阶段 |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| `dsv41_debugmodel_2p_ep2_fsdp2` | `deepseek_v4_1_debugmodel`，40 层 debug widths | FSDP2 / EP2 / TP1 / CP1 / PP1 | eager Attention Gym；Common MoE；RoPE workaround；virtual optimizer | eager | 2 | runner 检查子进程成功；无 target metric/route activation assertion | 无，`check_loss=False` | CI `models` suite 静态包含 |

矩阵投影：

| 并行 | Reference eager | Reference compile | AscendC sparse eager | AscendC compile |
| --- | --- | --- | --- | --- |
| FSDP2+EP2 | 完成性覆盖，数值 guard 缺失 | model config 明确不支持 | ratio-1 shared KV 未接入 | 不支持 |
| 8 卡 FSDP8+EP8 | 仅 example 手动入口，不属于 integration ST | 不支持 | 不支持 | 不支持 |

现有 `check_loss=False` smoke 即使成功，也不能单独证明 Attention Gym 数值等价、新 MoE arithmetic 精度、KL loss NPU gradient、checkpoint 兼容或 8 卡性能。

---

## 8. 测试格式审查

| 位置 | 类型 | 代码阅读结果 | 修改建议 |
| --- | --- | --- | --- |
| `tests/unit_tests/models/deepseek_v4_1/test_independence.py` | 测试结构 / 状态隔离 | identity oracle 把合法 Common package patch 与 V4.1 subpackage side effect 混在一起 | 改为「无额外 mutation + default-off 行为兼容」 |
| `tests/unit_tests/models/deepseek_v4_1/test_baseline_contract.py` | 文件位置 | 同时验证 V4.1 consumer 与 Common Router 新能力 | consumer 断言可保留；Common default-off/opt-in 契约另补 patch-level UT |
| `tests/unit_tests/models/deepseek_v4_1/test_indexer_distill_loss.py` | 测试结构 | 相邻 top-level test 间缺少标准空行 | formatter/ruff 修复 |
| `tests/integration_tests/deepseek_v4_1.py` | integration runner 收集 | 已接入 `build_models_test_list` 和独立 suite | 复用入口，只增强结果检查 |

---

## 9. Clean code / 文档 / 升级风险补充

1. `torchtitan_npu/models/deepseek_v4_1/__init__.py` module docstring 存在 `match the frozen the registered...` 残句，应清理。
2. example 文件名为 `4k`，代码固定 `SEQ_LEN=512`，README 又直接推荐该脚本，需统一。
3. `DEFAULT_VISION_IMAGE_PATHS = ("tests/assets/dsv4_vit_test.jpeg",)` 将测试 asset 作为 production recipe 默认输入；debug/smoke 可使用测试资源，正式 recipe 应通过 config/CLI 明确选择。
4. V4.1 当前 TP1/CP1/PP1、compile off 的 fail-fast 限制合理，不要在 shell/registry/runner 重复同一内部校验。
5. `attn-gym==0.0.9` 在 pyproject、requirements 和 CI workaround 中多处出现，应注明 CI workaround 的删除条件。
6. Common MoE 注释建议从「V4.1 keeps two extensions」改成能力级描述，例如 multimodal/conditional routing bias 与 deterministic/sorted top-k selection，避免 Common 层长期携带单模型 ownership 叙事。

---

## 10. 合入前置条件

### P0 — 修正真正的 ownership / activation 问题

1. **不要求把 `bias_vl/image_mask/sorted_topk` 移回 V4.1 model scope。** 保留 Common MoE generalization；补齐 default-off/opt-in 公共契约测试，并把注释改成通用能力语义。
2. 更正 `BatchedLinear` provenance：给出真实 upstream PR/commit/贡献目标及删除条件，不能继续以不包含该实现的 #3634 作为依据。
3. 去掉 `deepseek_v4_1 -> override.common.WorkaroundComplexRoPE` replacement 静态依赖，使 `override.imports` 重新承担实现选择职责。

### P1 — 收敛训练入口与状态契约

4. tokenizer/text/image 等训练输入统一到可见 config/CLI，移除隐藏环境变量 source-of-truth；example shell 保持薄封装。
5. 增加真实 tiny model 全量 state round-trip；旧 checkpoint 若不支持，明确 breaking boundary。
6. 对齐 example `4k`/512 命名、README 与实际 config。

### P1 — 补齐测试证据

7. 将 `test_v41_import_does_not_patch_v4` 改成「V4.1 子包无额外 mutation + Common default-off 对已有模型行为兼容」；Common MoE patch 增加 ownership 对应的 patch-level UT。
8. 复用 2 卡 `dsv41_debugmodel_2p_ep2_fsdp2`，为新实现恢复 deterministic 自动数值 guard，并检查 `indexer_kl_loss` metric。

完成上述 P0/P1 后再重新 review；当前不建议直接合入 `master`。

---

## 11. Self-check

- [x] 已按 PR base/head 做审查；
- [x] 已核对固定 TorchTitan `v0.3.0`；
- [x] 已核对官方 upstream #3634，确认 `BatchedLinear` provenance 不成立；
- [x] 已根据 maintainer 补充架构口径重新判断 Common MoE ownership：可通用的小差异优先扩展 Common，不再按 model-specific 污染处理；
- [x] 已检查 patches、Override、模型、训练入口、state/checkpoint、UT、ST、文档一致性；
- [x] tests 目录按 `.agents/skills/developer-tests-review` 的 `review测试` workflow 静态审查；
- [x] 未把作者执行记录当作本 reviewer 的执行结果；
- [x] 未修改生产代码、测试或 `master`；仅更新 `pr_833/review.md`。

删减者review结论:

> 范围：严格按 `torchtitan-npu-reviewer-skills/Reducer.md` 做静态删减审查；未执行测试。Reducer 的目标不是为每个新增对象找理由，而是先假设它不需要存在，再证明哪些语义必须留下。PR 当前基线为 `master@5ffd25ecc173eed7f92f8178f8796738bbdf8368 <- pr_833@75a4ad5102183a03f62ec91c9c9ab3f906ca5af9`。GitHub 镜像没有历史 review thread/review submission；下文“历史矩阵”针对本文件既有 maintainer review R1-R10 重新判定。

## 1. Reducer Findings

| ID | 优先级 | 对象 | Reducer 判定 | 为什么当前形态不应保留 | 最小化修改 | 测试动作 / 风险控制 |
| --- | --- | --- | --- | --- | --- | --- |
| RED-01 | **P0 / Blocker** | `torchtitan_npu/patches/torchtitan/models/common/moe.py` 中 `vision_enabled` / `bias_vl` / `image_mask` / `sorted_topk` | **MOVE** | 这里要区分“Common 抽象是否合理”和“是否允许落在 patch 目录”。前者可以合理；后者当前不成立。该文件自己的 module docstring 已明确写出这两项 V4.1 extension **“neither has an upstream counterpart”**，而仓库规则要求 `patches/torchtitan` 只临时承载已经有实际 upstream PR/commit、但 pinned TorchTitan 尚缺失的代码。#3634/#4095 不能为这两项 V4.1 能力提供 provenance。 | 保留“通用 Router 能力”这个设计方向，但先从 auto-applied TorchTitan patch 移出：优先放本仓 Common Extension/正常源码层，并让 V4.1 opt-in；若确实要进入 TorchTitan Common，则先提交一个真实包含这些字段/行为的 upstream PR，再让 patch 成为与该 PR 可机械对照的临时镜像。不要为了迁移再复制一套完整 MoE。 | 先改 ownership，再决定测试位置。现有 model-level default-off/opt-in 行为测试可保留一个；不要为了当前非法 patch 再新增一套 patch-level 测试。 |
| RED-02 | **P0 / Blocker** | `patches/torchtitan/models/common/linear.py::BatchedLinear` 以及本 PR 新增 `self.n_batches = config.n_heads` | **MOVE + DELETE** | 既有 review 已确认 #3634 实际不包含 `BatchedLinear`，官方 TorchTitan 当前也没有该符号；本 PR 又为 V4.1 的 `Attention.forward` 读取 `wo_a.n_batches` 增加一个 alias，相当于为了一个 caller 扩大临时 patch API。更关键的是 `sdmyzlp/torchtitan:br_dpsk_v4_1` 的 `BatchedLinear` 本身使用 `Config.n_batches` 和 `[B,O,I]` weight，和本仓“`n_heads` + flattened weight + alias”的混合形态并不相同，无法声称机械跟随 source branch。 | 不再给当前 patch 叠 alias。V4.1 若只需要 group count，直接在模型模块保存/推导 group count；若需要真正 upstream `BatchedLinear` 语义，则以实际 upstream PR/commit 为唯一来源整体对齐，而不是继续维护 hybrid API。没有真实 upstream PR 前，代码应离开 `patches/torchtitan`。 | ownership 修正后只保留该层实际数学行为的一个测试；不要为 alias 单独加测试。若 weight layout 改变，state-dict round-trip 必须覆盖 `wo_a.weight`。 |
| RED-03 | **P1** | `_make_v41_attn_config()` / `_make_indexer_config()` 的 `owns_compressor`、`owns_indexer`、`source_key`、`external_key`、`is_candidate_source`、`uses_candidates` 等布尔组合 | **SIMPLIFY + REUSE** | 多个耦合 bool 表达的是有限状态机，却暴露出大量非法组合，于是又产生 `source_key && external_key`、candidate owner 等防御校验。当前 `sdmyzlp/torchtitan:br_dpsk_v4_1` 已经有更收敛的 `HierarchicalIndexer.Mode.{FULL,REINDEX,REUSE}` 表达，可直接说明“不需要这些组合态”。 | 复用/对齐单一 role/mode；由 `layer_id + source layer sets` 一次派生 role，再构造 compressor/indexer。让结构消除非法状态，而不是继续添加校验。 | 删除针对非法 bool 组合的 UT；保留 FULL→REUSE、FULL→REINDEX 两个代表性行为检查。 |
| RED-04 | **P1** | `V41Model.Config` 的 `n_layers`、`kv_source_layers`、`index_source_layers`、`candidate_source_layer`、`candidate_topk_blocks`、`candidate_block_size` 与每层 config 同时保存 topology | **MERGE / DELETE DERIVED STATE** | model builder 已经把 ownership、candidate role 和 block 参数写进每层 `Compressor.Config/Indexer.Config`；运行时模型并不需要再次持有整套 source lists。`n_layers` 也可由 `len(layers)` 得到。重复快照制造了 `n_layers == len(layers)` 等二次一致性校验和测试。`compress_ratios` 仍有 dataloader alignment/state adapter consumer，应保留或由 layers 单点派生。 | 让 `layers` 成为 runtime topology 的单一事实源。state adapter 改为遍历 `model_config.layers`；删除无 consumer 的 model-level source/candidate mirror fields，并删除 registry 中刚构建后再比较 `n_layers/len(layers)` 的校验。 | 删除“配置字段等于常量”的 representation tests；保留真实 registry build/forward 与 adapter 全量 round-trip。 |
| RED-05 | **P1** | `_make_v41_config()` 中固定 topology 的大段 `ValueError` 矩阵，以及 `test_invalid_reuse_topology_is_rejected` 的 8 组 private-helper 负例 | **DELETE** | 三个公开 flavor 都把固定常量传给这个 private builder；这些非法 topology 不是 CLI/user contract。当前实现先允许私有 helper 接受任意组合，再用几十行 defensive programming 防住自己，测试又反向把这些内部错误文案固化。 | 删除 private topology validation matrix，让 builder 只构造合法固定 topology；保留真正用户可配置边界的 TP/CP/PP/compile fail-fast，以及 dataloader 的 `seq_len % alignment` 检查。若未来 topology 真成为 CLI 参数，再在用户入口恢复最小验证。 | 直接删除 8-case parametrized negative test；这部分删减不降低受支持路径覆盖。 |
| RED-06 | **P1** | `_v41_trainer_config(...seq_len/fsdp_shard_degree/context_parallel_degree/expert_parallel_degree/local_batch_size/steps...)`、`DSV41_TOKENIZER_PATH`/`DSV4_TOKENIZER_PATH`/`DSV41_VISION_TEXT`、example shell 同时覆盖训练参数 | **SIMPLIFY / DELETE SECOND SOURCE** | public registry 函数并不使用这些 helper 参数做 flavor 选择；用户本来已有统一 CLI 覆盖层。与此同时 tokenizer/text 又绕开 config 直接读环境变量，形成 registry、env、shell、CLI 四个入口。 | private trainer helper只保留 flavor 构建所需参数；训练数值直接写 recipe defaults，由 CLI 覆盖。tokenizer/text/image path 进入 dataloader config/CLI；不再新增新的 shell/config 变体。现有 example shell 只保留薄封装。 | 增加/保留一条真实 CLI/config parsing 到 dataloader 的检查即可；删除针对内部 helper 参数的测试需求。 |
| RED-07 | **P1** | `test_attention_threading.py`、`test_attention_policy.py`、`test_independence.py` 中大量 object identity、tuple slot、字段存在性和静态源码扫描 | **REDUCE / MERGE** | producer→consumer 连接是产品语义，但“40 层每一层返回同一个 Python object”“固定 tuple index 是 2/3/4/5/6”“某个 private module 没有某字段”并不是用户结果。这些测试会把未来把 tuple 封装、role 收敛或等价 tensor 重建都误判为回归。`test_v41_package_has_no_v4_references` 还扫描 tests/examples 源码，属于高维护的仓库结构 oracle；真实隔离 subprocess build 已经提供更强的行为证据。 | threading 只留 3 个代表节点：source→reuse、reindex、candidate pool，并在 consumer hook 检查数值/shape/最终 forward，而不是覆盖全部 40 层 identity。`test_independence` 保留隔离 subprocess 的真实 registry forward/backward；删除 AST/repo-wide grep 与 Common type identity 锁定。 | 这是主要测试删减来源；详见 Test Reduction Matrix。 |
| RED-08 | **P1** | `tests/integration_tests/deepseek_v4_1.py` 的 `use_golden=True` + `check_loss=False`，以及已删除专用 loss anchor | **SIMPLIFY, 不扩 ST 矩阵** | runner 只有 `check_loss=True` 才读取并比较 golden，也只有 `check_loss/check_resume` 才自动加 deterministic 参数；因此当前 `use_golden=True` 不提供数值 oracle，只制造“似乎有 golden”的配置噪声。另一方面本 PR 明确改变 MoE arithmetic 并新增 KL backward，单纯 completion 又不足以证明新训练语义。 | 不新增第二个 NPU case。复用现有 2P FSDP2+EP2 case：若定位为纯 smoke，就把 `use_golden` 明确设为 false；若它承担本 PR 数值回归（更符合本次计算变更），则让同一个 case 开 deterministic 并做短轨迹/目标 metric guard。两种语义二选一，不保留当前半 golden 状态。 | ST 数量保持 1；检查 `indexer_kl_loss` 存在且 finite。若恢复 loss guard，只记录新实现的预期，不再维护 old-vs-golden 双路径。 |
| RED-09 | **P2** | `test_state_dict_adapter.py` 的 ownership helper tests、deterministic duplicate round-trip；手写 HF 子集 | **MERGE** | `owns_*` helper 的真值表和“同一输入调用两次得到同值”都弱于真正 adapter 契约。当前更重要的风险是实际 tiny registry model 的 compressor/indexer/vision/MoE key 是否完整映射。 | 用一个真实 tiny model `state_dict -> to_hf -> from_hf` 全量 round-trip 替代 helper 真值表 + deterministic duplicate；必要时再加一个未知 key/breaking-boundary case，不做组合矩阵。 | 测试数量下降但 oracle 变强；覆盖 `wo_a`、compressor/indexer、vision markers、`bias_vl`、grouped expert weights。 |
| RED-10 | **P2** | `attn-gym==0.0.9` 在 `requirements.txt`、`pyproject.toml`、`.ci/unit_test.sh`、`.ci/smoke_test.sh` 的版本字面量；example 名称 `...4k...sh` 实际 `SEQ_LEN=512` | **SIMPLIFY / RENAME** | 新依赖本身必要，但版本号不应在 CI workaround 再复制两份；脚本名与真实配置不一致则是已经发生的文档状态分叉。 | 保留一个依赖版本 source-of-truth，CI 的 `--no-deps` workaround 从它读取；不要为 CI 再建脚本。把现有 example 直接重命名为 512 对应名称并同步 README 引用。 | 纯维护面删减；不新增测试。 |

## 2. Reduction Inventory

| 新增/修改对象 | 当前职责 | 判定 | Reducer 结论 |
| --- | --- | --- | --- |
| `DeepSeekV41Metadata(doc_ids_BL)` | packed-document isolation | **KEEP** | 单字段、两个真实 consumer，共享语义清晰；不要重新引入 `cu_seqlens`/context state。 |
| model/block/attention 显式 thread `cmp_k/idx_k/topk_indices/topk_scores/candidates` | 跨层共享 runtime state | **KEEP** | 这是从 mutable state 收敛到显式数据流，和 source branch 方向一致；不要为了缩短签名恢复 module state。 |
| 每层 `Compressor` / `Indexer`，非 source 无权重 | 统一 layer shape 与 ownership | **KEEP** | 语义真实，source/reuse 生命周期清楚；应删的是 role 表达冗余，不是每层对象本身。 |
| `IndexerKLLoss` | indexer distillation | **KEEP** | `Z*Y-p` 有独立数学 oracle，是训练语义而非 debug metadata。 |
| `CompressedSparseInnerAttention2` + `attn-gym` | window + selected compressed KV + sink | **KEEP** | 引入第三方 operator 替代本仓重复 sparse implementation，符合复用优先。 |
| `DeepSeekV41Metadata`/Attention Gym 的 `doc_ids` 接线 | varlen isolation | **KEEP** | 单一 metadata source。 |
| `_make_v41_attn_config` 的多 bool role surface | 构造每层 role | **SIMPLIFY/REUSE** | 收敛到单一 Mode/role。 |
| `V41Model.Config` source/candidate topology 镜像字段 | 重复保存 builder topology | **DELETE/MERGE** | layers 已编码；去掉 derived state。 |
| `n_layers` + `len(layers)` 双记录 | 层数 | **SIMPLIFY** | state adapter/metrics 改为遍历 layers 后可删除 `n_layers`。 |
| `_make_v41_config` topology validation matrix | 防御 private helper 的非法调用 | **DELETE** | 非用户入口；结构上消除非法组合。 |
| `_document_alignment` / `document_alignment` | CLI 改 seq_len 后仍保证 pooling 边界 | **KEEP** | 这是实际运行约束，不应因 Reducer 误删；只压缩重复测试。 |
| `_vision_encoder_anchor()` | 无 image scatter 时让 vision 参数参与 autograd | **INLINE/SIMPLIFY** | 行为要留，单调用 helper/长 docstring 可直接内联成清晰一行或短局部注释。 |
| `DEFAULT_VISION_IMAGE_PATHS=tests/assets/...` | synthetic/debug 默认图像 | **MOVE** | 测试 fixture 不应成为正式 flash recipe 的隐式 production 默认；只留 debug/integration config。 |
| Common MoE V4.1 fields in auto patch | multimodal/sorted routing | **MOVE** | 抽象可 Common，patch placement 当前无 upstream basis。 |
| `BatchedLinear.n_batches` alias | 让 V4.1 forward 读取 group count | **DELETE** | caller-specific alias；group count 本地保存即可。 |
| `attn-gym` dependency | selected attention backend | **KEEP** | 必要依赖；只去重版本声明。 |
| example shell | 单节点参考训练入口薄封装 | **KEEP + RENAME** | 不新增脚本；修正 4k/512 漂移。 |
| 2P FSDP2+EP2 integration case | NPU 完成性/数值代表场景 | **KEEP** | 只保留这一组，不扩组合；让语义明确。 |

## 3. Historical Review Fix Reduction Matrix

| 既有 finding | Reducer 复核 | 当前状态 | Reducer 动作 |
| --- | --- | --- | --- |
| R1 Common MoE generalization **Accepted** | “Common 抽象可复用”成立，但不能推出“允许放在 `patches/torchtitan`”。patch 本文明确承认两项能力无 upstream counterpart。 | **重新打开** | 对抽象 **KEEP**，对 patch placement **MOVE (P0)**；不要继续为非法 placement 增测试。 |
| R2 `BatchedLinear` provenance | 仍成立，而且本 PR 的 `n_batches` alias 又扩大了无 provenance API。 | **未修** | **MOVE + DELETE alias (P0)**。 |
| R3 model 静态依赖 `WorkaroundComplexRoPE` | Reducer 同意：这是实现选择重复，override 不再是单一入口。 | **未修** | **REUSE/MOVE** 到稳定 config + override seam。 |
| R4 hidden env + `4k`/512 | 仍成立；`_v41_trainer_config` 还额外复制一组本可 CLI 覆盖的参数。 | **未修** | **DELETE second source + RENAME**。 |
| R5 ST 无 numeric guard | 仍成立，但不应通过增加新 NPU case 修。 | **未修** | **MERGE** 到现有 2P case，保持 ST=1。 |
| R6 identity oracle | Reducer 更进一步：repo-wide AST scan、40-layer object identity、private tuple slot 都属于实现表示锁定。 | **未修** | **DELETE/REDUCE**，保留真实隔离 build + 代表 producer/consumer。 |
| R7 state adapter 子集 round-trip | 仍成立；解决方式不应是再叠 helper test。 | **未修** | **MERGE** 现有测试为一个 real-model full round-trip。 |
| R8 private `_wrap_block` | 仍是升级耦合。 | **未修** | **MOVE** private compatibility 到 Extension/compat seam；模型只声明 extra blocks。 |
| R9 patch-level tests | 前提需要修改：RED-01/02 ownership 未合法化前，不应先为非法 patch 扩测试。 | **需改方案** | 先 MOVE/明确 upstream basis；最终位置只留最小行为 UT。 |
| R10 defensive topology validation | 完全符合 Reducer 删除目标。 | **未修** | **DELETE** private validation matrix + 对应 negative tests。 |

GitHub 镜像当前没有 review thread/review submission，因此没有额外“历史 inline comment 已修复但残留代码”可做二次删减；本矩阵已覆盖 `review.md` 中可见的 R1-R10 历史决策。

## 4. Top Reduction Plan

| 优先级 | 改动 | 预期净效果 |
| --- | --- | --- |
| **P0** | 把无 upstream counterpart 的 V4.1 Router/MoE 扩展移出 `patches/torchtitan`，或先建立真实 upstream PR 后再以 patch 镜像；不复制完整 MoE。 | 恢复 patch 目录可删除、可机械对照的生命周期；去掉 package-wide model leakage。 |
| **P0** | 停止继续扩 `BatchedLinear` hybrid patch；删除 `n_batches` alias，V4.1 本地持有 group count；若要 upstream Common，则整体对齐真实 PR。 | 去掉无 provenance API 和一个跨层 patch 依赖。 |
| **P1** | role bool 收敛为一个 Mode；删除 model-level topology mirror fields、`n_layers/len(layers)` 二次状态、private topology validation matrix。 | 同时减少 config surface、运行时校验和大量 negative/representation tests。 |
| **P1** | Trainer recipe 只保留 flavor defaults + CLI；tokenizer/text/image 不再读 hidden env；现有 shell 保持单一薄入口。 | 训练入口重新单一可追溯，不新增 config/shell 变体。 |
| **P1** | 测试以行为为中心压缩：一个真实隔离 forward/backward、几个代表 state transitions、独立 KL/mHC 数学 oracle、一个 real state round-trip、一个 2P ST。 | 降低维护成本，同时提升 oracle 强度。 |
| **P2** | CI 从单一依赖版本源读取 `attn-gym`，重命名 `4k`→512 example，移动 tests asset 默认值到 debug/integration。 | 去掉文档/依赖重复状态。 |

### KEEP 清单（不要为了“删代码”误删）

- 显式跨层 tuple state，而不是恢复 mutable attention context；
- `DeepSeekV41Metadata.doc_ids_BL` 单字段 metadata；
- 每层 Compressor/Indexer 的 source/reuse 结构；
- Attention Gym `selected_attention` 复用；
- `IndexerKLLoss` 的 marginal-weighted KL 和独立 closed-form gradient oracle；
- mHC 的显式 branch-sum oracle；
- dataloader alignment **约束本身**；
- 现有唯一 2P FSDP2+EP2 代表 ST。

## 5. Test Reduction Matrix

| 测试文件 / case | 当前价值 | Reducer 动作 | 保留的最小 oracle |
| --- | --- | --- | --- |
| `test_attention_policy.py` | selection mask 手算是高价值；大量 config ownership/非法 topology 是 representation/defensive | **保留 1，删除/合并其余** | 保留 document-isolated causal selection 的精确 expected；删除 8-case private builder rejection，role 配置检查缩成一个真实 source→reuse/reindex 行为。 |
| `test_attention_threading.py` | producer/consumer 语义重要，但全 40 层 identity 与 tuple slot 强耦合 | **大幅缩减** | 仅 source→reuse、reindex、candidate pool 三个代表 transition + 最终 forward；删除重复 pass-through identity、private missing-state assertion matrix。 |
| `test_baseline_contract.py` | TP user boundary、实际 routing 行为有价值；内部 factory/type/default field 次要 | **MERGE** | 保留 TP rejection；Common/V4.1 routing 只保留一个最终行为 case。删除 `compress_ratios` private consistency 负例。 |
| `test_independence.py` | 隔离 subprocess real registry build/backward 很强；AST scan/type identity 较弱 | **保留 1~2，移动 1** | 保留 `test_v41_builds_and_runs_without_v4`；删除 repo-wide AST grep 和 Common type identity；`test_common_rope_imports_without_cann` 移到 common rope/override 对应测试目录。 |
| `test_indexer_distill_loss.py` | closed-form gradient 与 pooled-teacher 等价是核心算法 oracle | **KEEP + 小合并** | 保留 `Z*Y-p` + pooled sum；invalid-slot 可并入主 case；删除“每层 config 是否挂 loss”的 representation test。 |
| `test_mhc_v4_1.py` | explicit branch oracle 高价值；多个性质测试存在重叠 | **KEEP 2，合并/删重复性质** | `matches_the_explicit_branch_sum` + permutation case 足够区分错误 contraction axis；shape/Sinkhorn 可并入一个 smoke。 |
| `test_state_dict_adapter.py` | 当前 helper ownership + 手写子集过弱 | **替换而非增加** | 一个 tiny real registry model 全量 round-trip；删除 `owns_*` 真值表和 deterministic duplicate。 |
| `test_training_contract.py` | synthetic producer 与 FullAC gradient/routing 都是用户可观察行为 | **KEEP** | FullAC 前后 output/grad + image mask；synthetic loader 基础字段可留一个。 |
| `test_vision_loader_alignment.py` | alignment 约束真实，但 6 个 case 重复验证字段/透传/不改变数据 | **缩成 2** | 一个 registry-derived alignment 到 loader 的正向 case + 一个 CLI `seq_len=511` rejection。删除 `dataset.document_alignment` 字段存在、loader pass-through、unchanged-row 等重复检查。 |
| `dsv41_debugmodel_2p_ep2_fsdp2` | 唯一代表 NPU ST | **KEEP 1，不新增组合** | 明确 smoke 或 deterministic guard 二选一；本 PR 更适合在同一 case 增 KL metric + 短数值 guard。 |

### 测试 review 子结论

当前测试的主要问题不是“数量少”，而是**实现表示测试过多、真正 end-to-end oracle 反而需要更强**。建议用替换而不是叠加：删除 private validation/identity/helper 测试后，把预算集中到真实 registry forward/backward、real state round-trip 与现有 2P ST。按仓内 UT/ST review skill，CPU UT 不能替代真实 NPU 通信，但也不需要为每个 bool/field 再造 case。

## 6. “还能否删掉约 30%？”

**可以，但目标应是本 PR 新增的 validation/config/test scaffolding，而不是核心 V4.1 算法。** 不建议为了数字删除显式 state、metadata、KL、Attention Gym 或 mHC 语义。可直接形成约 30% 量级维护面下降的组合是：

1. 删除 `_make_v41_config` 的 private defensive validation matrix + 8 个 negative topology cases；
2. role bool 收敛为单一 Mode，并删除 model-level topology mirror fields/对应字段断言；
3. `test_attention_threading` 从“40 层 object identity 全展开”收敛到 3 个代表 transition；
4. 删除 `test_independence` 的 repo-wide AST scan/type identity oracle，保留真实隔离 subprocess；
5. state adapter 4~5 个 helper/subset checks 合并成 1 个 real-model full round-trip；
6. vision alignment 6 个 checks 收敛为正向 + 一个真实错误入口；
7. 删除 `BatchedLinear.n_batches` caller-specific alias、trainer helper 的重复可调参数和 CI 里的重复版本字面量。

这组删减预计可以明显超过“只做格式清理”的量级，并接近/达到 **新增 UT + defensive/config surface 的 30%~40%**；核心生产算法不需要同步缩掉 30%。真正的衡量标准是减少事实源、非法状态和表示耦合，而不是追求总代码行数百分比。

## 7. Reducer 最终结论

**当前结论：需要修改后再合入。** 最高优先级不是继续补测试，而是先把 RED-01/RED-02 的 patch ownership 修正；随后用 RED-03~RED-07 把角色/配置/入口/测试维护面收敛。已有 review 对 Common MoE “抽象方向可接受”的判断可以保留，但严格按仓库 patch 生命周期，当前无 upstream counterpart 的 V4.1 字段不能因此继续留在 `patches/torchtitan`。

Reducer 自检：

- [x] 已先读 PR metadata、完整 diff、固定 TorchTitan 版本与 source branch；
- [x] 已读取 `.agents/AGENTS.md` 与 UT/ST review skill/原则文档；
- [x] 已逐类盘点新增 class/config/field/helper/state/test/dependency/entry；
- [x] 已显式标出 KEEP，避免把核心语义误删；
- [x] 已复核既有 R1-R10，未把旧 review 结论机械继承；
- [x] 已给出 P0/P1/P2、Test Reduction Matrix 与约 30% 删减方案；
- [x] 本次只允许修改 `pr_833/review.md`；不修改生产代码、测试、`master`，不 approve/merge/close PR。

质疑者review结论:

> 范围：严格按 `torchtitan-npu-reviewer-skills/Challenger.md` 对 GitHub PR #15 当前 head `b8db3fe9f785f5a85fe815855611d7c4aa336cee` 做独立静态审查；未执行测试。固定依赖为 `torchtitan==0.3.0` / `attn-gym==0.0.9`。同时核对了 V4.1 source branch `sdmyzlp/torchtitan:br_dpsk_v4_1`、Attention Gym v0.0.9、TorchTitan 已合入的 PR #3864（merge commit `183efed45d7d8d2cd880dd3803a3334ed47aba60`）与 CANN `sparse_lightning_indexer_kl_loss_grad` reference。GitHub 镜像没有历史 human inline review / review submission；作者在 PR 描述中的运行记录只作为 claim，不作为本 reviewer 的执行证据。

**Challenger 结论：暂停并澄清，不建议当前 head 合入。** 最关键的新问题不是 `IndexerKLLoss` 的 closed-form 本身——`Z * Y - p` 已由 source branch 与 CANN reference 相互印证——而是它接入的 **AuxLoss 框架并不是 source/upstream 要求的那套语义**。当前 backport 用 global batch size 归一化 token-additive KL，并在多层同名 loss 汇总时覆盖而不是累加；这会同时改变训练梯度尺度和 `indexer_kl_loss/mean` 的含义。另一个 correctness 缺口是 packed-document：`doc_ids` 只能保证“已经正确分组的 compressed entries”不跨文档，不能修复 compressor 本身跨 document boundary pooling；source branch 明确要求每个 document segment 对齐 compression grid，而本仓模型接口没有建立这个前置条件。

**测试执行：未执行（仅静态审查）。** 当前 GitHub mirror head 也没有可见 commit status / workflow run；PR 描述中的 “CI unit/smoke 均绿”无法从该 mirror head 独立复核。

## 1. Claim Traceability Matrix

| Claim | 实际生产路径 | Challenger verdict | 依据 |
| --- | --- | --- | --- |
| V4.1 跨层状态已从 mutable context 收敛为显式数据流 | `V41Model.forward -> block -> Attention/Indexer` 显式传 `cmp_k/idx_k/topk_indices/topk_scores/candidates` | **SUPPORTED** | 当前实现没有重新写回 module-level mutable attention context；UT 也覆盖 source/reuse/reindex threading。 |
| CSA2 改用 Attention Gym 后 LSE 可用于 marginal-weighted KL teacher | `CompressedSparseInnerAttention2 -> selected_attention(... return_aux=AuxRequest(lse=True)) -> IndexerKLLoss` | **SUPPORTED** | attn-gym v0.0.9 的 LSE 明确覆盖 sparse + local window + sink 全分母；shape 为 `[B,H,L]`。 |
| Indexer distillation 的梯度契约是 `Z*Y-p` | `IndexerKLLoss._teacher/_kl` | **SUPPORTED** | V4.1 source branch同样定义 marginal `p` 与 token-summed KL；CANN reference 明确计算 `p_reduce = p.sum(...)`、`ds = softmax * p_reduce - p`。之前“kernel 可能是 `Y-p`”的质疑已排除。 |
| 本 PR 对齐上游/source 的 Indexer KL 训练语义 | `IndexerKLLoss -> LoggedAuxLoss.inject` | **CONTRADICTED** | source branch 与 TorchTitan merged #3864 都按 step 的 `global_valid_tokens` 归一化；本仓 patch 按 `global_batch_size` 归一化，token-additive raw sum 的系数随每序列有效 token 数放大。详见 CHAL-01。 |
| `indexer_kl_loss/mean` 表示所有 consumer layer 的平均值 | `register_aux_loss_zero_hook -> LoggedAuxLoss.zero_all -> collect_aux_loss_metrics` | **CONTRADICTED** | 当前 `_step_acc[key] = module._acc.item()` 对同一 key 反复覆盖；V4.1 多个 `IndexerKLLoss` 同 key，最终只保留最后一层，再除以实例数。详见 CHAL-01。 |
| `doc_ids` 足以替代旧 metadata 并保证 packed-document isolation | `get_attention_masks -> Compressor -> Indexer._selection_mask -> selected_attention` | **PARTIAL / CONTRADICTED ON MISALIGNED PACKING** | source branch明确声明每个 document segment 必须是 `compress_ratio` 的整数倍；本仓 compressor 直接按全局 token grid `unflatten`，模型没有验证任意 packed boundary 对齐。详见 CHAL-02。 |
| 2P NPU ST 覆盖推荐/默认的 V4.1 分布式训练入口 | example / config 默认 `spmd_types`；integration case 强制 `partial_dtensor` | **PARTIAL** | 两条入口使用不同 SPMD backend，而 `parallelize_deepseek_v4_1` 与 `Module.parallelize` 对 backend 有不同路径。详见 CHAL-03。 |
| NPU ST 可守护新增 KL/MoE 数值语义 | `dsv41_debugmodel_2p_ep2_fsdp2` | **UNVERIFIED** | `check_loss=False`、无 `expected_steps`，runner 对该 case 只要求子进程 rc=0；不会检查 aux metric 或数值轨迹。详见 CHAL-04。 |
| Common MoE default-off 不改变 V4/V3.2 | package auto patch + existing model configs | **PARTIAL** | CPU config/行为检查支持 default-off；作者声称 V4/V3.2 anchor 一致，但本 reviewer 未执行，mirror 也无 workflow 证据。 |
| checkpoint/state adapter 已覆盖新 namespace | real V4.1 `state_dict` -> adapter | **UNVERIFIED** | 现有 UT 仍以手写子集为主，真实 model full-state round-trip 未形成。 |

## 2. Challenger Findings

| ID | 严重度 | Claim / Semantic Unit | 位置 | Claimed behavior | Actual path / 问题 | Trigger / Impact | Evidence | Fix | Verification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CHAL-01 | **Blocker** | Indexer KL normalization + metric aggregation | `torchtitan_npu/patches/torchtitan/models/common/aux_loss.py`；`patches/.../decoder.py`；`models/deepseek_v4_1/indexer.py` | PR 声称 `coeff=0.01` 对齐 source/upstream，指标为 `indexer_kl_loss/mean`。 | `IndexerKLLoss._kl()` 是对 token rows 的 raw **sum**；当前 `LoggedAuxLoss.inject()` 却用 `_mesh_scale_factor / global_batch_size`，而 source branch 与 merged TorchTitan #3864 都用 main loss 相同的 **global valid-token count**。因此 aux gradient 的有效系数随序列有效 token 数变化；当前 512-token recipe 下相对 source 约放大到“每序列有效 token 数”这一量级。与此同时 `zero_all()` 用 `_step_acc[key] = module._acc.item()`，多个同名 `IndexerKLLoss` 会互相覆盖，`collect_aux_loss_metrics()` 再除 `_group_counts[key]`，得到的是“最后一层 / 层数”，不是 layer mean。 | **训练总是触发**：V4.1 从第一个 selection consumer 起即挂多个 per-layer KL。影响不只是日志：归一化差异直接改变 24 个 indexer 参数的新增梯度尺度；metric 又不能准确反映实际各层 aux loss。 | source branch `IndexerKLLoss` 明写 “summed over rows and normalized by step global valid-token count”；TorchTitan merged #3864 的 trainer 调 `AuxLoss.set_step_denominator(global_valid_tokens)`，`group_acc[key] += instance_acc`；本仓 patch仍是旧 `global_batch_size` + overwrite 实现。现有 `test_indexer_distill_loss` 主要用 `global_batch_size=1` 和极小 token case，无法区分这两种 step normalization，也没有两个同名 loss 的 roll-up oracle。 | 直接按 **merged #3864 / V4.1 source branch** 更新这个 temporary upstream patch：trainer 用 global valid-token denominator；group accumulator 对每个 instance 做 `+=`；移除为 global batch size 注入而新增的 decoder monkey patch。由于 #3864 已合入 upstream，patch 应保持与其可机械对照，而不是继续维护 forked semantics。 | 新增/复用 upstream-style CPU test：同一 per-token loss 重复成 2× token 数后，归一化后的参数梯度保持不变；构造两个同 key 的 aux loss，值分别 `a/b`，step metric 必须是 `(a+b)/2`（含 reduce 后语义）。V4.1 ST 至少检查真实 `indexer_kl_loss/mean` 存在且 finite。 |
| CHAL-02 | **High** | Packed-document isolation | `compressor.py::Compressor.forward`；`indexer.py::_selection_mask`；`model.py::get_attention_masks`；`vision_loader.py` | PR/代码注释描述 `doc_ids` 替代旧 metadata 后可做 document isolation。 | Compressor 先对整条序列按固定 `compress_ratio` 直接 `unflatten`/pool；如果 document boundary 落在 group 中间，该 compressed entry 已经混入两个 document。Indexer 随后用 `doc_ids[:, ::ratio]` 给 entry 贴上起始 token 的 doc id，只能 mask selection，无法把已混合的 KV 拆开；Attention Gym 文档也明确 `doc_ids` 只约束 local-window branch，sparse indices 的跨文档合法性由 caller 负责。 | `compress_ratio=2` 时，只要 packed positions 在奇数 offset reset（例如一个 3-token doc 后接下一个 doc）就触发。影响是 compressed KV 跨 document 内容混合，后续 sparse attention 可能产生跨样本信息泄漏/错误 teacher mass。 | V4.1 source branch 的 indexer module docstring明确写出：每个 document segment 必须是 `compress_ratio` 的整数倍，这也是 compressor reshape segment-exact 的前提。本仓 `_document_alignment()` 只保证**当前 synthetic loader 的整行长度**对齐，且该 loader 一行只有一个 document；`test_index_selection_mask_is_document_isolated_and_causal` 使用的是 4+4 token、ratio=2 的已对齐边界，未覆盖反例。 | 两种方案二选一：A) 若暂不支持任意 packed docs，在真实模型/dataloader boundary 对每个 document segment 验证其长度对所有 active compression ratios 对齐，并在 docs 明确 contract；B) 若要支持任意 packing，则 compressor 必须按 doc segment 分组/pad，不能按全局 token grid reshape。不要只在 synthetic loader 校验 row length。 | CPU UT 用 ratio=2、document lengths `3 + 3`（或 positions `[0,1,2,0,1,2]`）构造明显不同的 KV，确认要么入口明确拒绝，要么 compressed entries 绝不混 doc；再保留一个 aligned positive case。 |
| CHAL-03 | **High** | Distributed backend support boundary | `config_registry.py`；`examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh`；`tests/integration_tests/deepseek_v4_1.py`；`parallelize.py` | README/example 是 V4.1 推荐训练入口，现有 2P case 被描述为该真实训练路径的 ST。 | `_v41_trainer_config()` 没有覆盖 upstream `ParallelismConfig.spmd_backend`，因此默认是 `spmd_types`；推荐 8P shell 又显式 `SPMD_BACKEND="spmd_types"`。但唯一 V4.1 integration case 强制 `--parallelism.spmd-backend=partial_dtensor`。`parallelize_deepseek_v4_1()` 对 `full_dtensor/spmd_types` 与 partial+EP 走不同分支，state/input redistribution 也不是同一机制。 | 任何按默认 config/example 使用 `spmd_types` 的真实用户路径。影响是现有 NPU ST 不能证明推荐入口的 sharding/redistribution/typecheck 路径可运行；同时 config 没有拒绝 `full_dtensor`，支持矩阵边界不清晰。 | TorchTitan v0.3.0 `ParallelismConfig.spmd_backend` 默认 `spmd_types`；example 显式同值；integration case 明确改成 `partial_dtensor`。当前 UT 不能证明真实 HCCL/NPU distributed backend。 | 先定义单一支持面：若产品入口就是 `spmd_types`，把**现有同一个** 2P ST 调成 `spmd_types`，无需新增第二 case；若只支持 `partial_dtensor`，则 recipe/example/docs 统一 pin partial，并在 model config 对其它 backend fail-fast。若确实支持两者，再分别说明必要性和独立路径，但不要默认无边界地都接受。 | 静态检查最终 recipe 与 ST backend 完全一致；真实 2P NPU 完成 forward/backward/optimizer step。若保留 `spmd_types`，开启其 typechecking 的最小 CPU/设备测试覆盖新增 multimodal/router metadata。 |
| CHAL-04 | **High** | NPU ST oracle | `tests/integration_tests/deepseek_v4_1.py`；`run_tests.py` | PR 描述声称 30/30 steps、loss 下降且 `indexer_kl_loss` 全程有限，并据此支撑新 KL/MoE 路径。 | case 设置 `check_loss=False`、`expected_steps=None`。runner 的 `_check_phase_results()` 在这两个条件下直接 return，因此自动门禁既不检查 30 个 step 是否真的记录，也不读取 loss/aux metric；只要训练子进程 rc=0 就算通过。 | Aux loss 即使按 CHAL-01 错尺度训练、metric 即使只记录最后一层/层数，只要没有立刻 NaN/崩溃，当前 ST 仍会通过。 | `tests/integration_tests/README.md` 自己也把该 case 定义为 smoke、无 loss compare；`run_tests.py` 的逻辑与此一致。 | 不增加 ST 数量。复用 `dsv41_debugmodel_2p_ep2_fsdp2`：至少设置 `expected_steps` 或等价完成检查；解析并要求 `indexer_kl_loss/mean` 出现且 finite；在 CHAL-01 修复后，为明确改变的 MoE/KL 新语义建立短 deterministic trajectory guard。 | CI runner 自动失败于缺 step、缺 aux metric、非 finite；记录实际 backend/override/seed。不要把一次人工日志观察继续当门禁。 |
| CHAL-05 | **Medium** | Upstream/source reproducibility | PR 描述、README、patch headers | PR 多次使用“上游 V4.1 / 对齐上游”及逐位 A/B 数字。 | 官方 `pytorch/torchtitan` 默认分支当前没有 `deepseek_v4_1`；实际可核对实现来自开发 branch（例如 `sdmyzlp/torchtitan:br_dpsk_v4_1`），但仓内没有固定 source commit。与此同时本仓实际依赖的 AuxLoss 已经与该 source/merged upstream 分叉。 | 后续 source branch 继续变化、或 reviewer 想复现“逐位无损”数字时触发。影响是“对齐上游”的对象和精度基准无法稳定重放。 | source branch 可直接找到 `HierarchicalIndexer.Mode`、packing-alignment contract 和 token-normalized `AuxLoss`；这些已经与当前移植形态存在可见差异。 | 在 PR/README 固定 reference repo + commit SHA，并列出有意偏离点。凡放入 `patches/torchtitan` 的 upstream backport，必须引用实际包含相同实现的 merged/pending PR/commit。 | re-review 时按固定 source commit 做 compare；A/B 数字注明命令、commit、dtype/backend 和比较对象。 |

## 3. Coverage Matrix

| Dimension | 状态 | Challenger 结论 |
| --- | --- | --- |
| Claim | **DEEP** | 关键 KL “对齐 upstream/source” claim 被 CHAL-01 反证；其余逐项追踪。 |
| E2E | **DEEP** | 主要 forward 可静态闭合，但推荐 `spmd_types` 入口与 ST backend 不一致。 |
| Numerics | **DEEP** | KL closed-form 正确；step normalization 错位，MoE 新 arithmetic 又无 NPU numeric guard。 |
| Bwd | **DEEP** | KL 新增 backward 路径存在独立小例子；但 step-level gradient scale 不符合 source framework。 |
| Dist | **DEEP** | EP2 case 存在，但使用 `partial_dtensor`；默认/recommended `spmd_types` 未被该 ST 证明。 |
| State | **DEEP** | 显式 runtime state 是改进；full state adapter/checkpoint 仍缺真实 round-trip。 |
| Compile | **PASS / N/A** | V4.1 明确拒绝 compile，边界清楚。 |
| Checkpoint | **DEEP** | integration 关闭 checkpoint，adapter 仅子集 oracle。 |
| Kernel | **PASS** | 当前 fused distill op未接入，边界明确；CANN reference 只用于核对公式，不把它算作本 PR NPU kernel coverage。 |
| Perf | **N/A / UNVERIFIED** | 本 PR 未提供可独立验证的性能目标；8P example 不是性能门禁。 |
| Compat | **DEEP** | Common default-off 有静态/CPU证据；patch lifecycle、source pin 和 backend support仍不完整。 |
| UT | **DEEP** | 算法小例子较强，但遗漏 aux framework normalization、多实例 metric、misaligned packed boundary。 |
| ST | **DEEP** | 唯一 V4.1 ST 为不同 backend 的 completion smoke，且不检查目标 metric。 |
| Docs | **DEEP** | “packed isolation”“上游对齐”“4k/512”等 contract 仍有漂移。 |
| Historical | **PASS** | GitHub mirror 无 human review thread/submission，不虚构历史人工结论。 |

## 4. Test Evidence Matrix

| 测试 / 证据 | 能证明什么 | 不能证明什么 | Challenger 判定 |
| --- | --- | --- | --- |
| `test_indexer_distill_loss.py::test_student_gradient_is_z_times_y_minus_p` | 单行、小 K 下 `Z*Y-p` 与 loss value | Trainer step denominator、不同 seq_len 下 coefficient invariance、多层 metric 聚合 | **有效但范围窄** |
| `test_indexer_distill_loss.py::test_per_layer_losses_sum_to_the_pooled_teacher` | shared student logits 上多个 teacher gradient 的线性叠加 | `LoggedAuxLoss.zero_all()` 是否把多个实例 metric 正确相加 | **有效但不覆盖 framework** |
| `test_attention_policy.py::test_index_selection_mask_is_document_isolated_and_causal` | 对齐的 4+4 token、ratio=2 下 selection mask | document boundary 不落 compression grid 时 compressor 是否跨文档 pooling | **反例未覆盖** |
| `test_training_contract.py::test_full_ac_preserves_image_routing` | FullAC 下 image mask 传递、same-implementation output/grad 一致 | 独立 MoE oracle、真实 EP/HCCL、recommended `spmd_types` | **CPU contract 有效** |
| `test_baseline_contract.py::test_v41_moe_rides_the_common_stack_with_an_opt_in_vision_bias` | Common router default-off 与 V4.1 opt-in config/CPU行为 | 真实分布式 score alignment / NPU numeric equivalence | **部分证据** |
| `dsv41_debugmodel_2p_ep2_fsdp2` 定义 | 静态上能进入 V4.1、2P FSDP2+EP2、reference operator path | reviewer 未执行；且 runner 无 step/loss/aux metric oracle，backend 还是 partial_dtensor | **只可计为定义完整的 smoke case** |
| PR 描述中的 CPU/NPU/CI 数字 | 作者自验证背景 | 当前 reviewer 的独立通过证据 | **不计作执行证据** |

## 5. Historical Human Review Traceability

GitHub PR #15 当前查询到的 issue comments、inline review comments / review submissions 均为空，因此**没有历史 human review 可标记为 RESOLVED / STILL_OPEN**。`review.md` 里已有 Maintainer/Reducer 章节属于本镜像上的审查报告，不冒充 human review。

对已有报告中与本次 Challenger 复核重叠的技术项：R2 `BatchedLinear` provenance、R3 RoPE override ownership、R4 入口 source-of-truth、R5 ST numeric guard、R7 real state round-trip、R8 private AC API 均仍可从当前 head 静态复现，状态为 **STILL_OPEN**；Reducer 对 patch placement 的 RED-01/02 也未被后续 production commit 修改。CHAL-01/02 是在这些既有项之外新增的 correctness finding。

## 6. UNVERIFIED / Review Boundary

- **未执行任何测试。** 不能把 PR 描述的 69/246/4 passed、2×910C 30 steps、CI green 当成本次 review 的通过结果。
- 当前 GitHub mirror head `b8db3fe9...` 没有可见 commit status / workflow run；CI claim 只能标记为作者声明。
- 推荐 8P `spmd_types` 路径未由现有 V4.1 integration case覆盖；8P 稳定性/性能未验证。
- 若 `full_dtensor` 也被视为支持，目前 config没有拒绝、ST也未覆盖，支持结论为 UNVERIFIED。
- checkpoint/save-resume、真实 full-state HF round-trip 未验证。
- `.agents` / test-review 基线要求读取 `.ci/lint.sh`，但当前 branch 的 `.ci` 不存在该文件；本次只能以 `requirements.txt`、`.ci/unit_test.sh`、`.ci/smoke_test.sh` 固定 TorchTitan v0.3.0。该规则/文件漂移应另行清理，但不把它伪装成本 PR 的运行结论。

## 7. Re-review Conditions

1. **先修 CHAL-01。** 将 AuxLoss backport 对齐 merged TorchTitan #3864 / 固定 V4.1 source commit：global-valid-token normalization、多实例 group sum、对应 trainer hook；删除旧 global-batch-size monkey-patch 语义。没有这一条，不应继续用当前 `coeff=0.01` 声称与 source 对齐。
2. **明确 CHAL-02 的 packing contract。** 要么模型/真实 dataloader 对每个 document segment 做 compression-grid alignment fail-fast，要么 compressor 改成 segment-aware；补 misaligned boundary oracle。
3. **统一 distributed backend。** 默认 config、推荐 example、唯一 2P ST 使用同一个被声明支持的 SPMD backend；不支持的 backend 在 config 入口拒绝。
4. **增强现有 ST，不新增组合。** 同一 `dsv41_debugmodel_2p_ep2_fsdp2` 自动检查 expected steps、`indexer_kl_loss/mean` 存在且 finite，并在新语义确定后加入最小 deterministic numeric guard。
5. 同时关闭既有仍 open 的 R2/R3/R4/R7 等架构项：patch provenance、Override ownership、单一 config/CLI source-of-truth、真实 full-state round-trip。
6. README/PR 固定 V4.1 source repo + commit，并同步 packing alignment、SPMD backend、512/4k 与 checkpoint 支持边界。

完成以上 P0/P1 后再进行 Challenger re-review；当前 head 不满足合入条件。

Challenger self-check：

- [x] 在读取既有最终结论前先按生产路径独立核对了 KL、Attention Gym、MoE、sharding、state、tests；
- [x] 核对了 PR metadata、完整 changed-file 集、固定依赖、`.agents/AGENTS.md`、test-review skills、UT/ST runner/README/example；
- [x] 核对了 pinned TorchTitan v0.3.0、Attention Gym v0.0.9、V4.1 source branch、merged TorchTitan #3864、CANN KL reference；
- [x] `tests/` 结论严格按仓内 developer-tests-review 的静态 `review测试` 边界，不声称执行；
- [x] 已将被证伪的早期 KL-kernel 怀疑移除：CANN reference 支持 `Z*Y-p`；
- [x] 仅追加 `pr_833/review.md`，不修改生产代码/测试/`master`，不 approve/merge/close PR。

Maintainer最终review结论:

> **本节是 PR #15 的最终权威 Maintainer 结论。** 前面的 Maintainer、Reducer、Challenger 内容全部原样保留，供追溯使用；若旧章节与本节冲突，以本节为准。本次 final review 严格按 `torchtitan-npu-reviewer-skills/Maintainer.md` 收口，对 reviewer 结论重新做事实核验、root-cause 去重、冲突仲裁、changed-file 覆盖和最终 merge gate，而不是对既有报告做机械汇总。

## 1. Final Decision

**合入建议：修改后重新 Review。**

当前不是「补测试即可合入」：至少存在三项 P0。第一，新增 `IndexerKLLoss` 接入了与其 token-additive 目标不兼容的旧 AuxLoss 语义，实际改变训练梯度尺度，并且同名多层 metric 会覆盖；第二，本 PR 修改的 Common MoE / Linear patch 含没有对应 TorchTitan upstream PR 的实现，违反本仓 `patches/torchtitan` 硬边界；第三，默认/推荐入口走 `spmd_types`，唯一 V4.1 NPU integration case 却强制 `partial_dtensor`，且没有 step / aux metric / numeric oracle，因此关键默认分布式训练路径没有有效 NPU 门禁。

PR 的主方向仍保留：V4.1 与 V4 model-specific 实现解耦、跨层共享状态显式穿参、Attention Gym 复用、mHC 修正、每层 Compressor/Indexer 的 source/reuse 生命周期，都是应继续保留的结构。需要修改的是它们周边的训练语义、patch ownership、入口和验证边界，而不是退回旧 mutable context / golden 专用实现。

**测试执行：未执行（仅静态审查）。** PR 描述中的 CPU/NPU/CI 结果仅作为作者自验证背景；当前 GitHub mirror 的 final-review head 没有可见 commit status/check run，本 review 不将这些声明计作独立通过证据。

## 2. Final Review Baseline

| 项目 | Final review 事实 |
| --- | --- |
| PR | GitHub #15 / GitCode !833 mirror |
| base | `master@5ffd25ecc173eed7f92f8178f8796738bbdf8368` |
| final-review 输入 head | `pr_833@92df773ddb30e074c0eadf6bf0c55b562841f4e8` |
| current PR files | 50 个；其中 `review.md` 是 review-only 追加，production/test/docs 主改动仍来自 `7b1bf45c3d71594b40d39f1d57b44ab84058c0ac` |
| 固定 TorchTitan | `requirements.txt` / `.ci/unit_test.sh` / `.ci/smoke_test.sh` 均指向 `torchtitan==0.3.0` / `v0.3.0` |
| Attention Gym | `attn-gym==0.0.9` |
| V4.1 reference source | 实际可核对来源是 `sdmyzlp/torchtitan:br_dpsk_v4_1`，但仓内未固定用于本次移植的 commit；moving branch 不能作为可重放 source-of-truth |
| CI 状态 | 当前 mirror head 无可见 status/check run；作者的 CI green 声明未独立复核 |
| baseline 规则漂移 | `.agents/AGENTS.md` 要求从 `.ci/lint.sh` 读取固定 upstream，但当前 branch 不存在该文件；本 review 以 requirements + unit/smoke CI 固定 `v0.3.0`，并把 lint 文件缺失记录为已有规则漂移，不伪装成 PR 测试结果 |

## 3. Reviewer Completeness Gate

| Reviewer | Mandatory delivery | Final gate |
| --- | --- | --- |
| Challenger | `质疑者review结论:`，含 Claim Traceability、Findings、Coverage、Historical、Test Evidence | **COMPLETE** |
| Reducer | `删减者review结论:`，含 Reducer Findings、Reduction Inventory、Historical Fix Reduction、Top Reduction Plan | **COMPLETE** |
| Architect | Maintainer.md 要求正式 `架构师review结论:`，并至少含 Architecture Findings / Boundary Matrix / Entry / Patch / Upgrade Risk | **REVIEW-GAP**：当前 `review.md` 没有正式 `架构师review结论:` 交付；不能用前面的旧 Maintainer 章节冒充 Architect delivery |

Architect delivery 缺失不再单独阻断本次 final gate，因为本节已经按 Maintainer.md 要求补做最小 Architecture Supplement：见第 7 节的 Architecture Boundary Matrix、第 5 节的 patch/entry/upgrade findings，以及第 6 节的冲突仲裁。该补审只覆盖本 PR 所需决策，不替代今后独立 Architect reviewer 的职责。

## 4. Historical Human Review Closure Matrix

GitHub PR #15 的 conversation comments、inline review threads、review submissions 均为空，因此没有历史人工技术意见需要标记为 RESOLVED / STILL_OPEN；Agent 生成的前述 Maintainer/Reducer/Challenger 报告不冒充 human review。

| Historical review | Source | Final status | 说明 |
| --- | --- | --- | --- |
| HUMAN-0 | GitHub PR #15 | **NOT_APPLICABLE** | 无历史 human inline review、review submission 或 conversation technical comment；不存在 developer response/fix diff 可做人工闭环 |

## 5. Final Finding Registry

| ID | Priority | Root cause / 位置 | Final Maintainer 判断 | 唯一修改方向 | Re-review 证据 |
| --- | --- | --- | --- | --- | --- |
| **R1** | **P0** | `IndexerKLLoss -> LoggedAuxLoss`：`patches/torchtitan/models/common/aux_loss.py`、`patches/.../decoder.py`、`deepseek_v4_1/indexer.py` | **实际训练 correctness 错误。** `IndexerKLLoss._kl()` 对 token rows 求和；当前 `LoggedAuxLoss.inject()` 却按 `global_batch_size` 缩放，而 merged TorchTitan #3864 与 V4.1 source 都使用 main loss 相同的 step `global_valid_tokens`。同时 `zero_all()` 对同 `(reduce_mesh, metric_name)` 使用覆盖赋值，多个 per-layer `IndexerKLLoss` 只留下最后一层，再除实例数。`Z*Y-p` closed-form 本身是正确的，问题在 AuxLoss framework。 | 将 #3864 temporary backport 恢复为 upstream-compatible contract：step 前由 Trainer 设置 `global_valid_tokens` denominator；instance accumulator 在设备端按 metric group `+=` 汇总，日志时每 group 再 `.item()`；删除 `Decoder.Config` 注入 `global_batch_size` 的旧语义。若 pinned v0.3 仍需 FullAC replay compatibility，作为独立 Extension/compat delta 隔离并单测，不继续把 normalization/group semantics fork 在 upstream patch 内。 | CPU：token 数翻倍但 per-token 数据相同，归一化后 indexer 参数梯度不变；两个同 key loss 值 `a/b`，metric 必须等于 `(a+b)/2`；NPU ST 中 `indexer_kl_loss/mean` 必须存在且 finite。 |
| **R2** | **P0** | `patches/torchtitan/models/common/moe.py`、`linear.py` | **硬 patch-policy 违规。** Common MoE 抽象方向可以接受，但 patch 文件自己明确承认 `vision_enabled/bias_vl/image_mask/sorted_topk` 没有 upstream counterpart；官方 TorchTitan 当前也搜不到这些字段。`BatchedLinear` 标注 #3634，但 #3634 merge commit 中根本没有该类；当前实现又与 source branch 的 `Config.n_batches + [B,O,I]` layout 不同。 | `patches/torchtitan` 只保留能与真实 upstream PR/commit 机械对照的部分。把无 upstream PR 的通用 MoE 扩展和 `BatchedLinear` 移到 `torchtitan_npu/extensions/models/common/`（若最终只服务 V4.1，则放 model scope），由 V4.1 显式 opt-in；不要复制整套 MoE。只有真实 TorchTitan PR 已提交且实现对齐后，才允许以 temporary patch 形式回迁，并写清删除条件。 | static：每个 patch 字段/类都能定位到真实 upstream PR/commit；Common extension 有 default-off + opt-in 行为 UT；`wo_a.weight` 最终 layout 纳入 state round-trip。 |
| **R3** | **P0** | `config_registry.py` / example / `parallelize.py` / `tests/integration_tests/deepseek_v4_1.py` / runner | **默认生产分布式路径缺有效 NPU gate。** TorchTitan v0.3.0 默认 `spmd_backend="spmd_types"`，V4.1 recipe 未覆盖该默认值，example 还显式写 `spmd_types`；唯一 2P case 却强制 `partial_dtensor`。两者在 `parallelize_deepseek_v4_1()` 中进入不同实现。该 case 又 `check_loss=False` 且 `expected_steps=None`，runner 对结果直接返回，只剩 rc=0。 | 本 PR 收敛成一个明确支持面：**V4.1 当前只声明 `spmd_types` 为支持的 SPMD backend**，`partial_dtensor/full_dtensor` 先在 config/model boundary fail-fast，后续独立 PR 再扩。把现有唯一 `dsv41_debugmodel_2p_ep2_fsdp2` 改成 `spmd_types`，不新增第二组合；给同一 case 增 expected steps、`indexer_kl_loss/mean` finite 检查，并在 R1 修复后建立短 deterministic 新语义轨迹 guard。 | 同一 2P NPU case 真实完成 init/forward/backward/optimizer step；backend 与 recipe/example 完全一致；缺 step、缺 aux metric、non-finite 或 deterministic guard 偏离均自动失败。 |
| **R4** | **P1** | `compressor.py` + `indexer.py::_selection_mask` + `model.get_attention_masks` + data contract | 当前 synthetic vision loader 每 row 单 document 且 row length 对齐，因此默认 recipe 没有直接触发；但代码/测试宣称 packed-doc isolation，模型却允许任意 `positions` reset。若 boundary 不落 compression grid，Compressor 在 mask 之前已把两个 document pooling 到同一 entry，后续 `doc_ids[:, ::ratio]` 无法修复。 | 本 PR 不扩成任意 packing 实现。把支持范围收窄为「document segment 必须按 model-derived compression alignment 对齐」；alignment 由 active ratios 单点派生，并在真实 packing/data producer boundary pad/validate，每个 segment 而不是只检查整 row。文档明确自定义 dataloader 也必须满足该 contract；不要在 Compressor/Indexer 深处堆重复 defensive checks。 | CPU 用 ratio=2 的 `3+3` document boundary 证明 producer 明确 reject/pad，再保留 aligned positive case；最终模型文档不再声称未约束的 arbitrary packing。 |
| **R5** | **P1** | `deepseek_v4_1/__init__.py -> override.common.WorkaroundComplexRoPE` | 模型 config 已静态选择 replacement，而入口仍声明 `override.imports=...rope.workaround`；因此 override 不再是实现选择的单一入口，模型层反向依赖 override 层。 | model registry 只持有稳定 base/split-aware `ComplexRoPE.Config`（结构能力在 model/Common Extension 承载）；workaround/AscendC 只能由 `override.imports` 显式替换。禁止 model config import 具体 workaround implementation。 | UT 从真实 `Trainer.Config.override.imports` 入口构建：无 override 为 base config；启用 workaround 后才发生 replacement；其它 override 可通过同一 seam 选择。 |
| **R6** | **P1** | `config_registry.py` + example + README | tokenizer/text/image/training values 同时来自 registry defaults、环境变量、shell 和 CLI；`DSV41_TOKENIZER_PATH`/`DSV4_TOKENIZER_PATH`/`DSV41_VISION_TEXT` 是隐藏 second source。example 名称写 `4k` 实际 `SEQ_LEN=512`，production recipe 默认 image path 又指向 `tests/assets`。 | tokenizer/text/image paths 全部进入 dataloader config/CLI；registry 只给 recipe default，不再直接读 hidden env；example 保留一个薄 shell，通过 CLI 覆盖，不新建额外 config/script；脚本直接重命名为 512 对应名称并同步 README。测试 asset 只留 debug/integration 默认。README 同时固定实际 V4.1 reference repo + commit，并列出本仓 intentional deviations。 | config parsing UT 证明 CLI 值到达真实 dataloader；静态核对 README/example/config 同一 seq_len/backend/input source；source commit 可重放。 |
| **R7** | **P1** | `state_dict_adapter.py` / `vision_state_dict.py` / `test_state_dict_adapter.py` | 当前 adapter 已有 text/vision ownership mapping，但 UT 使用手写 HF 子集，无法证明真实 registry model 的 compressor/indexer、vision blocks、marker、MoE `bias_vl`、grouped experts、最终 `BatchedLinear` layout 全部可逆。integration 又关闭 checkpoint。 | 在 R2 最终 ownership/layout 确定后，用 tiny **真实 registered model** 的完整 `state_dict()` 做 `to_hf -> from_hf` 全量 round-trip，严格检查 key set、shape、dtype/value；删除弱 ownership helper/deterministic duplicate。若不在本 PR 验证 DCP resume，README 必须把 V4.1 checkpoint resume 标为 **UNVERIFIED**，不能写成已验证支持。 | real-model full-state round-trip；明确旧 checkpoint compatibility boundary。DCP save/resume 若仍宣称已验证，则需同一代表 NPU case 或独立已有框架给出恢复证据。 |
| **R8** | **P2** | role/config/test scaffolding | 当前多个 bool (`owns_* / source_key / external_key / candidate roles`) 实际表达有限状态机，同时 model-level 又镜像 topology，导致 private builder validation matrix 和大量 representation tests。source branch 已有 `FULL/REINDEX/REUSE` 单一 Mode 证明可更简。 | role 收敛为单一 Mode/role；由 layer topology 一次派生；删除无 consumer 的 model-level mirror state 和 private topology defensive matrix。测试只保留真实 source→reuse、reindex、candidate 三个代表行为 + 独立数学 oracle；不要恢复 mutable state。 | 行为 UT 不再依赖 40 层 object identity/private tuple slot；公开用户边界（TP/CP/PP/compile、document alignment）仍保留 fail-fast。 |
| **R9** | **P2** | `model.py::apply_activation_checkpointing_extensions`、依赖/文档维护 | Vision AC 直接调用 pinned TorchTitan `_wrap_block()` 私有 API，升级脆弱；`attn-gym==0.0.9` 在多个依赖/CI位置重复字面量，CI workaround 无清晰删除条件。 | 把 private AC 适配集中到 Extension/compat seam，model 只声明 extra blocks；推动/复用 upstream public hook。依赖版本以单一 source-of-truth 为主，CI `--no-deps` workaround 从该事实源派生或明确删除条件。 | pinned v0.3 compatibility UT + source upgrade smoke；静态检查版本不漂移。 |

## 6. Reviewer Finding Normalization / Conflict Arbitration

| 输入结论 | Final arbitration |
| --- | --- |
| 旧 Maintainer R1 接受 Common MoE generalization vs Reducer RED-01 要求 MOVE | **MERGED_INTO R2**：Common 抽象可以保留，但「可 Common」不等于「可放 `patches/torchtitan`」。最终接受抽象、拒绝当前 patch placement。 |
| 旧 Maintainer R2 只要求修 provenance vs Reducer RED-02 要求移出 patch | **R2 采用更强结论**：仅改注释不够。没有真实 upstream PR 前必须移出 patch；若后续 upstream PR 存在，再按其真实 API/layout 机械回迁。 |
| 早期对 fused KL 可能是 `Y-p` 的怀疑 | **SUPERSEDED / NOT A FINDING**：CANN reference 明确 `p_reduce = p.sum(...)`、`dI = Y * p_reduce - p`，即 `Z*Y-p`。最终 R1 只针对 AuxLoss normalization/aggregation framework。 |
| 旧 R5/CHAL-03/CHAL-04 与 RED-08 | **MERGED_INTO R3**：不扩 ST 矩阵；只改现有唯一 2P case，使其覆盖真正默认 `spmd_types` 路径并具备 step/metric/numeric oracle。 |
| RED-05 希望删除 defensive validation vs CHAL-02 要求 packing check | **分别处理**：固定 private topology 的 defensive matrix 按 R8 删除；document alignment 是真实输入 boundary，按 R4 保留在 producer/public boundary，而不是深层重复校验。 |
| 旧 R7 与 RED-09 | **MERGED_INTO R7**：不叠 helper test，替换为一个 real registered model full-state round-trip。 |
| 旧 R3/R4/R8/R10 | 分别归入 **R5/R6/R9/R8**，状态均 STILL_OPEN；当前 production head 后续只追加过 review.md，没有 production fix 可关闭这些项。 |

## 7. Architecture Supplement for REVIEW-GAP

由于缺少正式 Architect delivery，final Maintainer 补做以下最小 Architecture Boundary Matrix；它也是本 PR re-review 的目标形态。

| 能力 | 当前 ownership | Final target ownership |
| --- | --- | --- |
| V4.1 topology / Compressor / Indexer / mHC / vision | `models/deepseek_v4_1` | **保留 model scope**；这是模型语义，不进入 TorchTitan patch |
| Attention Gym selected attention | pinned external dependency | **保留复用**，固定版本并由 model wrapper 适配 |
| AuxLoss #3864 | forked `patches/torchtitan` + decoder config hack | pinned TorchTitan 缺失期间保留**机械 upstream backport**；step denominator 的 Trainer 接线按 upstream contract；本仓兼容 delta 进入 Extension，不污染 patch 语义 |
| multimodal route bias / sorted top-k | auto-applied Common MoE patch，无 upstream counterpart | 若语义通用，放 `extensions/models/common`；V4.1 config 显式 opt-in；有真实 upstream PR 后才临时 patch |
| `BatchedLinear` | 标错 #3634 的 Common patch hybrid | `extensions/models/common` 或 V4.1 model scope；真实 upstream PR 前不在 patch |
| RoPE workaround | model 静态依赖 replacement + CLI override重复 | model 用稳定 base config；具体实现仅由 `override.imports` 选择 |
| Vision AC private hook | model 直接 `_wrap_block` | Extension/compat seam；model 只声明 extra block |
| tokenizer/text/image/backend | env + registry + shell + CLI | 单一 `Trainer.Config` / dataloader config + CLI；example 仅薄封装 |
| supported SPMD backend | default/example `spmd_types`，ST `partial_dtensor` | 本 PR 当前只声明 `spmd_types`；其它 backend fail-fast，后续独立扩展 |
| state mapping | model adapter + hand-written subset UT | model adapter + real registered model full-state oracle |

该目标保持 torchtitan-npu 是 TorchTitan 的 NPU 适配层，而不是形成新的 TorchTitan fork：上游已有能力复用；拟上游但 pinned 版本缺失的代码才进入 patch；本仓增强进入 Extension；实现选择走 Override；模型特有语义留在 model scope。

## 8. Final Coverage Gate

| Semantic unit | CPU / static evidence | NPU / distributed evidence | Final status |
| --- | --- | --- | --- |
| V4.1 registry 可独立于 V4 build + forward/backward | isolated subprocess real registry path | 当前 2P case 为不同 backend smoke | **PARTIAL** |
| 显式 cross-layer state / source-reuse-reindex | threading UT 覆盖 producer/consumer | smoke only | **PASS on CPU contract** |
| mHC contraction | float64 explicit branch-sum + permutation oracle | 无独立 NPU numeric oracle | **PASS on algorithm contract** |
| Attention Gym + KL teacher closed-form | `Z*Y-p`、invalid slot、pooled teacher 有独立小例子 | 现 case不检查 aux metric | **PARTIAL；被 R1 阻塞** |
| AuxLoss step normalization / metric mean | **缺失且当前生产实现已证实错误** | smoke 不会发现 | **FAIL / P0** |
| Common MoE default-off + V4.1 opt-in | model-level CPU 行为有证据 | primary backend numeric guard缺失 | **PARTIAL；R2/R3** |
| Vision + FullAC image routing | CPU checkpoint/non-checkpoint output/grad 对照 | 未覆盖 primary `spmd_types` NPU path | **PASS CPU / UNVERIFIED NPU** |
| packed document isolation | aligned 4+4 selection mask；synthetic row 单 document | 无 | **PARTIAL；misaligned segment contract 未闭合** |
| primary distributed path | static code显示 default/example=`spmd_types` | integration 强制 `partial_dtensor` | **FAIL / P0** |
| HF/local state lifecycle | hand-written subset round-trip | checkpoint disabled | **PARTIAL / P1** |
| compile / TP / CP / PP | config 明确 fail-fast | 不要求 ST | **N/A：当前明确不支持** |
| 8P稳定性/性能 | example only | 无门禁证据 | **UNVERIFIED，不作为当前 merge 通过依据** |

### Changed-file coverage audit

当前 50 个 changed files 已全部纳入 final review，没有按 deletion/test/doc 类型跳过：

| Surface | 文件数 | Final coverage |
| --- | ---: | --- |
| CI / dependency：`.ci/*`、`pyproject.toml`、`requirements.txt` | 4 | pinned TT/attn-gym、CI入口、重复版本事实源 |
| examples/docs：旧 readme 删除、新 shell、新 readme | 3 | entry、backend、512/4k、source pin、checkpoint wording |
| `review.md` | 1 | reviewer delivery / history / final gate |
| integration + loss asset | 4 | loss anchor 删除、V4.1 case、runner、README matrix |
| unit tests：旧目录删除 2 + 新 V4.1 UT 9 | 11 | UT正向功能、oracle、格式/表示耦合、state/packing缺口 |
| 旧 `models/deepseek_v41` 删除 | 10 | 与新 package 替代关系、旧 golden/reference/context 删除 |
| 新 `models/deepseek_v4_1` | 15 | registry/attention/compressor/indexer/mHC/model/data/vision/sharding/parallel/state 全链路 |
| changed Common patches | 2 | MoE / Linear patch ownership、default-off、provenance |
| **合计** | **50** | **覆盖完成** |

## 9. Priority Summary

### P0 — 合入前必须修复

1. **R1 AuxLoss correctness**：global-valid-token denominator + group sum，去掉 global-batch-size fork。
2. **R2 patch hard policy**：无 upstream counterpart 的 Common MoE extras / `BatchedLinear` 离开 `patches/torchtitan`，或先有真实 upstream PR 再机械回迁。
3. **R3 primary NPU gate**：支持面统一为 `spmd_types`；现有唯一 2P ST 改同 backend，并增加 step / aux metric / deterministic numeric guard。

### P1 — re-review 前必须闭合

4. **R4** document segment alignment contract 在真实 producer/public boundary 闭合。
5. **R5** RoPE implementation choice 回归 `override.imports` 单一入口。
6. **R6** tokenizer/text/image/backend 等训练输入收敛到 config/CLI，修正文档/source pin/512 命名。
7. **R7** real registered model full-state round-trip；checkpoint 未验证就明确标为 UNVERIFIED。

### P2 — 同 PR 建议收敛，不能反向扩大设计

8. **R8** Mode/role 单一事实源，删 derived topology/private defensive matrix 和 representation-heavy tests。
9. **R9** private AC API 移到 compat Extension，去重依赖事实源并写删除条件。

## 10. UNVERIFIED / Non-derivable Claims

- 本 final reviewer **没有执行** CPU UT、NPU ST、8P example、checkpoint resume 或性能测试。
- PR 描述中的「69 passed / 246 passed / 4 passed / 2×910C 30 steps / CI green」不是本次独立执行证据。
- 当前 mirror head 没有 GitHub status/check run 可独立复核。
- 8P `spmd_types` 稳定性、吞吐/内存没有自动门禁，不得由 2P partial-dtensor smoke 推导。
- V4.1 DCP save/resume 目前没有本模型证据；HF adapter subset 也不能证明完整 state lifecycle。
- AscendC fused sparse attention 与 fused indexer KL 当前明确未接入，本 review 不评价其本 PR 数值/性能正确性。
- `full_dtensor` / `partial_dtensor` 若未来要成为 V4.1 支持面，需要各自独立说明不同生产路径和最小 NPU 证据；本 PR final target 不默认宣称支持。
- moving `sdmyzlp/torchtitan:br_dpsk_v4_1` 不能作为可重放 baseline；必须固定实际 source commit 才能复核逐位 A/B 结论。

## 11. Re-review Conditions

下一轮 Maintainer re-review 只在以下条件满足后进行：

1. R1 production fix 完成，并有 token-denominator invariance + multi-instance metric aggregation UT；不得只改文档/系数。
2. R2 ownership 完成：当前两个 changed Common patch 中，本 PR 新增且无真实 upstream basis 的部分已经移出 patch；若选择 upstream 路线，必须给出实际 PR/commit 且实现可机械对照。
3. R3 支持面完成：recipe/example/config/2P ST 统一为 `spmd_types`，其它 backend fail-fast；同一 ST 自动检查完整 step、`indexer_kl_loss/mean` finite 和新预期 deterministic guard。
4. R4 packing contract 完成，并有 misaligned segment 反例；默认 synthetic single-document case 不再被当成 arbitrary packed-doc 证明。
5. R5/R6 入口完成：model 不直接依赖 workaround replacement；数据输入不读隐藏 env；README/example/config 单一事实源且 source commit 固定。
6. R7 state oracle 完成；如果 checkpoint resume 仍写成支持，需要给出实际恢复证据，否则文档明确 UNVERIFIED。
7. 修复提交后重新核对所有既有 V4/V3.2 路径的 default-off compatibility；Common 能力移动/拆分后不得引入新的 package-wide副作用。
8. 提交新的 reviewer 证据时，测试结果需注明实际 commit、命令、backend、NPU 数和结果；本文件前面的作者自验证数字不自动继承为新 head 的通过证明。

**最终 merge gate：修改后重新 Review。当前 `92df773d...` final-review 输入 head 不满足合入条件。**

Maintainer final self-check：

- [x] 读取 Maintainer prompt、PR metadata、50 个 changed-file surface、完整 diff、`.agents/AGENTS.md`、requirements/CI、test-review UT/ST/format/report 规则、docs 规则；
- [x] 查询全部 GitHub human review threads/submissions/comments，结果为空并显式记录；
- [x] Challenger / Reducer delivery 已读取；Architect formal delivery 缺失已标记 REVIEW-GAP，并由 final Maintainer 做最小架构补审；
- [x] 核对 pinned TorchTitan v0.3.0、merged #3864、merged #3634、V4.1 source branch、CANN KL reference；
- [x] 已做 root-cause 去重和冲突仲裁，旧结论不再作为并行 merge gate；
- [x] final section 只追加 `pr_833/review.md`，未修改 production/tests/master，未 approve/merge/close PR。
