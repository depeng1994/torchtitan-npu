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