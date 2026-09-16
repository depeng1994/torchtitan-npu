# PR !788 / Mirror PR #7 Maintainer Review

> 审查对象：`pr_788` → `master`，镜像 PR https://github.com/depeng1994/torchtitan-npu/pull/7  
> 原 PR：GitCode !788（DeepSeek-V4 LoRA training and adapter checkpoints）  
> 固定上游：`torchtitan==0.3.0` / `pytorch/torchtitan@v0.3.0`  
> 测试执行：**未执行（仅静态审查）**。PR 描述中的 2026-09-15 双卡实测作为开发者提供的外部证据记录，但不替代本次静态审查和入库测试定义。  
> 说明：以下“暂停并澄清”针对原功能变更的可接受性；mirror PR 最终合入仅用于归档本 review，不代表对原功能 PR 的 APPROVE。

## 1. 顶层结论

| 项目 | 结论 | 说明 |
| --- | --- | --- |
| 合入建议 | **暂停并澄清** | LoRA 主功能方向成立，dense / batched / routed-expert 的公式、checkpoint resume、PEFT consumer 均有较强测试投入；但当前实现新增第二套 `MODULE/CONFIG` 训练入口，和仓库既有 `ExtensionConfig + CLI` 扩展机制冲突；同时混入与 LoRA 无关的 LI quantization 和 mHC schema 兼容变更，并且新增的 EP/FSDP LoRA sharding 没有落成仓内多 NPU ST。 |
| 架构 | **阻塞** | `torchtitan_npu/models/deepseek_v4/lora_config.py` 把 LoRA 做成独立 recipe/module。仓库 `torchtitan_npu/config/configs.py::ExtensionConfig` 已明确把新增语义组暴露为 `--extension.*` CLI；应复用该机制，保持 DeepSeek-V4 单一训练入口。 |
| 上游解耦 | **阻塞** | `models/deepseek_v4/lora.py` 同时承载通用 LoRA Linear/BatchedLinear/GroupedExperts 实现和 DSV4 target policy，并依赖上游私有 `_lora_adapter_sharding`、PyTorch 私有 `_is_non_strict_tracing`；升级耦合过重。 |
| PEFT 导出 | **阻塞** | final PEFT export 把训练用 `checkpoint.initial_load_path` 直接写成 `adapter_config.json.base_model_name_or_path`；当它是 native DCP 时，代码自己只 warning，但仍写出一个 Transformers/PEFT 不可自动加载的 base 路径。训练基座路径和 PEFT/HF 基座标识必须解耦。 |
| 测试 | **阻塞** | 已提交 ST 只有 `dsv4_lora_1rank`；但生产改动新增 routed-expert LoRA 参数 placement 和 DTensor/PEFT export 路径。PR 描述声称 EP2/FSDP2 手工验证，但该组合没有进入 integration runner，无法成为长期门禁。 |
| Scope | **阻塞** | A5 CPT/QAT LI quantization 参数修改、mHC fake schema 宽松兼容均与 LoRA PR 背景无关，应拆成独立 PR，各自给背景、风险和测试。 |
| patches 边界 | **符合** | 本 PR 没有把新增 LoRA/NPU/model-specific 代码塞进 `patches/`；这一点符合仓库边界。 |

## 2. 主要问题与修改建议

| ID | 严重级别 | 代码位置 | 问题点 | 影响 | 建议修改方案 |
| --- | --- | --- | --- | --- | --- |
| R1 | P0 | `torchtitan_npu/models/deepseek_v4/lora_config.py`；`docs/feature_guides/deepseek_v4_lora.md`；PR usage | 新增 `MODULE=torchtitan_npu.models.deepseek_v4.lora_config CONFIG=deepseek_v4_lora*`，形成第二套 DSV4 训练入口。仓库已有 `ExtensionConfig`，其注释明确约定新增语义组通过 nested dataclass 形成 `--extension.*` CLI；`TrainerEx` 也已有 CLI parse 后、Trainer 初始化前应用扩展的时机。 | 后续每个模型能力都可继续复制 `*_config.py`，训练入口扩散；基础 recipe、LoRA recipe 的参数/修复容易漂移，违反“入口单一、CLI 暴露能力”的架构目标。 | 删除 production `lora_config.py`。新增 `ExtensionConfig.lora`（enable/rank/alpha/rank_experts/targets/include_mtp/chunk 等），在现有 `TrainerEx` 的 pre-init 扩展点通过 model-owned converter/registry 应用 `DeepSeekV4LoRAConverter`；checkpoint PEFT policy 也由同一 CLI 配置投影。用户继续使用 `MODULE=torchtitan_npu.models.deepseek_v4 CONFIG=deepseek_v4_flash`，只追加 `--extension.lora.*`。 |
| R2 | P0 | `torchtitan_npu/models/deepseek_v4/peft.py::_save_last_step`；`state_dict_adapter.py::peft_adapter_config` | `_save_last_step()` 调用 `peft_adapter_config(base_model_name_or_path=self.initial_load_path)`；当 `initial_load_path` 是 native DCP 时，随后只打印 warning。也就是说导出的 `adapter_config.json` 主动写入一个 PEFT/Transformers 不可作为 base model 使用的路径。文档反而要求加载时显式传 HF base，说明该字段语义本身已不可信。 | 导出 artifact 元数据不自洽；`AutoPeftModel*`/自动基座解析会被误导，产物可移植性差。 | 将训练恢复基座与 PEFT 元数据拆开。增加明确 CLI 字段（如 `--extension.lora.peft-base-model-name-or-path`，或 checkpoint-owned 等价字段）；仅当 `initial_load_in_hf=True` 时才允许从 `initial_load_path` 推导。native DCP 场景必须提供 HF/Transformers 可加载标识，或使用真实 `hf_assets_path` 且确认其就是模型基座，而不是在导出后 warning。增加 UT 断言 native DCP 路径不会写入 `adapter_config.json`。 |
| R3 | P0 | `torchtitan_npu/models/deepseek_v4/lora.py` | 文件混合“通用 LoRA module implementation”和“DeepSeek-V4 target/export policy”。其中 dense/batched adapter、GroupedExperts adapter、chunking、adapter sharding 都不是 DSV4 policy；同时直接 import `torchtitan.components.lora._lora_adapter_sharding` 和 `torch.compiler._is_non_strict_tracing` 两个私有符号。 | 与上游 TorchTitan/PyTorch 内部实现强耦合；上游升级时容易 break，也阻碍其他模型复用。 | 把可复用实现移动到 `torchtitan_npu/extensions/components/lora.py`（目录直接镜像 upstream `components/lora.py`）或优先向 TorchTitan upstream 贡献；DSV4 目录只保留 target 列表、模型特定 converter policy/PEFT alias。不要依赖上游 `_...` 私有 helper；需要的 sharding contract 应通过公开 hook/extension 暴露，或在 extension 中封装成仓库自有稳定接口。`_is_non_strict_tracing` 同理，优先使用公开 tracing/compile contract。 |
| R4 | P0 | `torchtitan_npu/models/deepseek_v4/sharding.py::_GROUPED_EXPERTS_PARAM_LAYOUT`；`state_dict_adapter.py::to_peft`；`tests/integration_tests/deepseek_v4.py::dsv4_lora_1rank` | 生产代码新增 `w13_lora_* / w2_lora_*` sharding，并在 PEFT export 对 DTensor 做 expert/rank flatten + redistribute；但仓内唯一 LoRA ST 是 1 NPU。PR 描述中的 EP2/FSDP2 仅是一次手工验证，未进入 runner。 | 最容易受 mesh/placement、expert local/global shape、DCP consolidation 影响的路径没有长期门禁。CPU/Gloo UT 不能证明真实 NPU EP/FSDP collective。 | 不建议再叠一个重复 case；直接把 LoRA integration 主 case 提升为 2 NPU EP2/FSDP2（或若必须保留 1rank，再增加一个只覆盖分布式差异的最小 case）。至少真实执行 model build→parallelize→forward/backward→adapter update→final PEFT export；若成本允许继续复用两阶段 `check_resume`，同时守住 distributed DCP resume。case 必须注册在 `build_deepseek_v4_test_list()` 并由 `models` suite 选中。 |
| R5 | P0 | `examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a5.sh`；`examples/deepseek_v4/deepseek_v4_flash_qat_4k_a5.sh`；`torchtitan_npu/ops/ascendc/mhc.py`；`tests/smoke_tests/ops/test_mhc_meta_registration.py` | PR 背景是 LoRA，却额外改变 A5 CPT 的 LI 为 FP8、QAT 的 LI 从 FP8 改 MXFP4，并放宽 mHC fake schema。它们是独立运行时行为变更，PR 描述未解释为什么与 LoRA 必须原子合入。 | 出现回归时无法归因；量化数值和 op-plugin schema 兼容风险被 LoRA 大改掩盖。 | 三项从本 PR 移除，分别提交量化 recipe PR 和 mHC compatibility PR。各自补背景、支持版本/上游 issue、独立测试。LoRA PR 只保留 LoRA 必需变更。 |
| R6 | P1 | `torchtitan_npu/models/deepseek_v4/lora.py::GroupedLoRAExperts.forward` | 新 forward 接受 `**dimensioned_kwargs`，只认 `routed_scores_R`，并在 W2 前额外乘 routed score。固定上游 v0.3.0 的真实 `RoutedExperts.forward` 只以 `(routed_input_RD, num_tokens_per_expert)` 调 `inner_experts`，score 在 token dispatcher `combine()` 阶段处理；本 PR 没有新增生产 caller 传 `routed_scores_R`。UT 却直接构造了该额外输入。 | 形成没有真实 producer 的第二套语义；未来如果 dispatcher 仍执行 score combine，误接此参数可能造成重复加权。测试保护的是人工接口而非实际调用链。 | 保持与 pinned upstream `GroupedExperts.forward(x, num_tokens_per_expert)` 同签名，删除 `routed_scores_R` 和任意 kwargs 防御分支。若确实有目标 override 需要 pre-W2 score，必须先落真实 producer/callsite 并解释为何与 dispatcher combine 不重复，再按真实表示写测试。 |
| R7 | P1 | `tests/unit_tests/models/deepseek_v4/test_lora.py` | 新增约 859 行单文件，同时测试 `lora.py`、`lora_config.py`、`peft.py`、`state_dict_adapter.py`；并存在多条 `test_review_*` 永久测试名。仓内 test-review 规则要求 UT 目录按 production module 镜像组织，测试名表达产品行为而非 review 过程。 | 可维护性差；失败定位需要在超大文件中辨别四类产品语义；review/process 命名会长期污染测试 API。 | 拆成 `test_lora.py`、`test_peft.py`、`test_state_dict_adapter.py`，CLI 重构后 config 测试落到对应 extension/config 测试文件；将 `test_review_*` 改成产品行为名。公共 fixture 只提取确有重复的最小准备。 |
| R8 | P1 | `tests/unit_tests/models/deepseek_v4/test_lora.py`（registry/config entry tests） | 现有 UT 对公式、converter、PEFT consumer 的独立 oracle 很好，但 production LoRA entry 的 UT 主要停在 config/model spec 构建、target module 单独 build，没有一条 CPU UT 从真实 registry/config 生成可运行的小模型并进入主 forward/backward。 | converter tree wiring、冻结策略和完整模型 module composition 的错误只能等 NPU ST 发现。 | 增加一条最小 debug-model CPU 正向功能 UT：真实 `model_registry(... converters=[LoRA])` → build 小模型 → 固定输入 forward/backward → 断言 LoRA B/A（按模式）有梯度且 base 无梯度，并用无 LoRA base + 显式 delta 或独立小 reference 检查主要输出。不要复制完整 trainer。若 DSV4 CPU 主 forward 确实被设备算子阻断，应在报告中明确该限制，并由真实 NPU integration 作为唯一入口证明。 |
| R9 | P1 | `tests/integration_tests/README.md` LoRA row | 表格把 `dsv4_lora_1rank` 的 `Check Loss` 写成“是，含 grad_norm”，但 case 实际 `check_loss=False`、`check_resume=True`；runner 因此不读取仓内 golden loss，而是用第一阶段作为动态基线比较 resume 的 loss/grad_norm。README 后文对 dynamic baseline 的描述又是正确的。 | 文档和代码事实冲突，容易把 resume A/B 比较误认为仓内静态 golden。 | 改为 `Check Loss=否`，原因写“`check_resume=True`，动态比较 phase 0/1 的 loss + grad_norm，不读取 golden 文件”；如果真要静态 golden，再把 case 改成 `check_loss=True` 并提交稳定 golden。 |
| R10 | P2 | `torchtitan_npu/models/deepseek_v4/peft.py::_finalize_peft_directory` | 继承的 TorchTitan checkpointer 使用 `torchtitan.tools.filesystem` 抽象，而 PEFT finalize 重新用 `os.listdir/os.replace/os.remove/shutil.rmtree/open`，把 final export 隐式限制到本地 POSIX 路径。 | 如果 checkpoint backend/URI 不是普通本地路径，native checkpoint 可工作而 PEFT finalization 会失败，扩展破坏 base manager 的存储 contract。 | 优先复用 TorchTitan filesystem/HF writer/consolidation hook；若 PEFT export 明确只支持 local FS，则在 Config 初始化时显式校验并在文档声明，而不是运行到最后一步才由 `os.*` 失败。 |
| R11 | P2 | `tests/smoke_tests/models/deepseek_v4/test_lora.py` | smoke test 通过 toy model + `object.__new__` 手工拼 CheckpointManager，重复验证许多 checkpoint selection/roundtrip 语义；它没有进入真实 DSV4 trainer/config，因此不能替代 integration ST。 | 测试层级混杂，增加 NPU smoke 时长却没有增加真实模型路径证明。 | 纯 state-selection / key-filtering 逻辑放 CPU UT；真实 NPU save/load/export 留在 integration runner。只有确实需要真实 NPU DCP primitive 的最小边界才保留 smoke，并在测试名/注释里写清它不证明 DSV4 ST。 |
| R12 | P2 | `torchtitan_npu/ops/ascendc/mhc.py::_fake_mhc_pre_backward/_fake_mhc_sinkhorn_backward`；对应 smoke test | 用 `*_` 接受任意未来 positional arg，测试甚至刻意覆盖“15 = future append”。这会让未知 schema 增长静默通过，即使未来新增参数改变 meta 输出语义。 | 防御式兼容掩盖真实 contract 变化；op-plugin schema 演进后可能不再 fail-fast。 | 在独立 mHC PR 中只显式支持当前已知新增字段（如 `inner_precise`），并锚定明确 torch_npu/op-plugin 版本或上游 issue；未知 schema 继续报错。除非上游 contract 明确保证所有未来尾参都不影响 shape/meta，才允许 varargs，并把该 contract 链接到代码。 |

## 3. PR 背景与实现设计核对

| PR 声明/背景 | 代码事实 | Review 结论 |
| --- | --- | --- |
| dense + batched + routed-expert 都需要 LoRA | `lora.py` 对 `Linear`、现有 `BatchedLinear` 和 `GroupedExperts` 都构造 adapter；grouped expert 使用 expert-aware grouped MM。 | 需求成立；公式级实现有较充分 UT，但通用实现应从模型目录上提到 extension/upstream。 |
| 并行化后、optimizer 前冻结 base | `__init__.py::_parallelize_lora` 先调用 `parallelize_deepseek_v4()`，然后按 `"lora_"` 名称冻结其余 parameter；LoRA registry 分支关闭 `post_optimizer_build_fn` 的 MoE balancing hook。 | 时序与“sharding 后冻结”背景一致；routing bias 固定也与关闭 balancing hook 的设计一致。建议把“是否 LoRA”从独立 config module 改成 CLI extension 后保留该模型 policy。 |
| periodic native DCP 只保存 adapter + routing buffer + training state | `DeepSeekV4PEFTCheckpointManager._flattened_model_states_sd()` 选择 LoRA、trainable params、model buffers，并按 `save_training_state` 选择 non-model state；production LoRA config 把 `save_training_state=True`。 | 主语义成立；但 manager 放在 model 目录且 final PEFT 本地 FS/基座元数据存在问题。 |
| resume 先固定 base，再恢复 adapter/state | `dcp_load()` 对 adapter-only checkpoint 先递归加载 `initial_load_path`，再加载保存的 adapter/non-model keys。 | 设计合理；1-rank integration 有动态 loss/grad_norm resume 检查。需要 EP2/FSDP2 持续门禁。 |
| final export 兼容 PEFT 0.20 / Transformers 5.17 | optional deps 精确 pin；UT 有实际 Transformers + PEFT consumer，并与独立 merged model logits 比较。 | consumer oracle 很强；但 `base_model_name_or_path` 写 native DCP 的元数据问题必须修。 |
| CI smoke 单 NPU，multi-NPU 单独手工验证 | 仓内 case 只有 `dsv4_lora_1rank`；PR 描述给出 2026-09-15 2×910B3 结果。 | 手工记录可作为开发证据，不是长期回归保护；本 PR 修改了分布式 sharding，必须落成 integration runner case。 |

## 4. 语义变换与独立 oracle

| 语义变换 | 生产代码和应观察结果 | PR 中独立 oracle | 状态 | 修改建议 |
| --- | --- | --- | --- | --- |
| Dense/Batched LoRA `base + scale * B(A(x))` | `lora.py::_get_linear_lora_cls`；B zero-init 时初始输出等于 base，训练后 adapter 产生 delta。 | CPU UT 显式计算 reference output/gradient，并覆盖 dense/batched。 | 已覆盖 | 保留 oracle，重构文件位置后迁移。 |
| Routed expert gate/up/down LoRA | `GroupedLoRAExperts` 对 w13/w2 做 grouped MM adapter delta。 | CPU UT 按 expert 拆分构造 reference 并比较 output/grad。 | 部分覆盖 | 移除无真实 producer 的 `routed_scores_R` 测试分支；实际生产签名下保留 grouped reference。 |
| Freeze base / 只训练 adapter | `_parallelize_lora` 在 parallelize 后冻结非 `lora_`。 | integration test wrapper snapshot base/adapter；CPU UT 也覆盖 A/B 与 B-only。 | 已覆盖（单卡） | distributed case 再确认 sharded base 无 grad/adapter 更新。 |
| Adapter-only native DCP resume | `peft.py::dcp_load/_flattened_model_states_sd`。 | 1-rank integration 两阶段动态比较 step 2 loss + grad_norm；UT 有 key/tensor roundtrip。 | 已覆盖（单卡） | 将主 ST 提升到 EP2/FSDP2。 |
| PEFT mapping + consumer | `state_dict_adapter.py::to_peft/peft_adapter_config`。 | 实际 PEFT/Transformers consumer + independent merged-model logits。 | 已覆盖（单卡/CPU consumer） | 增加 distributed export ST，并修正 base metadata。 |
| LoRA + 真实 DSV4 registry 主 forward | converter 注入完整 config tree 后应由主模型消费。 | 当前 UT 多为 target module / config 构建；真实完整训练只在 NPU integration。 | 部分覆盖 | 增加最小 CPU main-forward UT；若设备限制不可行，明确记录并由 integration 兜底。 |

## 5. UT 正向功能覆盖

| 正向功能 | 生产代码和应观察结果 | PR 中的直接检查 | 状态 | 合入前置条件 |
| --- | --- | --- | --- | --- |
| Linear/Batched LoRA 数值与梯度 | Adapter delta 与显式矩阵公式一致。 | `tests/unit_tests/models/deepseek_v4/test_lora.py` 中 dense/batched output+grad reference。 | 已覆盖 | 重构后保留。 |
| GroupedExperts LoRA 数值与梯度 | 每个 expert 的 A/B delta 与 grouped MM 等价。 | grouped expert per-expert reference、clamp/gradient 检查。 | 部分覆盖 | 删除人工 `routed_scores_R` 路径后按真实 upstream caller 重新锚定。 |
| Converter target/strict/quantization guard | target 命中准确；不支持量化时早失败。 | converter target、strict/non-strict、empty、quantization reject。 | 已覆盖 | 保留；通用实现上提 extension。 |
| A/B 与 B-only 训练 | 可训练参数发生更新，冻结参数不更新。 | CPU optimizer step 覆盖两模式。 | 已覆盖 | 保留。 |
| Native adapter checkpoint | 只选择需要状态并可恢复。 | 多个 toy-manager/key-set/roundtrip UT。 | 部分覆盖 | 内部 selection UT 可保留，但至少一条正向构造应走真实 Config/build；真实设备语义由 integration 证明。 |
| PEFT FQN/shape/config | dense/batched/expert 映射正确、rank 约束正确。 | mapping/alias/config/mismatch tests。 | 已覆盖 | 增加 base path metadata 断言。 |
| PEFT 外部消费者 | PEFT 0.20 + Transformers 5.17 可加载，logits 对独立 merge。 | 实际 consumer test。 | 已覆盖 | 保留 exact pinned dependency 环境。 |
| Production LoRA entry 主 forward | 真实 registry/converter/build 后主模型前反向。 | 没有 CPU main-forward oracle。 | 部分覆盖 | 见 R8。 |

## 6. ST 触发判断

| 生产代码变更 | 受影响训练场景 | 为什么需要 ST | 现有代表测试 | 判断 | 合入前置条件 |
| --- | --- | --- | --- | --- | --- |
| Dense/Batched/Grouped LoRA 进入 DSV4 trainer | DSV4 LoRA forward/backward/optimizer | 真实 NPU dtype、模型组合、optimizer wiring 不能由 CPU module UT 完全证明。 | `dsv4_lora_1rank` | 复用但不足 | 主 case 改为/补成 EP2+FSDP2，仍保留真实训练完成性与 adapter update 检查。 |
| `_GROUPED_EXPERTS_PARAM_LAYOUT` 新增 LoRA placement | EP2/FSDP2 routed experts | mesh/placement、local expert shape 和 collective 是真实 NPU 分布式语义。 | 无 LoRA 多卡 case | **新增/调整** | 2 NPU EP2/FSDP2 integration case，进入 `models` suite。 |
| DTensor expert adapter → PEFT flatten/redistribute | 多卡 final export | CPU/Gloo key mapping不能证明 NPU DTensor layout/consolidation。 | PR 描述有一次手工 EP2/FSDP2；仓内无 case | **新增/调整** | 同一个 2-NPU LoRA case 完成 final PEFT export 并核对导出 adapter，不再另建重复 ST。 |
| Adapter-only DCP resume | 单卡/多卡 resume | DCP metadata、optimizer state、分片恢复需真实设备/分布式。 | `dsv4_lora_1rank` 动态 loss+grad_norm resume | 部分覆盖 | 若 2-NPU case保留两阶段，则一并覆盖；否则至少保留现有 1-rank resume 并给 2-NPU case覆盖 sharding/export差异。 |
| mHC fake schema 变更 | op-plugin / torch_npu import + fake/meta registration | 真实依赖边界与 schema 版本有关。 | `tests/smoke_tests/ops/test_mhc_meta_registration.py` | 拆 PR | 独立 PR 中锚定已知 schema，不接受任意未来尾参；必要时真实 fake/meta operator invocation。 |
| A5 quantization CLI 改动 | A5 CPT/QAT | 改变实际 LI quantization recipe，属于独立训练行为。 | 本 PR 无针对新增参数的 integration case | 拆 PR | 独立量化 PR 按该脚本实际使用路径验证并更新说明。 |

## 7. NPU ST 事实表

| 测试 | 模型/配置 | 并行数值 | 替换实现/融合算子 | 编译模式 | NPU 数 | 启用与完成检查 | golden | 执行阶段 |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| `dsv4_lora_1rank` | test-only `lora_config` → production LoRA debug config | EP1/FSDP1 | Golden/reference overrides；真实 LoRA model/checkpointer | eager | 1 | 两阶段真实训练；snapshot 冻结参数/routing bias；A/B 更新；final PEFT tensor 对比；runner 比较 resume loss+grad_norm | **无静态 loss golden**；`check_loss=False`，phase 0 为动态基线 | 已提交、静态确认进入 `models` suite；本次未执行 |
| 拟调整 `dsv4_lora_ep2_fsdp2` | 同一 production LoRA debug config（CLI refactor 后） | EP2 + FSDP2 | 同上 | eager | 2 | 至少 forward/backward + adapter update + final distributed PEFT；优先继续两阶段 resume | 动态 resume baseline 即可；无需为相同语义再造 loss 文件 | **缺失，合入前置条件** |

## 8. 测试格式/组织问题

| 位置 | 类型 | 代码阅读结果 | 修改建议 |
| --- | --- | --- | --- |
| `tests/unit_tests/models/deepseek_v4/test_lora.py` | 文件位置 | 单文件跨 4 个 production module，未按正式代码目录/模块一一对应。 | 拆为对应 `test_lora.py/test_peft.py/test_state_dict_adapter.py/...`。 |
| 同上，多条 `test_review_*` | 命名 | 测试名记录 review 过程而不是稳定产品行为。 | 改成 `test_<动作>_<条件>_<结果>`。 |
| `tests/smoke_tests/models/deepseek_v4/test_lora.py` | 测试入口 | `.ci/smoke_test.sh` 会收集，但 toy manager 不是 integration runner 中真实 DSV4 训练入口。 | 不计作 ST；纯逻辑迁 CPU UT，设备边界才留 smoke。 |
| `tests/integration_tests/deepseek_v4.py` | integration runner 收集 | `dsv4_lora_1rank` 已由 `build_deepseek_v4_test_list()` 返回，`models` suite 静态包含；`ngpu=1` 不会证明新增的多卡 sharding。 | 将主 case 提升到 2 NPU 或新增唯一差异 case。 |
| `tests/integration_tests/lora_config.py` | 辅助对象 | test-only Trainer 包装 `super()` 并加入 snapshot/export assertions，作为 oracle 有价值；但依赖 production `lora_config.py` 第二入口。 | CLI refactor 后测试仍可保留 assertion wrapper，但模型 recipe 应从标准 DSV4 config + `extension.lora` 建立。 |
| `tests/smoke_tests/ops/test_mhc_meta_registration.py` | 测试结构 | 直接调用 fake helper，且刻意把“未知 future append”定义为应成功。 | 独立 PR 只测试已知 schema；未知 schema fail-fast。 |

## 9. 逐文件审查

| 文件 | 审查结论 | 具体问题/建议 |
| --- | --- | --- |
| `docs/feature_guides/deepseek_v4_lora.md` | 需修改 | 内容基本完整，但围绕独立 `lora_config.py` 第二入口展开；CLI 架构调整后重写。文档已明确 native DCP 不能作为 PEFT base，这反向证明 `adapter_config.json` 不应写 native `initial_load_path`。 |
| `examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a5.sh` | 拆 PR | 新增 `--extension.quantization.li-quantization fp8` 与 LoRA 无关；需独立量化背景/验证。 |
| `examples/deepseek_v4/deepseek_v4_flash_qat_4k_a5.sh` | 拆 PR | LI 从 `fp8` 改 `mxfp4` 是训练行为变化；独立 PR。 |
| `examples/deepseek_v4/readme.md` | 可随 LoRA 文档保留 | 仅增加指南链接；最终链接应指向 CLI-refactor 后的指南。 |
| `pyproject.toml` | 基本合理 | `peft==0.20.0` / `transformers==5.17.0` 作为 optional extras 与产物兼容目标一致；继续保持可选，不应进入核心依赖。 |
| `tests/integration_tests/README.md` | 需修改 | LoRA row 的 `Check Loss` 与 `check_loss=False` 不一致，见 R9；多卡覆盖矩阵也应体现 LoRA EP2/FSDP2。 |
| `tests/integration_tests/deepseek_v4.py` | 部分通过 | case 正确注册并用 `check_resume` 比较动态 loss/grad_norm；但仅 1 NPU，未覆盖本 PR sharding 差异。 |
| `tests/integration_tests/lora_config.py` | 有价值但需跟架构调整 | wrapper 用于构造 base DCP、冻结快照、adapter 更新、PEFT tensor oracle是有效的 test helper；不要让它固化 production 第二 config 入口。 |
| `tests/smoke_tests/models/deepseek_v4/test_lora.py` | 建议精简/迁移 | toy manager + `object.__new__` 不是真实 DSV4 ST；纯逻辑回 UT，真实 save/resume/export 交 integration。 |
| `tests/smoke_tests/ops/test_mhc_meta_registration.py` | 拆 PR | mHC schema compatibility 与 LoRA 无关；且“future append must pass”过度防御。 |
| `tests/unit_tests/models/deepseek_v4/test_lora.py` | 覆盖强、组织需重构 | 数值/梯度、converter、checkpoint、PEFT consumer oracle 较好；但 859 行跨模块、`test_review_*` 命名、缺 production main-forward CPU entry。 |
| `torchtitan_npu/models/deepseek_v4/__init__.py` | 设计方向合理 | parallelize 后 freeze、关闭 routing-bias balancing hook符合 PR 背景；保留该时序，但 enable 应来自统一 CLI extension，不来自独立 config module。 |
| `torchtitan_npu/models/deepseek_v4/lora.py` | 阻塞修改 | 通用实现放错层、依赖上游/PyTorch private API、额外 `routed_scores_R` 无真实 producer；见 R3/R6。 |
| `torchtitan_npu/models/deepseek_v4/lora_config.py` | 阻塞删除/重构 | 第二训练入口；改 `ExtensionConfig.lora + CLI`。 |
| `torchtitan_npu/models/deepseek_v4/peft.py` | 阻塞修改 | native base 路径被写进 PEFT metadata；finalize 丢失 filesystem abstraction；见 R2/R10。 |
| `torchtitan_npu/models/deepseek_v4/sharding.py` | 需要多卡证明 | LoRA expert placement 与 base axis 对齐思路合理，但新增的是分布式生产语义，必须有 EP2/FSDP2 ST。 |
| `torchtitan_npu/models/deepseek_v4/state_dict_adapter.py` | 主要逻辑有强 UT，需分布式 ST | PEFT shape/FQN 映射和 consumer oracle充分；DTensor expert flatten/redistribute 没有仓内真实 NPU multi-mesh 门禁，且 base metadata 调用方需修。 |
| `torchtitan_npu/ops/ascendc/mhc.py` | 拆 PR/收紧 | 与 LoRA 无关；不要用 `*_` 静默接受未知未来 schema。 |

## 10. Clean code / 防御式编程 / 文档一致性复核

| 维度 | 结果 | 建议 |
| --- | --- | --- |
| 单一训练入口 | 不符合 | 删除 `lora_config.py` production recipe；CLI extension 开关 LoRA。 |
| 通用代码最小作用域 | 不符合 | 通用 LoRA module/sharding 上提 `extensions/components` 或 upstream；DSV4 只保留模型 policy。 |
| private API 依赖 | 风险高 | 去除 `_lora_adapter_sharding`、`_is_non_strict_tracing` 直接依赖。 |
| 防御式未知参数兼容 | 不符合 | mHC `*_` future args、GroupedExperts `**dimensioned_kwargs` 都应按真实 producer/schema 收紧。 |
| 文档-代码一致性 | 有错误 | integration README `Check Loss` 修正；LoRA guide/usage 改统一 CLI；PEFT base metadata 与文档一致。 |
| PR 原子性 | 不符合 | LoRA、量化 recipe、mHC schema 分成独立 PR。 |
| patches 使用 | 符合 | 本 PR 未新增 NPU/model-only patch；继续保持。 |
| 测试数量控制 | 可进一步精简 | 不用 toy smoke + 1rank ST + 新增多卡 ST 三套重复证明同一 checkpoint 语义；把纯逻辑留 UT，真实路径集中到最小 integration case。 |

## 11. 合入前置条件

| 优先级 | 前置条件 | 最小验收标准 |
| --- | --- | --- |
| 1 | 收敛训练入口 | production 不再要求 `MODULE=...lora_config`；标准 DSV4 MODULE/CONFIG + `--extension.lora.*` 即可启用/配置 LoRA。 |
| 2 | 修 PEFT base metadata | native DCP `initial_load_path` 不写入 `base_model_name_or_path`；有独立、可被 Transformers 加载的 base 标识，并有 UT。 |
| 3 | 通用 LoRA 与模型 policy 分层 | dense/batched/grouped 通用实现移到 extension/upstream；模型目录只保留 DSV4 targets/mapping/policy；去除不必要 private API。 |
| 4 | 去掉无真实 caller 的 routed score 分支 | GroupedLoRAExperts 与 pinned upstream caller contract 一致，测试使用真实 producer representation。 |
| 5 | 落成多 NPU LoRA ST | `models` suite 中存在并静态可选的 2-NPU EP2/FSDP2 LoRA case，真实训练并覆盖 distributed adapter/export；本次 review 不要求扩大到 4/8 卡。 |
| 6 | 清理 PR scope | A5 quantization 两处改动和 mHC 两个文件从 LoRA PR 移出。 |
| 7 | 整理测试 | UT 按 production module 拆分，去掉 `test_review_*`；smoke 只保留真实设备边界，README 修正 `check_loss` 描述。 |
| 8 | 完成后重新 review | 重新检查 diff、CLI help/config schema、integration runner 静态包含关系，以及新的 CI/status；当前 mirror head 无 GitHub status checks / Actions 记录。 |

## 12. 附录：固定版本与审查边界

| 项目 | 事实 |
| --- | --- |
| TorchTitan 固定版本 | `requirements.txt` 为 `torchtitan==0.3.0`，`.ci/setup_torchtitan.sh` checkout `v0.3.0`。 |
| 上游 LoRA 基线 | `v0.3.0/torchtitan/components/lora.py` 已有 `LoRAConverter`；本 PR 应扩展/上提，而不是把通用能力长期绑在 DSV4 目录。 |
| 上游 GroupedExperts 调用 | `v0.3.0/torchtitan/models/common/moe.py::RoutedExperts.forward` 以两个参数调用 `inner_experts`，routing score 由 dispatcher combine 路径处理。 |
| LoRA integration 静态入口 | `build_deepseek_v4_test_list()` 返回 `dsv4_lora_1rank`；`build_models_test_list()` 包含 DeepSeek-V4 list；`.ci/smoke_test.sh` 运行 `--test_suite models`。 |
| 当前 mirror CI | head `6a902d62a015967c7fbc9ec6a7b1c3064a602a1e` 查询不到 GitHub combined status，也没有 PR workflow run；不能声称 CI 通过。 |
| Review 测试模式 | 按 `.agents/skills/developer-tests-review` 的 `review测试` 执行：只做静态读取，不修改/运行 testcase。 |
