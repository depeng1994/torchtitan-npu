# PR 822 Maintainer Review — 归一化版

> 本文件为三轮 Review 后的 **authoritative conclusion**。此前阶段性 Review 中 R/S/A 编号之间存在少量“先局部认可、后横向复用审查推翻”的结论，本版已全部合并，不再以“以后文为准”保留冲突表述。

## 1. Review 范围与最终结论

| 项目 | 结论 |
|---|---|
| Review 对象 | GitHub 镜像 PR #14，`pr_822 -> master`；对应 GitCode PR !822 |
| 代码审查基线 | `master@4c4079dc932234ae9b438011125c935669820080` / 原 PR code head `8a0bcdbbde15e4383058a9b1a1e89112e82c7973` |
| 原始变更规模 | 26 files，+2692 / -144（后续 `review.md` 提交不计入产品代码规模） |
| TorchTitan 基线 | 仓库固定 TorchTitan `v0.3.0`；Review 按该版本核对 Trainer、Dataloader、RMSNorm、RoPE、MoE 等上游契约 |
| Review 维度 | correctness、训练入口、上游解耦、Override/Extension/patch 边界、可删除代码、primitive ownership、状态面、UT/ST、文档一致性 |
| 测试执行 | **未执行**。测试结论来自仓内 test-review skill 的静态审查；PR 描述中的 77/77 与 8 卡 A/B 仅作为开发者验证证据，不替代仓内可重复 UT/ST |
| patch 边界 | **本 PR 未新增 patch，方向正确**。V4.1/NPU 特有能力没有新增进入 `patches/torchtitan` |
| Maintainer 最终结论 | **当前提交不建议合入**。必须先收敛 P0/P1：训练入口、真实 ST、RMSNorm/RoPE ownership、CC12M 重复 ownership、Golden/文档语义及关键主路径测试 |

## 2. 归一化后的主要 Findings

| ID | 级别 | 代码位置 | 最终问题结论 | 最终建议 |
|---|---|---|---|---|
| R1 | **Blocker / architecture** | A3/A5 launcher；`config_registry.py::_cc12m_dataloader_config()` | **训练 source-of-truth 被拆到 recipe、shell、env、override imports 多处。** `STEPS` 环境变量替代标准 CLI，A5 又新增 wrapper；同一 recipe 通过不同入口得到不同算子栈。 | 保留一个训练入口。恢复 TorchTitan 标准 CLI；删除 `STEPS` 特殊入口和 A5 wrapper。融合通过显式 `--override.imports` 或单一 typed profile 表达，不再用 `USE_GOLDEN/ENABLE_A5_FUSION` 形成第二配置系统。CC12M 数据路径按 R6 的最终方案复用现有 Config，而不是继续增加自定义 env。 |
| R2 | **Blocker / correctness** | A3 `TASK_QUEUE_ENABLE=${...:-1}`；`scripts/run_train.sh` 强制 `TASK_QUEUE_ENABLE=2` | A3 宣称的 `TASK_QUEUE_ENABLE=1` **实际不会生效**，公共 launcher 命令前置变量覆盖 export。 | 只保留一个 owner：公共 launcher 改为可覆盖 `${TASK_QUEUE_ENABLE:-2}`，或统一固定 2 并删除 A3/README 中的 1。补 effective-config/launcher smoke。 |
| R3 | **Blocker / ST** | `tests/integration_tests/deepseek_v41.py`；新增 RMS/RoPE/MoE/mHC/sparse、CC12M、Muon | 新默认生产路径没有仓库内真实 NPU ST。现有 2 卡 golden 只覆盖 reference 轨迹，不能证明默认 fusion、CC12M、Muon、A5 sparse/Sinkhorn。 | 按现有 runner 增加 <=4 NPU 的最小生产 ST：A3 fusion；tiny CC12M Trainer→dataloader；Muon materialize+step；A5 suite 中 sparse/Sinkhorn 前后向。 |
| R4 | **High / primitive ownership** | `models/deepseek_v41/rope.py`；`override/deepseek_v41/rope.py`；attention/compressor/vision `rotary` Config | **`V41RoPERotation` 不是 V4.1 模型语义，而是在 V4.1 namespace 复制 common rotary arithmetic。** 仓库已有 upstream `ComplexRoPE/CosSinRoPE`、split-aware backport、`override/common/rope.py` 的 Ascend rotary。当前形成 `rope + rotary` 双 Config，并重复 `npu_rotary_mul` adapter。 | 不保留 V4.1-owned RoPE arithmetic primitive。decoder/compressor/indexer 优先复用现有 `RoPE.forward(..., inverse=...)` / split-aware API；vision 的 2D table 仍保留在 V4.1。若外部 cos/sin apply seam 确实缺失，在 upstream-shaped common/Extension 中补 model-neutral seam，由 `override/common` 提供 NPU 实现。完成后删除 `models/deepseek_v41/rope.py`、`override/deepseek_v41/rope.py` 和第二套 `rotary` Config。 |
| R5 | **High / primitive ownership** | `models/deepseek_v41/rms_norm.py`；`model_registry.py`；`override/deepseek_v41/rms_norm.py` | **只有少数 callsite 有特殊 FP32 operation-order，普通 q/kv/block/final/indexer norm 没有 V4.1 专属语义。** 当前把全部 RMSNorm 改成 `V41RMSNorm`，主要价值变成给 `exact=True` override 提供 model-specific type tag。 | 普通 norm 恢复 upstream `RMSNorm.Config` + `override/common/rms_norm.py`。仅将 pre-PR `_ReferenceRMSNorm/_golden_rms_norm` 对应位置收敛为语义明确的 FP32 RMSNorm 类型；其 NPU override 只覆盖这个特殊 contract。不要用“模型名 subclass”作为纯 override tag。 |
| R6 | **High / duplicate ownership** | `cc12m_loader.py::Config/__init__`；`config_registry.py`；A3 launcher | CC12M 重复持有 `manifest_path/data_dir/tokenizer_path/vocab_size`，并吞掉 Trainer 已传入的 tokenizer 后再次构造 HF tokenizer。 | **最终方案优先于早期“暴露三个自定义 path CLI”的建议**：复用 inherited `dataloader.dataset_path` 作为 manifest；复用 `hf_assets_path` + Trainer tokenizer；从 manifest parent 推导 data root；vocab 从 tokenizer 读取。删除第二个 tokenizer、重复 path/vocab Config 和无界 `**kwargs`。 |
| R7 | **High / non-production code** | `cc12m_loader.py::_sha256_prefix/digest_batches/_main` | A/B digest CLI 不属于 Trainer/dataloader 产品路径，却放在 production model module 并带来额外 CLI/hash/测试/README。 | 从 production loader 删除 digest CLI。若仍需 A/B 工具，放独立验证工具；正式回归由 integration test + 固定资产承担。 |
| R8 | **High / portability** | CC12M config/script/README/test；A5 affinity | `/data/p00465316/...` 私有路径进入生产配置、脚本、文档和 UT；A5 默认绑核写死某台机器 topology。 | 删除私有默认路径；README 使用占位符；CPU affinity 只保留公共可覆盖策略。unit test 禁止依赖私有 tokenizer 目录。 |
| R9 | **High / stale semantics** | Golden env、A3 注释、README、integration env、既有 UT、sparse docstring | config 层已删除 Golden env 读取，但测试/README 仍宣称 `USE_GOLDEN`/`TORCHTITAN_NPU_*_GOLDEN` 生效；`CONFIG=..._vision` 也并不会自动回 reference。sparse 文档又写 validation pending，与 PR 描述冲突。 | operator/reference 选择只保留一套机制（建议 `override.imports`）；删除 dead Golden env/test/comment；刷新 synthetic/reference 命令和 sparse 支持状态。 |
| R10 | **High / UT main path** | `test_cc12m_loader.py`、`test_muon_profile.py`、MoE/mHC override tests | CC12M 多数 UT 直测 internal dataset；Muon 只测手写 FQN；MoE/mHC 新 adapter 无对应 contract UT。 | 增加 recipe→真实 build 主链；Muon 执行 `OptimizerConfig.materialize()` 并与真实 `model.named_parameters()` 对账；MoE/mHC 增加 override activation/isolation、参数传递、空 token、route-score/Sinkhorn 等 CPU mock contract。 |
| R11 | **High / checkpoint correctness** | CC12M iterable dataloader | 无限 cycle dataset 进入 CheckpointManager，但没有验证 `state_dict/load_state_dict` 后样本序列连续。 | 增加多 rank + cycle boundary resume UT；若正式支持 checkpoint，再加最小 integration resume。否则降低 README 能力声明。 |
| R12 | **Medium / test oracle** | CC12M target_grid/prepare tests | 部分测试用生产 helper 生成 expected，是同实现自证。 | 增加冻结输入→固定 grid、固定 token/type/label 的独立 oracle；接线测试可保留，但不能代替 correctness oracle。 |
| R13 | **Medium / test layout** | `tests/unit_tests/models/deepseek_v41/test_*` | override-only 测试混在 model tests；同时未来 R4/R5 重构后，当前 `V41RoPERotation/V41RMSNorm` 测试 ownership 也会变化。 | model semantic 测试留 models；common rotary/common RMS override 测试迁到对应 common/override 路径；V4.1 专属 FP32 norm 若保留则只测其特殊 contract；sparse/MoE/mHC adapter 测试放 `tests/unit_tests/override/deepseek_v41/...`。 |
| R14 | **Medium / reproducibility vs complexity** | `_verify_meta()`；prepare meta | 当前 runtime meta verifier 是 optional half-contract：做很多 hash/字段验证但仍不能证明完整 image identity；继续“缺什么补什么”会让 loader 变成 dataset verifier。 | **最终选择简化方向**：训练 loader 只消费 manifest；meta 作为离线 preparation provenance，不在每个 rank 的 dataset constructor 中做 optional runtime verifier，删除 `_verify_meta()` 及其分支测试。若团队未来把 dataset identity 定义为正式 runtime contract，应另做 required、集中式 verifier，而不是继续扩张当前半契约。 |
| R15 | **Medium / defensive over-engineering** | `prepare_cc12m.py`；mHC/sparse adapters | preparer 同时承担 tar linter、重复 stem/member/key 防御、写盘后二次完整 hash；adapter 又重复检查模型固定 invariant。 | preparer 保留会影响样本集合的 decode/长度/seed/hash-dedup/extract；删除通用 tar-linter 式防御和额外读盘验证。adapter 只保留真实 kernel 外部限制/override 依赖检查，删除 typed/fixed model invariant 的重复 guard。 |
| R16 | **Medium / missed reuse** | CC12M sequence/label | `supervised[] + _pad_sample()` 又实现一次 next-token masking，而 V4.1 已有 `build_shifted_labels()`。 | 在未 pad 序列上直接复用 `build_shifted_labels()`，再统一 pad；删除 `supervised` 返回值和重复 shift/mask 实现及对应实现耦合测试。 |
| R17 | **Medium / Muon duplication** | V4.1 与 DSV4 optimizer profile | V4.1 独立复制 DSV4 多个完全一致的 layout primitive，但模型 policy 又确实不同。 | 不允许 V4.1 import V4；只在 `models/common` 抽 2~3 个完全一致的 model-neutral layout primitive，FQN/bucket policy继续各模型自持。 |
| R18 | **Medium / layering inversion** | `model_registry.py -> torchtitan_npu.override.common.rope.WorkaroundComplexRoPE` | reference model 直接 import override namespace，依赖方向反了；R4 的新增 primitive 没解决这一既有问题。 | 目标依赖为 `model -> common/extension <- override`。既然本 PR 正在重构 RoPE seam，建议同步把 reference-compatible split/workaround API 下沉到 upstream-shaped common/Extension；至少不得继续扩大 model→override 依赖。 |
| R19 | **Medium / MoE ownership** | `models/deepseek_v41/moe.py`、`override/deepseek_v41/moe.py` | **V4.1 MoE 核心定义本身合理**：VL image bias、sqrtsoftplus、sorted top-k、FP32/clamp/score-absorb/cast 顺序是真实模型语义；问题只在 `V41MoE/V41RoutedExperts` 复制 generic padding/dispatch/combine orchestration。 | 保留 V41 Router/expert semantic；不要搬进 patch。长期推动 upstream/common 暴露 router-extra-kwargs / score-absorb hook，逐步删除 generic orchestration copy。新 Asc grouped-GEMM override 位置可接受，但须补 R3/R10 测试。 |
| R20 | **Low / docs & typing** | README、recipe docstring、Muon type annotation | README 混入一次性 50-step benchmark、HBM 百分比、共享机故障；Muon profile 用 `Any` 绕过类型检查。 | README 只保留稳定支持矩阵/CLI/约束；实验数据留 PR/benchmark 文档。Muon 使用 `ModelSpec` + 明确 V4.1 config type。 |

## 3. Primitive Ownership 最终矩阵

| Primitive / 行为 | 最终 Owner | 结论 |
|---|---|---|
| 标准 RMSNorm | TorchTitan common + `override/common` | **复用，不做 V4.1 subclass** |
| 特殊 FP32 RMSNorm operation-order | model-neutral common/Extension 优先；若仅 V4.1 需要可暂留 V4.1 | **只在真实特殊 callsite 使用** |
| RoPE cache / YaRN / position reshape | TorchTitan common | **复用** |
| external cos/sin rotary arithmetic seam | common/Extension | **需要时新增 model-neutral seam，不放 V4.1** |
| V4.1 vision 2D position table | V4.1 model | **保留** |
| V4.1 Router：image bias / sqrtsoftplus / sorted top-k | V4.1 model | **保留** |
| V4.1 expert FP32/clamp/score-absorb arithmetic | V4.1 model | **保留** |
| generic MoE dispatch/combine/SP padding | TorchTitan/common | **V4.1 当前复制属于兼容债，目标删除** |
| standard Ascend RMSNorm / rotary kernel adapter | `override/common` | **复用，不按模型复制** |
| V4.1 grouped expert / sparse / mHC NPU implementation | `override/deepseek_v41` | **位置正确** |

**Maintainer 规则：不得为了获得 `@override(exact=True)` 的隔离效果而先造一个无真实语义差异的 `ModelXRMSNorm/ModelXRoPE/...` reference subclass。Override 边界必须跟随真实 semantic boundary。**

## 4. 冗余 / 可直接删除与复用清单

| 当前代码/设计 | 最终动作 |
|---|---|
| A5 22 行 wrapper | **删除**；硬件差异由同一入口的显式 override/config 表达 |
| `CC12M_MANIFEST_PATH/DATA_DIR/TOKENIZER_PATH` 第二套 path 配置 | **删除**；复用 `dataloader.dataset_path` + `hf_assets_path` |
| loader 内第二个 `HuggingFaceTokenizer` | **删除**；使用 Trainer tokenizer |
| CC12M `vocab_size` 独立 Config | **删除**；从 tokenizer 得到 |
| `data_dir` Config | **删除**；从 manifest parent 推导 |
| `_rank/_world` 无后续读取的成员 | **删除** |
| `assemble_caption_sequence()` 的 `supervised` 状态 | **删除**；复用 `build_shifted_labels()` |
| `_sha256_prefix/digest_batches/_main` | **从 production module 删除** |
| optional `_verify_meta()` | **删除**；meta 降为离线 provenance |
| `_verify_extracted()` 及通用 tar-linter 防御 | **删除/收窄** |
| CC12M 下仍传的 c4 dataset/path no-op 参数 | **删除** |
| dead Golden env / tests / comments | **删除** |
| `models/deepseek_v41/rope.py` + V41 RoPE override | **按 R4 common/Extension 收敛后删除** |
| 普通 `V41RMSNorm` 实例 | **恢复 upstream RMSNorm**；仅保留特殊 FP32 contract |
| defensive adapter invariant guards | **删除重复项，只留真实 kernel contract** |
| CC12M 与上述 defensive/digest/meta 一一绑定的 UT | **随生产代码同步删除** |
| README 一次性 benchmark/机器故障记录 | **删除** |

## 5. 测试覆盖最终矩阵

| 语义单元 | 当前证据 | 最终结论 |
|---|---|---|
| 特殊 FP32 RMSNorm 数值语义 | 当前 `test_rms_norm.py` 有手写/F.rms_norm oracle；2p golden 间接覆盖 | oracle 有价值，但测试应随 R5 ownership 收窄；普通 RMS 不再维护 V41-specific test |
| RoPE 数值语义 | 当前 V41 tests 有 quarter-turn/grad oracle | arithmetic oracle 可迁移复用，但不应成为 V41-owned primitive 的存在理由；R4 重构后放到 common/对应 callsite tests |
| fused/common RMS/RoPE NPU kernel | CPU mock only | **需要真实 NPU ST** |
| V4.1 grouped expert override | 本 PR 无完整 adapter UT/ST | **补 UT + ST** |
| mHC post/Sinkhorn | 本 PR 无完整 adapter UT/ST | **补 UT + A3/A5 ST** |
| sparse packed indices | 显式 tensor oracle 有效 | 保留；仍需 NPU forward/backward ST |
| CC12M sequence/shard | UT 较多 | 删除 defensive implementation tests，保留独立 protocol/label/shard oracle |
| CC12M Config→Trainer→dataloader | 缺 | **必须补** |
| CC12M checkpoint resume | 缺 | **必须补或降低能力声明** |
| Muon materialize/real parameter match | 缺 | **必须补** |
| launcher effective config | 缺，且已发现 TASK_QUEUE bug | **必须补 smoke** |

## 6. 架构/边界最终核对

| 维度 | 最终结果 | 说明 |
|---|---|---|
| 新增 `patches/torchtitan` 污染 | **通过** | 本 PR 未新增 patch |
| NPU 特有实现使用 Override | **机制通过、ownership 部分不通过** | MoE/mHC/sparse 的 V4.1 override owner 合理；标准 RMS/RoPE 不应为了 exact target 按模型复制 |
| Extension 使用 | **当前未使用，但 RoPE common seam 值得考虑** | 如果 pinned upstream 缺 external-cache rotary 等 model-neutral hook，应优先 Extension/upstream-shaped common |
| model→override 依赖方向 | **不通过（既有问题，本 PR 应避免加深）** | `model_registry` 直接 import `override.common.rope` |
| 单一训练入口 / CLI 优先 | **不通过** | R1/R2/R8/R9 |
| upstream 解耦 | **部分不通过** | CC12M `**kwargs` 隐藏 Trainer contract；RMS/RoPE 按模型 fork common primitive；MoE generic orchestration复制 |
| state/checkpoint | **部分不通过** | CC12M resume 未验证；新增 RMS/RoPE 本身无额外 persistent state，但 ownership 仍需重构 |
| 文档一致性 | **不通过** | Golden、TASK_QUEUE、sparse validation、私有路径/benchmark |
| clean code / 可删除性 | **不通过** | CC12M scaffolding、digest、meta verifier、A5 wrapper、primitive fork 存在明显缩减空间 |

## 7. 26 个产品改动文件逐文件最终结论

| 文件 | 最终状态 | 最终结论 |
|---|---|---|
| `examples/deepseek_v41/debug/deepseek_v41_flash_8p_cpt_4k_a3.sh` | **需重构** | 单入口/CLI、TASK_QUEUE、私有路径、Golden env、no-op c4 args |
| `..._a5.sh` | **建议删除** | 第二入口 + 私有 CPU topology；硬件差异改显式 override/config |
| `examples/deepseek_v41/prepare_cc12m.py` | **保留核心、明显瘦身** | 保留 deterministic scan/decode/length/seed/hash-dedup/extract；删除 tar-linter/重复验证/无长期消费者统计 |
| `examples/deepseek_v41/readme.md` | **需重写稳定部分** | 删除私有路径、过时 Golden 语义、一次性 benchmark/机器事故 |
| `scripts/run_train.sh` | **需修改** | `TASK_QUEUE_ENABLE` owner 冲突；CPU affinity 可覆盖方向本身合理 |
| `tests/integration_tests/README.md` | **覆盖不足** | 50→30 可接受；必须增加新生产 ST case |
| `test_cc12m_loader.py` | **大幅收敛** | 删除 defensive/digest/meta/private-asset tests；补 main-path/resume/独立 oracle |
| `test_muon_profile.py` | **需补强** | 必须 materialize + actual named_parameters 对账 |
| `test_rms_norm.py` | **需按新 ownership 拆分** | 普通 RMS 回 common；只保留特殊 FP32 model contract 和对应 common/NPU override tests |
| `test_v41_rope.py` | **需按新 ownership 迁移/收敛** | arithmetic oracle 有用，但不应守一个 V41-owned common primitive |
| `test_sparse_fused_indices.py` | **保留核心，迁 override tests** | explicit index oracle 有价值；仍缺 NPU backward/ST |
| `models/deepseek_v41/attention.py` | **需修改 ownership** | 删除散落 golden helper是对的；但不应通过 V41-specific `rotary` 第二 Config 收敛，改复用 common RoPE seam |
| `cc12m_loader.py` | **需大幅精简** | R6/R7/R11/R14/R16：单 tokenizer/path owner、删 digest/meta verifier、补 resume |
| `compressor.py` | **去重目标正确，owner 需改** | 删除本地 `_golden_rope/_golden_rms_norm` 是对的；RoPE 归 common，只有特殊 FP32 norm semantic 可保留 |
| `config_registry.py` | **需修改** | 单入口/私有 env、CC12M duplicate config、Muon typing/profile |
| `model_registry.py` | **需修改** | 普通 norm 不应全换 V41；同时消除或收敛 model→override RoPE 反向依赖 |
| `models/deepseek_v41/rms_norm.py` | **收窄而非全模型保留** | 仅保留真实 FP32 operation-order semantic；普通 RMS 用 upstream |
| `models/deepseek_v41/rope.py` | **不建议保留为 V4.1 primitive** | 抽 seam 目标正确，但 owner 应为 common/Extension；完成迁移后删除 |
| `vision.py` | **模型语义保留，common arithmetic 复用** | 2D position table 属于 V4.1；RMS/RoPE apply 应复用收敛后的 common seam |
| `vision_data.py` | **保留** | `target_grid()` 是真实 producer/consumer 共用决策；补独立 oracle即可 |
| `override/deepseek_v41/mhc.py` | **位置正确，需精简 guard+补测试** | V4.1+NPU 特有融合属于正确 owner |
| `override/deepseek_v41/moe.py` | **位置正确，需补测试** | V41 expert arithmetic × grouped GEMM 的交叉点合理；不搬 patch/common |
| `override/deepseek_v41/rms_norm.py` | **仅特殊 FP32 contract 可 V41-specific** | 标准 RMS fusion 应复用 `override/common/rms_norm.py` |
| `override/deepseek_v41/rope.py` | **不建议保留** | 与 common Ascend rotary 重复；随 R4 迁 common 后删除 |
| `override/deepseek_v41/sparse_attn/__init__.py` | **位置正确，需 ST** | lazy exact override 合理；支持状态需与文档统一 |
| `override/deepseek_v41/sparse_attn/ascendc.py` | **核心实现需硬件回归** | packed local-index 修复有 UT；真实 forward/backward 必须由 A5 ST 证明 |

## 8. 最终修改优先级

| 优先级 | 必须完成的事项 |
|---|---|
| **P0 / Blocker** | R1 单一训练入口与标准 CLI；R2 TASK_QUEUE 生效值；R3 新默认路径真实 NPU ST |
| **P1 / 合入前必须收敛** | R4 RoPE owner/common seam；R5 RMSNorm 只按真实 semantic 分型；R6 CC12M 单 tokenizer/path source-of-truth；R8 私有路径/绑核；R9 Golden/支持状态；R10 Muon/CC12M/MoE/mHC 主路径 UT；R11 checkpoint resume |
| **P2 / 同 PR 建议完成** | R7 production digest 删除；R12 独立 oracle；R13 test layout；R14 meta half-contract 删除；R15 defensive code；R16 labels 复用；R17 Muon common primitive；R18 model→override layering；R20 docs/typing |
| **长期 debt，不作为本 PR 独立 blocker** | R19：保留 V4.1 Router/expert model semantic，等待/推动 upstream hook 后删除 generic MoE orchestration copy |

## 9. 最终目标架构

```text
TorchTitan common / upstream-shaped Extension
    ├── standard RMSNorm
    ├── RoPE cache / scaling / split / generic apply seam
    ├── generic MoE dispatch/combine hooks
    └── common optimizer/layout primitives (仅真正通用部分)
             ↑                    ↑
             │                    │
DeepSeek V4.1 model               NPU Override
    ├── vision 2D positions       ├── common RMSNorm / rotary kernels
    ├── CSA2 topology             └── V4.1-only grouped expert / sparse / mHC
    ├── VL router/image bias
    ├── frozen expert arithmetic
    └── only truly special FP32 norm contract
```

禁止把目标结构做成：

```text
V41RMSNorm -> AscV41RMSNorm
V41RoPE    -> AscV41RoPE
V42RMSNorm -> AscV42RMSNorm
V42RoPE    -> AscV42RoPE
...
```

如果 reference semantic 与 common 相同，这种按模型复制 type 仅用于 override 隔离的模式不接受。

## 10. 最终合入判断与复审条件

| 结论 | 说明 |
|---|---|
| **当前状态** | **不建议合入** |
| **核心阻塞原因** | 训练入口/配置不单一；TASK_QUEUE 实际运行值与声明不一致；新增默认生产路径没有仓内真实 NPU ST |
| **核心架构原因** | RoPE common primitive 被 V4.1 重新 fork；普通 RMSNorm 被过度模型化；CC12M 重复 tokenizer/path/config ownership；reference model 仍存在 model→override 反向依赖 |
| **核心 clean-code 原因** | production digest、optional meta verifier、A5 wrapper、tar-linter/defensive guards、重复 label/tokenizer/config 都有明确删除空间 |
| **复审条件** | 完成 P0 + P1；P2 至少对未完成项给出有 owner/后续 PR 的明确计划。所有最终实现必须保持 reference golden 锚，不允许通过回写/重算 golden 隐藏数值变化 |
