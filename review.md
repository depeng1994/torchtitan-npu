# PR 822 Maintainer Review

## 1. Review 范围与结论

| 项目 | 结论 |
|---|---|
| Review 对象 | GitHub 镜像 PR #14，`pr_822 -> master`；对应 GitCode PR !822 |
| 基线 / Head | `master@4c4079dc932234ae9b438011125c935669820080` / `pr_822@8a0bcdbbde15e4383058a9b1a1e89112e82c7973` |
| 变更规模 | 26 files，+2692 / -144 |
| TorchTitan 基线 | `requirements.txt` 与 CI 均固定为 TorchTitan `v0.3.0`；本次 review 按该上游版本核对 Trainer / ParallelAwareDataloader 等调用契约 |
| PR 背景 | CC12M 真实图文训练入口；A3 默认 4 项融合，A5 额外 sparse attention + mHC Sinkhorn；Muon 显式可选；PR 描述附带 8 卡 A3/A5 手工 A/B 数据 |
| patches / Extensions 边界 | **通过**：本 PR 未新增 `patches/torchtitan` / PyTorch / torch_npu patch，也未把 V4.1/NPU 特有实现塞进 patch；NPU 特有算子适配位于 `torchtitan_npu/override/deepseek_v41`，方向符合仓库边界 |
| 测试执行 | **未执行**。按仓内 `.agents/skills/developer-tests-review` 做静态 review；PR 描述中的 `77/77` 与 8 卡 A/B 作为开发者验证证据参考，但不替代仓库内可重复的 UT/ST 入口 |
| 测试审查结论 | **补充测试后合入** |
| Maintainer 合入建议 | **当前提交不建议直接合入**。R1、R2、R5 为阻塞项；R3/R4/R6/R7/R8 需要在本 PR 内收敛，之后再复核 |

## 2. 主要 Review 意见

| ID | 级别 | 代码位置 | 问题点 | 影响 / 为什么需要改 | 建议修改方案 |
|---|---|---|---|---|---|
| R1 | **Blocker** | `examples/deepseek_v41/debug/deepseek_v41_flash_8p_cpt_4k_a3.sh`：`CONFIG`、`CC12M_ARGS`、`NPU_OPS_OVERRIDES`；`examples/..._a5.sh`；`torchtitan_npu/models/deepseek_v41/config_registry.py::_cc12m_dataloader_config()` | **训练入口的 source-of-truth 被拆到了 config + shell + 环境变量三处，违背仓库“训练入口单一/CLI 优先”的要求。** 当前 A3 脚本把默认 config 改成 CC12M，再用 `CC12M_MANIFEST_PATH/CC12M_DATA_DIR/CC12M_TOKENIZER_PATH` 注入数据；用 `STEPS` 环境变量耦合 training/scheduler，并主动拒绝标准 `--training.steps` / `--lr-scheduler.*` CLI；融合选择又通过 `USE_GOLDEN` / `ENABLE_A5_FUSION` 隐式拼接 `--override.imports`；A5 再增加一个 wrapper。结果是同一个 `deepseek_v41_flash_40layers_16experts_cc12m` config，直接走 `python -m torchtitan_npu.train` 与走脚本时算子栈并不一致。 | 训练语义无法仅从打印出的 Trainer Config 复现；新增环境变量和 wrapper 会持续扩大入口分叉；用户无法自然使用 TorchTitan 原生 CLI 做实验（例如独立调整 scheduler total/warmup）。这也是后续上游升级时最容易失配的一层。 | 保留**一个**训练 launcher。CC12M dataloader 已有 `manifest_path/data_dir/tokenizer_path` Config 字段，应直接通过 `--dataloader.manifest-path`、`--dataloader.data-dir`、`--dataloader.tokenizer-path` 暴露；不要在 config factory 读取机器环境变量。删除 `STEPS` 专用入口和“禁止标准 CLI”逻辑，warmup/total/steps 由正常 CLI 或 recipe 默认值控制。融合选择使用显式 `--override.imports`（或一个经过 maintainer 认可的 typed CLI profile），不要再用 `USE_GOLDEN/ENABLE_A5_FUSION` 形成第二套配置系统。若 CC12M 因 dataloader 类型确实需要独立 recipe，可以保留**一个**数据 recipe，但 recipe 中不能携带机器路径和隐藏算子选择。 |
| R2 | **Blocker / correctness** | `examples/deepseek_v41/debug/deepseek_v41_flash_8p_cpt_4k_a3.sh`：CC12M 分支 `export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}`；`scripts/run_train.sh`：torchrun 前置环境 `TASK_QUEUE_ENABLE=2` | **提交中的 CC12M 入口实际上没有使用文档/PR 所声明的 `TASK_QUEUE_ENABLE=1`。** A3 脚本先 export 为 1，但公共 launcher 在执行 `torchrun` 时用命令前置环境变量强制写死 `TASK_QUEUE_ENABLE=2`，后者会覆盖调用者 export。 | README 写“CC12M 入口自动固定 `TASK_QUEUE_ENABLE=1`”，PR 的 8 卡验证也以其入口语义为背景；当前仓库代码无法复现该运行时条件。若 task queue 值会影响数值、时序或稳定性，则验证结论与可执行提交不是同一个配置。 | 如果公共 launcher 允许调用方覆盖，应改为 `TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"`；如果项目决定统一固定为 2，则删除 A3 中无效的 export，并刷新 README/验证说明。最终只能保留一个真实生效的定义，并补一条 launcher/config smoke 覆盖。 |
| R3 | **High / portability** | `config_registry.py::_cc12m_dataloader_config()`；A3 脚本 CC12M 默认；A5 脚本 `CPU_AFFINITY_CONF`；`tests/unit_tests/models/deepseek_v41/test_cc12m_loader.py::REAL_TOKENIZER`；`examples/deepseek_v41/readme.md` | **多处把开发机私有路径/拓扑写成仓库默认值。** `/data/p00465316/...` 同时进入生产 config、脚本、文档和 unit test；A5 wrapper 更默认写死 `npu0:288-311 ... npu7:456-479` 的 CPU 拓扑。 | 默认训练脚本在绝大多数环境会直接失败；A5 在 CPU topology 不同的机器上可能错误绑核；unit test 会因私有路径存在与否表现不同。这些都不应成为公共仓库的默认契约。 | 数据路径默认留空并由 CLI 必填/显式传入；README 使用 `<cc12m_root>` 等占位符。CPU affinity 只允许用户显式覆盖，公共默认继续使用通用策略，不要提交某台机器的 core map。删除 unit test 中私有 `REAL_TOKENIZER` 路径依赖；真实 tokenizer 验证如果必须保留，应进入有明确资产契约的 integration 环境。 |
| R4 | **High / stale semantics** | `config_registry.py` 删除 `_golden_enabled()`；A3 脚本顶部注释及 README 的 `CONFIG=..._vision` “Golden 路径”；`tests/integration_tests/deepseek_v41.py::GOLDEN_ENV`；既有 `test_baseline_contract.py` / `test_training_contract.py` 的 `USE_GOLDEN` monkeypatch；`override/deepseek_v41/sparse_attn/ascendc.py` module docstring | **Golden 开关的调用点没有随实现刷新。** 本 PR 已移除 config 层对 `USE_GOLDEN/TORCHTITAN_NPU_*_GOLDEN` 的读取，但 integration case 仍设置并注释这些旧 env，两个既有 UT 仍 monkeypatch `USE_GOLDEN`；这些设置现在是 dead configuration。另一方面 README 说只设 `CONFIG=..._vision` 即回到 Golden，但脚本 `USE_GOLDEN` 默认是 0，所以该命令仍会加载 RMSNorm/RoPE/MoE/mHC fused overrides。另有 sparse adapter 文件头仍写着 “Hardware and numerical validation are pending”，与 PR/README 的“已验证并 A5 默认启用”直接冲突。 | 文档、测试命名和真实代码出现三套语义；后续 reviewer 看到测试通过会误以为 Golden env 仍被生产代码消费。 | 选定唯一机制：建议 operator/reference 选择只由 `override.imports` 决定。删除所有已失效的 Golden env 与注释；若保留 shell convenience，则 README 的 synthetic golden 命令必须显式给出 reference override/对应开关。同步刷新 sparse 文件头，使“支持状态/验证状态”只保留一个事实来源。 |
| R5 | **Blocker / ST** | `tests/integration_tests/deepseek_v41.py`、`tests/integration_tests/README.md`；本 PR 新增 `override/deepseek_v41/{rms_norm,rope,moe,mhc,sparse_attn}`、CC12M loader、Muon | **没有仓库内真实 NPU ST 进入本 PR 新增的默认生产路径。** 现有唯一 `dsv41_golden_2p_ep2_fsdp2` 是 2 卡 reference golden；它能看护本 PR 对 reference RMSNorm/RoPE 重构不破坏旧轨迹，这是有效覆盖，但它没有加载 A3 4 项 fusion、A5 sparse/Sinkhorn、CC12M dataloader 或 Muon。`tests/integration_tests/README.md` 只把 50 步文字改成 30 步，没有增加新 case。 | PR 的主行为是“默认启用融合”和“新增真实数据/Muon”，但 CI/ST 无法证明这些入口能被实际激活、能完成前后向、能在 FSDP/EP 下工作。PR 描述中的 8 卡手工 A/B 很有价值，但无法替代可持续回归。 | 按仓内 ST 规则补进现有 runner，控制在 <=4 NPU：① A3 四项 override 的 2p/4p debugmodel 训练 case，验证 activation + 完成 + 与 reference 的数值准则；② 使用仓内 tiny image/manifest/tokenizer 的 CC12M 2p 训练 case，真正从 Trainer 构造 dataloader；③ `--optimizer.name Muon` 的最小训练 case并确认 DistMuon 被实际 materialize；④ A5 环境若不在默认 CI 池，至少提供现有 runner 可调用的 A5 suite/case，并在 A5 pipeline 中跑 sparse+Sinkhorn 前后向，而不是只保留 PR 手工记录。 |
| R6 | **High / UT completeness** | `test_cc12m_loader.py`、`test_rms_norm.py`、`test_v41_rope.py`、`test_sparse_fused_indices.py`；新 `override/deepseek_v41/moe.py`、`mhc.py` | **UT 对新增主调用链覆盖不完整。** CC12M 大部分测试直接实例化 `_Cc12mCaptionDataset`，没有从 `deepseek_v41_flash_40layers_16experts_cc12m()` 得到真实 Trainer Config 再按上游 `Trainer` 的 build 签名构造 dataloader；所谓“real tokenizer integration”其实放在 `tests/unit_tests`，并依赖私有目录后 `skipif`。此外 RMSNorm/RoPE 有 CPU adapter 测试，sparse 有 indices 层 mock，但新 MoE grouped GEMM override 与 mHC post/Sinkhorn override 没有对应的 adapter/config isolation UT。 | 可能出现“内部 helper 全绿，但 recipe/Config/override.imports 接不起来”的假阳性；MoE/mHC 参数传递、空 token、route score、Sinkhorn 参数等错误只能等到手工 8 卡暴露。 | 增加 recipe→`config.dataloader.build(dp_world_size, dp_rank, tokenizer, seq_len, local_batch_size)`→首 batch 的正向 UT；真实 tokenizer 私有路径测试移出 unit_tests。为 MoE/mHC 增加 CPU mock contract：override 激活/隔离、kernel 参数、空 token fallback、routed score 语义、mHC pre/post shape 与状态不新增；数值 kernel 本身由 NPU ST 负责。 |
| R7 | **High / Muon path** | `config_registry.py::_v41_muon_profile()` / `_v41_optimizer_config()`；`test_muon_profile.py` | 当前 Muon UT 主要对**手写 FQN 样例、layout 字典和 bucket 数量**做检查，没有执行 `OptimizerConfig.materialize()`，也没有把 regex/layout 与**实际 V4.1 model.named_parameters()** 对账。 | PR 对外承诺的入口是 `--optimizer.name Muon`。如果 materialize 后 param group、factory kwargs 或真实参数命名有遗漏/重叠，现有测试仍可全绿；尤其 profile 是按 layer config 手工拼 FQN，最需要与实际模型参数做闭环。 | 从真实 recipe 取 config，设置 `optimizer.name="Muon"` 后调用与生产一致的 materialize 路径，断言得到 `[DistMuon, AdamW]` 且 scalar CLI 超参真实落入 kwargs；构造 CPU/meta V4.1 模型，用 `named_parameters()` 计算 regex 命中集合，与 `compute_sharding_by_fqn` / buckets 做一致性检查（允许明确列出的 AdamW-only 参数），禁止只测试自造字符串。再用 R5 的 NPU smoke 证明 optimizer 真正跑过 step。 |
| R8 | **High / checkpoint correctness** | `cc12m_loader.py::_Cc12mCaptionDataset.__iter__()` / `DeepSeekV41Cc12mDataLoader`；README “相同初始权重/保存加载”章节；上游 v0.3.0 Trainer 会把 dataloader 交给 CheckpointManager | CC12M dataset 是无限 cycle 的 `IterableDataset`，而 Trainer checkpoint 会保存/恢复 `ParallelAwareDataloader` 状态；本 PR 没有验证 `state_dict()/load_state_dict()` 后下一条样本位置是否连续。README 已明确引导用户开启 checkpoint/load，因此这不是理论路径。 | 如果 StatefulDataLoader 对该 stateless iterable 的恢复只做 fast-forward、或 cycle/DP shard 的位置恢复不符合预期，resume 后可能重复/跳过样本；loss 仍然能跑，但训练数据序列已改变，A/B 和恢复语义都失真。 | 增加 CPU UT：多 rank 分片分别消费 N 条→保存 loader state→重建 loader/load state→检查后续 sample id / input hash 与不中断序列逐项一致，并覆盖 cycle 边界；若 checkpoint resume 是正式支持能力，再增加最小 integration resume case。若暂不支持，则 README 不应给出“可用”的恢复说明，应显式声明限制。 |
| R9 | **Medium / UT oracle** | `test_cc12m_loader.py::test_target_grid_matches_full_decode_at_boundaries()`、`test_prepare_scan_uses_shared_budget_and_grid()` | 部分新增测试是**同实现自证**。`from_path()` 在本 PR 后直接调用 `target_grid()`，测试再拿 `from_path()` 的 grid 与 `target_grid()` 比较，两个值来自同一函数；prepare scan 测试也用生产 `target_grid()` + `assemble_caption_sequence()` 重新计算 expected。它们能证明“共用函数接线”，但不能证明函数本身算对。 | `_safe_resize` / 长度预算若同时引入错误，这些测试仍会通过，不满足 test-review skill 对独立 oracle 的要求。 | 保留 producer-consumer 接线测试，但另加独立 oracle：用冻结的边界输入和**固定期望 grid**（来自 PR 前等价算法/权威参考）验证 `target_grid`；对序列用手工构造的小 grid/小 caption 固定 token/type/index/supervision 结果，避免 expected 再调用同一生产组装函数。 |
| R10 | **Medium / test layout** | `tests/unit_tests/models/deepseek_v41/test_rms_norm.py`、`test_v41_rope.py`、`test_sparse_fused_indices.py` | override 专属行为混在 `tests/unit_tests/models/deepseek_v41`。其中 sparse 文件完全是在测 `torchtitan_npu.override.deepseek_v41.sparse_attn`；RMSNorm/RoPE 文件也同时混合 reference model arithmetic 与 fused override adapter。 | 不符合仓内 skill 要求的生产目录镜像；后续维护者无法通过目录判断测试是在守 model contract 还是 NPU override contract。 | reference `V41RMSNorm/V41RoPERotation` 的纯模型测试留在 `tests/unit_tests/models/deepseek_v41`；`AscV41*`、override.imports、kernel mock、sparse indices 等拆到 `tests/unit_tests/override/deepseek_v41/...` 对应目录。 |
| R11 | **Medium / reproducibility contract** | `cc12m_loader.py::_verify_meta()`；`prepare_cc12m.py` meta 生成；README/PR 对固定数据身份的描述 | runtime meta 校验当前只覆盖 manifest SHA、部分 tokenizer JSON hash、image-processing 参数；prepare 已记录 `seq_len` 与 tokenizer vocab，但 runtime 没核对它们；图片内容同路径替换完全不进入 meta identity，README 只能要求人工比较 digest，而且 digest 默认只看前若干 batch。 | 对“固定 8K A/B 数据身份”而言，这是弱于文档语义的契约：配置变化可能到单个样本才 late-fail，同路径图像替换则可能静默改变训练集。 | 至少自动校验 `meta.seq_len == training seq_len` 和 `tokenizer_vocab_size == config.vocab_size`。如果“固定 A/B 数据集”是正式能力，manifest/meta 应携带 selected image hashes（或稳定聚合 hash），runtime/digest 能完整验证；如果不想承担全量 runtime hash 成本，则把文档表述降为“manifest/tokenizer/preprocess preflight”，不要称为完整数据身份校验。 |
| R12 | **Medium / upstream decoupling** | `cc12m_loader.py::DeepSeekV41Cc12mDataLoader.__init__(..., **kwargs)`；上游 TorchTitan v0.3.0 `Trainer` 构造 dataloader 时会显式传 `tokenizer=` | 新 dataloader 用 `**kwargs` 吞掉 Trainer 传入的 `tokenizer`，但没有使用也没有校验；未来上游增加 dataloader build 参数时也会被静默吞掉。 | 这是典型的防御式“兼容”写法：短期少写一个参数，长期会让上游接口变化不再 fail-fast，和本仓强调的解耦/升级可感知相反。 | 按 v0.3.0 明确签名写出 `tokenizer` 参数（即使本 loader 因真实 caption 需要自有 tokenizer，也应注释为什么忽略 Trainer tokenizer），删除无边界 `**kwargs`；如果确需可扩展参数，只透传给 `super().__init__` 并让上游 `_validate_kwargs` 校验，不要静默丢弃。 |
| R13 | **Medium / clean code & docs** | `config_registry.py::_v41_muon_profile(model_spec: Any)`；`examples/deepseek_v41/readme.md`；CC12M recipe docstring | 新 Muon profile 直接使用 `Any`，而仓内 DSV4 同类实现已经用 `ModelSpec` + 具体 model config cast，当前写法主动绕开类型检查。README/recipe docstring 同时混入 FUSION50 手工验收数字、某台共享宿主机的锁页内存故障、当前 HBM 百分比等一次性实验信息。 | 会削弱 pyrefly 对 FQN/config 属性的帮助，并让稳定用户文档迅速过期；PR 验证日志和产品使用文档职责混在一起。 | `_v41_muon_profile/_v41_optimizer_config` 使用 `ModelSpec` 与明确的 `V41Model.Config`/对应具体类型；把性能/误差/共享机故障等保留在 PR 验证记录或专门 benchmark 文档，README 只保留稳定的支持矩阵、CLI 入口、约束和可复现命令。 |

## 3. 测试覆盖审查（按仓内 developer-tests-review skill）

| 语义单元 | 当前 UT | 独立 oracle | 当前真实 ST | Review 结论 |
|---|---|---|---|---|
| V4.1 reference RMSNorm 重构 | `test_rms_norm.py` 覆盖 FP32/native 前后向 | **有**：native path 对 `F.rms_norm`，reference_fp32 path 手写公式 | **有**：现有 `dsv41_golden_2p_ep2_fsdp2` 会经过 reference 模型并做 30-step exact loss | 基本充分；保留现有 golden 锚 |
| V4.1 reference RoPE 重构 | `test_v41_rope.py` 覆盖三种 mode、inverse、非连续输入和梯度 | **有/较好**：quarter-turn 固定关系 | 同上，reference golden 间接覆盖 | 基本充分 |
| fused RMSNorm | config replacement/state_dict key UT | 仅 CPU/reference；NPU kernel 未执行 | **无** | 需要 A3 NPU ST |
| fused RoPE | CPU mock 检查 dtype/shape/rotary_mode/batch position | mock 只证明 adapter 参数，不证明 kernel 数值 | **无** | 需要 A3 NPU ST |
| fused MoE grouped GEMM | 本 PR 无对应新增 adapter UT | **无** | **无** | 先补 UT，再补 ST |
| fused mHC post / Sinkhorn | 本 PR 无对应新增 adapter UT | **无** | **无** | 先补 UT，再补 A3/A5 ST |
| sparse attention packed index localization | `test_sparse_fused_indices.py` 覆盖 ratio 1/2 × single/odd/even packed | 对 local index 期望是显式 tensor，**有效** | **无**；测试文件自身 TODO 也说明 backward 需 NPU | NPU 前后向 ST 必须补 |
| CC12M sequence/label/padding/sharding | UT 数量较多，正向与多数边界均覆盖 | **部分**；`target_grid`/prepare 两项存在同实现自证（R9） | **无** | 补独立 oracle + Trainer/ST |
| CC12M config→Trainer→dataloader | 目前主要直接 new internal dataset | **无主路径闭环** | **无** | 必须补 |
| CC12M checkpoint resume | **无** | **无** | **无** | README 已宣称 checkpoint 可用，需补 |
| Muon profile | regex 样例、layout、bucket 数量 | **部分**；expected 与 production FQN 均为手工列表，未与真实参数闭环 | **无** | materialize + actual named_parameters + ST |
| launcher/operator selection | **无** | 不适用 | **无** | 已静态发现 `TASK_QUEUE_ENABLE` 实际值错误，需至少 smoke 校验最终生效配置 |
| 真实 V4.1 tokenizer | unit test 中私有路径存在时才跑 | 依赖开发机资产 | 不在 integration runner | 不应算可移植 UT/ST；改为正式资产契约或仅保留手工验证说明 |

## 4. 架构/边界逐项核对

| 维度 | 结果 | 说明 |
|---|---|---|
| `patches/torchtitan` 是否被放入 NPU/V4.1 私有逻辑 | **通过** | 本 PR 无 patch 改动 |
| torch_npu / PyTorch patch 新增必要性 | **通过 / 不适用** | 未新增 patch |
| NPU 特有实现是否走 Override | **通过** | RMSNorm/RoPE/MoE/mHC/sparse 均通过自有 V4.1 Config + `@override(exact=True)` 接入 |
| 是否依赖 V4 专有 override | **通过** | 新 V4.1 override 独立，未把 V4 类替换搬过来 |
| Extension 目录定位 | **不适用** | 本 PR 无 Extension 改动 |
| 单一训练入口 / CLI 优先 | **不通过** | R1/R3/R4：环境变量、A5 wrapper、私有路径和 CLI 禁用形成多入口 |
| 上游 TorchTitan v0.3.0 契约 | **部分通过** | Config/ParallelAwareDataloader 方向正确；R12 的 `**kwargs` 会隐藏未来上游接口变化 |
| state_dict 参数名兼容 | **通过（静态）** | RoPE adapter 无参数；RMSNorm 继承相同 weight；已有 state_dict key UT |
| 文档与代码一致性 | **不通过** | R2/R4/R13 |
| clean code / 可维护性 | **部分通过** | model primitive 抽象本身较干净；入口脚本和实验型 README 过重，Muon 使用 `Any` 可进一步收紧 |

## 5. 26 个改动文件逐文件覆盖

| 文件 | Review 状态 | 主要结论 / 关联意见 |
|---|---|---|
| `examples/deepseek_v41/debug/deepseek_v41_flash_8p_cpt_4k_a3.sh` | **需修改** | R1：环境变量替代 CLI；R2：TASK_QUEUE 设置无效；R3：私有数据路径；R4：synthetic Golden 注释与实际融合默认冲突 |
| `examples/deepseek_v41/debug/deepseek_v41_flash_8p_cpt_4k_a5.sh` | **需修改/建议删除 wrapper** | R1：增加第二训练入口；R3：默认 CPU affinity 写死单机拓扑 |
| `examples/deepseek_v41/prepare_cc12m.py` | **方向可接受，需配合测试/契约收敛** | 离线数据准备作为独立工具是合理的；R9：现有测试对共享 helper 的 correctness oracle 不独立；R11：meta identity 仍不完整 |
| `examples/deepseek_v41/readme.md` | **需修改** | R2/R3/R4/R13：运行参数、Golden 语义、私有路径和一次性实验记录需要刷新/精简 |
| `scripts/run_train.sh` | **需修改** | CPU affinity 改为可覆盖本身合理；但 `TASK_QUEUE_ENABLE=2` 仍硬覆盖上层，直接触发 R2 |
| `tests/integration_tests/README.md` | **修改本身正确但覆盖不足** | 50→30 与现有 case 一致；R5：没有新增生产 ST case |
| `tests/unit_tests/models/deepseek_v41/test_cc12m_loader.py` | **需修改** | 正向覆盖较多；R3 私有 tokenizer、R6 主路径不足、R8 无 resume、R9 oracle 自证、R10/格式问题 |
| `tests/unit_tests/models/deepseek_v41/test_muon_profile.py` | **需补强** | R7：只测自造 FQN/profile 元数据，未 materialize/实际参数闭环 |
| `tests/unit_tests/models/deepseek_v41/test_rms_norm.py` | **部分通过** | reference oracle 较好；override 部分应拆目录并补 NPU ST（R5/R10） |
| `tests/unit_tests/models/deepseek_v41/test_sparse_fused_indices.py` | **部分通过** | local-index oracle 有价值；override-only 文件位置不对，且 backward/NPU ST 缺失（R5/R10） |
| `tests/unit_tests/models/deepseek_v41/test_v41_rope.py` | **部分通过** | reference arithmetic/grad 测试较好；fused mock 应拆到 override tests，真实 NPU ST 缺失 |
| `torchtitan_npu/models/deepseek_v41/attention.py` | **通过（静态）** | 将旋转算术抽成 `V41RoPERotation`，保留 cache/position ownership；未发现新增 state |
| `torchtitan_npu/models/deepseek_v41/cc12m_loader.py` | **需修改** | 主体序列/DP 逻辑清晰；R8 checkpoint state、R11 meta identity、R12 `**kwargs` 需收敛 |
| `torchtitan_npu/models/deepseek_v41/compressor.py` | **通过（静态）** | 删除本地重复 golden RoPE/RMSNorm，复用 V4.1 primitive，方向正确；数值由现有 reference golden + 新 ST 看护 |
| `torchtitan_npu/models/deepseek_v41/config_registry.py` | **需修改** | R1/R3/R7/R13：环境路径、入口语义、Muon 主路径/type typing |
| `torchtitan_npu/models/deepseek_v41/model_registry.py` | **通过（静态）** | V4.1 专属 RMSNorm Config 替换范围明确，参数名保持；compressor `reference_fp32` 的语义有对应 UT |
| `torchtitan_npu/models/deepseek_v41/rms_norm.py` | **通过（静态）** | model-owned arithmetic 抽象合理，未放进 patch/override reference 双实现 |
| `torchtitan_npu/models/deepseek_v41/rope.py` | **通过（静态）** | 无参数、模式明确；reference UT 较完整 |
| `torchtitan_npu/models/deepseek_v41/vision.py` | **通过（静态）** | 删除重复 `_ReferenceRMSNorm/_apply_rope`，通过 Config 注入统一 primitive，clean code 方向正确 |
| `torchtitan_npu/models/deepseek_v41/vision_data.py` | **通过生产重构，测试需改** | `target_grid()` 提取共享决策是合理重构；R9 指向测试 oracle，而非该抽取本身 |
| `torchtitan_npu/override/deepseek_v41/mhc.py` | **实现归属正确，测试不足** | exact Config override 符合架构；R5/R6：post/Sinkhorn 缺仓内 UT/ST |
| `torchtitan_npu/override/deepseek_v41/moe.py` | **实现归属正确，测试不足** | 独立 V4.1 grouped GEMM override，不污染 V4/patch；R5/R6：缺 adapter UT 与生产 ST |
| `torchtitan_npu/override/deepseek_v41/rms_norm.py` | **实现归属正确，需 ST** | override 隔离明确；CPU config/state UT 有，NPU kernel 路径无仓内 ST（R5） |
| `torchtitan_npu/override/deepseek_v41/rope.py` | **实现归属正确，需 ST** | reshape/fold 逻辑有 CPU mock，但 kernel 数值/训练链路需 ST（R5） |
| `torchtitan_npu/override/deepseek_v41/sparse_attn/__init__.py` | **方向正确，需 ST** | lazy import + exact override 合理；R4 支持状态文档需一致，R5 缺 NPU case |
| `torchtitan_npu/override/deepseek_v41/sparse_attn/ascendc.py` | **核心实现需仓内硬件回归** | packed local index 处理有有效 UT；文件头“validation pending”与 PR 冲突；forward/backward 只能由 A5 ST 证明（R4/R5） |

## 6. 对 PR 描述中验证证据的处理

| PR 描述中的证据 | Review 评价 |
|---|---|
| deepseek_v41 UT 77/77 | 有价值，但本 review 不重新执行；且 private-tokenizer `skipif`、同实现 oracle、Muon 主路径、MoE/mHC adapter 等问题意味着“测试数量通过”不能等价为覆盖完整 |
| A3 8 卡 50-step fusion A/B | 可作为数值验收证据保留；应把最小可回归版本落到 `tests/integration_tests`，否则之后的 PR 无法自动守住默认融合路径 |
| A5 8 卡 sparse/Sinkhorn 50-step A/B | 同上；尤其 sparse 文件自身仍写 validation pending，必须把代码/文档/测试三者统一 |
| Muon 50-step 手工运行 | 能证明某次环境可跑，但 `--optimizer.name Muon` 的 materialize/真实 FQN 选择仍需 UT 闭环并进入最小 ST |
| CC12M digest / meta | 作为 A/B preflight 有价值；R11 说明目前不是完整数据身份校验，不应靠人工前几 batch digest 承担正式 reproducibility contract |

## 7. 建议修改优先级

| 优先级 | 必须完成的事项 |
|---|---|
| P0 | 修复 R1（统一训练入口/恢复标准 CLI）、R2（TASK_QUEUE 实际值）、R5（新增真实生产 ST） |
| P1 | 清理 R3 私有路径/绑核；统一 R4 Golden/支持状态；补 R6/R7 主路径 UT；确认并覆盖 R8 dataloader checkpoint resume |
| P2 | 修正 R9 独立 oracle、R10 测试目录、R11 meta 契约、R12 明确上游构造签名、R13 typing/README 清理 |

## 8. 最终结论

| 结论 | 说明 |
|---|---|
| **当前状态** | **不建议合入** |
| **核心原因** | 默认训练行为依赖 shell/env 而非单一 Config/CLI；存在 `TASK_QUEUE_ENABLE` 真实运行值与文档/验证不一致；新增默认融合/CC12M/Muon/A5 sparse 路径没有仓库内真实 NPU ST |
| **复审条件** | 至少完成 P0 + P1，并在 `review.md` 对应 R 项逐条给出修改位置/新增测试 case；现有 reference golden 必须继续保持不回写/不重算锚来“消除”差异 |

## 9. 冗余代码 / 可进一步精简专项 Review

### 9.1 总体判断

| 项目 | 专项结论 |
|---|---|
| CC12M 增量规模 | `prepare_cc12m.py` 348 行 + `cc12m_loader.py` 430 行 + `test_cc12m_loader.py` 527 行，三者合计新增 **1305 行**。真实训练核心只需要“读取 manifest → image preprocess → 序列组装/label → DP shard → ParallelAwareDataloader”，当前相当一部分代码实际在维护第二套配置、A/B preflight、数据完整性检查和这些检查对应的测试。 |
| 是否存在可直接删除代码 | **是**。A5 wrapper、CC12M 下无效/无消费方的 dataloader 参数、loader 内部第二个 tokenizer、`_rank/_world` 状态、production loader 内 `digest` CLI，以及若干只验证内部不变量的防御分支都有明确删除空间。 |
| 是否存在已有能力未复用 | **是**。TorchTitan `BaseDataLoader.Config.dataset_path`、Trainer 已构造并传给 dataloader 的 `tokenizer`、V4.1 已有 `build_shifted_labels()` 均可直接复用；当前 PR 为这些能力又建了一套字段/逻辑。 |
| 新增 abstraction 是否全部过度 | **否**。`V41RMSNorm`、`V41RoPERotation`、`ImagePatchProcessor.target_grid()` 以及 attention/compressor/vision 对本地重复实现的删除是净减复杂度，应保留；专项意见不建议把这些抽象回退。 |
| 专项总体建议 | **先做“删除式重构”，再补测试。** 不应在现有 1305 行 CC12M scaffolding 上继续增加更多 validator/test；先把 source-of-truth 收敛到 Trainer Config/CLI 和现有 helper，再只测试剩下的生产语义。 |

### 9.2 专项 Finding

| ID | 级别 | 代码位置 | 冗余 / 可精简点 | 建议修改方案 |
|---|---|---|---|---|
| S1 | **High / duplicate ownership** | `cc12m_loader.py::DeepSeekV41Cc12mDataLoader.Config`、`_Cc12mCaptionDataset.__init__()`；`config_registry.py::_cc12m_dataloader_config()`；A3 launcher | **CC12M 自己重复持有了 `manifest_path`、`data_dir`、`tokenizer_path`、`vocab_size` 四个本可由已有契约推导/提供的信息。** `ParallelAwareDataloader.Config` 已继承 `dataset_path`；`prepare_cc12m.py` 又保证 manifest 中 image 是 `images/...` 相对路径，因此 data root 可直接取 `Path(manifest).parent`；上游 Trainer 已通过 `config.tokenizer.build(tokenizer_path=config.hf_assets_path)` 构造 tokenizer，并显式传 `tokenizer=` 给 dataloader；当前 loader 却用 `**kwargs` 吞掉它，再构造第二个 `HuggingFaceTokenizer`，然后额外保存/核对 `vocab_size`。这是最明显的重复 source-of-truth。 | **进一步精简后的建议优先于前文 R1/R12 中“继续暴露三个 CC12M 自定义 path 字段”的方案。** CC12M recipe 使用 `HuggingFaceTokenizer.Config()` 作为 Trainer tokenizer；复用 `--hf-assets-path` 提供 tokenizer；复用现有 `--dataloader.dataset-path` 指向 manifest；loader 显式接收 `tokenizer: BaseTokenizer`，用 `tokenizer.get_vocab_size()/bos_id/eos_id`，并从 manifest parent 推导 image/meta root。删除 `manifest_path/data_dir/tokenizer_path/vocab_size` 四个自定义字段（或至少只保留一个 manifest 字段）、`_cc12m_dataloader_config()` 环境变量 helper、loader 内第二次 tokenizer 构造和 vocab 双重校验。`self._rank/self._world` 构造后未再读取，也直接删除。 |
| S2 | **High / missed reuse** | `cc12m_loader.py::assemble_caption_sequence()`、`_pad_sample()`；`vision_data.py::build_shifted_labels()` | PR 又实现了一套 `supervised: list[bool]` + `targets[1:]` + `masked_fill` 的 next-token label 逻辑，但 V4.1 已有 `build_shifted_labels(input_ids, token_types)`，其语义恰好是“图片协议 target mask 掉，TEXT target 保留”。当前 `supervised` 并非模型必需状态，主要只是为 `_pad_sample()` 和 prepare 的统计服务。 | 在**未 padding 的** `BOS + image protocol + caption + EOS` 上直接调用 `build_shifted_labels()`，此时 BOS→image target 会被 mask、最后 caption→EOS 会保留、EOS 位置自然为 -100；再统一右 pad token/type/index/label。这样可以删除 `assemble_caption_sequence()` 的 `supervised` 返回值、`_pad_sample()` 中自写 shift/mask 逻辑、prepare 对 `sum(supervised)` 的依赖以及一批只验证该重复实现的 UT。核心监督语义仍由现有共享 helper 守住。 |
| S3 | **High / non-production code in production module** | `cc12m_loader.py::_sha256_prefix()`、`digest_batches()`、`_main()` 及 `argparse/hashlib` 相关代码 | `digest` 是手工 A/B preflight/debug 工具，不参与 Trainer/dataloader 生产调用链，却占据 model production module 后半段并引入一套 `world/global_batch_size/num_batches` 消费量模型、tensor hashing 和 CLI parser；相应测试/README 又继续扩张。 | 从 `torchtitan_npu/models/.../cc12m_loader.py` **删除整套 digest CLI**。如果 A/B 数据检查确实需要保留，应放到一次性验证脚本/测试工具而不是模型 production module；更推荐让正式 integration case 和固定测试资产承担回归，不再维护第二个“模拟训练消费”的 CLI。删除后同步去掉 `test_digest_consumption_accounting` 与 README digest 教程。 |
| S4 | **Medium / half-contract complexity** | `cc12m_loader.py::_verify_meta()`；`prepare_cc12m.py` meta 写入；`test_dataset_verifies_meta_identity()` | 当前 `meta.json` 是“有则验证、没有也允许”的可选半契约；loader 为此做 manifest hash、tokenizer 多文件 hash、processor 字段逐项对比，但又不验证图片内容、`seq_len` 等完整身份。R11 若继续沿“再补更多字段/全量 hash”方向，会把训练 loader 继续膨胀成 dataset verifier。 | 从 clean-code 角度建议二选一，**不要继续维持中间态**。本仓更适合的简化方向是：训练 loader 只消费 manifest；`meta.json` 保留为离线 preparation provenance，不在每个 rank 启动时做可选校验，删除 `_verify_meta()` 及其分支测试。如果团队确实把固定数据身份定义为正式运行时能力，则反过来把 meta 变成 required、一次性集中校验的明确组件，但不要继续在 dataset constructor 里逐字段“有则检查”。本专项倾向前者。 |
| S5 | **Medium / defensive over-engineering** | `prepare_cc12m.py::_resolve_tars()`、`_iter_wds_samples()`、`_verify_extracted()`、meta 统计 | 离线 preparer 除了“过滤坏图/超预算、固定 seed、图片 hash 去重”这些真实产品语义外，还承担了通用 WebDataset tar linter：重复 stem、key 重新出现、同 key 重复 extension、non-regular member、写盘后再次完整读回 hash；同时对全部源 tar 做 SHA256、生成多组运行时不消费的统计。固定来源已经是 `pixparse/cc12m-wds`，这些分支大多是在防御“输入 tar 自身格式损坏/并发被篡改”而非训练需求。`seen_keys` 还会随 shard sample 数量增长。 | 保留真正改变样本集合的逻辑：稳定排序、完整 JPEG decode、长度预算、seed、image-hash 去重、选中图片提取。删除 tar-linter 式校验（至少 `seen_keys`/key re-appear、duplicate member、duplicate-stem 专门错误）、`_verify_extracted()` 第三次读盘；若仍想防止 scan/extract 间 tar 被替换，可在 extraction 同一次读 bytes 时比较已记录 hash，不要再单独遍历输出目录。meta 只保留真正有长期消费方的 provenance 字段；例如 `requested_count == selected_count == unique_images` 在当前成功路径上是重复事实，没有必要全部存。Python 3.12 下若仍保留文件 hash，也可直接使用 `hashlib.file_digest`，不必自维护 `_sha256_file()` chunk loop。 |
| S6 | **Medium / test code mirrors implementation** | `tests/unit_tests/models/deepseek_v41/test_cc12m_loader.py` | 527 行 UT 中相当部分是在逐条锁定上述防御分支和 private helper：invalid rank/world、duplicate id、bad manifest field、meta mismatch、loader guard、digest accounting、duplicate tar stem/member、私有 real-tokenizer `skipif` 等。测试数量多，但大量测试的存在理由是“生产代码先写了一个 guard”，形成实现与测试互相固化，而不是守核心训练语义。 | 在 S1-S5 删除后同步大幅删测试，不要为了维持 `77/77` 数量保留无意义 branch。核心 UT 收敛为：①固定独立 oracle 的 sequence/protocol/label；②真实 `ImagePatchProcessor` 边界；③ DP shard 顺序；④ tiny tar preparation 能过滤坏图/overlong 并生成可训练 manifest；⑤ recipe → Trainer Config → `config.dataloader.build(...)` 首 batch；⑥若正式支持 checkpoint，则 resume 顺序。其余 malformed-input linter 测试删除。 |
| S7 | **Medium / defensive checks in new adapters** | `models/deepseek_v41/rope.py::V41RoPERotation.__init__()`；`override/deepseek_v41/mhc.py::AscV41HcPre.__init__()`；`override/deepseek_v41/sparse_attn/ascendc.py::AscV41SparseAttention.forward()` | 新 adapter 中存在多层对内部模型不变量的重复验证。`V41RoPERotation.Config.mode` 已是 `Literal[...]`，运行时再次检查三值没有新增契约；V4.1 model 当前固定 `hc_mult=4/sinkhorn_iters=20`，`AscV41HcPre` 再校验 CANN 支持区间属于防御式代码；sparse adapter 又逐项检查 ratio 集合、CP1/batch/shape、attn_sink、ratio0/1 second-stream shape、plan 完整性等，而这些大部分已由 V4.1 crop/model/metadata 构造链保证。 | 删除由 typed/fixed model config 已保证的 guard，让错误在唯一 owner 处暴露。建议直接删 `V41RoPERotation.__init__` 的 mode check；`AscV41HcPre` 若只服务当前 V4.1 固定 shape，可删整个 override `__init__`；sparse 保留**真正表达 override 依赖或 kernel 外部限制**的检查（例如必须拿到 `AscV41Metadata`，以及若 dtype 确实是用户可配置而 kernel 只支持 BF16，则保留 dtype check），删除 q/kv shape、ratio 枚举、plan completeness 等模型内部 invariant 的重复校验。 |
| S8 | **Medium / duplicated Muon policy** | `config_registry.py::_v41_muon_profile()` 对照 `models/deepseek_v4/config_registry.py::_dsv4_muon_profile()` | V4.1 Muon profile 独立复制了 DSV4 中 dense DP axes、Owned layout、wq_b/wo_a BlockShard、expert sharding、shared/routed experts、router/hc FQN 和 regex 片段等一大段策略。V4.1 的确有“无 MTP/hc_head/APE、indexer 不进 Muon、bucket 拆成 attn/dense/routed”等差异，但这些差异不要求复制所有共同 layout 规则。后续 FSDP/CP axis 或 DistMuon contract 改动会要求两边手工同步。 | **不要让 V4.1 import V4**，independence 边界继续保持；但可以在 `torchtitan_npu/models/common` 提取很小的 model-neutral helper，例如 dense DP axes / Owned layout、expert layout、per-head BlockShard construction。V4/V4.1 各自保留自己的 FQN 选择与 bucket policy。不要做一个巨型“Muon framework”，只抽完全一致且有两个真实调用方的 2~3 个 primitive，确保净减少代码。 |
| S9 | **Medium / dead/no-op launcher code** | A3/A5 scripts | 在 CC12M config 下，A3 仍无条件传 `--dataloader.dataset c4_test --dataloader.dataset-path tests/assets/c4_test`，但 `DeepSeekV41Cc12mDataLoader` 根本不读取这两个值；同时 `HF_ASSETS_PATH` 被改成 CC12M tokenizer 路径，但 Trainer 此时仍构造 `SyntheticTokenizer`，其 `tokenizer_path` 参数被直接忽略，真正的 HF tokenizer又由 CC12M dataset 第二次加载。A5 文件本身只有“设两个 env → exec A3”22 行，是典型可删除 wrapper。 | S1/R1 收敛后删除 CC12M 环境分支、无效 c4 dataloader args、`STEPS` 参数拦截及 A5 wrapper。硬件差异直接由显式 `--override.imports` 表达；CPU affinity 保留公共 launcher 的通用可覆盖值，不再为 A5 新入口写机器专属 wrapper。 |
| S10 | **Low / documentation duplication** | `examples/deepseek_v41/readme.md` | README 把 PR 验证报告再次复制进稳定文档：FUSION50 误差数字、A5 50-step 性能/HBM、当前 host `73.5%`、`rtsMallocHost` 锁页内存事故等都会快速过期；这些文字又需要随每次 kernel/机器变化维护。 | 保留“支持矩阵、单一 CLI、必要硬件限制、数据准备方式、checkpoint/Muon 的稳定用法”。删除手工 benchmark 表、共享机故障记录、PR acceptance 过程；数值证据留在 PR/benchmark report。`TTNPU_DSA_ATTN_CHUNK=128` 如果确实是当前 40-layer 必需约束可以保留，但只写约束，不写某台机器的 HBM 百分比/历史 OOM叙事。 |

### 9.3 可直接删除 / 改写清单

| 当前代码 | 动作 | 理由 |
|---|---|---|
| `DeepSeekV41Cc12mDataLoader.Config.data_dir` | **删除** | `prepare_cc12m.py` 固定 manifest 与 `images/` 同 root，可由 manifest parent 推导 |
| `DeepSeekV41Cc12mDataLoader.Config.tokenizer_path` | **删除** | 复用 Trainer 已构造并传入的 tokenizer |
| `DeepSeekV41Cc12mDataLoader.Config.vocab_size` | **删除** | 直接取 `tokenizer.get_vocab_size()`；避免第二个 vocab source-of-truth |
| `DeepSeekV41Cc12mDataLoader.Config.manifest_path` | **优先删除/复用 inherited `dataset_path`** | 已有 BaseDataLoader CLI 字段可表达 manifest；若认为语义可读性必须保留，则最多保留这一个自定义字段 |
| `_Cc12mCaptionDataset._rank/_world` | **删除** | 只在构造 shard 时需要，存成成员后无读取 |
| loader 内 `HuggingFaceTokenizer(...)` | **删除** | 与 Trainer tokenizer 重复加载/重复所有权 |
| `assemble_caption_sequence()` 的 `supervised` 返回值 | **删除** | label 可复用 `build_shifted_labels()` 在未 pad 序列上生成 |
| `_sha256_prefix` / `digest_batches` / module `_main` | **从 production module 删除** | 不参与训练，只是手工 A/B preflight |
| `_verify_meta()` 可选 verifier | **建议删除** | 当前既不完整也非 required；不要继续扩张半契约 |
| `_verify_extracted()` | **删除** | scan→extract 后再次整文件读 hash 属于额外防御；需要时在 extraction 同次 bytes 上比较 |
| `V41RoPERotation.__init__` 的 Literal 值检查 | **删除** | typed Config 已限定合法值 |
| `AscV41HcPre.__init__` 的 hc_mult/iters guard | **删除或移到唯一 config owner** | 当前 V4.1 shape 固定，adapter 不应重复验证内部不变量 |
| A5 22 行 wrapper | **删除** | 硬件算子选择改由单入口显式 override |
| CC12M 下 `--dataloader.dataset c4_test` / c4 asset args | **删除** | 当前 loader 不消费，是 no-op |
| `test_cc12m_loader.py` 中与上述 guard/digest/meta 一一绑定的测试 | **同步删除** | 不应为了测试数量保留被删实现 |

### 9.4 明确建议保留的“抽象层”

| 代码 | 结论 | 原因 |
|---|---|---|
| `models/deepseek_v41/rms_norm.py::V41RMSNorm` | **保留** | 统一替代 compressor 本地 golden RMSNorm 与 vision `_ReferenceRMSNorm`，同时提供 V4.1-only exact override target；是净去重 |
| `models/deepseek_v41/rope.py::V41RoPERotation` | **保留主体** | attention/compressor/vision 三处原先各有旋转算术；统一 primitive 合理，只有 mode defensive check 可删 |
| `vision_data.py::ImagePatchProcessor.target_grid()` | **保留** | preparation/runtime 共用 raw-size→grid 决策，消除两套 shape 算法；R9 只要求独立 oracle，不要求回退抽取 |
| `attention.py` / `compressor.py` / `vision.py` 本 PR 对本地 RMSNorm/RoPE helper 的删除 | **保留** | 这些改动本身就是 clean-code 去重，不应因专项 review 被反向恢复 |
| `override/deepseek_v41/*` 的独立 Config override 边界 | **保留** | 符合 NPU 私有能力通过 Override 接入、V4.1 不依赖 V4 override 的仓库架构；需要精简的是 adapter 内重复 guard，而不是取消 Override 层 |
| sparse `asc_metadata` 与 `asc` 两个 override | **保留分离** | attention kernel 确实依赖额外 metadata 类型，显式两个 override 比 monkey patch/隐式全局状态更清晰 |

### 9.5 专项优先级

| 优先级 | 精简动作 |
|---|---|
| **S-P0** | S1：只保留一个 tokenizer/data-path source-of-truth；S3：从 production loader 移除 digest CLI；S9：随单入口重构删除 A5 wrapper/no-op args |
| **S-P1** | S2：复用 `build_shifted_labels`；S4：决定 meta 是“离线 provenance”还是“required runtime contract”，不要半契约；S7：清掉新 adapter 中重复 internal-invariant guards |
| **S-P2** | S5/S6：精简 preparer 与对应 UT；S8：仅抽最小 Muon common primitives；S10：README 去掉 PR/机器实验日志 |

### 9.6 专项结论

| 结论 | 说明 |
|---|---|
| **是否存在明显可删代码** | **是，而且不是零碎风格问题。** 最大头是 CC12M 的重复 tokenizer/config、production digest、可选 meta verifier、tar-linter 式校验和对应测试。 |
| **是否建议继续在当前实现上补 validator** | **不建议。** 先删除/复用，把 loader 缩回训练职责；否则 R11/R6 若按“缺什么补什么”继续做，会让 430 行 loader 与 527 行 UT 进一步增长。 |
| **对本 PR 合入的影响** | 专项本身不新增独立 correctness blocker，但 S1/S3/S7 与原 R1/R12/clean-code 职责直接相关，建议在本 PR 内一并收敛。尤其 S1 能同时解决路径 env、`**kwargs` 吞 tokenizer、双 tokenizer/vocab source-of-truth 等多个原 review 点。 |
| **精简后的目标形态** | 一个 launcher + 一个 CC12M recipe；数据位置走现有 dataloader CLI；tokenizer 由 Trainer 单点构造；dataset 只负责 manifest/image/sequence/shard；preparer 只负责确定性过滤/选样/提取；A/B 验证走正式测试而不是 production debug CLI。 |

## 10. V4.1 primitive ownership / 上游复用专项 Review

> **修正说明**：本节是在横向对比 TorchTitan v0.3.0 `models/common`、本仓 `override/common`、`patches/torchtitan/models/common` 后得出的架构结论。它**覆盖并修正**第 9.4 节中对 `V41RMSNorm` / `V41RoPERotation` 的暂定“保留”结论；若两节存在冲突，以本节为准。第 9.4 节当时只判断了“是否减少 V4.1 文件内部重复”，没有继续判断“是否在仓库/上游层面重复造了一套 common primitive”。

### 10.1 总体原则与归属判断

| Primitive / 行为 | 是否应由 V4.1 单独拥有 | Review 结论 |
|---|---|---|
| 普通 RMSNorm（q/kv、Transformer block、final norm、普通 indexer norm） | **否** | 数学/状态语义与 TorchTitan `RMSNorm` 相同，不应仅为了获得 V4.1 专属 exact override target 而换成 `V41RMSNorm`。应继续使用 upstream `RMSNorm.Config` + 本仓 `override/common/rms_norm.py`。 |
| “FP32 归一化 + FP32 weight multiply 后最后一次 cast”的 RMSNorm | **可以有独立语义类型，但不应泛化成所有 V4.1 norm** | vision 与 ratio>1 compressor 的确有冻结 reference 数值顺序，pre-PR 也分别维护 `_ReferenceRMSNorm` / `_golden_rms_norm`。应把这一个语义抽成窄的 `Fp32RMSNorm`/等价类型；若可复用则放 model-neutral common/Extension，否则只在真正需要的 V4.1 callsite 使用。 |
| RoPE cache / YaRN / position reshape | **否** | 已由 TorchTitan `RoPE/ComplexRoPE/CosSinRoPE` 和本仓 split-aware backport/override 负责；V4.1 不应再拥有一套 cache abstraction。 |
| “给外部 cos/sin 做 interleave / complex / half rotation”的算术 seam | **需要一个可替换 seam，但不应是 V4.1-owned** | 这是通用 rotary arithmetic，不包含 V4.1 模型语义。若上游 `RoPE.forward` 当前接口不能覆盖 vision 外部 2D table / 不同 frozen op-order，应在与 `torchtitan.models.common.rope` 对应的 common/Extension 层补一个 model-neutral seam，而不是新增 `models/deepseek_v41/rope.py` + `override/deepseek_v41/rope.py`。 |
| V4.1 vision 的 2D RoPE table 生成 | **是** | `_vision_rope/_vision_rope_batch` 描述 V4.1 vision position layout，属于模型结构；应留在 V4.1 vision。需要复用的是“旋转运算”，不是 2D table 语义。 |
| V4.1 Router（`bias_vl`、`image_mask`、sqrtsoftplus、sorted top-k） | **是** | 这些是明确的 V4.1 VL routing 语义，上游通用 router 不具备，不应为了去重塞进 patch/common。 |
| V4.1 expert 的 frozen dtype / clamp / score-absorb arithmetic | **是** | `V41GroupedExperts` / `V41FeedForward` 的 FP32 gate/up、clamp、router score 放置和 cast 顺序决定 reference 数值轨迹，属于模型语义。 |
| MoE 的通用 dispatch/combine、SP padding、token accounting orchestration | **原则上否** | 这些是框架能力；V4.1 当前因 router extra kwargs / score absorption / dtype contract 不得不复制一部分 upstream `MoE/RoutedExperts.forward`，属于可接受的兼容债，但目标应是通过 upstream hook/common seam 删除重复 orchestration，而不是长期 fork。 |
| Ascend RMSNorm / rotary kernel wrapper | **普通语义应 common；特殊语义才 V4.1-specific** | `override/common/rms_norm.py`、`override/common/rope.py` 已有 `npu_rms_norm/npu_rotary_mul`。V4.1 不应只为“限制 override 作用域”复制同一个 kernel adapter。 |
| Ascend V4.1 grouped-expert override | **是，位置正确** | 新 `override/deepseek_v41/moe.py` 针对 V4.1 frozen expert arithmetic 做 grouped GEMM 替换，属于 NPU + 模型语义交叉点，放 V4.1 override 合理。 |

### 10.2 专项 Finding

| ID | 级别 | 代码位置 | 架构问题 | 建议修改方案 |
|---|---|---|---|---|
| A1 | **High / architecture，建议本 PR 必改** | `models/deepseek_v41/rope.py`；`override/deepseek_v41/rope.py`；`attention.py` / `compressor.py` / `vision.py` 新增 `rotary` Config | **`V41RoPERotation` 解决的是 common abstraction 缺口，不是 V4.1 模型特性。** PR 前 attention、compressor、vision 确实各有不同 operation order 的本地 helper，抽出来是对的；但当前抽成 V4.1-owned Config 后，又在 `AscV41RoPERotation` 中重新实现一遍 `torch_npu.npu_rotary_mul`。横向看，TorchTitan `ComplexRoPE.apply_rotary_emb` 已有 adjacent-pair complex rotation、`CosSinRoPE` 已有 half rotation，本仓 `override/common/rope.py` 已有 interleaved reference arithmetic、`AscComplexRoPE`、`AscCosSinRoPE`、`AscPartialComplexRoPE`。因此当前是“V4.1 内部去重、仓库整体增重”。 | 保留“把 cache 生成和 apply arithmetic 解耦”的设计意图，但把 seam 提到 model-neutral 层：优先让 decoder/compressor/indexer 直接走 `RoPE.forward(query, key=None, positions, inverse=...)`/split-aware API；vision 的外部 2D cos/sin 若无法套现有 RoPE，则在与 upstream `models/common/rope.py` 镜像的 Extension/common 中补**一个** external-cache rotary seam，并让 common Ascend override 实现 `npu_rotary_mul`。完成后删除 `models/deepseek_v41/rope.py` 和 `override/deepseek_v41/rope.py`。 |
| A2 | **High / duplicate configuration** | `DeepSeekV41Attention.Config.rotary`、`Compressor.Config.rotary`、`Indexer.Config.rotary`、`VisionAttention/Block/Encoder.Config.rotary` | **同一 RoPE 语义被拆成 `rope`（cache/positions）+ `rotary`（apply）两套 Config source-of-truth。** `mode="interleave"/"complex"/"half"` 实际是各 callsite 冻结的数值实现细节，却被建模成可配置字段；理论上用户/override 可以组合出不匹配的 cache format + rotary mode。对于 frozen trajectory 来说，这不是需要开放的实验维度。 | 不要把 operation-order 当训练 CLI/config 维度。如果确实必须保留多种 arithmetic 以匹配 frozen reference，应由具体 RoPE implementation/type 固定，而不是每个 V4.1 owner 再携带一个 `rotary.mode`。配置层只表达一个 RoPE implementation。 |
| A3 | **High / over-specialization，建议本 PR 必改** | `models/deepseek_v41/rms_norm.py`；`model_registry.py` 把 q_norm/kv_norm/block/final/indexer norm 全部从 `RMSNorm.Config` 换成 `V41RMSNorm.Config`；`override/deepseek_v41/rms_norm.py` | **只有少数位置需要 V4.1 特殊 FP32 contract，但 PR 把整个模型的 RMSNorm 类型都 V4.1 化了。** `reference_fp32=False` 时 `V41RMSNorm.forward()` 只是 `super().forward()`，这些实例没有新增模型语义；唯一作用主要是给 `@override(target=V41RMSNorm.Config, exact=True)` 提供一个 model-specific tag。与此同时仓里已经有通用 `override/common/rms_norm.py::AscRMSNorm`。这会让以后 upstream RMSNorm API/实现优化必须额外验证一套 V41 wrapper。 | 普通 q_norm/kv_norm、block attention/ffn norm、final norm、无需 FP32 特殊顺序的 indexer norm恢复为 upstream `RMSNorm.Config`，直接复用 common Ascend override。只把 pre-PR 确实调用 `_golden_rms_norm` / `_ReferenceRMSNorm` 的位置收敛到一个**语义命名**的 FP32 norm 类型；其 NPU override 只覆盖这个特殊类型。不要使用“模型名 subclass”作为纯 override tag。 |
| A4 | **Medium / layering inversion（既有问题，本 PR 加深耦合）** | `model_registry.py` 从 `torchtitan_npu.override.common.rope` 导入 `WorkaroundComplexRoPE`；本 PR 再在 model layer 上叠 `V41RoPERotation` | V4.1 reference model 在 PR 前已经直接依赖 `override.common`，说明当前 RoPE 缺口本来就没有放在干净的层级：reference model 应依赖 upstream/common model contract，NPU override 应单向依赖 model，不应反向。新 `V41RoPERotation` 没有解决这一 inversion，只是在其上再加一层 model-specific apply abstraction。 | 把 reference-compatible `WorkaroundComplexRoPE` / split API 放回 upstream-shaped common/Extension（或在 pinned TorchTitan 已具备等价 API 后直接删除 workaround），让依赖方向变成 `model -> common/extension <- override`；不要让 `models/deepseek_v41` import `override.*`。该问题虽非本 PR 首次引入，但本 PR 正在重做 RoPE seam，适合一起收敛。 |
| A5 | **Medium / ownership boundary，MoE 基础定义总体合理** | `models/deepseek_v41/moe.py::{V41Router,V41RoutedExperts,V41GroupedExperts,V41FeedForward,V41MoE}` | **MoE 与 RMSNorm/RoPE 不同：V4.1 单独定义有充分理由，但当前文件混合了“模型语义”和“框架 orchestration”。** `V41Router` 的 VL bias/image_mask/sorted top-k、`V41GroupedExperts`/`V41FeedForward` 的 frozen dtype/clamp/score 顺序必须保留；而 `V41MoE.forward`、`V41RoutedExperts.forward` 的 padding、dispatch/combine、token accounting 大量镜像 upstream/patch，容易随 TorchTitan 升级漂移。 | 本 PR 不要求为了这个既有问题重写整个 MoE；**不要把 V4.1 VL 逻辑搬进 patch**。长期应推动 upstream/common 暴露最小 hook（例如 router extra kwargs / expert score-absorb seam），V4.1 只保留 router + expert arithmetic；一旦 pinned upstream 提供等价 hook，删除复制的 generic orchestration。 |
| A6 | **Medium / NPU override reuse** | `override/deepseek_v41/moe.py::AscV41GroupedExperts.forward()` | 新 grouped-GEMM override 的**归属是正确的**：它只替换 `V41GroupedExperts`，没有全局 monkey patch，也没有污染 patch 目录；但实现仍复制了 DTensor→local、offset cumsum、SPMD type mutate 这一段框架机械逻辑。 | 当前可接受，但若 V4/V4.1 或其他模型出现第三个同类 override，应立即抽一个 model-neutral grouped-expert helper/hook；不要继续复制。该项不建议把 V41-specific expert arithmetic搬到 `patches/torchtitan`，除非抽出的部分本身是明确可贡献上游的通用 seam。 |
| A7 | **High / anti-pattern rule** | 本 PR `V41RMSNorm` / `V41RoPERotation` 的 exact override 设计动机 | **“为了 exact override 隔离而先造一个模型专属 reference class”不应成为本仓通用模式。** Override 的边界应该跟随真实语义边界；否则每个模型都会出现 `ModelXRMSNorm/ModelXRoPE/...`，上游 common 的演进无法自然下沉，Extension/Override 也会退化成按模型复制算子 adapter。 | Maintainer 规则建议明确：只有当 forward/state/config contract 与 upstream/common 有**真实语义差异**时才定义 model-specific primitive；若差异只是 kernel implementation，优先复用 common Config + common override；若缺少可精确选择的 hook，则修 common/Extension hook，而不是用空 subclass/type tag 绕过。 |

### 10.3 针对三个用户点的最终判断

| 项目 | 为什么当前会单独维护 | 是否架构合理 | 本次建议 |
|---|---|---|---|
| **RMSNorm** | PR 想同时统一 vision/compressor 的 FP32 reference arithmetic，并给 V4.1 fusion 一个 exact override target | **一半合理、一半过度。** FP32 特殊 contract 合理；把所有普通 RMSNorm 都改成 V41 类型不合理 | 收窄到真正 FP32 特殊位置；普通 norm 回 upstream/common；删除“V41 类型仅作 override tag”的部分 |
| **RoPE** | attention/compressor/vision 原来三套 operation order，需要抽 seam 才能统一切换 fused kernel | **抽 seam 合理，但放成 V4.1 专属 primitive + 专属 NPU override 不合理** | 把 seam 提到 upstream-shaped common/Extension，复用现有 RoPE/cache/Ascend override；V4.1 只保留 vision 2D table 和模型侧调用关系 |
| **MoE** | V4.1 有 image-aware routing、sqrtsoftplus/sorted top-k、score absorption、特殊 FP32/clamp/cast reference contract | **核心模型定义合理** | 保留 V41 Router/expert semantic；逐步删除复制的 generic orchestration；新 Asc grouped-expert override 位置可接受，但补 R5/R6 的真实 UT/ST |

### 10.4 目标架构

| 层级 | 应承载内容 |
|---|---|
| TorchTitan / upstream-shaped common | 标准 RMSNorm、RoPE cache/position/scaling、通用 MoE dispatch/combine；若缺 external-cache rotary / router-extra-kwargs / score-absorb hook，应优先向这里补通用 seam并推动上游 |
| `torchtitan_npu/extensions/...`（按上游目录镜像） | pinned upstream 尚未提供、但**模型无关**且可作为上游增强的通用 seam；例如确有必要的 external-cache rotary abstraction。不得包含 V4.1/A3/A5 特有策略 |
| `torchtitan_npu/models/deepseek_v41` | V4.1 独有模型语义：vision 2D position table、CSA2 topology、VL router/image bias、frozen expert arithmetic、必要的 FP32 norm contract（若暂时没有更通用归属） |
| `torchtitan_npu/override/common` | 真正通用的 NPU implementation replacement，例如标准 RMSNorm / 通用 RoPE kernel；不按模型复制同一 torch_npu adapter |
| `torchtitan_npu/override/deepseek_v41` | 只有同时依赖 **V4.1 语义 + NPU kernel contract** 的实现，例如当前 grouped-expert / sparse-attn / mHC 特殊融合 |
| `patches/torchtitan` | 仅放准备贡献 upstream 的临时代码；V4.1 image bias、vision 2D RoPE、模型专属 arithmetic 不得为了“复用”搬进 patch |

### 10.5 本专项优先级与对前文的修订

| 优先级 | 要求 |
|---|---|
| **A-P1 / 本 PR 必须收敛** | A1/A2：不要保留 V4.1-owned duplicate RoPE + 第二套 `rotary` Config；A3：RMSNorm 只为真实特殊语义分型，普通 norm 回归 upstream/common |
| **A-P1 / 建议同 PR 收敛** | A4：既然本 PR 正在重构 RoPE seam，顺带消除 `models/deepseek_v41 -> override.common.rope` 的反向依赖，或至少在 PR 中给出明确迁移 TODO/后续上游落点 |
| **A-P2 / 非本 PR blocker** | A5/A6：MoE 的 model-specific semantic 保留；generic orchestration/hook 随 upstream 演进逐步去重。新 V41 grouped-GEMM override 本身不因“单独定义”被否定，但仍受 R5/R6 测试要求约束 |
| **对第 5/9.4 节的修订** | 原“`V41RMSNorm` 保留”“`V41RoPERotation` 保留主体”“V41 RMS/RoPE override 均归属正确”的表述仅从 V4.1 文件内去重看成立，**从全仓/上游架构看不再成立**。应按本节 A1-A4 的收敛方案复审。 |

### 10.6 专项结论

| 结论 | 说明 |
|---|---|
| **是否应该为 V4.1 单独维护三套 primitive** | **不应该一概维护。** RMSNorm 只保留真正特殊的 FP32 contract；RoPE 不应 V4.1-owned；MoE 的 VL/router/expert semantic 应 V4.1-owned。 |
| **本 PR 是否存在新的架构问题** | **有。** 新增的 `V41RoPERotation` + `AscV41RoPERotation`、以及把全部 RMSNorm 改成 `V41RMSNorm`，都把“operator override 隔离”转化成了 model-specific primitive fork，削弱了与上游 common 的复用。 |
| **是否否定本 PR 的去重目标** | **不否定。** 删除 attention/compressor/vision 原本散落的 `_golden_rope/_apply_rope/_ReferenceRMSNorm` 是正确目标；问题在于抽取后的 owner 层级不对。应“继续去重，但往 common/Extension 收敛”，而不是恢复三份本地 helper。 |
| **MoE 是否也要删掉 V41 定义** | **否。** V41Router/image bias、frozen expert arithmetic 是模型语义；真正应消除的是 generic dispatch/combine/padding 的复制，而不是把 V4.1 语义塞进 upstream patch。 |
