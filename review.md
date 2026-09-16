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
