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
