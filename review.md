# PR 787 Maintainer Review

## 1. 结论

| 项目 | 结论 |
| --- | --- |
| PR | GitHub PR #13 / GitCode !787：支持 Swap optimizer 的断点续训 |
| 基线 | `master@4c4079dc932234ae9b438011125c935669820080` |
| PR Head | `pr_787@5caddc9575beb9f959345098ee70ad6f6b08848d` |
| 固定上游 | `torchtitan==0.3.0`，CI 解析并 checkout `v0.3.0`；`torch==2.14.0.dev20260719`，`torch_npu==2.14.0.dev20260808` |
| 变更范围 | `torchtitan_npu/override/common/optimizer.py`、`torchtitan_npu/extensions/components/checkpoint.py`、`tests/unit_tests/override/common/test_swap_optimizer.py`、`docs/feature_guides/muon_optimizer.md` |
| 总体合入建议 | **补充测试后合入**。但不是“只补测试”即可：**R1 是确定的生产代码 correctness 问题，必须先修复；R2 是本仓架构要求下应修复的强耦合问题；R3/R4 是合入前测试前置条件。** |
| 测试执行 | **未执行（仅静态审查）**。PR 描述中的 13 个 CPU UT、2P/4P NPU 实测与性能数据仅作为作者提供的背景信息，本次静态 review 不把它们视为已复现的门禁证据。 |

本 PR 的总体方向是合理的：没有向 `patches/` 塞 NPU 特有逻辑，没有新增训练脚本/额外配置体系；仍通过现有 `swap_optimizer` override 和现有 CheckpointManager 扩展接入，训练入口没有分叉。将 NovaSwap CPU buffer 暴露为 DCP 可识别的 zero-copy tensor view，也比复制一份 optimizer state 再交给 DCP 更符合该功能的内存目标。

当前不能直接放行的原因是：partial optimizer state 的 materialize 逻辑与固定上游 `init_optim_state()` 的语义矛盾；checkpoint bridge 又直接穿透 `SwapEngine._handles` / `_wait` 和 DTensor 私有接口；同时仓内没有任何 NPU ST 同时进入 `swap_optimizer + checkpoint save + resume` 这条真实路径，DTensor shard metadata 也没有 CPU oracle。

## 2. 主要 Review Findings

| ID | 严重度 | 代码位置 | 问题点 | 影响 | 建议修改方案 | 合入要求 |
| --- | --- | --- | --- | --- | --- | --- |
| **R1** | **P1 / Blocker** | `torchtitan_npu/override/common/optimizer.py::_NovaSwapAdamW.ensure_all_state()`；同类风险也存在于 `_ensure_all_optim_state()` 对非 AdamW optimizer 直接调用上游 `init_optim_state()` 的路径 | `ensure_all_state()` 已显式识别 `missing` 参数，但随后调用固定上游 TorchTitan v0.3.0 的 `init_optim_state(self.optimizer)`。上游实现开头是 `if optim.state: return`，即 **optimizer 只要已有任意 state，就完全不会为缺失参数补 state**。所以“部分参数已初始化、部分参数尚未初始化”时，`missing` 虽非空，但 helper 是 no-op；后续 `_build_buckets(unbucketed_ids)` 只会处理已有 `exp_avg` 的参数，checkpoint 会缺掉未初始化参数的 `exp_avg/exp_avg_sq[/max_exp_avg_sq]`。这是代码路径可静态证明的逻辑矛盾，而不是测试风格问题。 | 可能生成不完整 optimizer checkpoint；DCP load 可能因为目标 state key 多于 checkpoint 而失败，或者 direct `load_state_dict()` 进入不完整恢复；即使当前 DeepSeek recipe 第一训练步通常让所有参数产生 grad，也不应让 common override 的正确性依赖这个未声明假设。 | 不要复用“all-or-none”的 `init_optim_state()` 来补 partial state。对 AdamW 建议在 `_NovaSwapAdamW` 内实现 **仅 materialize missing parameters** 的最小路径：临时将 param groups 收缩为 missing params（保留原 group hyperparameters），仅给 missing params zero grad，临时 `lr=0`，调用保存下来的原生 AdamW step，最后完整恢复 param groups / lr / grad；这样不能推进已初始化参数的 `step`/moment。不要手写 AdamW state schema。若决定 partial state 明确不支持，则必须把支持边界写成显式 invariant 并在第一次形成 partial state 时处理，而不是让 checkpoint 静默不完整；从通用 checkpoint 语义看，更建议正确补齐。 | **必须修复**，并增加 R4 中的 partial-state UT。 |
| **R2** | **P1 / Architecture** | `optimizer.py::get_checkpoint_view()` 直接读取 `SwapEngine._handles[tensor_name][0]` 并 import 私有 `_wait`；`_checkpoint_metadata()` 直接调用 `DTensor.__create_chunk_list__()` | checkpoint adapter 穿透两个实现私有层：NovaSwap engine 的 `_handles/_wait`，以及 DTensor 的 `__create_chunk_list__`。本仓对 Extensions 的要求是尽量通过稳定 hook/API 解耦，避免上游或 backend 升级后 patch/extension 轻易 break。这里 optimizer override 已经有 `swap_api` 这一公开 facade，却绕开它访问 engine 内部；同样，DTensor 私有 dunder helper 的兼容性不应散落在 optimizer override。 | NovaSwap handle representation、handle group、ownership/wait 语义一变，checkpoint 代码直接失效；PyTorch nightly/DTensor 内部实现升级也可能让 shard metadata 路径 break。更重要的是 CPU buffer ownership/lifetime 是 NovaSwap 自己的职责，外部读取 `_handles` 无法形成清晰契约。 | 在 `torchtitan_npu/extensions/novaswap/swap_api.py` 增加最小、稳定、只暴露 checkpoint 所需语义的 API，例如 `get_cpu_buffer_for_checkpoint(name) -> torch.Tensor`（内部负责等待完成、验证单 storage handle、返回当前 live CPU buffer；命名可自行调整）。optimizer override 只依赖 `swap_api`。DTensor -> DCP shard metadata 建议集中到独立 helper/compat 层，优先使用 PyTorch DCP/DTensor 已公开的 metadata/protocol；若固定 torch 版本确实只能调用私有 helper，也应封装在单一 compat 点并注明 pinned-version 原因，而不是散在业务 override 中。 | **建议作为本 PR 合入前修改**。至少先消除 `SwapEngine._handles` / `_wait` 直接依赖；DTensor 私有 API 若暂时保留，应集中封装并由 UT 锁定。 |
| **R3** | **P1 / ST Blocker** | `tests/integration_tests/deepseek_v4.py`、`tests/integration_tests/run_tests.py`、`.ci/smoke_test.sh` | 仓内现有两个相关 case 没有交叉：`dsv4_muon_swap_ep2_fsdp2` 启用 `swap_optimizer`，但不启用 checkpoint；`dsv4_checkpoint_resume_ep2_fsdp2` 做 save/resume，但使用 `GOLDEN_OVERRIDES` 且 **没有** `swap_optimizer` / Muon。并且 `build_models_test_list()` 没有包含 `build_deepseek_v4_checkpoint_resume_test_list()`，CI `.ci/smoke_test.sh` 只跑 `--test_suite models`，所以 README 中“checkpoint resume 已注册到门禁 models suite”的表述与执行代码不一致。PR 描述里的手工 NPU 断点续训不能替代仓内 ST。 | 当前 CI 可以分别证明“swap 训练能跑”和“普通 optimizer checkpoint resume 能跑”，但不能证明本 PR 新增的桥接链路在真实 NPU、真实 NovaSwap、真实 DTensor/FSDP/DistMuon 上能保存、退出进程、重新构建 optimizer 后恢复并继续训练。 | 在现有 integration runner 中新增/调整 **一个最小 2P case** 即可，不要加 shell 脚本。建议 `dsv4_muon_swap_checkpoint_resume_ep2_fsdp2`：使用稳定的 DeepSeek-V4 debug model + `GOLDEN_OVERRIDES`（隔离本 PR 与 fused op 噪声）+ `torchtitan_npu.override.common.optimizer.swap_optimizer` + `--optimizer.name=Muon` + EP2/FSDP2 + checkpoint interval=2 + 连续 4 step / 从 step2 新进程恢复到 step4。Muon 配置应同时产生 DistMuon 与 AdamW fallback，单 case 即覆盖两种 swapped state。优先复用现有 `check_resume` 比较连续训练与恢复后 step3/4 的 loss + grad_norm；若 deterministic 条件成立，要求逐值一致。该 case 应进入 `build_models_test_list()` 的默认门禁；不要修改 `.ci/smoke_test.sh` 加专用命令。 | **必须补充仓内 ST 定义并进入默认 suite**。 |
| **R4** | **P1 / UT Blocker** | `tests/unit_tests/override/common/test_swap_optimizer.py`；生产分支 `optimizer.py::_checkpoint_metadata()` / `_NovaSwapAdamW.ensure_all_state()` / `_swap_muon()` | 新增 UT 全部是普通 CPU Tensor + fake swap runtime；整文件没有 `DTensor` case。PR 最核心的新语义之一正是“从 DTensor 生成 global shape / global offset / local size 并交给 DCP reshard”，目前没有独立 CPU oracle。另外 Muon 新测只是把 `DistMuon` monkeypatch 成 `FakeMuon` 并手工塞 `_torchtitan_npu_checkpoint_locations`，它没有走 `_swap_muon()` 真实 producer，因此不能证明 momentum 创建时实际注册了 checkpoint location，更没有 DCP load round-trip。R1 的 partial state 也没有测试。 | shard offset 写错、placement 变化、rank1 offset 错误、producer 没记录 location、partial state 漏存等问题均可能在现有 13 个 UT 下漏过。 | 至少补三类 CPU UT：① `test_optimizer_state_swap_materializes_only_missing_adamw_state_before_checkpoint`：两个参数只初始化一个，保存前记录已初始化参数的 `step/exp_avg/exp_avg_sq`，调用 container `state_dict()` 后断言两个 FQN state 都存在且旧参数 state 完全未推进；② 2-rank Gloo/CPU DTensor `Shard(0)` case，使用真实 mesh/placement，检查 rank0/rank1 的 `global_shape/global_offsets/local_offsets/local_sizes`，并做 DCP save/load 的最小 round-trip；③ DistMuon producer-consumer 薄测试：让 `_swap_muon` 创建 momentum/location，再让 `state_dict()` 消费该 location，而不是手工注入。若真实 DistMuon CPU 构造不可行，可以把 location 记录逻辑抽成纯 helper 并对 producer + consumer 各自做真实对象契约，但不能用“手工塞最终 location”冒充 producer 覆盖。 | **必须补充**。 |
| **R5** | P2 / Test organization | `tests/unit_tests/override/common/test_swap_optimizer.py::test_optimizer_state_swap_rejects_pinned_memory_checkpoint`、`test_native_optimizer_allows_pinned_memory_checkpoint` | 这两条测试验证的是 `torchtitan_npu/extensions/components/checkpoint.py::CheckpointManager`，却被放进 optimizer override 测试文件。仓内 test-format 规则要求 CPU UT 与正式生产目录镜像；而且仓库已经有 `tests/unit_tests/extensions/components/test_checkpoint.py`。 | 测试职责混杂，后续维护者很难按生产模块找到 guard 覆盖；还会让 optimizer test 额外 import checkpoint extension，增加模块全局 patch/导入顺序耦合。 | 将两条 CheckpointManager capability 测试移动到现有 `tests/unit_tests/extensions/components/test_checkpoint.py`；optimizer 文件只保留 swap container/state adapter 行为。 | 合入前建议整理。 |
| **R6** | P2 / Test isolation | `tests/unit_tests/override/common/test_swap_optimizer.py::_build_checkpoint_model()` | helper 直接 `torch.manual_seed(seed)`，修改进程全局 RNG 且没有恢复。仓内 test-format 明确要求 RNG 可恢复，不能污染后续测试。 | 测试顺序变化时可能影响其它依赖随机初始化的 CPU UT，产生隐式顺序依赖。 | 用 `with torch.random.fork_rng(): torch.manual_seed(seed); model = ...` 包住模型初始化；或者使用仓库已有 RNG 保存/恢复 fixture。 | 合入前建议修复。 |
| **R7** | P2 / Clean code | `optimizer.py::_CheckpointableTensor`、`make_checkpointable_view()` 最后的 `isinstance(view, _CheckpointableTensor)` | 本地重新定义了一份与 PyTorch DCP protocol 同形的 runtime protocol，然后在给 `view` 手工 `setattr` 四个字段后立刻做 `isinstance`。这个检查只会再次确认“刚刚 set 的属性存在”，不能验证 shard bounds/offset 语义；真正的 DCP protocol 已由 PyTorch 自己验证。这属于低价值 defensive programming，同时复制上游协议会形成第二份定义。 | 增加维护面；PyTorch protocol 字段变化时本地副本不会自动暴露编译/导入错误，反而可能延迟发现兼容性问题。 | 删除本地 `_CheckpointableTensor` 和该 runtime check。若类型标注确实需要，优先直接 import 固定 PyTorch 版本提供的 `torch.distributed.checkpoint.protocol.CheckpointableTensor`；否则只附 metadata，让 DCP 在真正消费时按自己的 protocol/validator 校验。保留 byte range / compact layout 这类真正保护零拷贝 view 正确性的检查即可。 | 建议清理。 |
| **R8** | P2 / Scope & docs | PR 描述 vs `docs/feature_guides/muon_optimizer.md` | PR 描述写“继续复用标准 DCP 的 save/load、resharding flow”，容易被理解成已经支持跨 world-size reshard；文档又把“跨 world size 恢复及更复杂并行拓扑”明确列为能力边界。后者更准确。 | 支持范围容易被误读；后续用户可能把“复用 DCP planner”当成“跨拓扑已经验证”。 | 将 PR 描述/最终 feature guide 统一为：当前实现携带 DCP shard metadata、使用 DCP 标准 planner/format；**已验证范围仍限定当前列出的同拓扑 checkpoint/resume，跨 world-size/复杂拓扑未验证，不承诺**。如果未来要声明跨 world-size 支持，再补对应 distributed UT/ST。 | 非代码 blocker，但应在合入前统一措辞。 |

## 3. 语义变换与独立 Oracle

| 正向语义单元 | 旧行为 | PR 后行为 | 应使用的独立 oracle | 当前证据 | 结论 |
| --- | --- | --- | --- | --- | --- |
| AdamW swapped moments 保存 | `OptimizerStateSwapContainer.state_dict()` 原先拒绝/不支持 checkpoint | 从 NovaSwap bucket 的 CPU byte buffer 构造 dtype/shape/stride 对应的 zero-copy tensor view，替换 flat optimizer state 中 `exp_avg/exp_avg_sq/max_exp_avg_sq` | 保存前 clone 原生 optimizer state；DCP/load 后逐 key 比 tensor 值、param-group 值与继续一步更新结果 | 新增 async DCP round-trip + direct state_dict round-trip 对普通 Tensor 有直接检查；已有 stock AdamW 对照测试保护 swap step 数值 | **部分覆盖**：DTensor shard / partial state 缺口见 R4 |
| AdamW load 后保持 NovaSwap NPU/DTensor 对象 identity | 普通 `load_state_dict()` 会按 optimizer 语义重建/替换 state 引用，不满足 swap runtime 已注册对象的 identity 要求 | checkpoint 数据先写入 CPU swap buffer view，`load_flat_optim_state_dict()` 时把 swapped key 指回原 NPU/DTensor tensor，finally 再恢复原引用 | 保存原 state tensor identity；load 后 `is` 相同，并与独立 checkpoint 值、继续一步结果一致 | 新 UT 检查普通 Tensor identity 与继续一步更新 | **已覆盖（普通 Tensor）**；真实 DTensor/NPU 由 ST 补齐 |
| DistMuon momentum checkpoint | 原先 momentum 只参与 swap | `_swap_muon()` 创建 momentum 时记录 `(tensor_name, byte_offset)`，state_dict 用 CPU buffer view | producer 实际注册 location + consumer state_dict 读取同一 location + save/load 后 momentum/继续训练一致 | 新 UT 手工构造 FakeMuon 并手工注入 location，只验证 consumer | **部分覆盖** |
| DTensor DCP shard metadata | 无 swap checkpoint metadata | 通过 `__create_chunk_list__()` 写 `global_shape/global_offsets/local_offsets/local_sizes` | 2-rank Gloo + 明确 Shard placement，独立按 global shape/mesh rank 计算期望 offset/size；再 DCP round-trip | 无 DTensor UT | **未覆盖** |
| DCP `async` snapshot 生命周期 | swap CPU buffer 后续训练会复用/更新；异步保存必须在返回后不再依赖 live buffer | 交给 `dcp.async_save` 后应完成 CPU snapshot，允许训练继续复用 NovaSwap buffer | `async_save()` 返回后立即污染 live buffer，future 完成后 load 出来的值仍等于调用时 snapshot | 新 UT 正是这样做，检查 restored params/state | **已覆盖（CPU/no_dist）**；真实多 rank NPU 仍需 ST |
| `async_with_pinned_mem` | native optimizer 允许 | swap optimizer 明确拒绝，避免 staging 与 live NovaSwap CPU buffer 重用发生生命周期竞争 | 配置选择 + 构造 CheckpointManager 的可观察 ValueError；native path 仍委托上游 | 新 UT 有正/反两条 | **已覆盖**，但测试应移动目录 |
| save 前补齐 lazy optimizer state | 原先 checkpoint 不支持 | PR 试图 `_ensure_all_optim_state()`，保证 save/load 目标 tensor 都存在 | 构造 partial optimizer state，保存前后检查 missing 被补齐且 existing state 不变化 | 无测试，而且生产实现存在 R1 bug | **未覆盖 / 当前实现错误** |

## 4. CPU UT 正向功能覆盖

| 正向功能 | 生产代码和应观察结果 | PR 中的直接检查 | expected 来源 | 状态 | 合入前置条件 |
| --- | --- | --- | --- | --- | --- |
| CLI/override 选择 swap optimizer | `override.imports=...optimizer.swap_optimizer` 将 `OptimizersContainer.Config` 替换为 `OptimizerStateSwapContainer.Config` | `test_optimizer_state_swap_override_replaces_optimizer_config` 从 `apply_overrides` 真实入口检查 replacement 数量和 config 类型 | override registry 的用户可见配置结果 | 已覆盖 | 无 |
| 普通 AdamW swap 仍与 stock AdamW 更新一致 | bucket swap wrapper 不改变 AdamW 数学更新/state schema | 既有 `test_optimizer_state_swap_adamw_keeps_stock_state_and_updates` 使用相同梯度对比 stock AdamW 与 swapped AdamW 参数/state | stock PyTorch AdamW | 已覆盖 | 无 |
| AdamW async DCP save/load + snapshot + identity | state_dict 输出 CPU views；DCP async snapshot；load 回原 swapped tensor；继续一步一致 | `test_optimizer_state_swap_async_dcp_round_trip` 检查 view data_ptr、metadata、lr、state tensor identity、params/state、继续一步参数 | save 前 clone + source continued step | 已覆盖（single-rank/plain Tensor） | DTensor/真实 NPU 由下两行补齐 |
| direct `state_dict/load_state_dict` round-trip | 不经 DCP 也能恢复 flat state 与 hyperparameters，并保持 swapped tensor identity | `test_optimizer_state_swap_direct_state_dict_round_trip` 检查 lr、identity、state、继续一步参数/state | save 前 clone + source continued step | 已覆盖（single param/plain Tensor） | 无额外组合 |
| lazy state：全部为空 | save 前为所有参数创建 state，再 bucket/D2H | async round-trip 的 target 侧在 `state_dict()` 时会触发 empty-state materialize | round-trip final state | 部分覆盖 | 与 partial-state case 合并保护即可 |
| lazy state：**部分已存在** | 只补 missing state，不得推进 existing state | 无 | 应以保存前 existing `step/moment` clone + missing FQN 完整 key 为 oracle | **未覆盖 / 实现错误** | 新增 R4①，并修 R1 |
| DTensor shard metadata | Shard 后 global offset/size 正确，DCP 能按 logical global tensor 保存/加载 | 无 DTensor 测试 | 由 mesh rank + global shape + Shard 规则独立计算 | **未覆盖** | 新增 R4② |
| DistMuon location producer -> checkpoint consumer | `_swap_muon.momentum()` 首次创建 momentum 时注册 NovaSwap name/location，state_dict 读取该 location | `test_optimizer_state_swap_muon_uses_registered_checkpoint_view` 手工注入最终 location，只测 consumer | 当前 fixture 自己提供 location，不是 producer oracle | **部分覆盖** | 新增 R4③ |
| swap + `async_with_pinned_mem` 拒绝 | CheckpointManager 在 enable + pinned mode + swap capability false 时拒绝 | 新 test 直接构造 manager，检查 ValueError | 明确配置契约 | 已覆盖 | 移到 `tests/unit_tests/extensions/components/test_checkpoint.py` |
| native optimizer 不被 capability guard 误伤 | 无 capability flag 时继续上游 init | 新 test monkeypatch base `__init__` 检查转发 | 上游调用被 spy 捕获 | 已覆盖 | 同上，移动测试目录；避免继续扩展 implementation-detail 断言 |

## 5. NPU ST 触发判断

| 生产代码变更 | 受影响训练场景 | 为什么需要 ST | 现有代表测试 | 判断 | 合入前置条件 |
| --- | --- | --- | --- | --- | --- |
| AdamW NovaSwap checkpoint view + DCP load | NPU AdamW state D2H 到 pinned CPU，checkpoint 后新进程重新建 optimizer，再 load/H2D/step | CPU fake 不能证明真实 pinned allocator、NPU event、H2D/D2H ownership、DTensor state identity | `dsv4_muon_swap_ep2_fsdp2` 有 swap 但无 checkpoint；`dsv4_checkpoint_resume_ep2_fsdp2` 有 checkpoint 但无 swap | **未覆盖** | 新增/调整一个 2P swap checkpoint resume case，见 R3 |
| DistMuon momentum checkpoint + FlexShard redistribution | EP/FSDP + Muon 下 momentum DTensor、NovaSwap 与 FlexShard transfer stream 交互 | FakeMuon 不进入真实 `_momentum/_prepare_local/_enqueue_storage_to_compute`；CPU 无法证明 NPU stream/event 生命周期 | `dsv4_muon_swap_ep2_fsdp2` 只跑 2 step swap smoke，不保存/恢复 | **未覆盖** | 同一个 Muon resume case同时覆盖 DistMuon + AdamW fallback，不再复制第二个 case |
| DCP `async` 与 live CPU swap buffer 重用 | save future 与后续训练 overlap | CPU test可证明 DCP CPU snapshot 合约，但不能证明真实 NPU D2H/H2D + multi-rank 时序 | 无 swap+async checkpoint ST | **未覆盖** | 建议新 resume case使用 `--checkpoint.async-mode=async`；sync 路径已有普通 checkpoint 基线，不必再复制一套矩阵 |
| `async_with_pinned_mem` 明确不支持 | 初始化配置直接拒绝 | 纯配置/构造分支，无需真实 NPU 才能判定 | CPU UT | 无需 ST | 无 |

### 建议的最小 ST 形态

| 测试 | 模型/配置 | 并行数值 | Override / optimizer | checkpoint | NPU 数 | 完成/精度检查 | golden/expected | 执行阶段 |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| `dsv4_muon_swap_checkpoint_resume_ep2_fsdp2`（拟新增或由现有 resume case调整） | DeepSeek-V4 debug model，优先 Golden operator recipe，避免 fused-op 噪声 | EP2 + FSDP2 | `GOLDEN_OVERRIDES` + `optimizer.swap_optimizer`；`--optimizer.name=Muon` | enable，interval=2，`async_mode=async`；phase0 连续 4 step，phase1 `load_step=2` 后跑 3/4 | 2 | 两阶段正常退出；恢复后的 step3/4 loss + grad_norm 与连续训练相同；若 runner 可读取 checkpoint metadata，再检查 optimizer FQN state 存在 | 同一次连续训练动态基准，不新增静态 golden 文件 | 加入 `build_models_test_list()`，由现有 `.ci/smoke_test.sh -> --test_suite models` 自然收集 |

为什么一个 case 足够：`--optimizer.name=Muon` 的现有 swap case已经被仓库定义为“DistMuon and AdamW NovaSwap”，即同一训练实例包含 DistMuon 主路径与 AdamW fallback；再叠加 save/resume 就能进入本 PR 两类 state 的真实生产路径。第二个 swap checkpoint case 若没有新的 state 类型/并行分支，只会重复资源成本。

## 6. 逐文件代码审查

| 文件 / 位置 | 改动审查 | 结论 / 建议 |
| --- | --- | --- |
| `torchtitan_npu/override/common/optimizer.py`：checkpoint view helpers | 用 CPU uint8 buffer + byte offset + dtype/shape/stride 恢复 logical tensor view，且检查 byte range 和 compact contiguous layout；这与当前 NovaSwap D2H raw-byte storage 契约匹配。PinnedCpuStorage 默认 512-byte alignment，也满足常见 optimizer dtype view 的对齐要求。 | 设计方向可接受。保留真正与 zero-copy safety 有关的 byte range/contiguous 检查；删除 R7 的重复 protocol 防御检查。 |
| 同文件：`get_checkpoint_view()` | `_wait(handle)` 后直接拿当前 CPU buffer，避免额外 state copy；但读取 engine 私有 `_handles`。 | zero-copy 目标合理，边界应下沉到 `swap_api`，见 R2。 |
| 同文件：`_checkpoint_metadata()` | 普通 tensor 生成 local==global metadata；DTensor 从 chunk 得 global offset/size，契合 DCP CheckpointableTensor 所需字段。 | 算法意图正确，但私有 DTensor API + 无 DTensor oracle 是当前风险。 |
| 同文件：AdamW bucket refactor | 把 state 收集、打包、bucket append 拆开，并记录每个 moment 在 flat bucket 中的 byte offset；offset 用 `storage_offset * element_size`，与 raw CPU byte buffer 匹配。 | 分解比在 `_build_buckets` 内继续堆逻辑清晰；建议保持 helper 数量，不再新增 checkpoint 专用 facade 到 container。主要问题是 R1。 |
| 同文件：`_replace_swapped_states_with_checkpoint_views()` | 仅替换实际被 NovaSwap 管理的 state key，non-swapped `step` 等仍走上游 flat state dict；对空 local shard跳过。 | 职责清晰。缺 location 直接报错是必要 invariant，不属于多余防御。 |
| 同文件：`_state_for_optimizer()` / `_restore_optimizer_references()` | direct load 时把 incoming tensor copy 到 CPU view，然后将 flat state key指回原 NPU/DTensor tensor，finally 恢复 state identity 和 `param_names`。 | identity 保持是必要的 NovaSwap contract；`try/finally` 合理。建议补 DTensor/Muon producer-consumer test，而不是再加更多内部字段断言。 |
| `torchtitan_npu/extensions/components/checkpoint.py::__init__` | 通过 optimizer capability flag 拒绝 `async_with_pinned_mem`，native optimizer 默认允许；没有新增 CLI/config，只使用既有 `checkpoint.async_mode`。 | 这是较小且清晰的 integration point，符合“入口单一/CLI 暴露”要求。Capability flag 比在 CheckpointManager 里 `isinstance(OptimizerStateSwapContainer)` 解耦更好。测试位置需按 R5 调整。 |
| `tests/unit_tests/override/common/test_swap_optimizer.py` | 新增 async DCP round-trip、direct state round-trip、Muon checkpoint view、pinned mode guard。多数断言关注值、identity、继续一步，而不只是“能跑”。 | 测试价值总体较好；但关键 DTensor/partial-state/Muon producer 路径缺失，且 checkpoint manager 测试放错目录、RNG 污染。 |
| `docs/feature_guides/muon_optimizer.md` | 移除“swap optimizer 不支持 checkpoint”的旧限制，说明 sync/async 支持以及 pinned mode、cross-world-size/复杂拓扑暂不支持。 | 与实现方向基本一致；建议按 R8 收紧“resharding”支持措辞。 |

## 7. 架构、解耦与 Clean Code Checklist

| 维度 | 审查结果 | 结论 |
| --- | --- | --- |
| 与上游 TorchTitan 解耦 | 继续复用 v0.3.0 的 `get_flat_optim_state_dict/load_flat_optim_state_dict/init_optim_state`，没有复制整套 TorchTitan checkpointer。 | **总体正确**；但 R1 暴露了对上游 helper 语义理解不完整，R2 的 DTensor 私有 API 需收口。 |
| `patches/` 定位 | 本 PR 没有修改 TorchTitan/PyTorch/torch_npu patch。 | **符合要求**。NPU/NovaSwap 特有代码留在 override/extensions。 |
| Extensions 目录定位 | CheckpointManager 增强放在 `extensions/components/checkpoint.py`，目录镜像上游 components。 | **符合要求**。NovaSwap CPU buffer 访问应通过 extension 自己的稳定 API，而不是上层穿透 engine 私有成员。 |
| Override 机制 | optimizer 能力仍由 `override.imports=...swap_optimizer` 选择，没有新增奇怪 config 或专用 shell。 | **符合要求**。 |
| 训练入口单一性 | 无新增训练入口；checkpoint 继续使用既有 `checkpoint.*` CLI。 | **符合要求**。 |
| Monkey patch / upgrade break 风险 | Checkpoint extension 现有模块替换机制不是本 PR 新增；本 PR 新增的最大 break 面是 `SwapEngine._handles/_wait` 与 `DTensor.__create_chunk_list__`。 | **需按 R2 收敛**。 |
| 防御式编程 | byte range、buffer dtype/device、compact layout 是 zero-copy 必要 invariant；本地 runtime protocol `isinstance` 是重复校验。 | 删除 R7；不要为每个理论非法状态继续堆 guard。 |
| 文档刷新 | Muon feature guide 已刷新主要限制。 | 基本通过；PR 描述与能力边界按 R8 对齐。 |
| 性能声明 | PR 描述给出 checkpoint 时长、显存/CPU 内存对比，但仓内没有 benchmark artifact/命令作为可重复证据。 | 不作为 correctness blocker；建议 PR 描述保留硬件、world size、checkpoint mode、命令和采样口径，避免只留孤立数字。 |

## 8. 测试格式审查

| 位置 | 类型 | 代码阅读结果 | 修改建议 |
| --- | --- | --- | --- |
| `tests/unit_tests/override/common/test_swap_optimizer.py` 中两个 CheckpointManager test | 文件位置 | 生产对象属于 `extensions/components/checkpoint.py`，而仓库已有对应 `tests/unit_tests/extensions/components/test_checkpoint.py`。 | 移动到现有 checkpoint test 文件。 |
| `_build_checkpoint_model()` | 状态隔离 | `torch.manual_seed(seed)` 修改全局 RNG，未恢复。 | 使用 `torch.random.fork_rng()` 或等效保存/恢复。 |
| `test_optimizer_state_swap_muon_uses_registered_checkpoint_view` | 辅助对象 | `FakeMuon` + 手工写 `_torchtitan_npu_checkpoint_locations` 跳过本 PR `_swap_muon()` producer。 | 保留 consumer unit test可以，但必须再有 producer-consumer 薄测试；不要把此 test 命名/结论扩展为完整 Muon checkpoint 支持。 |
| async/direct round-trip helpers | 测试结构 | helper 将 fake swap runtime、模型构建、state clone、identity 检查分开，主要断言仍在 test 中，可读性尚可。 | 可保留；不要继续把更多产品逻辑复制进 test helper。 |
| `object.__new__(OptimizerStateSwapContainer)` | 测试入口 | unit-level adapter 测试绕过 container 构造是可接受隔离，但不能证明真实 config/optimizer composition。已有 override selection test只证明 config 被替换。 | 不要求所有 UT 改成完整 Trainer；由 R3 的真实 NPU ST 补足 composition/运行链路。 |

## 9. 合入前置条件

| 优先级 | 动作 | 最小验收标准 |
| --- | --- | --- |
| **1** | 修复 R1 partial-state materialization | 构造两个 AdamW 参数，只让参数 A 先产生 state；checkpoint 前 B 无 state。调用 `state_dict()` 后 A/B 的 `exp_avg/exp_avg_sq`（以及 amsgrad 时 `max_exp_avg_sq`）均存在；A 原有 `step`、moments、参数值完全不变；随后 direct 或 DCP round-trip 后继续一步与 stock/reference 一致。 |
| **2** | 收口 R2 私有实现依赖 | optimizer override 不再直接访问 `SwapEngine._handles` / import `_wait`；通过 `swap_api` 的稳定 checkpoint accessor 获取 live CPU buffer。DTensor chunk 私有接口至少集中在一个 compat helper，不散落业务代码。 |
| **3** | 补 DTensor/Muon CPU oracle | 有明确 mesh/placement/global shape/rank offset 的 DTensor test；Muon location 必须由真实 producer 逻辑产生后再被 state_dict consumer 使用。 |
| **4** | 增加默认门禁 NPU resume case | 一个 <=4 NPU（建议 2P）的 `swap_optimizer + Muon + checkpoint save + 新进程 load + 继续训练` case，经现有 `run_tests.py` 注册并进入 `models` suite；不要在 `.ci/smoke_test.sh` 加一次性命令。 |
| **5** | 清理测试组织/隔离 | CheckpointManager tests 移至 `tests/unit_tests/extensions/components/test_checkpoint.py`；修复 global RNG 污染。 |
| **6** | 对齐文档/PR support wording | 不把“复用 DCP reshard planner”写成已支持跨 world-size；保留当前未验证边界。 |

## 10. ST 不能替代证明的内容

| 项目 | 即使新增 R3 ST 仍不能单独证明什么 | 需要的证据 |
| --- | --- | --- |
| partial optimizer state | 常规模型 4 step 很可能所有参数第一步就已建 state，因此不会触发 R1 | 专门 CPU UT 构造 partial state |
| DTensor shard metadata 的每个 offset/size | 训练恢复成功只能说明当前拓扑可用，不能穷举 metadata | 2-rank CPU DTensor independent oracle |
| 跨 world-size / 更复杂 mesh reshard | 同拓扑 EP2/FSDP2 resume 不代表 world-size 改变仍正确 | 明确设计支持后，再增加对应 distributed test；当前文档保持“不支持/未验证” |
| 性能收益 | smoke/resume pass 不说明 checkpoint 时延和 host/NPU memory 更优 | 独立 benchmark，固定硬件/软件/数据/模式并记录重复测量 |

## 11. 附录：静态入口事实

| 项目 | 静态事实 |
| --- | --- |
| TorchTitan 固定版本 | `requirements.txt` 固定 `torchtitan==0.3.0`；`.ci/setup_torchtitan.sh` 与 `.ci/smoke_test.sh` 均解析版本并 checkout `v0.3.0`。 |
| 上游 `init_optim_state` 关键语义 | v0.3.0 `torchtitan/components/optimizer/utils.py` 在 `optim.state` 非空时立即 return，因此不能用于 partial-state 补齐。 |
| 上游 CheckpointManager | v0.3.0 `_flattened_model_states_sd()` 只展开 model，optimizer container 作为 Stateful 对象直接交给 DCP；本 PR 覆盖 `state_dict/load_state_dict` 是正确接入点。 |
| PyTorch DCP protocol | `CheckpointableTensor` 的关键字段就是 `global_shape/global_offsets/local_offsets/local_sizes`；DCP 自己会校验 shard 数量、维度和 bounds，因此无需本地复制 protocol 做一次同形 `isinstance`。 |
| 当前 default NPU suite | `.ci/smoke_test.sh -> python -m tests.integration_tests.run_tests ... --test_suite models`；`build_models_test_list()` 当前只拼接 DeepSeek V4/V4.1/V3.2 常规列表，没有拼 `build_deepseek_v4_checkpoint_resume_test_list()`。 |
| 当前 swap ST | `dsv4_muon_swap_ep2_fsdp2`：2P、EP2/FSDP2、Muon、`swap_optimizer`、2 steps；无 checkpoint、无 resume、`check_loss=False`。 |
| 当前 checkpoint ST | `dsv4_checkpoint_resume_ep2_fsdp2`：2P、EP2/FSDP2、step2 -> step4 resume、loss + grad_norm 动态对照；无 `swap_optimizer`，且独立 suite 未进入 default `models`。 |
| Review 执行边界 | 本次遵循仓内 `developer-tests-review` 的 review workflow，仅做静态代码/入口审查，不执行 pytest、NPU ST、lint 或 benchmark。 |

---

**Maintainer final:** 当前实现的 zero-copy DCP bridge 思路可以保留；请先修 R1，收口 R2，再用 R4 的 CPU oracle 和 R3 的单一真实 NPU resume case把“值正确、metadata 正确、真实 swap 生命周期正确”三层证据补齐。完成这些前置条件后再重新 review 合入。