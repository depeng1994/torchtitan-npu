# PR !810 / GitHub PR #10 Maintainer Review

| 项目 | 结论 |
| --- | --- |
| Review 范围 | `master@4c4079dc932234ae9b438011125c935669820080` → `pr_810@08b081ce63b03be87b1129d7e5acb1e56a883866`，12 个文件，+465/-0；同时复核 PR body、历史行间评论、TorchTitan v0.3.0、固定 TorchFT `90d7f68961e7c0bcd4278f7ebba83b5cd876c099`、TorchFT upstream PR #344，以及 torch_npu HCCL 实现。 |
| 总体结论 | **暂不建议合入；补充/收敛下述阻塞项后再合入。** 同步 quorum 的双层 flag、checkpoint 前置约束、optional import 隔离、trainer MRO 和 patch 目录定位本身可以成立；主要风险是新增的 HCCL 可恢复通信生命周期没有任何真实 NPU 路径验证，而且连 process-group 的最小 contract UT 都缺失。 |
| 测试结论 | **补充测试后合入**。按照仓内 `.agents/skills/developer-tests-review` 的 `review测试` 流程执行静态审查；**测试执行：未执行（仅静态审查）**。 |
| 明确禁止项 | 本 review 仅写入 `pr_810`；未向 `master` 提交，未合并 PR，未修改 PR 状态。 |

## 1. 需要修改的问题

| ID | 严重级别 | 代码位置 | 问题点 | 影响 / 核实依据 | 建议修改方案 | 是否阻塞 |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | Blocker | `torchtitan_npu/experiments/torchft/process_group.py`：`ProcessGroupHCCLEx._create_pg()`、`_run_context()`、`_wrap_work()`、`abort()`、`shutdown()` | **HCCL 的“超时后可恢复”语义目前没有被证明。** 这里同时启用了 HCCL backend 自身的 `options._timeout` 和 TorchFT 的 `_WorkAcceleratorTimeout/context_timeout`，并在 abort 时手工串联 `backend.abort() -> backend.shutdown() -> backend.clear_workmeta_list()`。固定 TorchFT 的 NCCL 实现会专门处理 native timeout，目的是让用户态 abort 在 NCCL watchdog 终止进程之前生效；而 torch_npu 的 `WorkHCCL` native timeout 路径会设置异常，在 blocking wait 等路径下会 abort communicator 并按 `TearDown` 传播异常。当前 PR 没有证据证明 Python timeout、HCCL watchdog 和手工 cleanup 三者的竞态一定能落在“可恢复而非进程退出/残留状态”的路径上。 | 这是该 PR 的核心能力，不是普通边缘路径。目标场景明确是 elastic DP + sync quorum；如果 timeout 后不能可靠 reconfigure 并继续下一次 collective，TorchFT 接口本身就失去意义。当前 torch_npu 也存在专门的 communicator reinit/recovery 逻辑，说明 HCCL 生命周期不是仅靠对象置空即可假定安全。 | 优先把 HCCL 生命周期收敛到 torch_npu 当前支持的最小、明确可重建序列；若确实必须保留 `abort + shutdown + clear_workmeta_list`，需要在代码注释中说明每一步解决的 HCCL 状态，并补真实 NPU ST：制造 collective timeout/失败 → `errored()` 可观察 → 新 quorum `configure()` → 新 PG 建立 → 后续至少一个 all-reduce/训练 step 成功，且无 watchdog 残留/旧 workmeta/旧 store 状态干扰。还应验证 native `options._timeout` 不会先于 TorchFT wrapper 走到进程 tear-down；必要时实现 HCCL 专属的 timeout 屏蔽/接管策略。 | **是** |
| R2 | Blocker | `tests/unit_tests/experiments/torchft/test_integration.py`；`tests/integration_tests/*`（本 PR 无改动） | **没有真实 NPU ST，也没有完整的正向调用链测试。** 当前 trainer UT 直接 monkeypatch 掉 `FaultTolerantTrainer.__init__`，因此没有覆盖 `init_distributed -> FTManagerEx -> get_dp_info -> FSDP all-reduce hook -> TorchFT optimizer/quorum -> checkpoint` 这条真实链；`ProcessGroupHCCLEx` 更是 0 个直接 UT。现有 DeepSeek-V4 integration matrix 仍走普通 `scripts/run_train.sh`/既有 config，没有 TorchFT trainer/manager 的触发入口。 | 仓内 test-review skill 对分布式/NPU 行为明确要求用 `tests/integration_tests` 的真实训练路径证明；CPU mock UT 不能证明 HCCL communicator 创建、timeout、abort、reconfigure、FSDP hook 或 checkpoint recovery。PR body 也明确承认本轮未跑完整 NPU ST。 | 不要新增独立 TorchFT shell 脚本；沿现有单一 `torchtitan_npu/train.py` + CLI/config manager + `tests/integration_tests/run_tests.py` 扩展最小 case。至少覆盖：checkpoint + per-replica dataloader checkpoint 开启、2+ FT replica/FSDP、sync quorum、一次真实失败/超时或 membership reconfigure、恢复后继续下一 step 并可提交 checkpoint。若该“基础接口 PR”暂时没有可选择 `FaultTolerantTrainerEx` 的真实入口，建议把 HCCL recovery 代码与后续入口集成 PR 一起落地，而不是先合一段无法在仓内 ST 触发的生命周期代码。 | **是** |
| R3 | Major | `.ci/setup_torchtitan.sh`：`_setup_torchtitan()` | PR 将 `_install_torchft` **无条件挂到通用 `_setup_torchtitan()`**，与“TorchFT 是 opt-in optional dependency”这一边界冲突。`.ci/unit_test.sh` 已经显式调用 `_install_torchft`，所以这里的额外调用并不是本 PR UT 所必需。 | 普通 torchtitan setup 从此会隐式要求 apt/protobuf、Rust/maturin、额外网络镜像和 TorchFT build；即使调用方完全不测试 TorchFT，也被附加这组失败面。当前仓内显式 UT 路径并不依赖这次无条件调用，因此这是可以去掉的架构耦合。 | 保持 `_setup_torchtitan()` 只负责固定 TorchTitan baseline；TorchFT 安装只在明确需要 FT 的 CI job/helper 中显式调用。若未来要让整套 unit test 收集 FT 测试，可保留 `.ci/unit_test.sh` 的显式 `_install_torchft`，不要污染 generic setup helper。 | 否，但建议本 PR 修复 |
| R4 | Major | `.ci/setup_torchtitan.sh`：`_install_torchft()` | CI 使用 `pip install --no-deps <torchft git pin>`，随后只手工补 3 个 OpenTelemetry 包，**把 TorchFT 的依赖真相拆成了 pyproject + shell 中的部分依赖闭包**。固定 TorchFT 自身还声明 `torchmetrics`、`torchx`、`aiohttp`、`requests`、`pydantic` 等依赖；当前脚本实际依赖基础镜像“恰好已经有剩余依赖”。此外临时 `cargo_home` 没有清理。 | 基础镜像或 TorchFT pin 一更新，CI 可能以缺包/版本漂移的形式脆弱失败；shell 里的 3 个版本下限也不会随 TorchFT 元数据自动刷新。 | 尽量让依赖声明保持单一来源：要么在受控镜像策略下安装 `.[torchft]` 的真实依赖闭包，要么把 CI 的镜像/constraint 逻辑集中成明确的 constraints/requirements，而不是 shell 内手抄部分 transitive deps。临时 cargo 目录加 cleanup/trap。 | 否 |
| R5 | Major | `tests/unit_tests/experiments/torchft/test_integration.py` | 单个 `experiments/torchft` UT 文件同时测试 `experiments`、`extensions`、`patches` 三类生产代码，不符合仓内测试目录“跟随生产目录职责”的规则。并且模块顶层 import `torchtitan_npu.experiments.torchft` 会在 pytest collection 阶段自动执行 accelerator monkey patch，修改全局 TorchFT module binding，后续其他测试可能在不知情的情况下运行于 patched 状态。 | 历史评论“UT 合并一下”已经在物理文件数量上落实，但现在属于**过度合并**：测试所有权和全局状态隔离变差。test-review skill 也要求 patch/global state 可恢复或隔离。 | 按生产职责拆成少量而非“一测试一文件”：`tests/unit_tests/experiments/torchft/` 测 manager/process_group；`tests/unit_tests/extensions/torchft/` 测 trainer；`tests/unit_tests/patches/torchft/` 测 accelerator patch。自动 apply/import 隔离建议放 subprocess，避免 collection 阶段污染全局解释器。 | 否，但应随 R2 一并整改 |
| R6 | Major | `tests/unit_tests/experiments/torchft/test_integration.py`：`test_manager_*`、`test_apply_rebinds_*` | 现有 UT 的断言面过窄：manager 只检查“外层 True / 底层 False”，没有检查 `init_sync=True`、timeout 传递、`ManagedProcessGroup.register("dp_replicate")`、unsupported config；patch 只检查 `manager/process_group.synchronize` 两个 binding，没有覆盖 `get_stream_context`、`record_event`、`checkpointing.http_transport`、`collectives`、`futures`、`TorchFTProcessGroup._register`、幂等 apply。 | accelerator patch 的核心难点就是 TorchFT 提前 `from utils import ...` 后的 consumer rebinding；只验证两个消费者不足以防未来 pin 更新时漏绑定。upstream TorchFT PR #344 本身为这类行为提供了约 10 个专项测试，可作为最小测试面参考。 | 复用 upstream #344 的 contract 思路补本仓 patch UT；对 `consumer_bindings` 中所有 eager consumer 做表驱动断言，并验证 `_register` 使用 `cpu + current accelerator`、不重复 device、无 accelerator 时仅 CPU、apply 幂等。manager 同时固化 `init_sync=True` 和 `dp_replicate` 注册。 | 否 |
| R7 | Major | `torchtitan_npu/extensions/torchft/trainer.py`：`FaultTolerantTrainerEx.Config.__post_init__()` | 这里直接调用 `Trainer.Config.__post_init__(self)`，刻意绕过 `TrainerEx.Config.__post_init__()`，当前理由是 TorchFT optimizer config 没有 NPU Muon `materialize()`。**今天功能上合理，但把“绕过整个 NPU Config 后处理”绑定到一个 optimizer 差异上**；以后 `TrainerEx.Config.__post_init__` 若增加与 optimizer 无关的公共校验/规范化，TorchFT config 会静默漏掉。 | 这是典型的 fragile grandparent call。当前 `TrainerEx.Config.__post_init__` 除 `Trainer.Config` 外确实主要做 `optimizer.materialize()` + Muon 约束，所以不是现行功能 bug；风险在后续演进。 | 从 `TrainerEx.Config.__post_init__` 抽出不依赖具体 optimizer 类型的公共 hook，再由普通 TrainerEx/TorchFT Config 分别调用；或者让 optimizer config 暴露统一 capability（例如可选 `materialize`）而不是跳过整个父类后处理。 | 否 |
| R8 | Major | PR body / 变更说明 | **PR 描述已经与当前 squash 内容不一致。** Body 仍称 manager/trainer/HCCL/CPU healing state “统一放在 `experiments/torchft`，不再新增 extensions”，但实际 trainer 已按历史 review 移到 `extensions/torchft/trainer.py`，也没有本次新增的 `manager.py`/CPU healing-state 生产文件；Body 写“7 passed、覆盖 CPU snapshot”，当前 `test_integration.py` 只有 5 个 test function，其中一个 2-way parametrize，即静态可见 **6 个 pytest node**，且没有 CPU snapshot test。 | 这会直接误导后续 maintainer 判断已解决项和测试覆盖。Checklist 还勾选了文档已更新，但 PR body 本身未刷新。 | 刷新 PR 描述/测试结果，使目录结构、当前测试 node 数、实际覆盖面与代码一致；若 CPU snapshot 已被移出本 PR，应明确删除对应宣称。 | 否 |
| R9 | Minor | `torchtitan_npu/patches/torchft/accelerator.py`：module docstring、`apply()` | patch 的目录定位是对的：它对应 upstream TorchFT PR #344 的 generic accelerator-neutral 修复，且当前 pin 正好是该 upstream PR 的 base commit；但 `apply()` docstring 写成“fixed-version equivalent of pending PR”略过度，因为本地只 backport utils + `_register` + consumer rebinding，并没有完整 backport #344 中 generic `ProcessGroupWrapper.abort()` 等全部改动。 | 对当前 NPU HCCL 路径，`ProcessGroupHCCLEx.abort()` 自己覆盖了 abort，所以这不是现行功能缺陷；但过宽的注释会让维护者误以为 upstream PR 的所有语义都已等价复制。 | 把注释改成“backport the accelerator utils / registration subset required by this NPU TorchFT path”，并注明当前 pin/上游 PR 合入后的删除条件。 | 否 |

## 2. 重点语义核实结果

| 检查项 | 结论 | 核实说明 |
| --- | --- | --- |
| 同步 quorum | **通过（设计成立）** | `FTManagerEx.use_async_quorum=True` 是 TorchTitan v0.3.0 的 legacy dispatch flag，用来启用 FSDP all-reduce hook、TorchFT optimizer wrapper 和 loss sync；真正传给 pinned TorchFT `Manager` 的是 `use_async_quorum=False`。固定 TorchFT 的 `OptimizerWrapper.zero_grad()` 会调用 `Manager.start_quorum()`，而 `Manager.start_quorum()` 在 `_use_async_quorum=False` 时立即 `wait_quorum()`，因此底层是同步 quorum。这里不应把外层 bool 名称机械理解为实际 quorum 调度模式。 |
| `init_sync=True` | **设计合理，但 UT 未固化** | 目标是同步训练 + 同步 quorum，启动时同步 membership/state 符合目标；当前测试没有断言该参数，建议补上。 |
| FSDP/replicate PG 接入 | **部分通过** | `ManagedProcessGroup(self._manager)` + `register("dp_replicate")` 与 TorchTitan v0.3.0 `maybe_set_all_reduce_hook()` 的使用方式匹配；accelerator patch 使注册 backend 支持当前 accelerator，而不是固定 CUDA。缺真实 FSDP/NPU ST。 |
| HCCL `group_id` | **未判定为缺陷** | 本地使用 `torchft_quorum_{quorum}_rank_{group_rank}`。虽然 torch_npu 常规 backend creator 的 `group_id` 来自共享 `group_name`，但 pinned TorchFT 的 `ProcessGroupNCCL` 也会按 quorum + `group_rank` 设置 backend `group_name`，因此 rank-scoped identity 有 TorchFT 语义依据；只需通过 HCCL ST 固化，不应据此单独报 bug。 |
| checkpoint 约束 | **通过** | `FaultTolerantTrainerEx` 默认启用 `checkpoint.enable=True` 和 `enable_ft_dataloader_checkpoints=True`，并在 `super().__init__()` 前拒绝关闭任一项；错误发生在分布式/模型/SDC 等资源初始化之前，符合 PR 目标范围。 |
| semi-sync / LocalSGD / DiLoCo | **通过** | `FTManagerEx` 在构造时拒绝 `semi_sync_method != None`，边界清晰，不把未支持能力伪装成可用。 |
| optional dependency 导入隔离 | **通过** | `torchtitan_npu/__init__.py` 不 import 新 `experiments.torchft`；`patches/__init__.py` 也不自动 import `patches.torchft`；只有显式 import FT experiment 才先检查 `torchft` 并应用 patch。缺包错误还区分了“torchft 自身缺失”和“torchft 内部某个 nested dependency 缺失”，不会把后者错误包装成安装提示。 |
| upstream TorchTitan 是否硬依赖 TorchFT | **否** | TorchTitan v0.3.0 的 TorchFT 是实验/可选能力，主包不硬依赖 TorchFT。因此本仓把 TorchFT 放在 `[project.optional-dependencies].torchft` 是合理方向；问题只在 CI generic helper 又把它无条件装回去了。 |
| accelerator patch 目录定位 | **通过** | `patches/torchft/accelerator.py` 对应 generic upstream TorchFT PR #344，不是 NPU-only patch；HCCL 专属实现放在 `experiments/torchft/process_group.py`，没有污染 patches。符合本仓“patch 仅存临时上游贡献代码”的定位。 |
| accelerator consumer rebinding | **机制合理，测试不足** | 因为 `import torchft` 会先让多个模块执行 `from torchft.utils import ...`，只替换 `torchft.utils` 不够，必须重绑定已加载 consumer。当前 `sys.modules` + 固定 consumer map 的方案在 exact pin 前提下可接受；后续升级 pin 时要重新审计 consumer 集合。 |
| `FaultTolerantTrainerEx` 父类顺序 | **当前顺序正确，不应按旧评论反转** | 当前 `class FaultTolerantTrainerEx(TrainerEx, FaultTolerantTrainer)` 的 MRO 为 `FaultTolerantTrainerEx -> TrainerEx -> FaultTolerantTrainer -> Trainer`。`TrainerEx.__init__()` 调 `super()` 后会进入 upstream `FaultTolerantTrainer.__init__()`，然后返回继续构建 NPU SDC/HF32 等扩展。反过来写成 `(FaultTolerantTrainer, TrainerEx)` 时，upstream `FaultTolerantTrainer.__init__()` 自己完整实现初始化且不调用 `super().__init__()`，会直接跳过 `TrainerEx.__init__()` 的 NPU 逻辑。 |
| 训练入口单一性 | **当前 PR 未新增旁路入口，方向正确** | 本 PR 没有新增独立 TorchFT shell/config runner；仓内统一入口仍是 `torchtitan_npu/train.py -> torchtitan.train.main()`。后续把 `FaultTolerantTrainerEx` 接入真实 recipe 时，应继续通过现有 CLI/config manager 暴露，不要再增加专用训练 shell。 |

## 3. 历史行间评论复核

| 历史评论 | 当前状态 | Maintainer 复核结论 |
| --- | --- | --- |
| “UT 测试代码合并一下” | **部分解决，但现在过度合并** | 已合为一个 `test_integration.py`，但该文件跨 `experiments/extensions/patches` 三类生产职责，违反当前仓 test-review 的路径镜像规则。建议按生产 ownership 拆成少量文件，不要退回一测试一文件。 |
| `trainer.py` 应放 extension | **已解决** | 当前在 `torchtitan_npu/extensions/torchft/trainer.py`，位置正确。PR body 仍写“统一放 experiments、不再新增 extensions”，需要刷新。 |
| `FaultTolerantTrainerEx` 父类是否应改成 `(FaultTolerantTrainer, TrainerEx)` | **旧建议不应采用** | 独立沿 MRO 和 upstream `FaultTolerantTrainer.__init__` 核实后，当前 `(TrainerEx, FaultTolerantTrainer)` 才能确保 NPU `TrainerEx.__init__` 被执行；反转会跳过 NPU 初始化。 |
| upstream TorchTitan 是否直接依赖 TorchFT | **已核实** | v0.3.0 主包没有 hard dependency；本仓 optional extra 是合理的。 |
| 目录结构是否应放 `experiments/torchft` | **基本解决** | manager/process group 属于 experiment，trainer 属于 extension，generic upstream backport 属于 patches；当前三者职责分层比“全部塞 experiments”更符合本仓架构。 |

## 4. 逐文件 review

| 文件 | 状态 | Review 结果 / 修改建议 |
| --- | --- | --- |
| `.ci/setup_torchtitan.sh` | **需修改** | `_install_torchft` 本身可以作为 FT 专用 helper，但不应由 generic `_setup_torchtitan()` 无条件调用；`--no-deps` + 手工 3 个 OTEL 依赖也应收敛为单一依赖真相，并清理临时 cargo home。见 R3/R4。 |
| `.ci/unit_test.sh` | **基本可接受** | UT suite 新增 FT 测试后显式 `_install_torchft` 有直接目的；但最好保证这是 FT 测试需要的显式依赖，而不是反向依赖 `_setup_torchtitan()` 的隐式安装。 |
| `pyproject.toml` | **通过** | TorchFT 放在 `[project.optional-dependencies].torchft` 且 pin 到精确 commit，符合可选依赖和 patch 可复现要求。pin 正好对应 upstream #344 base，后续 #344 合入并升级 pin 后应删除本地 patch。 |
| `tests/unit_tests/experiments/torchft/test_integration.py` | **需修改** | 正向覆盖不足、没有 HCCL contract UT、patch consumer 覆盖不足、目录职责混放、collection 阶段 monkey patch 全局状态；当前静态可见 6 个 pytest node，不是 PR body 的 7。见 R2/R5/R6/R8。 |
| `torchtitan_npu/experiments/__init__.py` | **通过** | 只增加 package marker/docstring，不会触发 TorchFT import，保持普通 import 隔离。 |
| `torchtitan_npu/experiments/torchft/__init__.py` | **通过** | opt-in import 才检查依赖；只在 `error.name == "torchft"` 时转换为安装提示，nested missing dependency 继续原样抛出，行为正确；随后显式加载 TorchFT patch，入口清晰。 |
| `torchtitan_npu/experiments/torchft/ft_manager.py` | **通过但需补测试** | sync quorum 双层 flag 设计经 pinned TorchFT 调用链核实成立；HCCL-only、semi-sync rejection 边界明确。建议补 `init_sync=True`、timeout、`dp_replicate` register、unsupported config 的 UT。 |
| `torchtitan_npu/experiments/torchft/process_group.py` | **阻塞** | 结构上沿用了 TorchFT accelerator PG wrapper 模式，但 HCCL native timeout 与 wrapper timeout、abort/shutdown/clear-workmeta 的恢复语义未被验证；没有直接 UT/ST。见 R1/R2。 |
| `torchtitan_npu/extensions/torchft/__init__.py` | **通过** | extension package 本身不自动 import trainer/TorchFT，未破坏 optional import 边界。 |
| `torchtitan_npu/extensions/torchft/trainer.py` | **基本通过** | 目录位置和 MRO 正确；checkpoint 前置约束正确；`init_distributed` 先设置 NPU SPMD backend 再进入 upstream FT init 合理。`Config.__post_init__` 的 grandparent call 建议重构，见 R7。 |
| `torchtitan_npu/patches/torchft/__init__.py` | **通过** | patch 仅在显式 FT experiment 导入后才加载，没有加入全局 `patches/__init__.py` auto-import；符合 optional dependency 隔离。docstring 可更准确强调这是 generic upstream backport。 |
| `torchtitan_npu/patches/torchft/accelerator.py` | **基本通过但需补测试** | generic accelerator-neutral patch 与 upstream #344 方向一致，consumer rebinding 的必要性成立；但当前本地测试覆盖远弱于 upstream PR，且“equivalent”描述范围过宽。见 R6/R9。 |

## 5. UT / ST 覆盖矩阵

| 语义单元 | 当前覆盖 | 判定 | 不能证明的内容 | 建议 |
| --- | --- | --- | --- | --- |
| 普通 `import torchtitan_npu` 不要求 TorchFT | subprocess 阻断 `torchft` import | **已覆盖** | 无明显缺口 | 保留 subprocess 隔离方式。 |
| 显式 FT import 缺包时提示 extra 安装 | 同一 subprocess | **已覆盖** | 未覆盖 torchft 内部 nested dependency 缺失，但生产代码逻辑本身已正确区分 | 可选补一条 nested missing dependency 不被重写的 UT。 |
| outer legacy hook flag=True / inner sync quorum=False | mock `Manager` kwargs | **部分覆盖** | 未覆盖 `init_sync`、真正 `start_quorum()->wait_quorum` 行为、Managed PG 注册 | 本仓只需固化 wiring；TorchFT 自身行为可依赖 pinned upstream tests。 |
| checkpoint 必须开启 | 2-way parametrize | **已覆盖** | 没有真实 checkpoint save/load/recovery | 后者必须进 NPU ST。 |
| trainer MRO 保留 NPU SDC/HF32 | monkeypatch upstream FT init | **部分覆盖** | 上游 FT 初始化、mesh、FSDP hook、optimizer/checkpoint 完全被 mock 掉 | 保留轻量 MRO UT，同时新增真实 integration path。 |
| HCCL `_create_pg` | 无 | **未覆盖** | Options timeout/group id/global ranks、CUSTOM backend 注册、sequence number | 增加 mock contract UT。 |
| HCCL `_wrap_work/_run_context` | 无 | **未覆盖** | per-op timeout、callback abort、同步/异步 Work 行为 | 增加 contract UT，并用 NPU ST 验证 watchdog 竞态。 |
| HCCL `abort/shutdown -> configure` | 无 | **未覆盖** | communicator/workmeta 是否可重建、是否进程退出/挂死 | **必须补 NPU ST**。 |
| accelerator utils | 间接/极少 | **部分覆盖** | stream/event/synchronize 对 current accelerator 的行为 | 参考 upstream #344 utils tests。 |
| accelerator consumer rebinding | 只测 manager/process_group 的 `synchronize` | **部分覆盖** | 其余 consumer、`get_stream_context`、`record_event`、`_register` | 表驱动覆盖 consumer map + registry。 |
| DeepSeek-V4 Flash + elastic DP + checkpoint + sync quorum | 无真实 integration case | **未覆盖** | PR 的最终目标场景全部不能由当前测试证明 | 使用既有 `run_tests.py`/CLI 增一个最小 NPU case，不新增独立 shell。 |

## 6. 建议的最小修改集

| 优先级 | 修改项 | 验收条件 |
| --- | --- | --- |
| P0 | 给 `ProcessGroupHCCLEx` 增加可恢复 timeout/reconfigure 的真实 NPU ST，并根据结果收敛 native timeout 与 abort/shutdown lifecycle | 失败/超时后同进程可建立下一 quorum PG，并成功执行后续 collective/训练 step；没有 watchdog 直接结束进程，也没有旧 workmeta/communicator 影响新 PG。 |
| P0 | 补 HCCL process-group mock contract UT | `_create_pg` options/backend 注册、timeout wrapper、abort/shutdown 状态转换、重新 configure 均有独立 oracle。 |
| P1 | 从 `_setup_torchtitan()` 移除无条件 `_install_torchft` | 普通 setup 不再安装 optional TorchFT；FT unit/integration job 显式安装。 |
| P1 | 收敛 TorchFT CI dependency 安装方式 | 不再依赖 shell 手抄“部分 transitive deps”；依赖版本来源清晰且可随 pin 更新。 |
| P1 | 调整 UT 目录归属和 patch 测试隔离 | experiments/extensions/patches 测试跟随生产 ownership；不会在整个 pytest collection 中无边界污染 TorchFT globals。 |
| P1 | 扩充 accelerator rebinding/registration UT | 覆盖 `consumer_bindings` 全集、current accelerator registry、apply 幂等。 |
| P2 | 重构 `FaultTolerantTrainerEx.Config.__post_init__` | 不通过直接跳到 grandparent 来规避 optimizer-specific hook；公共 NPU config 后处理可被 FT config 继承。 |
| P2 | 刷新 PR body | 目录、文件、测试 node 数和真实覆盖与当前 squash 一致；移除当前不存在的 CPU snapshot/healing-state 声明。 |

## 7. 上游/基线对照

| 对照项 | 版本 / 状态 | Review 用途 |
| --- | --- | --- |
| TorchTitan | `v0.3.0`（本仓 `requirements.txt` 固定基线） | 核对 `FaultTolerantTrainer` MRO、`TorchFTManager.use_async_quorum` 的 legacy dispatch 用法、FSDP hook、optimizer/checkpoint 调用链。 |
| TorchFT | `90d7f68961e7c0bcd4278f7ebba83b5cd876c099` | 核对 `Manager.start_quorum()` 同步等待、`OptimizerWrapper`、`ProcessGroupNCCL` timeout/group-name/reconfigure 行为。 |
| TorchFT PR #344 | `https://github.com/meta-pytorch/torchft/pull/344`，review 时仍为 open；base 正是本 PR pin 的 TorchFT commit | 证明本仓 `patches/torchft/accelerator.py` 属于可上游的 generic 临时 backport；同时对照 upstream 专项测试面。 |
| torch_npu HCCL | `Ascend/pytorch` 当前实现 | 核对 `ProcessGroupHCCL.Options`、backend registration、native `WorkHCCL` timeout/abort/shutdown 语义，识别 R1 的恢复风险。 |
