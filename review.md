# PR !829 / Mirror PR #11 Maintainer Review

| 项目 | 结论 |
| --- | --- |
| PR | `fix: derive DeepSeek V4 CP sizes from symbolic input shapes` |
| Mirror | `https://github.com/depeng1994/torchtitan-npu/pull/11` |
| 审查代码 | `pr_829@642e8f58d697218f3522e93b29d2306b4e8123bf`（相对 `master@4c4079dc932234ae9b438011125c935669820080`） |
| 固定 TorchTitan | `torchtitan==0.3.0`，CI 实际 checkout `v0.3.0` |
| 顶层结论 | **补充测试后合入** |
| 测试执行 | **未执行（仅静态审查）** |
| master 操作 | **无；本报告只允许提交到 `pr_829`** |

## 1. 结论与语义变换

| 维度 | 审查结论 | 依据 |
| --- | --- | --- |
| 生产代码正确性 | **基本成立** | PR 的两个根因与改动一一对应：编译路径不再从 split tensor `.tolist()` 生成 unbacked SymInt，而改为从 shape-carrier 的输入维度读取；`select()` 不再直接消费按值特化的 Python `out_width`，而从 `container_shape.shape[0]` 读取。eager 仍返回原 host split list，row selection / zero padding 语义未改变。 |
| 架构边界 | **符合仓库定位** | 变更留在 DeepSeek-V4 `metadata.py` / `token_dispatcher.py`，没有把模型限定逻辑塞入 `torchtitan_npu/patches`，没有新增 PyTorch/TorchTitan patch，也没有新增用户训练入口、config 文件或 shell recipe。 |
| 与上游解耦 | **生产代码可接受；测试代码有一处应收敛** | 生产路径仍沿本仓 DSV4 CP dispatcher 扩展，不 monkey-patch 上游；但新增 CPU worker 直接调用固定上游 `SelectiveAC._wrap_block()` 私有方法，见问题 R2。 |
| clean code / 精简 | **主实现较小且职责清楚** | `_shape_tensor()` 被 `CompressedBlockLayout` 和 `ExchangePlan` 复用，避免两套符号尺寸实现；`container_shape` 通过 dataclass pytree 进入编译输入。CP2/CP4 两套完整多进程 Inductor 回归可进一步减重，见 R3。 |
| 防御式编程 | **未发现新增低价值防御校验** | `select()` 中的 `container_shape is not None` 属于内部 plan 构造不变量，与原 `out_width is not None` assert 等价，不是新增用户输入拒绝矩阵。 |
| 文档一致性 | **代码注释已跟随实现；PR 测试说明缺失** | PR body 勾选“已经自己测试过”，但 `## 如何测试` 为空；需要补实际命令/结果，见 R4。 |

### 语义变换与独立 oracle

| 变换 | 旧路径 | 新路径 | 应保持/证明的结果 | PR 中独立检查 |
| --- | --- | --- | --- | --- |
| uneven all-to-all split 的编译期表示 | `send_splits_tensor.tolist()` / `recv_splits_tensor.tolist()`，SelectiveAC 重计算时可能生成未被 backward graph 捕获的 unbacked SymInt | 每个 peer 用零元素 CPU `[size, 0]` tensor 承载；编译/trace 时读取 `shape.shape[0]`，eager 仍返回 host list | collective 的 send/recv split、row 顺序、forward、backward 均不变；不同非 0/1 split 只复用同一图 | `cp_compile_worker.py` 使用真实 2/4-rank Gloo、SelectiveAC、Inductor；用独立构造的全局 source rows 精确比较 output，并手算 input gradient；前 6 个动态 batch 断言 `len(graphs) == 1` |
| compressed container width 的编译期表示 | `plan.out_width` Python int 直接参与 padding shape，packed batch 改变宽度时按值特化 | `CompressedBlockLayout.__post_init__()` 构造 `container_shape=[out_width,0]`，`select()` 使用 `container_shape.shape[0]` | `compressed_rows` 选择与 trailing zero padding 不变；宽度变化不应导致普通动态 batch 每次重新编图 | worker 逐 step 改变 `out_width`，独立构造 expected container，精确比较 selected rows + zero padding，并纳入同一 graph-count 断言 |
| eager 行为 | host `send_splits` / `recv_splits` | eager 分支继续原样返回 host list | 不引入 eager collective API 行为变化 | 现有真实 Gloo / dispatcher UT 继续经过 eager 路径；本 PR 未改 eager 返回值 |

## 2. UT 正向功能覆盖

| 正向功能 | 生产代码和应观察结果 | PR 中的直接检查 | 状态 | 合入前置条件 |
| --- | --- | --- | --- | --- |
| SelectiveAC + Inductor 下 dynamic split 不产生丢失的 unbacked SymInt，且 collective forward/backward 正确 | `ExchangePlan.__post_init__` → `splits_for_collective()` → `CPTokenDispatcher.gather()` → `_all_to_all()`；compiled/tracing 路径把 shape-derived sizes 交给真实 `all_to_all_single` | `cp_compile_worker.py`：真实 Gloo process group + CPU DeviceMesh；`DispatchBlock` 两次真实 dispatcher gather；手工 global-row oracle；exact output / exact input-grad；6 个普通动态 batch 只生成 1 个 Dynamo backend graph | **部分覆盖** | 数值/collective/compile 证据是有效的；但 SelectiveAC 通过上游私有 `_wrap_block()` 进入，而生产入口是 `ActivationCheckpointing.apply()`，建议按 R2 改为 public production entry |
| `out_width` 随 packed batch 变化时 container selection/padding 保持正确并复用动态图 | `CompressedBlockLayout.__post_init__` → pytree flatten → `CPTokenDispatcher.select()` | worker 每 step 构造不同 `CompressedBlockLayout(out_width=...)`，`__post_init__` 真实生成 carrier；独立 expected 验证 duplicate selection、末尾 zero pad；与 split 一起进入 graph reuse 断言 | **已覆盖（CPU 语义）** | 无额外 CPU 数值测试要求；真实 NPU Inductor 兼容性转 ST 覆盖 |
| shape carrier 能随真实 plan producer 进入 consumer | 非 CP `build_kernel_layout()` 和 CP `_assemble_block_plan()` 都以 `out_width` 构造 `CompressedBlockLayout`；CP window/block exchange 都经 `_build_exchange_plan()`；dataclass pytree 注册枚举全部 fields，因此 `container_shape` 是 tensor leaf | 现有 `build_cp_plan` / dispatcher UT 保护 plan geometry；新增 worker直接使用同一 `CompressedBlockLayout.__post_init__` / `_build_exchange_plan` 构造路径 | **已覆盖** | 无需再复制 plan-builder 数值矩阵 |
| 不把 CPU fake 当成真实 collective | `CPTokenDispatcher._all_to_all()` 的 compiled/non-spmd 分支使用 functional `all_to_all_single` | 新 worker 直接初始化多进程 Gloo，没有覆盖 `_all_to_all`；旧 `test_multiprocess_gloo` 也保留真实进程组 | **已覆盖** | 无 |
| 任意 CP degree 的 split-list 长度不被写死为 2 | `send_split_shapes` / `recv_split_shapes` 是按 host split list 长度构造的 list | 参数化 `world=2,4` 都跑完整 SAC + Inductor + Gloo worker | **已覆盖，但测试成本可缩减** | 见 R3：CP4 只需保留能证明“peer 数量不写死”的低成本 contract 检查，不必复制整套重编译回归 |

## 3. ST 触发判断

固定上游 `torchtitan v0.3.0` 中：`Trainer.Config.activation_checkpoint` 默认是 `SelectiveAC.Config`；标准 DeepSeek-V4 路径复用上游 `parallelize_deepseekv3`，其顺序是先 `ac_config.build(...).apply(model)`、再 `apply_compile(...)`；`CompileConfig.backend` 默认是 `inductor`。因此 PR body 声称的 **CP + SelectiveAC + Inductor** 是真实训练路径，不是测试专用组合。

| 生产代码变更 | 受影响训练场景 | 为什么需要 ST | 现有代表测试 | 判断 | 合入前置条件 |
| --- | --- | --- | --- | --- | --- |
| `ExchangePlan.splits_for_collective()` 改变 compiled collective 的 split 表示 | DeepSeek-V4 SMLA，CP>1，默认 SelectiveAC，NPU `torch.compile`/Inductor，backward 含 uneven all-to-all | CPU Gloo 能证明 collective 语义和 AOT/Inductor 核心回归，但不能证明 torch_npu/HCCL/Ascend Inductor codegen 对 shape-derived split SymInt 的真实兼容性 | `dsv4_smla_cp2_ep2_fsdp2`：CP2 + EP2 + FSDP2 + `spmd_types`，默认 SAC，但显式 `--compile.backend=aot_eager` | **部分覆盖** | **新增**一个最小 CP2 + NPU Inductor + 默认 SAC case，沿现有 `tests/integration_tests` runner，不新增脚本；见 R1 |
| `CPTokenDispatcher.select()` 的 output width 改为 input tensor dimension | packed document 边界变化导致每 rank `max_kept/out_width` 变化的 CP 训练 | 这是 PR 的第二个直接根因；现有 NPU CP case 只有 `aot_eager` 且仅 1 step，不能进入 Inductor 的 value-specialization/codegen 路径 | 同上 | **未覆盖目标 backend** | 与上一行合并到同一个新增 case；建议至少 2 个 training step，使 packed batch 有机会变化；性能上的“始终不重编译”仍由 CPU graph-count UT 证明，ST 只负责真实 NPU 完成性 |

## 4. Review 问题与修改建议

| ID | 级别 | 代码位置 | 问题点 | 影响 | 建议修改方案 |
| --- | --- | --- | --- | --- | --- |
| **R1** | **阻塞** | `tests/integration_tests/deepseek_v4.py::build_deepseek_v4_test_list`；现有 `dsv4_smla_cp2_ep2_fsdp2` | PR 的目标故障明确发生在 **SelectiveAC + Inductor**。现有真实 NPU CP testcase 虽然默认使用 SelectiveAC，但强制 `--compile.backend=aot_eager`；因此它不会进入本 PR 修改所针对的 Inductor codegen 路径。新增 CPU Gloo test 不能替代真实 NPU/HCCL/torch_npu backend 的 ST。 | 目前无法从仓库测试矩阵确认修复在真实训练环境可工作；可能出现 CPU 通过、NPU Inductor 仍在 backward/codegen 失败的情况。 | 在现有 runner 中新增最小 `dsv4_smla_cp2_inductor_sac`（名称可等价）：`use_golden=False`、CP2、`spmd_types`、`--compile.enable`、backend 使用固定上游默认 `inductor`（或显式 `--compile.backend=inductor`）、默认 SelectiveAC、2 steps、2 NPU、`check_loss=False`。无需为了本 bug 同时叠加 EP/FSDP；目标是让真实 DSV4 CP sparse-attention 的 init/forward/backward/optimizer step 完成且不出现未定义 SymInt/codegen 错误。同步更新 `tests/integration_tests/README.md` 矩阵。 |
| **R2** | **应修改（上游解耦）** | `tests/unit_tests/models/deepseek_v4/cp_compile_worker.py::main`：`SelectiveAC.Config().build()._wrap_block(...)` | 新测试直接绑定固定上游的 **private** `_wrap_block()`。真实生产路径调用 public `ActivationCheckpointing.apply(model)`；`apply()` 还执行 `_disable_dynamo_lru_cache()` 后再逐 layer 调 `_wrap_block()`。测试绕过 public entry，一方面与真实路径不完全一致，另一方面 TorchTitan 升级后 private 方法签名/行为变化会让本仓测试脆弱。 | 违反本仓“尽量与上游解耦、优先复用稳定入口”的维护目标；将来升级 TorchTitan 时容易出现与业务无关的测试 break。 | 构造一个最小 holder module，暴露 `layers`（如 `ModuleDict({"0": DispatchBlock(mesh)})`），调用 `SelectiveAC.Config().build().apply(holder)`，再编译 holder/layer。这样仍保持最小 CPU regression，同时覆盖生产实际 public AC 入口及其 LRU-cache 设置。 |
| **R3** | **建议** | `tests/unit_tests/models/deepseek_v4/test_cp_dispatch.py::test_cp_dispatch_sac_inductor_outputs_gradients_and_graph_reuse` 参数 `world=[2,4]` | CP2 和 CP4 都启动完整多进程 Gloo + SAC + Inductor，并执行多批动态编译；两者走的是同一生产 branch，CP4 的新增价值主要是证明 split-list 长度不是硬编码为 2。 | 全量 `tests/unit_tests` CI 会额外承担 2-rank + 4-rank 两套重编译/子进程成本；第二个组合的维护/运行成本高于它新增的语义覆盖。 | 保留 CP2 的完整 real-Gloo/SAC/Inductor/output+grad+graph-reuse 回归；把 CP4 降为低成本 contract 测试（例如参数化现有 `test_cp_gather_compile_reuses_graph_across_dynamic_splits_gloo` 的 peer 数，或单独验证 4-way shape-derived split list 可进入同一 compiled graph）。若坚持保留 CP4 全链路，应在测试注释中明确它能捕获 CP2 无法捕获的具体 failure。 |
| **R4** | **流程/文档** | PR body `## 如何测试` | Checklist 勾选“已经自己测试过”，但测试章节为空，无法核对作者实际执行了哪些命令、CP2/CP4 是否都运行、是否有 NPU 结果。 | PR 描述与证据不一致；后续定位该 compiler 回归时缺少可复现命令。 | 在原 PR 描述中补实际测试命令和结果。至少列 CPU 目标 pytest node；补 R1 后再列真实 NPU integration 命令。不要仅写“self-tested”。 |
| **R5** | **应修改（编译输入证据）** | `torchtitan_npu/models/deepseek_v4/metadata.py:L134,L149-L163,L347-L385`；`torchtitan_npu/models/deepseek_v4/token_dispatcher.py:L490-L527`；`torchtitan_npu/models/deepseek_v4/model.py:L63-L72,L193-L248`；`tests/unit_tests/models/deepseek_v4/cp_compile_worker.py:L23-L27,L44-L74`；固定上游 `torchtitan/distributed/compile.py@v0.3.0:L45-L66`、`torchtitan/components/checkpointer/base.py@v0.3.0:L118-L126`；`torchtitan_npu/override/deepseek_v4/sparse_attn/ascendc.py:L200-L229` | **(a) ExchangePlan 的 shape carrier 是否真是 graph input：未核实。** `CompressedBlockLayout`/`CompressedVarlenMetadata` 有 pytree 注册，flatten 会把 `exchange` 字段作为子值继续处理，但 `ExchangePlan` 本身没有 `register_pytree_node_for_dataclass`；其 `send_split_shapes`/`recv_split_shapes` 是 `field(init=False)` 的 tensor list，`splits_for_collective()` 再从对象属性读取 `shape[0]`。本仓静态代码不能证明固定 Torch `2.14.0.dev20260719` 的 Dynamo 最终把这些维度提升为 runtime `SymInt` graph inputs，而不是以 user-object attribute / identity guard 方式捕获；当前 worker 的 backend 只计 graph module 数，不检查 `example_inputs`/placeholder/guard，而且本 review workflow 未执行该 testcase，因此该点明确记为 **未核实**。**(b) 状态污染面：FSDP2/checkpoint 未发现问题；DTensor/GraphTrainer 需区分。** 这些 plan 在 `build_attention_masks()` 中按 batch 构造并写入 `extra_kwargs["attention_masks"]`，不是 `nn.Parameter`/buffer；固定上游 checkpoint wrapper 只收集 `model.state_dict()`，所以新增 CPU tensors 不进入 checkpoint state，也不构成 FSDP2 module state。AscendC `_mark_dynamic()` 的固定字段列表确实不含 `container_shape`、`send_split_shapes`、`recv_split_shapes`，但 `_shape_tensor()` 自身已经执行 `torch._dynamo.maybe_mark_dynamic(dim0)`；固定上游 GraphTrainer `dynamic_shapes.py` 会读取 Dynamo dynamic 标记，因此仅“未加入 `_mark_dynamic` 名单”不能判为 bug。且 DSV4 GraphTrainer 当前显式不支持 CP，所以 `ExchangePlan` leaf 不进入该 CP GraphTrainer 路径。standard Dynamo 下嵌套 `ExchangePlan` 的实际提升方式仍回到 (a)，**未核实**。**(c) worker 与生产输入形态不等价。** worker 编译 `DispatchBlock.forward(x, window, container)`，把 `WindowPlan`/`CompressedBlockLayout` 直接作为顶层参数；生产固定上游则对每个 `TransformerBlock` 做 `torch.compile(fullgraph=True)`，block 的参数是 `attention_masks`，真实 `ExchangePlan`/`container_shape` 位于 `attention_masks.window` 与 `attention_masks.plans[ratio]` 的嵌套层级。当前 test 不能证明这两种 Dynamo source/guard 路径等价。 | 如果固定 Torch 在真实嵌套路径上把 `ExchangePlan` 或其 tensor 属性做 identity/value guard，而不是把 shape 变成可复用的符号输入，仍可能出现 packed batch 触发重编译，甚至 SAC backward 缺失符号；这正是本 PR 要修的故障面。另一方面，**没有发现 FSDP2/checkpoint state_dict 膨胀或持久化污染**。 | 扩展 CPU regression，使被编译函数签名接近生产：`forward(x, attention_masks)`，传入真实 `CompressedVarlenMetadata`（包含 `window` + `plans[ratio]` + `ExchangePlan`），每次调用创建新的 metadata 并改变非 0/1 split 与 `out_width`，继续断言 exact output/grad 和同一 graph reuse。为回答 (a)，修复验证阶段应额外检查 counting backend 的 `example_inputs`/FX placeholders 或 Dynamo guard 信息，确认 shape-derived size 是 runtime symbol 而不是对象 identity/value guard；长期 UT 不必硬编码脆弱的内部 placeholder 数量。R1 的真实 NPU Inductor ST 仍必须保留，作为 production nesting + backend 的最终完成性证据。 |
| **R6** | **应修改（边界覆盖）** | `torchtitan_npu/models/deepseek_v4/metadata.py:L64-L68,L303-L316`；`torchtitan_npu/models/deepseek_v4/token_dispatcher.py:L254-L276,L622-L638`；`torchtitan_npu/override/deepseek_v4/sparse_attn/ascendc.py:L301-L310`；`tests/unit_tests/models/deepseek_v4/cp_compile_worker.py:L45-L74`；基线 `token_dispatcher.py@4c4079dc:L500-L527` | **(a) `out_width=0/1` 是真实构造边界，当前 container test 未覆盖。** 非 CP 直接设置 `out_width=seq_len // ratio`，CP `_container_slots()` 从 `max_kept=0` 开始计算，因此构造层面可以得到 0 或 1。AscendC fused metadata 会在 `ratio>1 && gather_indices.numel()==0` 时提前拒绝“无完整 compression block”的 batch，因此常规 fused NPU 路径的纯 width=0 往往在 `select()` 前失败；但 width=1 仍是合法情况。`select()` 对 0/1 的 row-selection + zero-padding 公式本身没有特殊分支，只要 `out.shape[0] <= out_width` 就仍成立；但 worker 使用 `out_width=5+step`，其最小值是 3（CP4）/4（CP2），并没有覆盖 container width 0/1。由于 worker 自己注明 Dynamo 会 specialize size 0/1，`container_shape.shape[0]` 在这些边界退化为常量并可能在 1→N 时额外重编图。**(b) zero peer split 的 collective 数值语义不变，但符号/缓存语义改变。** 基线 compiled 分支从 split CPU tensor `.tolist()` 取得 0；新实现从 `[0,0]` shape carrier 的 `shape[0]` 取得 0，传给 all-to-all 的 split 值仍相同。差异是新路径对 0/1 dim 按 Dynamo 规则特化。worker 前 6 个用于 `len(graphs)==1` 的 batch 固定 zero/one pattern，后续 `-2/-1/6` 才故意触发 zero/one transition且不再限制 graph 数，因此当前“一图复用”结论只覆盖普通 `>1` 动态尺寸。**(c) `dtype=torch.uint8, device="cpu"` 本身不参与数值语义；NPU 混合设备 guard/codegen：未核实。** carrier `numel()==0`，consumer 只读 `.shape[0]`，因此 dtype 不参与 split/padding 计算；CPU/uint8 元数据固定也不应随 batch 变化形成 guard churn。但真实 NPU Inductor 是否会把这些 CPU carrier/shape symbols 作为可接受的 mixed-device graph inputs，以及其 guard/codegen 是否稳定，当前 CPU Gloo worker 无法证明，现有 NPU CP ST 又是 `aot_eager`，因此明确标为 **未核实**。 | 对正常 `>1` 尺寸，现有思路有明确收益；但边界上不能宣称“任意 packed batch 都不重编译”。width=1 或 peer split 0/1 的图特化是当前设计已知行为；若生产数据能跨越这些边界，可能产生有限额外 recompile。CPU carrier 在 NPU Inductor 的 mixed-device 行为仍需真实 backend 证据。 | 增加最小边界测试而不是扩大大矩阵：① 对 `CPTokenDispatcher.select()` 单独覆盖 `out_width=1`（以及 reference/允许路径下的 0）并验证 padding 结果；再用 1→2/3 的 compile 调用明确记录“允许一次边界重编译”还是“必须单图”，不要把二者混为一谈。② 保留 worker 的 zero-split transition，但把注释/断言明确为：普通 `>1` split 单图，0/1 transition 只要求正确输出/梯度，不要求同 graph。③ R1 的 NPU CP2 + SelectiveAC + Inductor ST 同时承担 CPU `uint8` shape carrier 在真实 NPU mixed-device codegen 下的完成性验证。 |

## 5. 每处 diff 的代码阅读结果

| 位置 | 改动 | 结论 | 说明 / 建议 |
| --- | --- | --- | --- |
| `torchtitan_npu/models/deepseek_v4/metadata.py::_shape_tensor` | 新增零元素 CPU shape carrier，并 `maybe_mark_dynamic(dim0)` | **通过** | 用 tensor dimension 作为 SymInt 来源能避免编译路径对 tensor data 做 `.tolist()`；零元素避免 payload 存储。该仓已有 DSV4 AscendC `_mark_dynamic()` 使用 `torch._dynamo.maybe_mark_dynamic`，本 PR 没有额外引入一种完全不同的动态 shape 机制。 |
| `metadata.py::CompressedBlockLayout.container_shape / __post_init__` | 从 `out_width` 派生 symbolic width carrier | **通过** | `register_pytree_node_for_dataclass()` 基于 `dataclasses.fields()` flatten，新增 field 会自然进入编译输入；非 CP `build_kernel_layout()` 和 CP `_assemble_block_plan()` 都仍由单一 `out_width` 构造 plan，因此当前真实 producer 不会漏建 carrier。保留 `out_width` 作为 eager/说明字段、`container_shape` 作为 compiled shape 表示可以接受；后续若允许 post-init 修改 `out_width`，需要避免两者成为可漂移的双 source of truth。 |
| `token_dispatcher.py::ExchangePlan.__post_init__` | 单个 split tensor 改为每 peer 一个 `[size,0]` tensor | **通过** | 直接对应 `.tolist()` 产生 unbacked SymInt 的根因；按 peer 独立 dimension 能保持 uneven all-to-all 的每个 split 独立符号化。 |
| `token_dispatcher.py::ExchangePlan.splits_for_collective` | compile/trace 分支读取 `shape.shape[0]`，eager 保留 host lists | **通过** | 入口分工明确，没有为 eager 引入 D2H；返回值继续满足 collective split list 形态。 |
| `token_dispatcher.py::CPTokenDispatcher.select` | padding width 从 `plan.out_width` 改为 `plan.container_shape.shape[0]` | **通过** | row selection 和 `torch.cat([out,pad])` 没变，只替换 shape 来源；内部 assert 合理。 |
| `tests/unit_tests/models/deepseek_v4/cp_compile_worker.py` | 新增真实 Gloo + SAC + Inductor 动态 shape regression | **测试思路有效，但需按 R2 收敛入口** | 独立 output/gradient oracle、duplicate row、zero/one split transition、graph-count 都有区分能力；不是“不抛异常”测试。不要直接绑定上游 private `_wrap_block`。 |
| `tests/unit_tests/models/deepseek_v4/test_cp_dispatch.py::test_multiprocess_gloo` | PYTHONPATH 增加 `TORCHTITAN_DIR` | **通过** | standalone worker 不继承 pytest sys.path；补固定上游 checkout 路径符合 `.ci/unit_test.sh` 的环境模型，不改变被测语义。 |
| `test_cp_dispatch.py::test_cp_dispatch_sac_inductor_outputs_gradients_and_graph_reuse` | 启动 CP2/CP4 worker、按 rank 独立 log、timeout/finally cleanup | **通过，运行成本见 R3** | pytest 收集路径正确；进程/文件句柄在 `finally` 回收；每 rank log 隔离。 |

## 6. 测试格式审查

| 位置 | 类型 | 代码阅读结果 | 修改建议 |
| --- | --- | --- | --- |
| `tests/unit_tests/models/deepseek_v4/cp_compile_worker.py` | 文件位置 | helper 与生产 `models/deepseek_v4` 路径对应，文件名不以 `test_` 开头，不会被 pytest 当成独立 testcase 收集 | 保持 |
| `tests/unit_tests/models/deepseek_v4/test_cp_dispatch.py` | pytest 收集 | `.ci/unit_test.sh` 静态执行 `python -m pytest ... tests/unit_tests`，新增 testcase 会被正常收集；参数 id 为稳定的 `cp2/cp4` | 保持 |
| `cp_compile_worker.py` | 辅助函数 | worker 使用真实生产 `CPTokenDispatcher` / `_build_exchange_plan` / `CompressedBlockLayout`，没有 mock 掉本 PR 声称验证的 collective/shape 逻辑 | SelectiveAC setup 改为 public `apply()`，见 R2 |
| `test_cp_dispatch_sac_inductor_outputs_gradients_and_graph_reuse` | 状态隔离 | MASTER port 动态分配；每 rank 独立 log；worker 在 `finally` destroy process group；父进程在异常/timeout 时 kill 并 wait | 保持 |
| 同上 | 命名 | 名称准确表达 outputs / gradients / graph reuse，未夸大为真实 NPU | 保持 |
| 同上 | 测试入口 | CPU UT 只能证明 CPU/Gloo/Inductor contract，不能记作 NPU ST | 增加 R1 的 integration testcase，而不是把 worker 移入 smoke shell |

## 7. 真实 NPU ST 事实表（<= 4 NPU）

| 测试 | 模型/配置 | 并行数值 | 替换实现/融合算子 | 编译模式 | NPU 数 | 启用与完成检查 | golden | 执行阶段 |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| 现有 `dsv4_smla_cp2_ep2_fsdp2` | DeepSeek-V4 debugmodel | CP2 + EP2 + FSDP2，`spmd_types` | `NPU_OVERRIDES`：Asc RMSNorm/RoPE、DSV4 sparse-attn metadata/LI/core、MHC、token dispatcher | **`aot_eager`** | 4 | 1 training step 完成；`check_loss=False` | 无 | `build_deepseek_v4_test_list` → `build_models_test_list` → `.ci/smoke_test.sh` |
| **拟新增** `dsv4_smla_cp2_inductor_sac` | DeepSeek-V4 debugmodel | **CP2**，其它并行维度保持最小 | 同 `NPU_OVERRIDES`，进入真实 DSV4 CP dispatcher | **`inductor` + 默认 SelectiveAC** | **2** | 建议 2 steps；至少完成 init/forward/backward/optimizer，且无 unbacked SymInt / undefined-symbol / Inductor codegen failure | 无（该 case 是完成性/编译路径 canary） | 现有 `build_deepseek_v4_test_list` / models suite |

### 按模型/运行方式投影

| 并行方式 | 参考/Golden eager | NPU replacement + `aot_eager` | NPU replacement + Inductor |
| --- | --- | --- | --- |
| 1 rank | `dsv4_golden_1rank` | `dsv4_smla_1rank_aot_eager` | 本 PR 无需新增；未触发 CP dispatcher exchange |
| EP2/FSDP2 | `dsv4_golden_ep2_fsdp2` | `dsv4_smla_ep2_fsdp2` | 本 PR 无需新增；目标问题是 CP split/container |
| CP2 | Golden 路径不代表 NPU CP compiler contract | `dsv4_smla_cp2_ep2_fsdp2`（含 EP/FSDP，4 NPU） | **缺口：新增 R1 的 CP2 + Inductor + SelectiveAC 2-NPU canary** |

## 8. ST 不能单独证明的内容

| 内容 | 边界 |
| --- | --- |
| “不同 packed batch 永远不重新编图” | 普通 integration completion 不能精确证明 graph-cache 数量；本 PR 的 CPU UT 用 backend graph counter 对 6 个普通动态 batch 做直接断言，这才是该性能/编译性质的主要证据。NPU ST 负责真实 backend 不崩溃。 |
| HCCL/NPU collective 数值与 CPU Gloo 完全等价 | R1 的 ST 可以证明真实 NPU 路径完成，但没有独立 NPU dense oracle 时不能把它描述成逐元素数值等价证明。 |
| 所有 CP degree 都已在 NPU 覆盖 | 不需要复制 CP2/CP4/... 的 NPU 矩阵；shape-list 长度泛化由 CPU contract 测试保护，真实 NPU 最小 CP2 canary 足够覆盖 backend 路径。 |

## 9. 合入前置条件

| 顺序 | 必须动作 | 最小实现与验收 |
| ---: | --- | --- |
| 1 | **新增真实 NPU CP2 + SelectiveAC + Inductor integration case** | 文件：`tests/integration_tests/deepseek_v4.py`；沿 `_build_case`/`build_deepseek_v4_test_list`，不新增 shell/runner。2 NPU、CP2、`spmd_types`、NPU overrides、compile enable、Inductor、默认 SAC、建议 2 steps、`check_loss=False`。验收：真实训练入口完成 forward/backward/optimizer，日志无本 PR 描述的 SymInt/codegen failure。README 矩阵同步。 |
| 2 | **CPU regression 不再直接调用上游 private `SelectiveAC._wrap_block`** | 文件：`tests/unit_tests/models/deepseek_v4/cp_compile_worker.py`；用最小含 `layers` 的 module + public `SelectiveAC.apply()` 进入 checkpoint 路径，再执行现有 `torch.compile` / exact output / exact grad / graph-count 断言。 |
| 3 | **PR body 补“如何测试”** | 写明实际 CPU pytest 命令和新增 NPU integration 命令/结果；当前 checklist 与空测试章节不一致。 |

R3（CP4 全链路回归降成本）为建议项，不单独阻塞；若保留，需要在代码中说明 CP4 相对 CP2 的独立 failure mode。

## 10. 附录：固定版本与静态入口

| 项目 | 静态事实 |
| --- | --- |
| Torch | `torch==2.14.0.dev20260719` |
| torch_npu | `torch_npu==2.14.0.dev20260808` |
| TorchTitan | `torchtitan==0.3.0`；`.ci/setup_torchtitan.sh`、`.ci/unit_test.sh`、`.ci/smoke_test.sh` 均按版本 checkout `v0.3.0` |
| 固定上游 AC 路径 | `Trainer.Config.activation_checkpoint` 默认 `SelectiveAC.Config`；`ActivationCheckpointing.apply()` 是生产 public entry；DeepSeek V3/V4 standard parallelize 在 compile 前 apply AC |
| 固定上游 compile | `CompileConfig.backend` 默认 `inductor` |
| CPU UT 入口 | `.ci/unit_test.sh` → `python -m pytest -v --tb=short tests/unit_tests`，静态包含 `test_cp_dispatch.py` |
| NPU integration 入口 | `.ci/smoke_test.sh` → `tests.integration_tests.run_tests --test_suite models` → `build_models_test_list()` → `build_deepseek_v4_test_list()` |
| patch/extension 边界 | 本 PR 未改 `torchtitan_npu/patches` / `Extensions`，未新增 monkey patch；符合模型限定 CP 逻辑留在主仓模型实现的要求 |
| 审查限制 | 本次按 `developer-tests-review` 的 review workflow 只做静态阅读，**未执行测试**，因此本文不声称任何 testcase 实际通过 |
