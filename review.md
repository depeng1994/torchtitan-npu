# PR #23 Maintainer Review

> PR: `feat(deepseek_v4_1): support Engram HF checkpoint import and export`  
> Base: `master@828c0c59e49a8e91a46e8d107c33496426058b0e`  
> Reviewed head: `pr_891@613babc714256fd3da2858528b3c57c03b46b9b2`  
> Review result: **Request Changes**  
> 测试执行：**未执行（仅静态审查）**。按仓内 `.agents/skills/developer-tests-review` 规则，本轮只审查生产改动、现有 UT/ST 与调用链；PR 描述中的手工/临时测试结果不作为 committed regression 证据。

## 1. 总体判断

本 PR 的产品目标是合理的：补齐 DeepSeek-V4.1 Engram 的 HF key 映射、logical rows/padding、EP CPU shard 的 HF 导入/导出，以及官方 Engram `embed.weight + embed.scale`（1×32）和 `wkv.weight + wkv.scale`（32×32）反量化。现有实现对这些基本语义的映射和 shape 检查总体自洽，PR 描述也明确说明“完整官方量化整包仍受非 Engram 权重读取限制”，没有把局部能力包装成完整模型导入能力。

但当前实现没有满足本仓的架构与 Reducer 要求：**上游已有的 DCP shard protocol 没有复用，仓内已有的 MX reader 没有复用，永久 Engram 语义又被接入了临时 upstream patch 文件**。此外，两个真正高风险的生产闭环——分布式在线 HF save、Trainer/NPU 初始 HF load 后的 Host Engram/MXFP8 cache 重建——尚无 committed regression。建议先按下表收敛实现再合入。

## 2. Review Findings

| ID | 级别 | 代码位置 | 问题点 | 依据 / 风险 | 建议修改方案 |
|---|---|---|---|---|---|
| R1 | **阻塞** | `torchtitan_npu/models/deepseek_v4_1/engram/checkpoint.py:28-85`；`state_dict_adapter.py:207,295` | **`EngramCheckpointTensor` 这一整套 Tensor subclass 基本不应存在，PR 重做了当前固定 PyTorch 已提供的 `CheckpointableTensor` 能力。** | 仓库固定 `torch==2.14.0.dev20260719`（`requirements.txt:6`）。PyTorch 在 2026-07-16 的 commit `ff12fa13ad483fa07d47e9a5642b984f4898d8b0` 已新增并从 `torch.distributed.checkpoint` 导出 `CheckpointableTensor`，上游测试就是在普通 local `torch.Tensor` 上设置 `global_shape/global_offsets/local_offsets/local_sizes` 后直接做多 rank DCP save/load；其目的与本 PR 的 `__create_chunk_list__ / __create_write_items__ / __get_tensor_shard__` 完全一致。当前 PR 额外引入 `_make_wrapper_subclass`、`__torch_dispatch__` 和 3 个 DCP dunder hook，既扩大维护面，也把行为绑定到更多私有 planner 细节。 | **删除 `EngramCheckpointTensor` class 及其 `__torch_dispatch__`/DCP dunder 实现。** 在 `to_hf` 对普通 local shard Tensor 设置上游 protocol 所需四个 metadata 字段即可；如 `from_hf` 仍需要恢复原 native shard key，只保留一个极小的 adapter-local marker（例如原 key），不要再造 Tensor subclass。有效行数可直接从 `local_sizes` 推导并 zero padding。合入前在当前 CI wheel 中确认 `from torch.distributed.checkpoint import CheckpointableTensor` 可用；若固定 wheel 异常缺失，应先修依赖基线，而不是永久保留另一套 protocol。上游参考：https://github.com/pytorch/pytorch/commit/ff12fa13ad483fa07d47e9a5642b984f4898d8b0 |
| R2 | **阻塞** | `torchtitan_npu/models/deepseek_v4_1/engram/checkpoint.py:87-183`；`state_dict_adapter.py:175-182` | **新增 183 行模型私有 `EngramHuggingFaceStorageReader` 与仓内已有 `torchtitan_npu/extensions/mx_storage_reader/MXHuggingFaceStorageReader` 平行，违反“先复用、再扩展”的目录和 Reducer 原则。** | DeepSeek-V4 已经通过 `torchtitan_npu/extensions/mx_storage_reader/hf_storage.py` 统一读取 `.weight + .scale` MX 权重；该 reader 已有 E4M3/E8M0、HF shard metadata、切片读取、scale 校验等能力。Engram 的真正新增语义只有：table scale 是 **1×32**，wkv scale 是 **32×32**，且超大 Host table 需要 CPU dequant/CPU target；这不足以证明需要再复制一套 HF metadata 扫描和 Quantized reader。新实现还直接依赖 `safetensors.torch._getdtype`、`torch.distributed.checkpoint._hf_utils._HFStorageInfo` 以及父类的 `_weight_map/_tensor_full_shapes/_load_quantization_metadata/_weight_scale_mapping/_process_read_request` 等私有状态，升级 PyTorch 时会形成第二处 break 面。另有一个可删点：非量化 Engram 本来可继续用标准 `HuggingFaceStorageReader`，DCP 的 global shape 校验已能发现 table shape 不匹配，不需要为了 shape 校验把所有 Engram load 都切到新 reader。 | **把 Engram 所需的 block geometry/CPU dequant 能力下沉到现有 `extensions/mx_storage_reader`，由 scale shape 派生 row block（例如 `[R,K/32]` => row_block=1；`[ceil(R/32),ceil(K/32)]` => row_block=32），不要新增用户 Config。** CPU target 走通用 CPU block dequant；现有 NPU row-MX fast path可保留。V4.1 adapter 只负责“选择/描述格式”，不再拥有一套 reader。同步把当前对所有 `.scale` 的宽泛 skip 收窄到已识别 weight/scale pair；未知 Engram quant key 应 fail loud，避免官方格式演进时静默丢 tensor。对应 UT 放到已有 `tests/unit_tests/extensions/mx_storage_reader/`，模型 adapter 只留 mapping/selection seam 测试。 |
| R3 | **阻塞** | `state_dict_adapter.py:146-173`；`torchtitan_npu/patches/torchtitan/scripts/checkpoint_conversion/convert_to_hf.py:185-187`；`torchtitan_npu/scripts/checkpoint_conversion/export_quantized_hf.py:389,404-405` | **永久的 DeepSeek-V4.1 native-DCP→full-model 重组语义被接进“待上游升级后删除”的 TorchTitan patch，并通过两处 `getattr(..., "prepare_dcp_state_dict", None)` 建立了仓内私有隐式协议；删除 patch 或漏接 caller 时会静默退回错误路径。** | 上游 `torchtitan/protocols/state_dict_adapter.py@v0.3.0` 没有 `prepare_dcp_state_dict`。当前 patch 文件头仍写着 `Pending upstream PR #3985` / “Remove this module after dependency includes the PR”，而 pytorch/torchtitan#3985 当前已经 `merged=true`（merge commit `1b9eef3bd5d1533da05bffcc585fe74280ff8414`）；固定 v0.3.0 暂时仍需要 backport 不等于可以继续往该 patch 塞模型语义。更危险的是 `getattr(..., None)`：一旦升级时 patch 被按注释删除、adapter 方法改名、或某个 exporter 忘记接 hook，代码不会提示 Engram shard 需要重组，而是直接把 canonical full-table target 交给 DCP；原生 checkpoint 实际保存的是 `.ep_shard_XXXXX_of_YYYYY` FQN，两边 key 不一致。相同 duck-typed hook 又复制到了 quantized exporter，形成第二个 wiring 点。 | **不要在 `patches/torchtitan` 增加 Engram/NPU 特有逻辑。** 先把“DCP export 前根据 metadata 构造 load targets”的逻辑集中到 NPU-owned extension/helper（目录按上游 `tools/checkpoint_conversion` 结构镜像），标准 HF export 与 quantized HF export 共用同一实现；若现有临时 `convert_to_hf` 已因 dotted model name 等 NPU 语义无法随 #3985 删除，应把它从 patch 晋升为 NPU extension，而不是继续维持“未来整文件删除”的错误生命周期。能力缺失时必须 fail loud：检测到 `*.engram.table.weight.ep_shard_*` metadata 就必须执行重组，否则明确报错，不能 `getattr(..., None)` 静默跳过。并删除 `_load_dcp_state_dict(..., sd_adapter=None)` 这个无真实产品语义的可选分支，生产 caller 已始终有 adapter。不要为此新增 Config/env/shell；保持现有 CLI 入口。 |
| R4 | **重要** | `tests/unit_tests/models/deepseek_v4_1/test_state_dict_adapter.py:158-206`（现有两段测试中的第一段）；生产路径见 TorchTitan v0.3.0 `components/checkpointer/dcp.py:dcp_save(... to_hf=True)` | **“在线 HF 保存无需汇聚整表”仍没有 committed 的真实分布式 save regression。** | 当前 `test_engram_hf_shard_load_and_export` 的 HF→EP 部分只把 wrapper/protocol 对象作为 `dcp.load` target；native→HF 部分则先在**单进程**里把两个 EP shard key 一起 `dcp.save`，再 `prepare_dcp_state_dict` 成 canonical full table，最后保存的是完整普通 Tensor。它没有让 rank0/rank1 各持一个 local Engram shard 后执行 `sd_adapter.to_hf -> HuggingFaceStorageWriter(save_distributed=True) -> dcp.save`，因此没有覆盖本 PR 最关键的 global-offset 拼装、空/尾部 logical rows、writer consolidation，也没有覆盖当前 `__torch_dispatch__`（若 R1 修复后则无需测试该自定义 dispatch）。PR 描述声称做过 2-process Gloo，但该脚本/worker 未提交，不能防回归。 | 按仓内 distributed UT 既有模式新增 **2-rank Gloo committed UT**：每个 rank 只拥有自己的 `...ep_shard_{rank}_of_2`；调用真实 adapter 和 `HuggingFaceStorageWriter(save_distributed=True,...)` 做 DCP save，再由 rank0/所有 rank readback 校验 HF tensor 的完整 logical rows、rank boundary、padding 不落盘。若保留不同 `export_dtype` 路径，也应至少覆盖一次非-fp32 保存。优先放在与 DCP protocol/adapter 最接近的现有测试目录，不要新建一次性 shell。 |
| R5 | **重要** | PR 新增 HF load 路径最终由 `state_dict_adapter.py:175-225` 进入上游 `CheckpointManager.dcp_load`；相关既有 NPU 逻辑 `torchtitan_npu/override/deepseek_v4_1/engram/mxfp8.py:106,158-160` | **PR 自己明确承认“Trainer 加载后训练及 MXFP8 缓存重建尚待验证”，这正是本 PR 改到的生产闭环，不能只靠 CPU adapter UT。** | 上游真实调用链是 `CheckpointManager.dcp_load: state_dict -> adapter.to_hf -> reader/dcp.load -> adapter.from_hf -> states[MODEL].load_state_dict`。Host Engram 在 `load_state_dict` 后还需要 `MXFP8HostOffloadEngramTable` 的 post-hook 重建 `_quantized_storage/_quantized_scale`。现有新增 UT 到 `adapter.from_hf` 即止，没有执行真实 Model/Trainer load_state_dict，更没有 NPU fetch/训练一步；因此无法证明“HF 权重加载成功”在实际 A5/quantized training 中成立。按仓内 test-review 规则，CPU mock 不能证明 NPU runtime/backend 语义。 | 补一个 **tests/integration_tests 下的真实 NPU ST**，尽量复用现有 runner/CLI，不新增模型专用 shell：用 V4.1 debug model + 小型 HF fixture，启用 Engram 与相应 MXFP8 override，走真实 `CheckpointManager` initial HF load，随后至少做一次真实 lookup/forward+backward/optimizer step，验证 post-load cache 已建立且训练可继续；如该场景同时承诺 online HF save，再把 last-save/readback 纳入同一最小闭环。PR 阶段控制在 ≤4 NPU。完整官方 quantized package 仍未支持，所以 fixture/断言必须写清验证的是 Engram seam，而不是宣称整包加载。**测试审查结论：补充测试后合入。** |
| R6 | **重要** | `examples/deepseek_v4_1/readme.md:55-72,130-150`；`examples/deepseek_v4_1/deepseek_v4_1_flash_cpt_4k_a3.sh:114-125`（本 PR 未更新，但功能说明已发生变化） | **文档与实际新能力/限制没有同步；照现有主入口说明使用官方量化 HF checkpoint 会走错 flag 或产生过度预期。** | 主多机脚本当前只传 `--checkpoint.initial-load-in-hf`，没有 `--checkpoint.initial-load-in-hf-quantized`。本 PR 的官方 Engram MXFP8 reader 只有 `from_quantized=True` 才启用；不加该标准 CLI flag 时，遇到 Engram F8 weight 会明确报“requires from_quantized=True”。另一方面 PR 描述已经说明非 Engram 官方量化权重仍可能失败，所以即使加 flag 也不能把当前能力描述成“官方 V4.1 整包可直接训练”。README 目前只说 `CKPT_INIT_LOAD_PATH` 指向 HF checkpoint，未给出这两个关键边界。 | **不要新增 env 或 shell 分叉。** 在现有 README checkpoint 段补充标准 CLI：量化 HF 输入需要额外传 `--checkpoint.initial-load-in-hf-quantized`；明确当前 PR 只补 Engram 的官方 MXFP8 seam，完整官方量化 package 仍受非 Engram reader 支持范围限制；同时给出 native DCP resume、unquantized HF initial load、quantized-HF(Engram seam) 三者的最小示例/支持矩阵。若在线 HF export 是正式能力，也记录使用现有 `last_save_in_hf/last_save_model_only`，不要再造新入口。 |

## 3. 现有行间 Review 的归并

| 已有行间意见 | 本报告归并位置 | 处理 |
|---|---|---|
| `state_dict_adapter.py`：`prepare_dcp_state_dict` 不在上游 adapter protocol，且临时 patch 删除后会静默退化 | **R3** | 保留并提升为阻塞项；同时把 quantized exporter 中第二处相同 duck-typed hook 一并纳入，要求单点实现 + fail loud。 |
| `patches/.../convert_to_hf.py`：#3985 已合入，上游升级后本 patch 生命周期怎么办 | **R3** | 与上条合并。当前事实是 #3985 已 merged；不能再把新增 Engram 永久语义放进声称“升级即删”的文件。 |
| `test_state_dict_adapter.py`：未覆盖 wrapper 的真实 distributed save / PR 描述的 2-rank Gloo 未提交 | **R1 + R4** | R1 先从根上删除不必要的 wrapper；R4 仍要求对 adapter→HF writer 的真实 2-rank 产品 seam 做 committed regression。 |

## 4. 按文件覆盖检查

| 文件 | 本 PR 改动目的 | Review 结论 |
|---|---|---|
| `torchtitan_npu/models/deepseek_v4_1/engram/checkpoint.py` | 新增 EP shard 的 HF logical view；新增 Engram MXFP8 reader / CPU dequant | **需重构。** shard view 应复用 PyTorch `CheckpointableTensor`（R1）；MX reader 应并入已有 extension（R2）。`dequantize_engram_weight` 的数学逻辑本身对 1×32/32×32 两种 scale shape 是一致的，可保留为通用 CPU dequant 实现但不应留成模型私有 reader 的组成部分。 |
| `torchtitan_npu/models/deepseek_v4_1/state_dict_adapter.py` | 新增 Engram HF↔native key；logical rows/padding；EP shard export；HF reader 选择；native DCP shard 重组 | key mapping、logical row 截断和 padding zero 的方向正确；`_engram_tables` 替代单纯 bool 是有真实 shape 语义的。问题集中在 R1/R2/R3：不要通过模型私有 Tensor subclass、第二套 reader 和隐式 adapter hook 来实现。 |
| `torchtitan_npu/patches/torchtitan/scripts/checkpoint_conversion/convert_to_hf.py` | offline DCP→HF 前调用 adapter 重组 EP Engram shard | **不接受当前落点。** patch 目录生命周期与永久模型语义冲突，且 #3985 已 merged；见 R3。 |
| `torchtitan_npu/scripts/checkpoint_conversion/export_quantized_hf.py` | quantized export 同样支持 Engram native shard；Host table 不强制转 master dtype | Host table 保持 FP32 load container 与当前 Host master-weight 语义一致；但新增的 `sd_adapter=None + getattr hook` 是重复 wiring，应与标准 export 共用一个 NPU-owned helper 并删掉可选分支（R3）。本脚本现有 recipe 也没有把 Engram embed/wkv 当作新增发布量化格式，和 PR 描述“沿用 export_dtype，不新增 FP8 发布格式编码”一致。 |
| `tests/unit_tests/models/deepseek_v4_1/test_state_dict_adapter.py` | 增加 unquantized EP shard/HF roundtrip 与 Engram quantized load | 两组 expected tensor 不是对生产函数自算结果的同义断言，shape/padding/scale-file 分离也有价值；但第一组把 HF-load、native offline regroup、single-process HF save 三个 seam 混在一个 test，且没有覆盖真正 distributed online save。按 R1/R2 重构后，把 generic MX cases 下沉到 extension UT，adapter UT 留 mapping/shape seam，再补 R4 的 2-rank test 与 R5 的真实 NPU ST。 |

## 5. 建议的最小收敛形态

| 层级 | 应保留的新增语义 | 应删除/合并的复杂度 |
|---|---|---|
| V4.1 adapter | Engram HF↔native FQN、logical rows、padding、EP shard 的 global offset/native key 恢复 | 删除 `EngramCheckpointTensor`；不维护模型私有 DCP protocol。 |
| `extensions/mx_storage_reader` | 从 scale shape 派生 1×32/32×32 block geometry；CPU block dequant；继续复用现有 HF shard metadata 读取 | 删除 `EngramHuggingFaceStorageReader` 的平行 metadata/reader 实现。 |
| checkpoint conversion extension | 根据 native DCP metadata 为 Engram 构造 full-model load target；标准/quantized exporter 共用 | 不再往 `patches/torchtitan` 注入 Engram 语义；删除两处 `getattr(...prepare_dcp_state_dict...)` 复制和 `sd_adapter=None` fallback。 |
| CLI / docs | 复用现有 `initial-load-in-hf[-quantized]`、`last-save-in-hf` 等标准 CLI | 不新增 Config、env、shell 入口；只补支持矩阵和示例。 |
| tests | adapter CPU seam + generic MX reader UT + 2-rank Gloo online-save regression + ≤4 NPU Trainer load/ST | 不以 PR 描述中的一次性脚本/人工结果替代 committed regression。 |

## 6. Merge Gate

**当前结论：Request Changes。** 至少完成 **R1、R2、R3** 的架构收敛，并补齐 **R4**；针对 PR 已明确承诺的训练初始加载路径，**R5** 需要真实 NPU 证据；**R6** 同步文档后再考虑合入。整个修复过程不需要新增配置类、环境变量或专用 shell，优先复用 PyTorch DCP protocol、仓内 MX extension 和现有 TorchTitan checkpoint CLI。

## Final Cross-Check

本小节是对 R1–R6 的最终交叉复核；若与前文同 ID 的措辞存在差异，**以本节裁决为准**。本轮仍为静态审查，未在仓库 CI 镜像中实际执行 Python import、Gloo 或 NPU 测试。

### 1. 逐条证据复核与最终裁决

| ID | 最终状态 | Final Cross-Check |
|---|---|---|
| R1 | **修正，仍保留为阻塞项，但修复方向改为条件式** | 上游源码证据已再次直接核实：PyTorch commit `ff12fa13ad483fa07d47e9a5642b984f4898d8b0`（2026-07-16，`[DCP] Add CheckpointableTensor protocol (#189492)`）确实存在；该 commit 明确新增 `torch.distributed.checkpoint.protocol.CheckpointableTensor`，并在 `torch/distributed/checkpoint/__init__.py` 通过 `from .protocol import CheckpointableTensor` 导出；上游 test 也确实用普通 local Tensor + `global_shape/global_offsets/local_offsets/local_sizes` 做 distributed default-DCP save/load。因此“protocol 存在、导出路径正确”不是推断。**但原 R1 对 HF writer 的适用范围表述过强，需要纠正**：同一 commit/PR #189492 明写 “HF safetensor: see 2nd PR in ghstack”，后续 PR #189945（`[DCP] Support CheckpointableTensor in HF safetensors storage`）截至本次复核仍为 **open / merged=false**。它新增的内容正包括 `HuggingFaceStorageWriter`/reader/consolidation 对 logical FQN、global shape 和 CheckpointableTensor shard 的支持。由此不能仅凭 `ff12fa13` 就断言本 PR 的在线 HF save 场景已经能无条件删除 wrapper。另一方面，本仓虽然固定 `torch==2.14.0.dev20260719`，但本轮没有进入实际 CI wheel 执行 `import torch.distributed.checkpoint.CheckpointableTensor` 或 HF writer 路径，因此 wheel 实际内容也不能当作已运行验证。**最终要求改为**：先在固定 CI wheel 上用 R4 所要求的 2-rank `HuggingFaceStorageWriter(save_distributed=True,...)` 精确路径验证 plain Tensor + CheckpointableTensor metadata 是否足够；若可用，删除 `EngramCheckpointTensor`；若该 wheel 的 HF storage 尚缺能力，则允许保留一个最小 compatibility shim，但必须证明每个 `_make_wrapper_subclass/__torch_dispatch__/DCP dunder` 都是该 pinned HF writer 的真实必要条件，并把退出条件绑定到上游 #189945/后续版本，而不是把这套 wrapper 当成长期模型语义。原 R1 中“上游 protocol 已完整覆盖 HF save，因此 wrapper 必删”的绝对表述不再成立。参考：https://github.com/pytorch/pytorch/commit/ff12fa13ad483fa07d47e9a5642b984f4898d8b0 、https://github.com/pytorch/pytorch/pull/189945 |
| R2 | **修正，核心结论维持阻塞** | 已再次核对现有 `extensions/mx_storage_reader/MXHuggingFaceStorageReader`。原建议“只需从 scale shape 派生 row_block 即可无 Config 泛化到 1×32/32×32”过于乐观。现有 `_discover_mx_tensors` 计算 `expected_scale_shape = (*qdata.size[:-1], num_blocks)`，因此 Engram table 的 row-MX `[R,K] + [R,K/32]` 与现有模型兼容；但 wkv 的 block-MX `[R,K] + [ceil(R/32),ceil(K/32)]` 会因为第一维不等于 R 被当前校验直接拒绝。并且现有 `read_data -> _read_data_npu` 会对 target 创建 device stream，隐含“每 rank target 位于单一 NPU device”的前提；Host Engram table 的 CPU target 不能直接落进这条 fast path。**因此不再强制要求删除整个 `EngramHuggingFaceStorageReader`。** 更保守、可接受的收敛路径是：优先把两套实现重复的 HF safetensors metadata 扫描、E8M0 dtype 处理、weight/scale sidecar discovery、跨文件 region 读取等通用能力下沉到 `extensions/mx_storage_reader`；然后二选一：(a) 若改动仍小，把 descriptor 泛化为显式 block geometry + CPU/NPU backend，让 Engram 直接复用；(b) 若为支持 32×32 + CPU target 会显著复杂化通用 reader，则保留一个**很薄的 Engram-specific reader/strategy**，只负责 Engram 的 block alignment 与 CPU dequant，公共 metadata/private-API glue 不再复制。仍不建议为此新增用户 Config；官方两种 geometry 可由已知 Engram key + 经严格 shape 校验的 scale metadata 决定。R2 的阻塞点从“必须完全并入现有 reader”修正为“必须消除两套 HF metadata/private dependency 平行实现，并证明剩余 Engram-specific 层是最小差异”。 |
| R3 | **维持** | #3985 的 merged 状态与 merge commit 已再次通过 GitHub PR 数据直接核实，不是推断：`pytorch/torchtitan#3985` 当前 `state=closed`、`merged=true`，`merge_commit_sha=1b9eef3bd5d1533da05bffcc585fe74280ff8414`。因此“当前 patch 文件宣称等待 #3985、而 #3985 已合入；永久 Engram 语义不应继续绑在未来应删除的 patch 生命周期上”这一依据成立。两处 `getattr(sd_adapter, "prepare_dcp_state_dict", None)` 与 quantized exporter 的 `sd_adapter=None` 可选分支也均已在 diff 中复核。R3 维持。 |
| R4 | **维持** | 现有 `test_engram_hf_shard_load_and_export` 的 save 部分确为单进程组织两个 native shard / full target，没有 committed 的 rank0/rank1 各持 local shard 后执行真实 `adapter.to_hf -> HuggingFaceStorageWriter(save_distributed=True) -> dcp.save` 路径。R4 维持。**与修正后的 R1 不冲突**：R4 要求测试的是产品语义（distributed online HF save、offset/consolidation/padding）；如果验证后删掉 wrapper，就不需要专门测试 `__torch_dispatch__`；如果因为 pinned HF writer 的能力缺口必须保留 compatibility shim，R4 的真实 2-rank 路径应自然覆盖它，而不是再为每个 dunder 建实现细节测试。 |
| R5 | **维持** | `mxfp8.py:106` 的 `register_load_state_dict_post_hook` 与 `:158-160` 的 cache refresh 已复核；当前新增 adapter UT 没有进入真实 `CheckpointManager -> model.load_state_dict -> post-hook -> NPU fetch/train` 闭环。R5 维持，且它与 R2 不重复：R2 是 reader 架构/复用问题，R5 是真实 Trainer/NPU runtime 证明问题。 |
| R6 | **维持** | 主多机脚本仍只有 `--checkpoint.initial-load-in-hf`，没有 `--checkpoint.initial-load-in-hf-quantized`；README 也没有明确“当前只补 Engram MXFP8 seam、非 Engram 官方量化权重仍可能不被完整读取”的支持边界。R6 维持。 |

### 2. 冲突与重复检查

| 组合 | 检查结果 | 最终裁决 |
|---|---|---|
| R1 ↔ R4 | 原 R1 的“删除 wrapper”与 R4 提到“当前 `__torch_dispatch__` 未覆盖”表面上容易被读成冲突。 | **已消除。** R4 的 gate 是 distributed HF save 产品行为，不是强制保留并单测 `__torch_dispatch__`。R1 修正后：先用 R4 的真实路径验证上游 protocol/HF writer；可删则删，不能删才保留最小 shim。 |
| R1 ↔ R2 | 同在 `engram/checkpoint.py`，但一个处理 native EP shard 如何表示给 DCP/HF writer，另一个处理 quantized HF weight/scale 如何读取和反量化。 | **不重复。** 可以分别修。 |
| R2 ↔ R5 | R2 要求 reader 复用/下沉；R5 要求 NPU Trainer load 后 cache rebuild 和训练闭环。 | **不重复。** CPU reader UT 不能替代 NPU ST。 |
| R3 ↔ R4 | R3 是 offline native-DCP→HF conversion wiring 与 patch 生命周期；R4 是在线 distributed HF save regression。 | **不重复。** 两条入口不同。 |
| R4 ↔ R5 | 一个验证 save，一个验证 initial HF load + training。 | **不重复。** 都是 PR 声称能力的独立生产 seam。 |
| R3 ↔ R6 | R3 处理代码归属/协议，R6 处理用户可见 CLI 文档边界。 | **不重复。** |

未发现 R1–R6 之间需要合并为同一条的重复问题；除 R1/R2 的修复方向需要按上述证据收敛外，严重级别之间也无新的矛盾。

### 3. 第 3 节“现有行间 Review 的归并”一致性复核

镜像 PR 当前共有 **3 个未解决行间 review thread**，与第 3 节三行一一覆盖，无遗漏：

| 行间 thread | 第 3 节归并 | Final Cross-Check |
|---|---|---|
| `tests/unit_tests/models/deepseek_v4_1/test_state_dict_adapter.py:159`：真实 distributed HF save / wrapper 路径未 committed | `R1 + R4` | 归并仍成立，但**主归属应理解为 R4**（缺少产品行为 regression）；R1 仅承接“wrapper 是否为最小必要抽象”的 Reducer 维度。 |
| `patches/.../convert_to_hf.py:184`：#3985 已合入后 patch 的退出生命周期 | `R3` | 精确对应，维持。 |
| `state_dict_adapter.py:146`：`prepare_dcp_state_dict` 私有协议 + 待删 patch + silent fallback | `R3` | 与上一 thread 指向同一底层生命周期/wiring 问题，合并到 R3 是正确去重，不应拆成两条。 |

因此第 3 节没有“一个行间意见被漏掉”或“同一底层问题被重复计数”的问题；其中两个 patch/hook thread 合并为 R3 是有意去重。

### 4. 最终统计与 Merge Gate

| 类别 | 数量 | IDs |
|---|---:|---|
| **维持** | **4** | R3、R4、R5、R6 |
| **修正** | **2** | R1、R2 |
| **撤回** | **0** | - |
| **新增** | **0** | - |

**最终 Merge Gate：Request Changes，结论不变。**

最终阻塞含义按本节修正为：

1. **R1**：不再要求“无条件删除 wrapper”；要求先用 pinned CI wheel + 真实 HF writer 路径证明上游 protocol 是否足够。足够则删；不足则只保留可解释、可退出的最小 compatibility shim。
2. **R2**：不再要求“无条件把 Engram reader 完全塞进现有 MX reader”；要求至少消除重复的 HF metadata/private-API plumbing，并把剩余 Engram 32×32/CPU dequant 层收敛到最小。
3. **R3**：永久 Engram conversion 语义不得继续依赖声称随上游升级删除的 patch 生命周期；两处 private hook wiring 必须统一且 fail loud。
4. **R4**：必须有 committed 的真实 2-rank distributed online HF save regression，它同时也是裁决 R1 最终实现形态的关键证据。
5. **R5**：补真实 NPU initial-HF-load → post-load cache rebuild → 至少一步训练的 ST。
6. **R6**：同步量化 HF load 的标准 CLI flag、支持范围与限制文档。

没有新增 Config/env/专用 shell 的必要性；final 方案仍应优先减少抽象和重复代码。

