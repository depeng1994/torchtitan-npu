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

## Re-review on Updated Head (e535f4e)

> Re-review source: `pr_891_v2@e535f4ea3301198fdaadbe1ba92c3a79e43af818`  
> Previous reviewed source: `613babc714256fd3da2858528b3c57c03b46b9b2`  
> 测试执行：**未执行（仅静态审查）**。本轮按 `.agents/skills/developer-tests-review` 的 review 测试规则，只核对更新后的生产代码、CPU UT、现有 integration 入口和固定 TorchTitan v0.3.0 调用链。  
> 若本节与前文 R1–R6 的旧状态冲突，**以本节对 updated head 的裁决为准**。

### 1. R1–R6 Updated-Head 裁决

| ID | 最终状态 | 新 head 证据 | 裁决 / 剩余修改 | 已回写 GitCode 评论 |
|---|---|---|---|---|
| R1 | **维持** | `torchtitan_npu/models/deepseek_v4_1/engram/checkpoint.py:28-84` 仍保留 `EngramCheckpointTensor`、`_make_wrapper_subclass`、`__torch_dispatch__` 和 3 个 DCP dunder；`state_dict_adapter.py:245-267` 的 EP>1 路径仍构造该 wrapper。新 worker `tests/unit_tests/models/deepseek_v4_1/engram_hf_worker.py:32-39` 已开始真实使用它：`:34` 的 `detach().clone().to(...)` 会进入 dispatch，随后 distributed DCP save 会消费其 shard protocol。 | **维持 Final Cross-Check 中的条件式 R1，不恢复成“wrapper 必删”的绝对要求。** 新测试证明的是当前 wrapper 在 pinned 环境目标路径上可工作，不是“plain Tensor + `CheckpointableTensor` metadata 在当前 HF writer 上也可工作/不可工作”的对照证据。仍应先用同一 2-rank HF save 场景做最小 counterfactual：若 fixed CI wheel + HF writer 已能直接消费上游 protocol，则删 wrapper；若不能，才保留最小 compatibility shim，并把退出条件绑定到上游 HF-safetensors protocol 支持。 | **需要修正文案，但不是关闭 R1。** 旧行间意见里“`__torch_dispatch__` / write shard 路径从未执行”的事实已经过时，应删除这句；Reducer/上游复用问题仍保留。 |
| R2 | **维持** | `engram/checkpoint.py:103-183` 的 `EngramHuggingFaceStorageReader` 完整实现未变，仍自己维护 quantized metadata 扫描、`_HFStorageInfo`、weight/scale mapping 和 block-aligned read；`state_dict_adapter.py:146-153` 仍在配置 Engram 时直接选择它。updated head 没有把这些 common HF metadata/sidecar 能力下沉到 `extensions/mx_storage_reader`，也没有把 Engram-specific 层缩薄。 | 维持 Final Cross-Check 后的 R2：**不强制把 32×32 + CPU target 硬塞进现有 MX reader，但必须消除两套 HF metadata/private-API plumbing 的平行实现。** 可以保留薄 Engram strategy/subclass，只承载真正不同的 block alignment / CPU dequant。 | **无需修改**（若已有 R2 评论，结论与修复方向均未被新提交改变）。 |
| R3 | **已解决** | 临时 patch 已变成纯兼容 shim：`torchtitan_npu/patches/torchtitan/scripts/checkpoint_conversion/convert_to_hf.py:6-24` 只 re-export plugin-owned 实现，docstring 明确只有 shim 可在 caller 迁移后删除。永久实现位于 `torchtitan_npu/scripts/checkpoint_conversion/convert_to_hf.py:7-12`，明确说明 upstream EMA backport 删除后仍需保留；`:152-180` 新增 `load_dcp_model()`，对 missing/incomplete/geometry mismatch 分别 `raise ValueError`，不存在 silent fallback；`:213-217` 标准 exporter 直接调用该函数。quantized exporter `export_quantized_hf.py:384-401` 同样复用 `load_dcp_model`，原 `sd_adapter=None` 参数和两处 `getattr(...prepare_dcp_state_dict...)` 已消失。adapter 中原 `prepare_dcp_state_dict` 方法也已删除。UT 在 `test_state_dict_adapter.py:210-213` 额外保护 incomplete shard fail-loud。 | 原 R3 的三个底层问题都已闭环：**永久模型语义离开待删 patch、两处隐式 private hook 合并成单点显式 helper、unsupported layout fail loud。** 这个 helper 是 offline converter 的实现细节，不再污染 StateDictAdapter protocol。兼容 shim 的存在本身可接受，因为其业务实现已经不在 patch 内，且 shim 有独立退出条件。 | **需要更新并标记已解决。** 原来两条关于 “#3985 patch 生命周期” 和 “`prepare_dcp_state_dict` 私有协议/silent fallback” 的 GitCode 评论均已被新提交实质修复。 |
| R4 | **部分解决** | 新增 `test_state_dict_adapter.py:357-367::test_engram_hf_distributed_roundtrip`，通过 `torch.distributed.run --nproc-per-node=2` 启动 `engram_hf_worker.py`。worker `:21-32` 建立真实 2-rank Gloo、每 rank 仅持本地 24 行 EP shard；`:36-39` 实际调用 `dcp.save(... HuggingFaceStorageWriter(save_distributed=True, enable_consolidation=True))`；`:40-42` readback 断言最终 HF tensor 只有 41 logical rows；`:44-50` 再按 rank 恢复 native shard，rank0 验证 24 行、rank1 验证 17 行并要求尾部 padding 全零。因此 **distributed writer、rank boundary、logical truncation、padding 不落盘** 这几个原始缺口已经覆盖。 | 仍不能完全关闭 R4，原因有两个，而且第一个比 dtype 更关键：**(1) 新 worker 在 `:29` 用 `DeepSeekV41StateDictAdapter(config, None)`，所以 `fqn_to_index_mapping=None`。固定 TorchTitan v0.3.0 的真实 HF save 在有官方 `model.safetensors.index.json` 时会走另一条分支：`dcp.py:263-279` 把 writer 指向 `sharded/`、传 `fqn_to_index_mapping` 且关闭 writer 内部 consolidation，随后 `:317-323` 调 `consolidate_safetensors_files_on_every_rank`。V4.1 正常传 `hf_assets_path` 时 adapter 会从 index JSON 建 mapping，因此当前测试只覆盖了“无 mapping 的单文件内部 consolidation”，没有覆盖官方资产最接近的 mapped multi-file consolidation。** 最小修复是在同一 2-rank test 里给 adapter 一个 mini `model.safetensors.index.json`，让 Engram key 进入 mapping，并走与上游完全相同的 mapped writer/consolidation 分支。**(2) `:33-34` 注释声称覆盖 export dtype，但测试是在 `adapter.to_hf` **之后**把 wrapper 转成 `float64`；而固定 v0.3.0 的真实 last-save 路径在 `dcp.py:767-776` 先把 native state 转成配置的 export dtype，再于 `:786-791` 进入 `to_hf`，且 CLI 支持的是 fp16/bf16/fp32，不包含 float64。** 若要保留“已覆盖 export_dtype”的声明，请改成 bf16/fp16 并按真实顺序转换，最好与上面的 mapped branch 一次完成。 | **需要更新，但暂不标记 resolved。** 旧评论中“没有 committed 2-rank test / wrapper save 路径从未执行”应改掉；新的剩余意见应聚焦 **official index mapping 对应的 mapped consolidation 分支**，以及可选的真实 export_dtype 顺序。 |
| R5 | **维持** | NPU runtime 语义未变：`torchtitan_npu/override/deepseek_v4_1/engram/mxfp8.py:106` 仍注册 load-state post-hook，`:158-160` 在 load 后重建 quantized storage。updated head 没有新增/修改 `tests/integration_tests`；当前 `tests/integration_tests/run_tests.py:42-60` 的默认/独立 suite 仍只注册 DeepSeek-V4、V3.2、EMA、Qwen3.5，没有 DeepSeek-V4.1 case；目录中也没有 V4.1 integration testcase。新增的 Gloo worker 是 CPU UT，不是 NPU ST。 | R4 的 CPU distributed save 不能替代 R5。仍需要真实 NPU initial-HF-load → `model.load_state_dict` → MXFP8 Host Engram post-hook/cache rebuild → lookup/forward/backward/optimizer step 的 integration 证据。按 test-review 规则，状态仍是 **补充测试后合入**。 | **无需修改**（若已回写 R5 评论，证据和结论均未变化）。 |
| R6 | **维持** | updated head 无 `examples/` diff。主入口 `examples/deepseek_v4_1/deepseek_v4_1_flash_cpt_4k_a3.sh:114-125` 仍只有 `--checkpoint.initial-load-in-hf`，没有 `--checkpoint.initial-load-in-hf-quantized`；README `:72` 仍只说明 `CKPT_INIT_LOAD_PATH` 指向 HF checkpoint，没有新增 quantized flag、Engram-only seam 与完整官方量化 package 限制的支持矩阵。 | 文档/CLI 边界仍需按原 R6 同步；不应新增新 env 或 shell 分叉。 | **无需修改**（若已回写 R6 评论，结论未变化）。 |

### 2. R3 重构专项复核

这次 R3 的修改方向符合原 review 要求，**不再保留 R3 blocker**：

1. `patches/torchtitan/.../convert_to_hf.py` 只承担旧 import/CLI 路径兼容，不再拥有 Engram 或 NPU converter 实现。
2. `torchtitan_npu/scripts/checkpoint_conversion/convert_to_hf.py` 成为 canonical plugin implementation，生命周期不再依赖 #3985 patch 是否删除。
3. `load_dcp_model(state_dict, reader)` 是 converter-owned helper，不再向上游 `StateDictAdapter` 偷塞新 protocol。
4. standard 与 quantized exporter 共用该 helper；没有 `getattr`、没有 `sd_adapter=None` 的静默可选路径。
5. unsupported/missing/incomplete Engram shard layout 会在 load 前 `ValueError`，并已有 incomplete-layout CPU regression。

从 Reducer 角度看，这次反而删掉了一个跨模块隐式 abstraction（adapter hook）和一个可选参数分支，复杂度方向正确。

### 3. R4 新增 distributed UT 的覆盖边界

| 语义 | 新测试状态 | 证据 |
|---|---|---|
| 2-rank、每 rank 只持一个 native EP shard | **已覆盖** | `engram_hf_worker.py:21-32` |
| `adapter.to_hf` 产生 distributed Engram wrapper | **已覆盖** | `:31-34` |
| `HuggingFaceStorageWriter(save_distributed=True)` + real DCP save | **已覆盖** | `:36-39` |
| 41 logical rows 跨 24-row rank boundary 正确拼接 | **已覆盖** | `:40-42` |
| HF 不落 native padding；load 回 rank1 后 7 行 padding 归零 | **已覆盖** | `:44-50` |
| 官方 HF index mapping 对应的 `fqn_to_index_mapping != None` 分支 | **未覆盖** | worker `:29` 显式传 `hf_assets_path=None`；固定上游会因此走不同 consolidation 分支 |
| 真实 `export_dtype` 配置顺序与受支持 dtype | **部分覆盖/表述不准确** | worker `:33-34` 在 wrapper 生成后转 `float64`；上游真实路径先转换 native state，且配置只允许 fp16/bf16/fp32 |

所以 R4 从“缺少真实 distributed regression”降为一个**窄得多的部分解决项**：不要再增加新的 testcase，直接扩展现有 `test_engram_hf_distributed_roundtrip` 即可。

### 4. EMA UT 变更检查

`tests/unit_tests/ema/test_ema_initial_load.py:92-101` 只把 `_load_checkpoint_conversion_module()` 从旧的 `torchtitan_npu.patches.torchtitan.scripts.checkpoint_conversion.convert_to_hf` 改为 canonical `torchtitan_npu.scripts.checkpoint_conversion.convert_to_hf`。这与 R3 的 ownership 重构一致：`ParallelFileSystemReader` 的行为测试现在直接保护真正实现，而不是保护一个将来删除的 shim。

**未发现需要新增 review finding。** 唯一需要明确的是：这意味着现有 EMA UT 不再直接证明旧 shim import path；但 shim 当前只是 1:1 re-export。按 Reducer 原则，不建议仅为了 27 行 re-export 再新增一套重复 testcase。若仓内/外确有必须长期兼容旧 Python import path 的独立用户语义，应补一个极薄的 import/identity contract；若没有真实 caller，后续迁移完成后直接删 shim 比增加测试更合适。

### 5. Updated-Head 最终统计

| 状态 | 数量 | IDs |
|---|---:|---|
| **已解决** | **1** | R3 |
| **部分解决** | **1** | R4 |
| **维持** | **4** | R1、R2、R5、R6 |
| **新增** | **0** | - |

**Merge Gate 仍为 Request Changes。**

相比上一轮，架构 blocker 已明显收敛：R3 可以关闭，R4 的主体 distributed-save 缺口也已补上。当前仍需处理的是：

- **R1**：用当前 pinned HF writer 做 plain-CheckpointableTensor counterfactual，决定 wrapper 是删除还是保留最小 compatibility shim；
- **R2**：消除 Engram reader 与通用 MX reader 重复的 HF metadata/private plumbing；
- **R4**：把现有 2-rank test 扩到 `fqn_to_index_mapping` 的真实 mapped-consolidation 分支，并把 export-dtype 检查改成真实顺序/受支持 dtype；
- **R5**：真实 NPU initial HF load + MXFP8 cache rebuild + 至少一步训练 ST；
- **R6**：补标准 quantized-HF CLI flag 与支持边界文档。

本轮不需要新增 Config、env、专用 shell 或新的 testcase 文件；R4 应继续收敛在已经新增的 2-rank UT 上。

## Re-review Round 2 (dc480ec)

> Re-review source: GitCode head `dc480ec`, mirrored as squash branch `pr_891_v2@e0ba27de437a3827daa8fdcb0c7810bbf9cf01e9`.  
> Previous re-review source: `e535f4ea3301198fdaadbe1ba92c3a79e43af818`.  
> 测试执行：**未执行（仅静态审查）**。本轮按 `.agents/skills/developer-tests-review` 的 review 测试规则核对生产代码、CPU UT、NPU integration 定义、runner 注册与固定 TorchTitan v0.3.0 调用链。  
> 本节只重新裁决上一轮仍回写在 GitCode 上的 R1 / R2 / R4 / R5 / R6；R3 保持上一轮“已解决”。

### 1. 五条意见最终裁决

| ID | 最终状态 | 新 head 证据 | 最终裁决 | 已回写 GitCode 评论处理 |
|---|---|---|---|---|
| R1 | **已解决** | `torchtitan_npu/models/deepseek_v4_1/engram/checkpoint.py:6-19` 已不再定义 Tensor subclass；当前 `checkpoint_shard()` 对普通 `weight.detach()` 设置 `global_shape/global_offsets/local_offsets/local_sizes`，并仅附加 `_engram_native_key/_engram_valid_rows` 两个 adapter-local marker。原 `_make_wrapper_subclass`、`__torch_dispatch__`、`__create_chunk_list__`、`__create_write_items__`、`__get_tensor_shard__` 均已删除。 `state_dict_adapter.py:252-269` 的 EP shard 路径改为调用 `checkpoint_shard()`；`:180-183` 仅用最小 marker 恢复 native key/padding。新 2-rank worker `tests/unit_tests/models/deepseek_v4_1/engram_hf_worker.py:53-67` 先按生产顺序把 native state 转 BF16，再 `to_hf`，并在 `:56` 明确断言 Engram shard 的运行时类型就是普通 `torch.Tensor`。 | 这正是 Final Cross-Check 对 R1 的条件式目标：**能复用 pinned DCP 的 CheckpointableTensor duck-typing 时删除 wrapper，只保留独立产品语义所需的最小 marker。** 当前实现已做到；不再保留 R1 blocker。 | **需刷新重发并关闭原意见。** 原评论关于 wrapper/subclass/dunder 复杂度已被实质修复。 |
| R2 | **已解决** | 共享层 `torchtitan_npu/extensions/mx_storage_reader/common.py` 已集中承载：E8M0 dtype 兼容 `:36-39`、`.weight/.scale` sidecar discovery `:46-52`、safetensors/DCP shard metadata 扫描与 raw storage map `:55-114`、跨文件 region read `:116-148`。通用 reader `hf_storage.py:82-106` 通过 `SafetensorsReaderMixin` 复用 raw metadata，`:133-162` 只保留通用 MX shape/packing 识别。Engram reader 已移到 `extensions/mx_storage_reader/engram.py`，`:32` 继承同一个 mixin；`:50-90` 只做 Engram scale/geometry 校验与 mapping，`:104-136` 只做 1×32 / 32×32 block alignment 和 CPU dequant。新增 `tests/unit_tests/extensions/mx_storage_reader/test_engram_reader.py:34-76` 覆盖 scale 与 weight 跨文件、row/block 两种 geometry、非零 slice offset 和 unknown/orphan scale fail-loud。 | 上一轮要求的两点均满足：**公共 HF metadata/private plumbing 下沉共享；Engram-specific 层只保留真正不同的 geometry/CPU dequant 语义。** `_HFStorageInfo/_getdtype` 等上游 private 依赖仍存在，但已集中到一个 common seam，不再维护两份平行实现；这符合前一轮收敛方案。R2 可关闭。 | **需刷新重发并关闭原意见。** 原“两套 reader 平行维护”的事实已过时。 |
| R4 | **已解决** | `engram_hf_worker.py:33-46` 在 rank0 生成 mini `model.safetensors.index.json`，为 Engram embed 与 q_weight 指定两个 HF shard；`:50-58` 同时遍历 mapped/unmapped 两种 adapter，其中 mapped 传真实 assets path，得到非空 `fqn_to_index_mapping`。`:59-73` 与 TorchTitan v0.3.0 的生产分支一致：mapped 时先写 `sharded/`、传 `fqn_to_index_mapping`、关闭 writer 内 consolidation，再调用 `consolidate_safetensors_files_on_every_rank`；unmapped 走 writer 内 consolidation。`:53-55` 已修正 dtype 顺序：**先把 native state 转 BF16，再调用 `adapter.to_hf`**，且使用的是支持的 `bfloat16`，不再是上一轮的 post-`to_hf` float64 模拟。`:74-88` readback 同时检查 41 logical rows、mapped q_weight、rank boundary 与 native padding 清零。 | 上一轮 R4 剩余的两个缺口——**mapped consolidation 分支**和**真实 export_dtype 顺序/受支持 dtype**——均已补齐；原 2-rank distributed save 主体也保留。R4 可关闭。 | **需刷新重发并关闭原意见。** 旧评论里“未覆盖 mapped consolidation / dtype 顺序”的内容已失效。 |
| R5 | **部分解决** | 新增的 ST 结构本身已覆盖原要求的大部分真实链路：`tests/integration_tests/engram_hf.py:20-23` 直接包装仓内真实 `CheckpointManager.dcp_save/dcp_load`；`:26-53` snapshot 同时抓 Engram 参数与 `_quantized_storage/_quantized_scale` 有效行字节，并在无 cache 时 fail；`:69-77` 调用真实 load 后逐 rank 与 source snapshot 精确比较；`:80-91` 注册 4-NPU `dsv41_engram_mxfp8_hf_ep4`；`:95-125` 再比较 native-load 与 HF-load 两次 3-step 训练的 loss/grad_norm。 `run_engram_hf.sh:17-23` 显式启用 `host_offload_mxfp8`、EP4/FSDP4、4 NPU；`:36-47` 复用真实 A5 Trainer 入口。 `run_tests.py:52-67` 已把它注册为 `deepseek_v4_1_engram_hf` suite。**但 launcher 当前有一个确定的静态错误：**基础 A3 脚本 `examples/deepseek_v4_1/deepseek_v4_1_flash_cpt_4k_a3.sh:30-32` 默认把 `CKPT_INIT_LOAD_PATH` 设成 `/path/to/init_load_ckpt`，`:119-125` 又无条件传 `--checkpoint.initial-load-path ... --checkpoint.initial-load-in-hf`。而 `run_engram_hf.sh:24-34` 的 source phase 只覆盖 `load-only/last-save`，**没有清空 initial-load-path、也没有关闭 initial-load-in-hf**；native phase 虽改成 native DCP 路径，**同样没有关闭 initial-load-in-hf**。固定 TorchTitan v0.3.0 会因此让 source 尝试从占位路径初始加载，让 native phase 把 DCP 目录按 HF safetensors 读取，无法形成设计中的 fresh-source → native-load → HF-load 三阶段闭环。 | **ST 设计与断言已达到可以关闭 R5 的强度，但当前启动参数使它尚不能证明目标路径。** 最小修复应只改现有 `run_engram_hf.sh`：source phase 显式给 falsy `initial_load_path` 并传 `--checkpoint.no-initial-load-in-hf --checkpoint.no-initial-load-in-hf-quantized`；native phase 显式传 `--checkpoint.no-initial-load-in-hf --checkpoint.no-initial-load-in-hf-quantized`；HF phase 保持 `initial-load-in-hf` 并最好显式关闭 quantized flag。修复后按 README 的 4-NPU 命令实际运行并保存 PASS 证据即可关闭 R5。另有一处同类文档小不一致：`tests/integration_tests/README.md:6` 仍写“当前注册 DeepSeek-V4 与 DeepSeek-V3.2”，表格 `:10-24` 也未列新 V4.1 case，建议随 launcher 修复一起刷新；不单独新增 review ID。 | **需刷新重发，但保持 unresolved。** 原评论“没有真实 NPU ST”已不准确，应替换为“ST 已落地，但 source/native checkpoint mode 被基础脚本默认 HF 参数污染”；修复并有真实运行证据后再关闭。 |
| R6 | **已解决** | `examples/deepseek_v4_1/readme.md:74-82` 新增完整 Checkpoint 加载范围：`:78` 原生 DCP、`:79` 普通 HF、`:80` 明确写出 `--checkpoint.initial-load-in-hf-quantized` 及其与训练 MXFP8 override 独立，`:82` 明确限定“当前只补 Engram 表/gate 的 MXFP8 seam，非 Engram 官方整包量化权重仍不能据此直接训练”，并说明 HF export 不生成官方 MXFP8 发布格式。 `tests/integration_tests/README.md:130-143` 也补了 Engram HF 专项验证范围和不覆盖项。 | 原 R6 的 CLI 使用方法与支持边界均已补齐，没有新增 env/config/shell 用户入口。R6 可关闭。 | **需刷新重发并关闭原意见。** 原文档缺口已修复。 |

### 2. R5 新 ST 的覆盖边界

从静态代码看，launcher 参数问题修正后，这个 case 已经具备关闭原 R5 所需的全部观测点：

| R5 要求 | 当前实现 |
|---|---|
| 真实 NPU，而不是 CPU/Gloo 替代 | `engram_hf.py:83-91` 声明 4 NPU；`run_engram_hf.sh:18-23,36-47` 进入真实 A5 训练 launcher 与 NPU override |
| 真实 `CheckpointManager` initial HF load | `engram_hf.py:20-23,69-77` 包装真实 extension CheckpointManager 的 `dcp_load`，不替换 load 实现；HF phase `run_engram_hf.sh:33` 指向 source 生成的 HF checkpoint |
| MXFP8 Host Engram post-load cache rebuild | `host_offload_mxfp8` override 在 `run_engram_hf.sh:17` 被显式选中；snapshot `engram_hf.py:39-53` 要求 `_quantized_storage/_quantized_scale` 存在，并比较 load 前 source 与 load 后每 rank 有效行的原始字节 |
| load 后真实训练 | native/HF 两个 phase 都跑 3 steps；`engram_hf.py:95-125` 比较 step 1-3 的 `loss_metrics/global_avg_loss` 与 `grad_norm`，要求 finite 且逐值相等 |
| 现有 integration runner 注册 | `run_tests.py:52-67` 注册 opt-in suite；README `:134-139` 给出标准 runner 调用 |

因此 **R5 不是“缺测试设计”了，而是“测试设计已经足够，但当前参数接线让测试不能按设计运行”**。该专项 suite 是明确的硬件型 opt-in case，不进入默认 `models` CI 本身不构成新的 blocker；但本轮是静态 review，不能把“代码里有 case”写成“ST 已通过”。修复 flags 后仍需实际运行并留存结果。

### 3. Round-2 架构/Reducer 复核

本轮没有发现需要新增 R7 的生产问题：

- `checkpoint.py` 从约 190 行 wrapper/reader 混合文件收敛到 19 行纯 shard-protocol helper，**删除复杂度方向正确**。
- R2 的共享抽取把原来两份 safetensors plumbing 收到一个 `common.py`；新增 `engram.py` 的独立文件有明确剩余语义（Engram 1×32/32×32 geometry + CPU dequant），不是重复 wrapper。
- 当前 `state_dict_adapter.py` 不再存在 `prepare_dcp_state_dict`，也没有 `getattr(...prepare_dcp_state_dict...)` fallback；它只消费 `checkpoint_shard` 与 shared Engram reader。
- 当前 canonical `torchtitan_npu/scripts/checkpoint_conversion/convert_to_hf.py` 仍是上一轮确认过的 plugin-owned 实现（约 276 行）；**从 e535f4e 到本轮 squash 的 13-file diff 并没有再次修改这个文件**。其 `load_dcp_model()` 仍显式 fail-loud，standard/quantized exporter 继续直接复用；patch 仍只是 28 行 re-export shim。
- 对 `state_dict_adapter.py`、canonical converter、quantized exporter、patch shim 再次全文检查，均没有重新出现 `prepare_dcp_state_dict` 或针对它的 `getattr` 静默兜底。

### 4. 最终统计与 Merge Gate

| 状态 | 数量 | IDs |
|---|---:|---|
| **已解决** | **4** | R1、R2、R4、R6 |
| **部分解决** | **1** | R5 |
| **维持** | **0** | - |
| **新增** | **0** | - |

**最终 Merge Gate：Request Changes。**

现在只剩一个实质 gate：修正 `tests/integration_tests/run_engram_hf.sh` 的三阶段 checkpoint mode 继承问题，并用注册好的 4-NPU `deepseek_v4_1_engram_hf` suite 实际跑通；同时把 integration README 顶部“已注册模型/测试矩阵”同步到 V4.1。测试专项结论为：**补充测试后合入**（更准确地说，是修正已新增 ST 的启动参数并完成真实执行证据）。

### 5. GitCode 已回写评论处理汇总

| ID | 处理方式 |
|---|---|
| R1 | **需刷新重发**：说明 wrapper 已删除并改用 upstream CheckpointableTensor duck-typing，随后关闭该意见 |
| R2 | **需刷新重发**：说明 common I/O 已抽取、Engram 层已缩薄，随后关闭该意见 |
| R4 | **需刷新重发**：说明 mapped consolidation 与真实 BF16 dtype 顺序均已覆盖，随后关闭该意见 |
| R5 | **需刷新重发并保持 unresolved**：删除“没有 NPU ST”的旧表述，改为 source/native phase 继承基础脚本 HF 初始加载参数导致路径错误；修复 + 实跑后关闭 |
| R6 | **需刷新重发**：说明 CLI/支持边界文档已补齐，随后关闭该意见 |

