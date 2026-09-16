# PR 694 Maintainer Review

## 1. Review scope

| Item | Value |
|---|---|
| Mirror PR | `depeng1994/torchtitan-npu#9` (`pr_694 -> master`) |
| Original PR | `cann/torchtitan-npu!694` |
| Reviewed mirror head | `d9d174ef57db920e8926361daa0098e3f68c7f88` |
| GitCode latest increments | `833f18d0` (qwen3_5 naming/test relocation), `c4d1c6b2` (vision-mask patch rationale/docs), appended after `ebad0c8b` |
| Base | `master@89c39094e9dd0751127f6d9e8c4c9763dd891774` |
| Diff | 27 files, `+2292 / -65` |
| PR title | `[feat] Qwen3.5-VL NPU Context Parallel 适配迁移至 master` |
| Review dimensions | Correctness; upstream TorchTitan decoupling; Override/Extension/patch ownership; single/simple training entry; clean code; defensive compatibility; documentation consistency; UT/ST validity and coverage |
| Test workflow | `.agents/skills/developer-tests-review` → `review测试`; `AGENTS.md`, `ut-review.md`, `st-review.md`, `format-review.md`, `report-output.md` re-read on current head |
| Test execution | **未执行（仅静态审查）**。PR 描述中的 `26 passed`、NPU smoke `2 passed`、训练 loss/截图等仅作为开发者提供的验证证据核对，本文没有重新执行这些命令。 |
| Upstream pin fact source | `requirements.txt` pins `torchtitan==0.3.0`; skill/AGENTS also asks to cross-check `.ci/lint.sh`, but that file is absent on current branch. |
| Write scope | 本 review 只写入 `pr_694` 分支的 `review.md`；不修改 `master`，不合并 PR，不修改 PR 状态。 |

## 2. Overall verdict

| Conclusion | Result |
|---|---|
| Maintainer verdict | **Request Changes / 当前不建议合入** |
| Test-review conclusion | **补充测试后合入**（维持上一轮实质结论：需要补充有效模型 ST 与关键语义 UT） |
| Main reason | `833f18d0` / `c4d1c6b2` 修复了命名一致性、NPU runtime pytest 收集位置，并补充了 vision-mask workaround 的根因说明；但上一轮核心阻塞仍存在：**PR 能力边界与实现不一致、32K/CP8 验证命令无法由当前代码复现、训练入口重复且绕过仓库 NPU 入口、NPU/Qwen 专属逻辑仍放入 `patches/torchtitan`、多个 import-time/global monkey patch、FlexAttention fallback 全局扩散、实际 Qwen 模型 ST 仍全部 disabled**。 |

## 3. This-round delta recheck (`833f18d0` / `c4d1c6b2`)

### 3.1 Path migration and naming normalization

| Previous path / identifier | Current path / identifier | Re-review result | Effect on previous finding |
|---|---|---|---|
| `examples/qwen3_6/run_train_qwen35.sh` | `examples/qwen3_6/run_train_qwen3_5.sh` | **Resolved naming only** | All launcher architecture comments are migrated to the new path. The duplicate launcher / `-m torchtitan.train` issue is unchanged. |
| `tests/integration_tests/test_qwen35_npu_runtime.py` | `tests/smoke_tests/ops/test_qwen3_5_npu_runtime.py` | **Resolved file-placement/collection issue** | The former orphan pytest is now under the smoke pytest root and marked `pytest.mark.smoke`; it is valid op/component smoke evidence, but still cannot count as Qwen model ST. |
| `tests/unit_tests/models/qwen3_5/test_qwen35_cp.py` | `tests/unit_tests/models/qwen3_5/test_qwen3_5_cp.py` | **Resolved naming, not coverage quality** | The test body is still only `read_text()` + substring checks. Previous “low-value/source-string test” finding remains unchanged. |
| `test_qwen35_npu_adapter.py` | `test_qwen3_5_npu_adapter.py` | **Resolved naming** | Isolation improvements remain useful; functional-oracle gaps remain. |
| `test_qwen35_torchvision_dependency.py` | `test_qwen3_5_torchvision_dependency.py` | **Resolved naming** | Dependency checks remain; third-party JPEG/resize assertions still should not be counted as adapter product coverage. |
| Local `qwen35` identifiers such as launcher/log/test symbols | Local `qwen3_5` identifiers (`QWEN3_5_TRAIN_ARGS`, etc.) | **Resolved** | Correctly keeps upstream-owned `qwen35_*` config/class/function names unchanged. This directly addresses the naming-consistency concern. |
| Former `qwen35_debugmodel_cp2` integration entry | Removed | **Resolved misleading test declaration** | Better than keeping a knowingly unsupported VL CP case. However it strengthens, rather than removes, the need to correct the PR title/scope because no VL CP ST exists and production explicitly rejects VL CP. |

### 3.2 `c4d1c6b2` vision-mask documentation recheck

| Question | Current state | Maintainer judgment |
|---|---|---|
| Does the patch now explain **why it is needed**? | Yes. `vision_encoder.py` records that upstream vision-mask creation still goes through `torch.compile(create_block_mask)` and that the CANN 9.0.0 environment reproduces a build failure while eager mask creation works. | **Improved / accepted as rationale evidence.** This is materially better than an unexplained workaround. |
| Does it state **scope/removal condition**? | Yes. It says only mask creation bypasses compile, model-body compile remains intact, and the patch should be removed when pinned upstream has the required behavior. | **Improved.** Maintenance intent is clearer. |
| Does it provide an **actual upstream PR**? | No. Header still says `Upstream PR: pending publication...`. | **Not resolved.** `patches/torchtitan` contract requires a real upstream basis/PR, not a planned publication. |
| Does the new documentation resolve **directory ownership**? | No. The implementation is still explicitly “NPU-safe Qwen3.5”, and the failure condition is NPU/CANN-specific. | **Not resolved.** Generic upstream fix and NPU runtime workaround still need to be split; the NPU-specific portion should live under Override/Extension/workaround rather than `patches/torchtitan`. |

**Net result:** `c4d1c6b2` partially responds to the previous review by documenting necessity and lifecycle, but it does **not** clear the patch-ownership/upstream-provenance blocker.

## 4. Blocking / high-priority findings

| ID | Severity | Code / description location | Problem | Why it matters | Required change |
|---|---|---|---|---|---|
| B1 | BLOCKER | PR title/body; `torchtitan_npu/override/qwen3_5/parallelize.py::parallelize_qwen3_5_npu()` / `parallelize_qwen3_5_cp()` | PR 仍以 **Qwen3.5-VL NPU Context Parallel** 为标题与主要能力描述，但当前 CP 路径对 `model.vision_encoder is not None` 明确抛 `NotImplementedError`；`833f18d0` 又删除了原有歧义的 VL CP2 integration 条目。当前实际能力是“Qwen3.5 VL 非 CP 适配 + text-only long-context CP”，不是 VL CP。 | 能力声明是验收边界。当前标题、最初描述、调用链文档会让使用者误以为多模态 VL+CP 可用，而代码主动禁止。 | 二选一：①真正完成 multimodal/VL CP 并增加可运行 ST；或 ②把 PR 标题、描述、README、测试命名和验收项全部收敛为 **Qwen3.5 NPU adapter + text-only CP**，明确 VL CP 不在本 PR 范围。 |
| B2 | BLOCKER | PR 描述“32K/CP8 最终验证”；现有 `examples/qwen3_6/run_qwen3_6_4k.sh`; 新增 `torchtitan_npu/models/qwen3_5/config_registry.py::qwen35_27b_long_text_sft()` | PR 声称使用 `SEQ_LEN=32768 NGPU=16 bash ./examples/qwen3_6/run_qwen3_6_4k.sh` 完成 CP8/DP2 训练，但当前脚本明确限制 `SEQ_LEN <= 4096`，并硬编码 `--config qwen35_27b_4k_sft`，不会进入 `qwen35_27b_long_text_sft`；描述中的 `DATA_FILES=DATA_FILES=/...` 还会把变量值设成带 `DATA_FILES=` 前缀的错误路径。 | 这是本 PR 最关键的 text-only CP 功能验证，但记录无法由当前 branch 复现，不能作为合入证据。 | 给出**与当前分支完全一致**的入口与命令，确保实际选择 `qwen35_27b_long_text_sft`、实际 CP degree 与描述一致，并重新记录 step/loss/退出状态。若不希望再增脚本，优先用通用 `scripts/run_train.sh` + CLI/env 选择 module/config。 |
| B3 | BLOCKER | `examples/qwen3_6/run_train_qwen3_5.sh`; existing `scripts/run_train.sh`; `torchtitan_npu/train.py`; `torchtitan_npu/__init__.py` | `833f18d0` 仅重命名 launcher；当前脚本仍复制 CANN env、NPU env、compile、torchrun、日志等整套通用入口，并最终执行 `-m torchtitan.train`；仓库已有通用 `scripts/run_train.sh`，默认入口为 `torchtitan_npu.train`。同时 `torchtitan_npu/__init__.py` 明确写明 patches **MUST be imported earlier than anything else**。 | 这违反本仓“训练入口单一/简单”的维护原则，也让 Qwen 路径先加载 upstream `torchtitan.train`，再因 `--module` 导入 adapter，扩大“上游符号已提前绑定、patch 安装过晚”的风险。 | 不新增第二套通用 launcher。用 `scripts/run_train.sh` 作为唯一 torchrun/NPU 入口，通过 `MODULE` / `CONFIG` / `--override.imports` / dataset CLI 暴露 Qwen 差异。若保留 example shell，也应只组装 Qwen 专属参数并 `exec scripts/run_train.sh ...`，不能复制/分叉通用启动逻辑，更不能改成 `torchtitan.train`。 |
| B4 | BLOCKER | `torchtitan_npu/patches/torchtitan/hf_datasets/multimodal/mm_collator.py`; `.../models/qwen3_5/vision_encoder.py`; `.../distributed/context_parallel.py`; `.../trainer.py` | `patches/torchtitan` 中仍混入明显的 Qwen-specific / NPU-specific / NPU runtime compatibility 逻辑。`c4d1c6b2` 已补充 vision mask patch 在 CANN 9.0.0 下的复现原因、影响边界和移除条件，**这一点予以认可**；但 `vision_encoder.py` 仍只写 `Upstream PR: pending publication`，没有真实 upstream PR，而且实现本身仍是 NPU/Qwen-specific。`context_parallel.py` 还针对 “worker cannot reinitialize NPU” monkey-patch PyTorch private helper，`trainer.py` 新增 “NPU Torch pipeline schedule” private API bridge。 | 按本仓目录约束，`patches/torchtitan` 只能临时承载**已提交上游但当前固定版本尚未包含**的通用代码；NPU 或模型限定逻辑必须在 Override/Extension/workaround/其他合适主仓路径。说明“为什么需要”不等于满足“代码应放哪里/是否已向上游提交”的要求。 | 保留 `c4d1c6b2` 的根因说明，但将 NPU/model-specific 代码迁出 `patches/torchtitan`。真正通用的 TorchTitan 修复先提交 upstream PR，并在 patch 文件中记录**真实 PR + exact upstream commit/base**；NPU 部分通过 Override/Extension/workaround 消费上游 hook。`context_parallel.py` 必须把 #3430 generic backport 与 NPU eager-mask workaround 拆开。 |
| B5 | BLOCKER | `torchtitan_npu/models/qwen3_5/__init__.py`; `override/qwen3_5/gated_delta.py::npu()`; `override/qwen3_5/parallelize.py::_patch_spmd_type_annotation()` | 仅导入 `torchtitan_npu.models.qwen3_5` 就会安装 collator/vision patch 并修改 upstream `config_registry.model_registry`；导入 parallelize 又永久改写 upstream Qwen SPMD annotation；选择 `GatedDeltaKernel.Config` 的 override 时还会全局赋值 `qwen3_5.GatedDeltaNet._causal_conv = _npu_causal_conv`。 | Override 应是显式、局部、可组合的能力选择，不应通过一个 config override 或 model import 永久改变同进程所有 upstream 类/registry。上游升级时这些隐式绑定也最容易 break。 | 把行为收敛到显式 Override/Extension：优先替换 Config/ModelSpec/具体子模块，而非改写 class method/module global；需要 early hook 的能力应向 upstream 提通用 hook。至少保证“未选择 Qwen NPU override”时不安装 Qwen 专属全局行为。 |
| B6 | BLOCKER | `torchtitan_npu/patches/workaround/eager_flex.py::_run_eager_flex_attention()` | dense-SDPA fallback 条件是 `mask_mod` 显式 marker **或** `(torch.compiler.is_compiling() and q.device.type == "npu")`。因此 compile 下任何满足单 KV block、Q/K 等长、无 aux 的 NPU BlockMask 都会自动改走 dense SDPA，即使不是 Qwen vision mask。代码还对全 mask row 强行 OR diagonal，并引入固定 16M bool 元素上限。 | 这是全仓 NPU FlexAttention 语义变化：可能改变其他模型的 mask 结果、显存和可运行上限；README 却声称“只处理 Qwen3.5 vision 显式标记 mask，其他模型走 native”。`c4d1c6b2` 解释了 Qwen vision mask 为什么需要 eager builder，但没有解释/约束这个更宽的 compile-time global fallback。 | fallback 必须只对明确 opt-in 的 Qwen/已验证 mask 生效。若 compile 时 marker 丢失，应修复 marker/调用链，而不是用 `q.device.type == "npu"` 放大全局范围。补独立 oracle：普通 mask、全 mask row、边界长度、compile/eager 一致性，并证明非 Qwen FlexAttention 不被改写。 |
| B7 | BLOCKER | `tests/integration_tests/qwen3_5.py`; `.ci/smoke_test.sh`; `tests/integration_tests/README.md` | `833f18d0` 删除了误导性的 CP2 entry，但当前 Qwen integration suite 只剩 `qwen3_5_debugmodel_1rank`，且仍 `disabled=True`；因此加入 `models` suite 后实际运行 **0 个 Qwen 模型 ST**。注释称 smoke runner 不安装 torchvision，但 `.ci/gitcode.dockerfile` 会安装 `requirements.txt`，而本 PR 已把 torchvision 加入该文件。没有 long-text CP、MoE/EP+TP、compile/aot_eager 的注册 ST。 | 删除虚假 entry 是正确修复，但不能替代真实 ST。本 PR 改动 2.2K+ LOC，覆盖模型入口、CP、MoE、vision、compile、trainer/dispatcher，全都缺失真正生产入口的持续看护。开发者手工验证不能替代仓库 ST。 | 至少启用一个真实 `qwen3_5_debugmodel` 1-rank 模型 ST，并新增 text-only long-context CP 的最小可执行 ST。若 CI 缺 multimodal 依赖，应在 CI 明确安装需要的 extra，而不是永久 disabled。最多保持必要的 2~4 个 NPU cases，覆盖关键组合即可。 |
| H1 | HIGH | `torchtitan_npu/patches/torchtitan/distributed/context_parallel.py::_varlen_from_masks()` / `patched_cp_shard()` | dict 路径仅确认所有非空 value 的**类型**都是 `VarlenMetadata`，随后无条件拿第一个 metadata 生成一个 `CPVarlenMetadata` 并替换所有 Varlen value。v0.3 Qwen 当前恰好让 quadratic/deltanet 共用同一对象，但该 patch 改写的是全局 generic `cp_shard`，函数本身没有保证“shared”。 | 以后任意模型传入两个不同 varlen metadata 时会静默串用错误 `cu_seqlens`，比显式报错更危险。 | 若设计只支持 shared metadata，则验证所有 value 是同一对象（至少 `cu_seq_q/cu_seq_k` identity/等价）并 fail fast；否则逐 key 构造 CP metadata。不要用“当前 Qwen 恰好相同”作为 generic API contract。 |
| H2 | HIGH | `torchtitan_npu/models/qwen3_5/config_registry.py::_parallelize_long_text()` | text-only recipe 在模型**已经 build 后**执行 `model.vision_encoder = None`。 | 运行时删除已构造子模块会改变 module tree、state_dict/checkpoint/HF export/FSDP 边界，且并非 config 的单一事实源。 | 在 ModelSpec/Config 构造阶段就明确构造 text-only 结构（或提供一个 text-only model extension/config override），不要在 parallelize 阶段摘掉已构造模块。补 checkpoint/state_dict round-trip 测试。 |
| H3 | HIGH | `torchtitan_npu/patches/torchtitan/trainer.py::patched_pp_forward_backward_step()` | 为兼容“older NPU pipeline schedule”，代码依赖 `_has_backward`、`_backward_requires_autograd`、`_stages/_stage`、`clear_runtime_states()`、`_step_microbatches()` 等 private API，并全局改写 `Trainer.pp_forward_backward_step`。文件头记录的 #3634 是 DeepSeek-V4（已于 2026-09-02 merged），#3985 是 EMA，均不能说明这个 PP bridge 的 upstream provenance。 | 私有 API bridge 对所有模型 PP 生效，却没有真实 PP ST；升级 PyTorch/TorchTitan 时极易 break。 | 将 NPU runtime compatibility 移到 Extension/workaround，并只在确有旧 schedule contract 时显式启用；更优是向 upstream/PyTorch 提稳定 hook。若认为属于 TorchTitan patch，必须提供实际对应的 upstream PR。 |
| H4 | HIGH | `torchtitan_npu/patches/torchtitan/models/common/moe.py`; `.../token_dispatcher.py` | 本 PR 继续向 common MoE patch 叠加 DTensor localize、token count、legacy sequence padding 兼容；实现大量使用 `hasattr`/optional kwargs 兼容“older/newer dispatcher schema”。但当前仓固定 `torchtitan==0.3.0`，文件头的 #3634 已 merged，#4095 仍是 draft，且新增 legacy 兼容不等同于 #4095 的 pre-W2 router-score upstream 设计。 | patch 目录会逐渐变成“多版本兼容聚合层”，无法判断哪些代码能真正 upstream，也增加主线升级成本。 | 对固定 v0.3.0 contract 做最小实现；历史兼容逻辑若只服务 NPU 环境应移出 upstream patch。把 #3634 已合入的部分与 #4095 draft 部分拆清楚，更新注释和移除已不需要的 backport。 |
| H5 | HIGH | PR description; `requirements.txt`; patch docstrings | PR 顶部和“如何测试”仍称上游固定为 commit `2807d3f...`，但当前 branch 的运行时依赖是 `torchtitan==0.3.0`；若干新 patch 又标 `Upstream base: b175497e...`。另外测试 skill/AGENTS 要求同时从 `.ci/lint.sh` 核对 pin，但当前仓没有该文件。 | Review/维护无法确定真正兼容基线；大量反射、`try/except ImportError`、`hasattr` 也正是多基线混用的结果。 | 把当前唯一支持基线写清楚：以 `requirements.txt` 的 v0.3.0 为事实源，并补齐/修正规范要求的 CI pin 来源；历史 2807 只能放历史记录。每个 upstream patch 标注真实 upstream PR/commit，删除针对不再支持版本的防御式分支。 |

## 5. PR-description claims vs current branch

| Claim in PR description / docs | Current code observation | Review result |
|---|---|---|
| “Qwen3.5-VL NPU Context Parallel” | CP path明确拒绝任何带 `vision_encoder` 的模型；`833f18d0` 已删除此前的 VL CP2 suite entry。 | **不一致**。实际是 VL non-CP + text-only CP。 |
| 32K, CP=8, DP=2 long-text command uses `run_qwen3_6_4k.sh` | Script refuses `SEQ_LEN > 4096` and hardcodes `qwen35_27b_4k_sft`; new long-text config is a different factory and defaults CP=4. | **不可复现 / BLOCKER**。 |
| Qwen integration command “runs 1-rank smoke” | The only registered Qwen case is `disabled=True`, so runner filters it out. | **不一致**。 |
| “VL CP2 entry is retained but disabled” (`examples/qwen3_6/readme.md`) | `833f18d0` removed the CP2 entry entirely. | **文档过期**。 |
| “Only explicitly marked Qwen vision masks use dense fallback; other models stay native” | compile+NPU branch bypasses marker and applies fallback globally when shape predicates match. | **不一致 / correctness risk**。 |
| launcher “adds adapter to PYTHONPATH and prints actual TorchTitan/NPU import paths” | `run_train_qwen3_5.sh` does neither. | **文档过期**。 |
| varlen GDN CPU metadata loop remains an eager graph-break | Current `_causal_conv1d_varlen()` has a compile branch using device-side `cu_seqlens`, specifically to avoid that CPU loop. | **文档过期**。 |
| upstream pin `2807d3f...` | Current dependency is `torchtitan==0.3.0`; implementation uses v0.3 mesh/SPMD APIs. | **基线描述过期**。 |
| “NPU runtime test moved into valid CI entry” (`833f18d0`) | Current file is `tests/smoke_tests/ops/test_qwen3_5_npu_runtime.py`, marked smoke, and `.ci/smoke_test.sh` runs `pytest tests/smoke_tests`. | **已修复，认可**。 |
| “qwen3_5 naming unified” (`833f18d0`) | New files/functions use `qwen3_5`; upstream-provided `qwen35_*` names are kept as upstream API. | **已修复，认可**。 |
| “vision mask patch remains required under CANN 9.0.0 and only bypasses mask creation” (`c4d1c6b2`) | Current docstring now records this rationale and a removal condition. | **必要性说明已改进；但 upstream PR/目录归属仍未解决**。 |

## 6. Test review (per repository `developer-tests-review` skill)

### 6.1 UT positive-function coverage

| 正向功能 | 生产代码和应观察结果 | PR 中的直接检查 | 状态 | 合入前置条件 |
|---|---|---|---|---|
| Video MRoPE contiguous temporal positions | Patched collator should produce contiguous temporal MRoPE positions and preserve document boundaries. | `test_qwen3_5_video_mrope_uses_contiguous_temporal_coordinates()` uses a small independent expected tensor. | **部分覆盖** | Add mixed image+video, multi-document/reset and batch>1 boundaries if this patch remains local. |
| Qwen model import installs early compatibility patches | Import currently replaces upstream collator/vision functions and registry symbols. | Isolated subprocess asserts monkey-patched function identities. | **部分覆盖** | Prefer removing global patching. If retained, add direct producer→consumer functional behavior tests, not only identity. |
| GDN `npu` override selection | Real `override.imports` should select the Triton GDN kernel and NPU causal-conv behavior. | Test checks config becomes `TritonGatedDeltaKernel.Config`; source-string tests check path names. | **部分覆盖** | Through real config/override build, run small GatedDeltaNet forward/backward and compare to independent/reference conv + kernel semantics. |
| Varlen causal convolution | Per-document packed conv should match independent segmented causal convolution in forward/backward, including compile metadata path. | Existing test covers malformed offsets and valid output shape. | **部分覆盖** | Compare exact eager output/grad to per-segment independent convolution oracle; add compile-path device-offset case. |
| NPU Flex AuxOutput / dense fallback | Marked mask fallback must preserve expected attention semantics; unmarked other-model masks must remain native. | CPU direct call checks aux protocol only. | **部分覆盖** | Add dense fallback numerical oracle, full-mask-row behavior, threshold boundary, explicit opt-in and “unmarked mask is untouched” regression. |
| Qwen parallelizer | Real Qwen config/model should select the intended NPU parallelizer and mesh/FSDP behavior. | Uses `SimpleNamespace`/dummy model + mocks to inspect resolver/FSDP calls. | **部分覆盖** | Build upstream `qwen35_debugmodel` through real registry/config and exercise real supported parallelize entry for CPU-observable contract. |
| Context Parallel metadata | Global Varlen metadata must become correct rank-local CP/Qwen metadata consumed by Qwen attention. | `tests/unit_tests/models/qwen3_5/test_qwen3_5_cp.py` only searches source strings; other tests mostly verify rejection/wiring. | **检查无效** | Construct global Varlen metadata → CP metadata → Qwen metadata and assert exact local boundaries/index behavior; add 2-rank CPU/Gloo where communication semantics are involved. |
| `varlen_attention.py` Q-sharded/KV-replicated branch | CP exchange/head slicing should match non-CP reference forward/grad. | No direct numerical/communication test. | **未覆盖** | Add two-rank forward/grad equivalence against non-CP reference or exact gather/shard oracle. |
| Vision `owning_vision_encoder_forward().clone()` workaround | Wrapper should return owning tensor and preserve representative backward/FSDP hook behavior. | NPU smoke tests `_compute_learned_pos_embeds` backward, not the wrapper/FSDP boundary. | **未覆盖** | Test actual wrapper result ownership plus backward hook/gradient through representative module/FSDP boundary. |
| Common MoE DTensor/localize + legacy padding | Dispatcher/MoE should preserve row/score/token-count alignment and output shape/gradients. | No PR-specific functional tests under `tests/unit_tests/patches/torchtitan/models/common`. | **未覆盖** | Add forward/grad equivalence and sequence-shape restoration; use minimum 2-rank CPU/Gloo for dispatcher communication if feasible. |
| PP pre-split bridge | Patched Trainer should select the private pre-split schedule path only for the intended schedule contract and preserve Trainer semantics. | Fake `Stage/Schedule` calls `_run_presplit_pipeline_schedule`; patched Trainer wrapper/private API selection is not covered. | **部分覆盖** | Test actual `patched_pp_forward_backward_step` against representative schedule contract or remove/narrow global bridge. |
| Long-text recipe removes vision encoder | Text-only recipe should have deterministic module/state_dict/checkpoint/export topology. | No state_dict/checkpoint/export test. | **未覆盖** | Prefer architecture fix; otherwise prove checkpoint/state_dict/HF export behavior. |

### 6.2 ST trigger / integration coverage

| 生产代码变更 | 受影响训练场景 | 为什么需要 ST | 现有代表测试 | 判断 | 合入前置条件 |
|---|---|---|---|---|---|
| Qwen3.5 model adapter + launcher | Full Qwen3.5 VL 1-rank training | Import/vision/GDN/optimizer/model integration only exists on real NPU path. | `qwen3_5_debugmodel_1rank` exists but `disabled=True`. | **未覆盖** | Make the registered case runnable/non-disabled through existing integration runner and require at least one completed train step. |
| Text-only CP adapter + varlen/GDN CP | Qwen3.5 long-text CP | Requires real distributed/NPU sequence exchange and backward. | No integration case. | **新增** | Add a minimal CP2 text-only case using the same config/override path as production; CI does not need 32K sequence. |
| Common MoE DTensor/localize + legacy padding | Qwen3.5 MoE EP/TP path claimed in PR | Requires real dispatcher/DTensor/NPU training interaction. | No Qwen integration case. | **未覆盖** | Add one minimal combination only if this PR owns the common MoE changes; otherwise split generic compatibility out. |
| Flex/vision compile workaround | Qwen vision with `aot_eager`/compile | Compile fallback and mask builder differ specifically on NPU. | No registered Qwen model compile ST. | **未覆盖** | Add one active compile smoke if compile behavior remains acceptance scope. |
| GDN Triton op | GDN forward/backward | Kernel/autograd require real NPU. | `tests/smoke_tests/ops/test_qwen3_5_npu_runtime.py::test_merged_gdn_forward_backward`. | **已覆盖（op smoke）** | Keep as smoke; do not count it as model ST. |
| Vision learned-position op | NPU backward for learned position embedding | NPU-specific autograd behavior. | Same smoke file. | **已覆盖（op smoke）** | Useful component signal; still does not exercise vision wrapper/FSDP workaround. |
| Global PP schedule bridge | Any PP model using affected schedule | Global Trainer patch can affect non-Qwen PP training. | No explicit PP runtime case tied to this change. | **未覆盖** | Add representative PP ST or remove/narrow bridge from this PR. |

### 6.3 Test format / structure findings

| 位置 | 类型 | 代码阅读结果 | 修改建议 |
|---|---|---|---|
| `tests/smoke_tests/ops/test_qwen3_5_npu_runtime.py` | 文件位置 / pytest 收集 | `833f18d0` relocation is correct: file is now under smoke pytest root and marked `smoke`; `.ci/smoke_test.sh` statically includes it. | Keep as op/component smoke; do not use it as evidence for Qwen model ST. |
| `tests/unit_tests/models/qwen3_5/test_qwen3_5_cp.py` | 测试结构 | Rename is correct, but entire test remains source-string/path inspection (`read_text` + substring). It does not execute CP production behavior. | Replace with functional metadata/parallelization tests; do not count source-text assertions as product coverage. |
| `tests/unit_tests/models/qwen3_5/test_qwen3_5_npu_adapter.py` | 测试结构 / 状态隔离 | Good subprocess isolation for import-time global side effects; MRoPE has a real expected output. Much of the file still validates shell/source strings/schema/mocks. | Keep true contract/isolation tests; move product behavior to real config/model construction and independent oracles. |
| `tests/unit_tests/models/qwen3_5/test_qwen3_5_torchvision_dependency.py` | 测试结构 | Manifest dependency checks are relevant, but native torchvision JPEG decode/resize mostly tests third-party behavior. | Keep a minimal environment compatibility check; focus product UT on adapter behavior. |
| `tests/unit_tests/models/qwen3_5/test_qwen_gdn_triton.py` | 测试结构 | Checks file existence and source strings. | Replace with import/API/kernel-selection semantics; file-layout strings are not functional regression protection. |
| `tests/integration_tests/qwen3_5.py` | integration runner 收集 | Suite is registered but only case is `disabled=True`; disabled cases do not count as ST under the repository skill. Its comment says torchvision is absent, while CI image installs base `requirements.txt` containing torchvision. | Fix dependency/path setup and enable the case; add minimal text CP case. |
| `tests/integration_tests/README.md` | 测试入口 / 注释 | Supported-model table still documents only DeepSeek suites and not the newly registered Qwen suite/disabled status. | Refresh README with the real active matrix; explicitly mark disabled entries as non-coverage. |

## 7. Per-file review of all 27 changed files

| # | File | Assessment | Issue / recommendation |
|---:|---|---|---|
| 1 | `README.md` | OK | Only updates repository layout to mention `override/qwen3_5`; directory classification itself is correct. Keep it aligned after patch ownership refactor. |
| 2 | `examples/qwen3_6/readme.md` | **Needs changes** | Multiple statements no longer match code: CP2 said retained although `833f18d0` removed it; 1-rank integration described as runnable although disabled; PYTHONPATH/import-path diagnostics not implemented; varlen graph-break description stale; “only explicitly marked mask fallback” contradicts `eager_flex`; upstream baseline wording is mixed. Rewrite after implementation settles. |
| 3 | `examples/qwen3_6/run_train_qwen3_5.sh` | **BLOCKER** | Rename is correct, architecture issue is unchanged: duplicates common launcher and invokes `torchtitan.train` instead of repository NPU entrypoint. Collapse onto `scripts/run_train.sh`; expose only Qwen-specific defaults via CLI/env. |
| 4 | `pyproject.toml` | Mostly OK | Declaring `multimodal` extra is reasonable. `833f18d0`/PR notes explain the chosen torchvision nightly pairing, but clarify why torchvision is also placed in base `requirements.txt`; if multimodal is optional, CI should install the extra explicitly rather than partially globalizing it. |
| 5 | `requirements-dev.txt` | Needs cleanup | Adds torchvision for Qwen tests, but active model ST is still disabled for “missing multimodal stack”. Align declared test dependencies with what CI actually needs (`.[multimodal]` or one canonical list) instead of divergent comments. |
| 6 | `requirements.txt` | OK with documentation fix | Torch/torchvision pins are explicit and current TorchTitan runtime baseline is `0.3.0`; PR/patch docs must stop presenting `2807d3f...` as current runtime baseline. |
| 7 | `tests/integration_tests/__init__.py` | Architecture concern | Adds generic `train_script/train_args` solely to support a model-specific duplicate launcher. If Qwen reuses `scripts/run_train.sh`, this test-framework extension becomes unnecessary and should be removed. |
| 8 | `tests/integration_tests/qwen3_5.py` | **BLOCKER (coverage)** | `833f18d0` correctly removed the misleading VL CP2 entry, but the only remaining case is disabled, so no Qwen model ST runs. Enable a real case and add text-only CP case. |
| 9 | `tests/integration_tests/run_tests.py` | Mostly mechanical | Qwen suite registration is structurally fine, but registering an all-disabled suite gives false confidence. Revert model-specific launcher plumbing if common entry is reused. Unrelated formatting churn should be minimized. |
| 10 | `tests/smoke_tests/ops/test_qwen3_5_npu_runtime.py` | **Positive fix, insufficient for model ST** | `833f18d0` fixes the prior orphan integration pytest and naming. Keep op-level GDN/position backward checks; they do not validate model/CP/vision-wrapper behavior. |
| 11 | `tests/unit_tests/models/qwen3_5/test_qwen3_5_cp.py` | Low-value test | Rename is correct; body still only checks source substrings. Replace with exact CP metadata and communication semantics. |
| 12 | `tests/unit_tests/models/qwen3_5/test_qwen3_5_npu_adapter.py` | Partial | Naming/isolation are improved, but too many tests validate source text/mocks. Add real config → override → build → forward paths, dense fallback oracle, CP producer/consumer, and wrapper semantics. |
| 13 | `tests/unit_tests/models/qwen3_5/test_qwen3_5_torchvision_dependency.py` | Partial / over-tests dependency | Naming and manifest checks are fine; native torchvision JPEG/resize mostly tests third-party behavior. Do not treat it as adapter feature coverage. |
| 14 | `tests/unit_tests/models/qwen3_5/test_qwen_gdn_triton.py` | Low-value test | File existence/string checks should be replaced by executable public API/kernel-selection semantics. |
| 15 | `torchtitan_npu/models/qwen3_5/__init__.py` | **BLOCKER architecture** | Import-time installation of collator/vision patches plus global reassignment of upstream `config_registry.model_registry`. Replace with explicit adapter/override registration; importing a model package should not silently mutate generic upstream modules. |
| 16 | `torchtitan_npu/models/qwen3_5/_fla_compat.py` | High maintenance cost | Local naming cleanup is good. The module still synthesizes temporary fake `fla` modules in `sys.modules` to make upstream Qwen importable without FLA. Cleanup is careful, but this is tied to upstream import internals. Prefer upstream lazy/optional FLA import; retain only as narrowly scoped temporary workaround with one pinned baseline/removal condition. |
| 17 | `torchtitan_npu/models/qwen3_5/config_registry.py` | **HIGH** | Runtime `model.vision_encoder = None` in `_parallelize_long_text` changes built model topology; dynamic re-export loop also makes adapter API grow implicitly with upstream. Build text-only topology at config/spec level and prefer explicit recipe exports. |
| 18 | `torchtitan_npu/override/qwen3_5/gated_delta.py` | **HIGH/BLOCKER architecture** | NPU conv implementation belongs in override, but `npu()` globally mutates `GatedDeltaNet._causal_conv` as a side effect of a kernel-config override. Introduce an NPU GatedDeltaNet Config/subclass or proper hook so selection is local and composable. Add exact forward/grad oracle for eager and compile varlen conv. |
| 19 | `torchtitan_npu/override/qwen3_5/parallelize.py` | Needs refactor | Reimplements most of upstream Qwen parallelizer, permanently monkey-patches upstream SPMD annotation, temporarily monkey-patches global `torch.nn.Module.compile`, and carries legacy/current mesh compatibility. Minimize fork by calling/reusing upstream phases or exposing an upstream hook; make CP-specific annotation explicit instead of suppressing metadata in a global function. |
| 20 | `torchtitan_npu/override/qwen3_5/varlen_attention.py` | Semantics plausible, coverage missing | Q-sharded/KV-replicated branch is meaningful CP adaptation, but no independent two-rank forward/grad test proves exchange/chunk mapping. Add one before merge. |
| 21 | `torchtitan_npu/patches/torchtitan/distributed/context_parallel.py` | **BLOCKER ownership + correctness** | Generic #3430-related backport is mixed with NPU-specific private PyTorch mask workaround and Qwen hybrid-mask logic. Upstream #3430 scopes varlen CP to full-DTensor; local patch additionally implements spmd_types/dict behavior. Split responsibilities, enforce shared metadata contract, and move NPU workaround out of upstream patch. |
| 22 | `torchtitan_npu/patches/torchtitan/hf_datasets/multimodal/mm_collator.py` | **BLOCKER ownership** | Qwen3.5 contiguous-video MRoPE implementation has no actual upstream PR, only “pending publication”. That does not meet patch-directory contract. Submit upstream first or move to model extension/override; keep independent MRoPE oracle. |
| 23 | `torchtitan_npu/patches/torchtitan/models/common/moe.py` | **HIGH** | Adds Qwen/NPU-driven legacy/DTensor compatibility into common upstream patch, with schema introspection for multiple versions. Separate current v0.3 upstreamable delta from NPU/version compatibility and add functional tests. Update stale #3634 marker. |
| 24 | `torchtitan_npu/patches/torchtitan/models/common/token_dispatcher.py` | **HIGH** | Extends combine signature and reconstructs legacy SP token indices, but has no PR-specific dispatcher oracle/2-rank test. Keep only if this exactly matches an upstream proposal; otherwise move compatibility out of upstream patch. |
| 25 | `torchtitan_npu/patches/torchtitan/models/qwen3_5/vision_encoder.py` | **BLOCKER ownership; docs improved** | `c4d1c6b2` materially improves the rationale: it records the CANN 9.0 compile failure, eager workaround scope and removal condition. However the code remains explicitly NPU + Qwen-specific and still has no published upstream PR (`pending publication`). Documentation does not fix directory ownership. Move NPU-specific behavior to Qwen extension/override/workaround; upstream only the generic fix, and prove actual vision FSDP backward. |
| 26 | `torchtitan_npu/patches/torchtitan/trainer.py` | **BLOCKER ownership / blast radius** | Adds private NPU schedule bridge to global Trainer under unrelated/stale upstream PR headers. Move out, narrow activation, or provide a real upstream PR/hook; add PP behavior coverage. |
| 27 | `torchtitan_npu/patches/workaround/eager_flex.py` | **BLOCKER correctness blast radius** | Correct directory for an NPU workaround, but implementation changes compile-time Flex behavior globally and docs understate scope. Restrict to explicit opt-in, reduce multi-version reflection against fixed Torch pin, and add semantic regression tests. |

## 8. Upstream/patch provenance check

| Patch / reference | Current upstream status / local evidence | Maintainer assessment |
|---|---|---|
| `pytorch/torchtitan#3430` | Open; design states Varlen+CP works under **full_dtensor**, Q sharded and K/V Replicate. | Local `context_parallel.py` additionally carries spmd_types/hybrid-dict and NPU mask-workaround behavior; it is not an exact backport anymore. |
| `pytorch/torchtitan#3634` | Merged 2026-09-02; DeepSeek-V4 model support. | `moe.py` / `trainer.py` still label it “Pending”; refresh/remove backport portions already upstream. It also does not justify the new NPU PP bridge. |
| `pytorch/torchtitan#3985` | Open; EMA implementation. | Relevant to pre-existing EMA Trainer patch, not to newly added NPU private PP schedule bridge. |
| `pytorch/torchtitan#4095` | Open draft; pre-W2 router-score absorption. | Can justify some common MoE/dispatcher work, but not all Qwen legacy-sequence/multi-version compatibility added here. Keep exact diff/provenance clear. |
| Qwen3.5 video-MRoPE patch | File still says upstream PR is pending publication. | Does not satisfy `patches/torchtitan` ownership rule. |
| Qwen3.5 vision-mask/FSDP patch | `c4d1c6b2` adds a concrete CANN 9.0 failure explanation and removal condition, but still says upstream PR is pending publication. | **Partial response only**: necessity evidence improved; upstream tracking and directory ownership remain unresolved. Generic upstream fix and NPU runtime workaround should be separated. |

## 9. Clean-code / simplification opportunities

| Area | Current complexity | Simplification direction |
|---|---|---|
| Training entry | `scripts/run_train.sh` + a second ~100-line `run_train_qwen3_5.sh` + integration runner `train_script/train_args` abstraction | Keep `scripts/run_train.sh` as single runner; Qwen example should be a CLI/env recipe or a thin `exec`, not a parallel launcher framework. |
| Qwen parallelization | Large copy of upstream parallelizer plus mesh-version branches and monkey patches | Reuse upstream `parallelize_qwen3_5` phases where possible; upstream a small hook for CP/compile/FSDP differences; local override only supplies NPU deltas. |
| FLA optionality | Temporary synthetic `sys.modules` tree | Upstream lazy import or explicit kernel backend factory. One pinned contract, one fallback. |
| Flex compatibility | Aux API reflection + old/new protocol normalization + global fallback | Repository pins one Torch nightly; implement that exact contract, remove unnecessary multi-version defensive branches unless multiple versions are officially supported/tested. |
| MoE compatibility | `hasattr`, dataclass field introspection, optional kwargs for older/newer schemas | Support pinned TorchTitan v0.3 schema directly; separate any true historical compatibility into a narrowly selected extension if still required by a supported environment. |
| Patch installation | Many import-time global assignments | Prefer Config/ModelSpec Override and upstream hooks. Importing `torchtitan_npu.models.qwen3_5` should be as close to declarative registration as possible. |

## 10. Required changes before re-review

| Priority | Required change | Acceptance evidence |
|---:|---|---|
| 1 | Resolve feature-scope mismatch: implement VL CP or rename/re-scope to text-only CP + VL non-CP adapter. | PR title/body/README/config/tests all describe the same support matrix. |
| 2 | Replace invalid 32K validation record with a command executable on current head that actually selects `qwen35_27b_long_text_sft`. | Command, selected config/CP degree, successful exit, representative loss/step logs. |
| 3 | Collapse Qwen training onto repository single NPU entry (`scripts/run_train.sh` / `torchtitan_npu.train`). | No duplicated torchrun/env launcher; integration runner no longer needs model-specific script plumbing unless independently justified. |
| 4 | Move NPU/Qwen-specific code out of `patches/torchtitan`; ensure remaining patch code maps to real upstream PRs/commits. | Keep the new c4d1c6b2 rationale, but add actual upstream PR links and correct ownership; no “pending publication” placeholder for code retained in `patches/torchtitan`. |
| 5 | Remove/narrow import-time/global monkey patches (`model_registry`, GatedDeltaNet method, SPMD annotation, private PyTorch CP helper). | Override/Extension selection is explicit and local; unselected models are unaffected. |
| 6 | Restrict Flex dense fallback to explicit opt-in and prove semantics. | UT showing marked-mask equivalence, empty-row behavior, threshold, and unmarked/other-model path unchanged. |
| 7 | Enable real Qwen model ST and add minimal text-only CP ST. | Registered, non-disabled cases reachable from `.ci/smoke_test.sh`; successful training completion, not merely import/kernel tests. |
| 8 | Replace source-string tests with functional producer/consumer tests for CP, GDN, vision wrapper, MoE/dispatcher and PP bridge. | Independent expected outputs/gradients and minimum distributed tests where communication semantics are changed. |
| 9 | Stop removing `vision_encoder` after model build; make text-only topology/config explicit. | State dict/checkpoint/export behavior deterministic and tested. |
| 10 | Refresh docs and upstream-version/provenance comments after code changes. | No contradiction among `requirements.txt`, PR body, examples README, integration README, patch headers and actual runner/test registration. |

## 11. Positive changes retained / newly confirmed

| Change | Review |
|---|---|
| `833f18d0`: new files/owned identifiers renamed from local `qwen35` to `qwen3_5` while preserving upstream `qwen35_*` API names | **Accepted.** Correct direction; removes local naming drift without renaming upstream contracts. |
| `833f18d0`: orphan NPU runtime pytest moved to `tests/smoke_tests/ops/test_qwen3_5_npu_runtime.py` and marked `smoke` | **Accepted.** It is now statically reachable from `.ci/smoke_test.sh`. |
| `833f18d0`: misleading VL CP2 integration entry removed | **Accepted.** Better than retaining an unsupported test declaration; PR scope/docs still need matching correction. |
| `c4d1c6b2`: vision-mask workaround now documents concrete CANN 9.0 root cause, scope, and removal condition | **Accepted as documentation improvement.** It does not by itself satisfy upstream-PR or patch-directory ownership requirements. |
| MoE NPU dispatcher remains opt-in instead of default-on | Correct direction; NPU performance override should remain explicit. |
| Qwen-specific GDN kernel code lives under `override/qwen3_5` rather than a `torch_npu`/PyTorch patch | Correct ownership direction; remaining issue is the global class-method mutation inside the override. |
| No new `torch_npu` or PyTorch patch introduced by this PR | Good; no extra low-level patch acceptance burden added. |

---

**Final maintainer conclusion: Request Changes.** The two new GitCode commits improve naming, test collection, and workaround documentation, but they do not alter the core architecture/coverage verdict. The highest-value next steps remain: make the support scope truthful and reproducible, collapse to the single training entry, restore patch-directory ownership/provenance, restrict global monkey-patch blast radius, and establish two active model-level STs (basic Qwen + text CP).
