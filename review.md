# PR 807 Maintainer Review

> Review target: `pr_807 -> master` (mirror PR #12 / GitCode !807)  
> Reviewed head: `106e81d7360de5345041130878282bafc697a9e0`  
> Actual merge base: `master@4c4079dc932234ae9b438011125c935669820080`  
> Fixed TorchTitan dependency: `torchtitan==0.3.0` / CI checkout `v0.3.0`  
> Test review mode: repository `developer-tests-review` skill, static review only.  
> **测试执行：未执行（仅静态审查）**

## 1. 结论

**总体合入建议：暂停并澄清。**

本 PR 的 Engram 数值/分布式测试设计总体较完整，默认 Torch Host 路径也保持在 `deepseek_v41` 模型域内；NPU 特有实现放在 `ops/ascendc` + `override/deepseek_v41`，没有把模型/NPU 代码塞入 `patches/torchtitan`，这一点符合仓库分层要求。

但当前实现仍有 4 个合入前必须处理的架构/入口问题：

1. Host sparse gradient clipping 通过全局 monkey patch 永久改写上游 `torchtitan.distributed.utils.clip_grad_norm_`，污染整个 Python 进程；
2. 为注入 FSDP `ignored_params`，通过 `FunctionType` 复制上游 `apply_fsdp_to_decoder.__code__/__globals__`，与上游实现细节强耦合；
3. Engram enablement 同时存在于 Trainer Config 和 Model Config，且 Model `update_from_config()` 反向修改 optimizer param groups，破坏配置分层；
4. 正式 V4.1 flavor 默认启用 Engram 后，正常训练入口隐式要求用户先额外生成 `engram_token_id_map.npy`，现有 example/README 没有把这个前置条件接入单一训练流程。

此外，可选 AscendC 路径仍依赖 operator-pack 私有属性/未验收行为，PR 自述也明确“未作新算子包 NPU 验收”；如果该路径要与 Torch 路径一起作为生产能力合入，应完成稳定 capability contract 和真实 NPU acceptance，或者本 PR 先只合入 Torch Host 路径。

### 1.1 主要 review 意见

| ID | 级别 | 代码位置 | 问题点 | 影响 | 建议修改方案 |
| --- | --- | --- | --- | --- | --- |
| R1 | **Blocker** | `torchtitan_npu/extensions/components/optimizer.py::_install_host_sparse_grad_clip`, `register_host_sparse_table` | 首次并行化 Host Engram table 时执行 `dist_utils.clip_grad_norm_ = _clip_grad_norm_with_host_sparse_tables`。`WeakSet` 只解决 table 生命周期，**没有恢复被替换的上游全局函数**；一旦安装，进程中后续非 V4.1/非 Engram Trainer 也会经过该 wrapper。 | 违反 Extensions “增强而非污染上游全局状态、优先稳定 Hook”的定位；测试进程、多 Trainer/多模型进程会受到隐式影响；上游 `clip_grad_norm_` 签名/语义升级也会把兼容风险扩散到全仓。PR 描述虽然强调 FSDP 不改共享全局符号，但 gradient clip 实际仍改了共享符号。 | 不接受永久 monkey patch。优先向 TorchTitan 增加一个通用、可上游的 grad-clip extension point（如 Trainer/ModelSpec 的 clip hook 或可注入 clip callable），临时上游 patch 仅在已有/准备提交上游 PR 的前提下放 `patches/torchtitan`；Engram-specific norm/reduction 留在本仓 Extension/Model 侧。若短期只能本仓实现，也应将行为限制到 V4.1 Trainer 实例生命周期，不能替换模块全局函数。 |
| R2 | **Blocker** | `torchtitan_npu/models/deepseek_v41/parallelize.py::_apply_fsdp_with_ignored_params` | 为给 TorchTitan v0.3.0 `apply_fsdp_to_decoder()` 注入 `ignored_params`，复制 `__globals__`、替换其中 `fully_shard`，再用 `FunctionType(apply_fsdp_to_decoder.__code__, ...)` 重建函数。 | 这是对上游函数内部实现的“影子复制”。任何上游对函数 body、globals、closure、默认参数或 `fully_shard` 调用方式的调整都可能直接 break；代码 review 也无法再通过函数签名判断真实依赖。与本仓“上游升级不应轻易 break”的核心职责冲突。 | 把需求收敛成通用上游能力：为 `apply_fsdp_to_decoder` 增加 `ignored_params`（或参数过滤 callback）并直接透传给各 `fully_shard` 调用。若暂未上游合入，可按 patches 目录定位暂存**可直接贡献上游**的最小 patch，并附上上游 issue/PR；不要复制 `__code__`/`__globals__`。 |
| R3 | **Major** | `torchtitan_npu/models/deepseek_v41/config_registry.py::DeepSeekV41TrainerConfig`; `model.py::V41Model.Config.update_from_config` | `engram_enabled` 同时出现在 `DeepSeekV41TrainerConfig` 与 `V41Model.Config`；Trainer `__post_init__` 把值拷进 Model，随后 Model `update_from_config()` 又直接删除 `config.optimizer.param_groups` 中 pattern 等于 `.*\.engram\.table\.weight$` 的项。 | 两个 source of truth；Model config 越层修改 optimizer recipe，配置语义变得顺序敏感。用户若提供自定义同 pattern group，也会被 Model 层按字符串静默删除。以后 config converter/override 顺序调整容易产生隐蔽回归。 | 只保留一个 enablement source。若因为上游 `model_spec` CLI suppress 必须保留 V4.1 Trainer CLI 字段，应在同一个 Trainer/config transform 阶段一次性派生 model topology + optimizer groups；`V41Model.Config.update_from_config()` 只处理模型自身（层、token-map、并行约束），不要写 `config.optimizer`。长期建议向上游补通用 model-config CLI exposure/transform，而不是继续增加 model-specific Trainer config 传递字段。 |
| R4 | **Major** | `model.py::update_from_config`; `engram.py::_load_token_id_map`; `scripts/generate_engram_token_map.py`; `requirements.txt`; `examples/deepseek_v41/debug/deepseek_v41_flash_8p_cpt_4k_a3.sh` | 30/40-layer flavor 默认启用 Engram；当 `token_id_map_path` 未显式给出时自动指向 `${hf_assets_path}/engram_token_id_map.npy`，但标准 HF assets 并不包含该文件，本 PR 也未提交该 asset。example 仍只接受普通 tokenizer path，不生成/检查 map。用户必须知道并手工先跑新 Python 脚本。与此同时 `tokenizers==0.22.2` 被加入全局运行时 requirements，但生产训练代码只读 `.npy`，它主要服务离线脚本/测试。 | 现有单一训练入口被隐式拆成“先生成 asset，再训练”两个步骤；默认开启的新能力会把过去可启动的正式 V4.1 命令变成运行时 `FileNotFoundError`。这与本仓训练入口简单、CLI 明确的方向冲突。 | 首选把确定性的 token-map 构建并入现有 HF asset/模型初始化流程：从 `${hf_assets_path}/tokenizer.json` 直接构建（可内存生成或通过现有 asset workflow 缓存并校验 checksum），训练命令无需额外脚本。若坚持离线生成，则必须把该步骤写入正式文档/example preflight，提供明确 CLI/path，并把仅工具使用的 `tokenizers` 依赖移到合适的 tooling/dev 依赖，而不是无条件扩大训练运行时依赖。 |
| R5 | **Major** | `torchtitan_npu/override/deepseek_v41/engram/ascendc.py::_submit_engram_storage`, `_dedicated_engram_group`; `tests/integration_tests/deepseek_v41.py::build_engram_ascendc_test_list` | AscendC path 用 `getattr(elastic_buffer, "_engram_storage_ref", None)` 判断 operator-pack 是否具备 direct registration，且注释直接依赖外部 PR 10952/11226 的行为；这是对 pack 私有属性的 capability probe。PR 同时声明本轮没有做新算子包 NPU 验收。 | 外部 pack 升级/改私有字段即可静默退化到 per-forward write；具体支持矩阵无法从版本/公开 API 判断。对于本 PR 新增的 466 行 custom-op + override 生产代码，仅 CPU fake contract 不能证明真实 HCCL/ElasticBuffer 语义。 | 二选一：A) 本 PR 先只合入 Torch Host Engram，AscendC override 在 operator-pack 稳定 API + NPU acceptance 后单独合；B) operator pack 暴露稳定 capability/version contract（不要探测 `_engram_storage_ref`），仓内记录最小兼容版本，并实际运行 `engram_ascendc` 2-card case 验证 forward/backward/optimizer/resume。 |
| R6 | **Major** | `tests/integration_tests/README.md`; PR 描述 “基于 master 8b0da62” | 文档没有随测试/入口变化刷新：测试矩阵缺少 `dsv41_engram_torch_ep2_fsdp2_resume`、4-card replica case 和显式 `engram_ascendc` suite；对 V4.1 test scope 的描述仍停留在旧 golden 结构。PR body 还写旧 master `8b0da62`，实际 merge base 已是 `4c4079d...`。 | Reviewer/CI 使用者无法从仓内文档获知哪些 Engram case 属于 default `models` suite、哪些仅显式运行，以及 token-map/AscendC 前置条件；违反“文档与代码一致”。 | 在本 PR 更新 `tests/integration_tests/README.md` 的矩阵、入口、Engram Torch/AscendC coverage 与限制；同步说明 token-map 资产来源、HF/DCP 限制。PR 描述基线信息也应刷新或删除易过期 SHA。 |
| R7 | **Minor** | `tests/integration_tests/__init__.py::OverrideDefinitions.requires_engram_ops`; `tests/integration_tests/deepseek_v41.py::_build_engram_case`; `tests/unit_tests/tooling/integration_tests/test_case_definitions.py` | `requires_engram_ops` 只被 testcase 定义和 UT 读取，runner 完全不消费它；它没有 dependency gating、skip 或环境校验效果。另 Engram case 设 `use_golden=False`，但 `env_vars` 继承 `GOLDEN_ENV`，实际 V4.1 主干仍走 golden/reference operator path；字段语义与真实命令不一致。 | 增加“看起来有保护、实际无行为”的配置状态，测试元数据容易被误当作 runner contract。 | 若 suite 隔离已经足够，删掉 `requires_engram_ops`；若确实需要 dependency gate，则让 runner 明确消费并在缺 pack 时 fail/skip。同步修正 `use_golden` 元数据，或移除这个 runner 已不使用的字段，避免用 dead metadata 表示执行路径。 |
| R8 | **Minor** | `tests/unit_tests/models/deepseek_v41/test_engram_replica_grad.py`; `test_engram_host.py` | 多进程 Gloo UT 启动子进程后，timeout/failure 路径没有统一的 process-group cleanup。尤其 `Popen(...).communicate(timeout=180)` 抛 `TimeoutExpired` 时没有 `finally` terminate/kill；`subprocess.run(torch.distributed.run, timeout=...)` 也只保证 launcher 处理，不等价于仓内 integration runner 的整进程组清理。 | 失败时可能遗留 worker/端口，污染后续 UT；与 test-review 的“文件/进程必须隔离”规则不符。 | 封装一个测试侧 launcher：`start_new_session=True`，`try/finally` 中 TERM/KILL 整个进程组；或者使用可 join/terminate 的 `torch.multiprocessing` helper。两组 Gloo worker 复用同一清理 helper。 |
| R9 | **Minor** | `engram_host.py::supports_model_compile`; `override/.../ascendc.py::supports_model_compile`; `engram_host.py::wire_sparse_grad_replicas`; 多处 EP divisibility 检查 | `supports_model_compile` 只赋值、不被任何生产路径消费，而 PR 明确声明 model compile 不在支持范围；同时 `wire_sparse_grad_replicas` 使用 `DeviceMesh._flatten()` 私有 API；EP divisibility 在 model update、Host parallelize、AscendC buffer init 重复校验。 | 形成无效能力标志、私有 API 耦合和重复防御式校验，增加后续维护成本。 | 未实际消费的 capability flag 删除；若未来要支持 compile，再接入统一 validation。能用公开 mesh API 时不要依赖 `_flatten()`。同一 invariant 保留一个配置边界 + 一个外部 backend 边界即可，不要层层重复。 |
| R10 | **基本符合（无新增 blocker）** | `torchtitan_npu/extensions/components/optimizer.py::HostSparseOptimizersContainer`, `_materialize_missing_adam_state`; `config_registry.py::_v41_optimizer_config` | **优化器分层总体符合 maintainer 约定。** `HostSparseOptimizersContainer` 位于 `extensions/components/optimizer.py`，直接继承上游 `OptimizersContainer`；参数选择继续使用上游 `ParamGroupConfig`/first-match-wins 机制，`_build_param_groups()` 先调用上游实现再只对 `SparseAdam` group 去掉 unsupported `fused/foreach` 并补 checkpoint suffix，`step()`/`zero_grad()`/`state_dict()`/`load_state_dict()` 也均围绕 `super()` 扩展，没有复制一套 mixed-optimizer 构建、参数归属或 flat state-dict 框架。`_materialize_missing_adam_state()` 的存在有固定上游 v0.3.0 依据：上游 `init_optim_state()` 只要 `optim.state` 非空就直接返回，因此“已有部分参数 state、另一些参数从未参与本步”的场景会留下 DCP state 缺口；当前 helper 在 optimizer/checkpoint state 边界内补齐缺失 state，**没有越到 model 层**。但它手工写死 Adam/AdamW/SparseAdam 的 `step/exp_avg/exp_avg_sq/max_exp_avg_sq` schema，属于与 PyTorch optimizer internals 耦合的兼容补丁，不宜长期固化。 | 该设计没有违反“优化器修改独立出来、和优化器放在一起、基于已有混合优化器架构开发”的主要求；相反，继承/复用关系是清楚的。剩余的明确分层违反点就是 R1：同一个 optimizer extension 为接入 sparse grad clip，越过容器/实例边界去永久改写 `torchtitan.distributed.utils.clip_grad_norm_` 的进程级全局符号，使优化器特性反向污染 Trainer/全仓运行时。`_materialize_missing_adam_state()` 则是维护债而非分层越界。 | 保留当前 `HostSparseOptimizersContainer -> OptimizersContainer`、`ParamGroupConfig` 的扩展方式，不要把 SparseAdam 分组/step/checkpoint 逻辑搬回模型代码。R1 按实例级/上游 hook 解决。对 `_materialize_missing_adam_state()`，优先向上游修复 `init_optim_state()` 的“partial state”语义或提供通用 missing-state helper；在上游能力到位前可暂留本扩展，但要继续限定支持的 optimizer 类型并用现有 unused-parameter state UT 锁住行为。 |
| R11 | **语义正确（无新增 blocker）** | `torchtitan_npu/models/deepseek_v41/engram_host.py::wire_sparse_grad_replicas`, `reduce_sparse_gradient_across_replicas`; `parallelize.py::_shard_engram_tables`; fixed TorchTitan v0.3.0 `resolve_sparse_fsdp_mesh` / `disable_fsdp_gradient_division` / Trainer loss normalization | **eFSDP 同分片副本梯度同步的数学与时序是正确的。** Host table 只沿 EP 切连续 row shard，`weight` 又被作为 `ignored_params` 排除在 FSDP 管理之外，因此相同 EP rank 的 table shard 会在 `dp_replicate`/`efsdp` 轴形成副本。full-DTensor/SPMD 路径的上游 sparse storage mesh 明确是 `dp_replicate × efsdp × ep`，而 `DataParallelMeshDims` 只把 `dp_replicate` 与 `efsdp` 标成数据并行轴；本实现按 `(replicate, shard)` 选轴并显式排除 `ep`。partial-DTensor 路径传入的 `edp_mesh` 本身就只含 `dp_replicate/efsdp`，同样不会把不同 EP owner 混在一起。`reduce_sparse_gradient_across_replicas()` 先 all-reduce 每个 replica 的 row count，再按最大行数 pad，all-gather `(ids, values)`，最后在 CPU 构建 COO 并 `coalesce()`；因此不同 replica 命中同 row 时做求和，不同行做并集，零命中 replica 仍参加 collective。 | 不存在与稠密 FSDP 的 double-counting：稠密/gate 参数在 backward 中由 FSDP 自己 reduce-scatter/all-reduce；Host table weight 被 ignored，FSDP 不会再规约它，手工 sparse reduction 是它唯一的 DP 归约。时序上，TorchTitan 完成 backward 后才进入 global clip；Engram-aware clip 在计算 sparse norm/optimizer step **之前**调用 `reduce_sparse_gradient_across_replicas()`，所以拿到的是已跨副本求和的 table grad，不会漏掉 replica。缩放也自洽：固定上游 Trainer 的 loss 是 `local loss SUM / global_valid_tokens`，同时 `apply_fsdp_to_decoder()` 调用 `disable_fsdp_gradient_division()`，即稠密参数依赖跨 DP **求和**得到全局梯度而不是再除 world size；Host sparse grad 要与其保持同一标度，也必须对 replica **sum、不能 average**。因此当前没有再除以 `_replica_size` 是正确的。现有 4-rank Gloo worker 还直接验证了 `efsdp` 同 shard 求和、不同 `ep` shard 隔离和 empty replica 参与。 | 保留当前求和语义，不要新增 `/ replica_size`。R1/R2 重构时必须保持顺序为“backward/FSDP dense reduce 完成 -> Host sparse replica sum -> global norm/clip -> SparseAdam step”，并继续排除 EP 轴；4-rank Gloo regression 应作为该语义的固定 guard。R9 的 `DeviceMesh._flatten()` 私有 API 仍应替换为公开接口，但这是升级鲁棒性问题，不影响这里的数学正确性。建议在代码注释中把“不除副本数”的依据明确绑定到上游 `global_valid_tokens` loss normalization + `disable_fsdp_gradient_division()`，避免未来上游 scaling 变化后旧假设被静默保留。 |

## 2. 语义变换与独立 oracle

| 语义变换 | 生产实现 | 独立 oracle / 直接检查 | 结论 |
| --- | --- | --- | --- |
| tokenizer ID -> compressed ID -> n-gram hash row | `engram.py`, `generate_engram_token_map.py` | `test_engram.py` 的 NumPy hash/reference；`test_token_map.py` 使用真实 committed tokenizer 生成 dense compressed IDs | 已覆盖 |
| Engram context gate（RMS normalize + signed sqrt + sigmoid + residual） | `EngramContextGate` | `test_engram.py` 独立 reference，含 FP32/BF16、zero signed-sqrt | 已覆盖 |
| 图像 span / packed document 边界 | `EngramTable.hash`, `V41Model.forward` | `test_engram.py` image-mask/boundary case；`test_training_contract.py` 从真实 registry 走 model forward | 已覆盖 |
| EP owner routing + duplicate-row backward | `engram_lookup.py` | `engram_host_worker.py` 真实 2-rank Gloo All-to-All，正反向及 resume/state 验证 | 已覆盖 |
| 同 EP shard 的 eFSDP/replicate sparse-grad 求和 | `HostEngramTable.reduce_sparse_gradient_across_replicas` | `engram_replica_grad_worker.py` 真实 4-rank Gloo mesh，验证同 shard replica sum / 不同 EP shard 隔离 / empty replica；固定上游 loss normalization/FSDP no-divide 与该 sum 语义一致 | **语义正确且已直接覆盖**，见 R11 |
| Host sparse grad 进入 global clip + SparseAdam | `extensions/components/optimizer.py`, `engram_host.py` | `test_engram_sparse_grad.py` 独立 dense+sparse norm/clip、SparseAdam row update、state dict | 已覆盖功能语义；R1 架构实现不接受 |
| Native DCP 同 EP 恢复 | host table state-dict suffix + optimizer state | CPU/Gloo state tests + integration 两阶段 resume dynamic baseline | 已覆盖声明范围 |
| AscendC Fetch/FetchGrad wrapper | `ops/ascendc/engram.py`, `override/.../ascendc.py` | CPU fake operator contract + 显式 `engram_ascendc` integration case 定义 | **部分覆盖**：静态 case 存在，但 PR 自述未完成真实新算子包 NPU 验收，见 R5 |

## 3. UT 正向功能覆盖

| 正向功能 | 生产代码和应观察结果 | PR 中的直接检查 | 状态 | 合入前置条件 |
| --- | --- | --- | --- | --- |
| Engram hash/gate/residual | `engram.py`, selected layer before attention residual | 独立 NumPy/hash + gate reference、BF16 backward、zero gate、image mask | 已覆盖 | 无新增 UT |
| 正式/Debug geometry 与 registry attach | `config.py`, `engram_config.py`, `model_registry.py` | registry 构建、公开 geometry、disable path、optimizer group | 已覆盖 | R3 重构后更新相同 UT，不新增组合 |
| Torch Host local/EP lookup | `engram_host.py`, `engram_lookup.py` | 本地 lookup + 2-rank Gloo A2A forward/backward | 已覆盖 | 无新增 UT |
| sparse replica reduction | `wire_sparse_grad_replicas`, `reduce_sparse_gradient_across_replicas` | 4-rank Gloo | 已覆盖，且 R11 确认 sum/no-divide 语义与上游训练标度一致 | 修复子进程 cleanup（R8）；R1/R2 重构不得改变 reduction 时序/轴选择 |
| global clip / SparseAdam / checkpoint key | `HostSparseOptimizersContainer` | sparse norm、step、zero_grad、state dict suffix | 已覆盖语义；R10 确认容器分层基本符合 | R1 改成稳定 hook 后，测试应针对 hook 的实例作用域补一个“退出/另一个 Trainer 不受影响”断言 |
| FSDP ignored Host table | `parallelize.py` | `test_decoder_fsdp_keeps_shared_helper_unchanged` 证明当前 hack 不直接改原函数 object | 部分覆盖 | R2 修为显式上游参数后，测试改为断言 `ignored_params` 实际透传；不要继续测试 `FunctionType` hack |
| AscendC custom-op wrapper | `ops/ascendc/engram.py`, `override/.../ascendc.py` | Fake ElasticBuffer forward/backward/storage/capacity | 已覆盖 CPU contract | 真实 NPU 语义由 ST 覆盖，见 R5 |
| HF adapter rejection | `state_dict_adapter.py` | Engram enabled 时报拒绝；disable 保持原 HF path | 已覆盖 | 文档补齐限制 |

**测试专项静态结论：补充测试/验收后合入。** 不是缺少默认 Torch Host 的 UT/ST 结构，而是可选 AscendC 生产 override 尚缺真实 operator-pack NPU acceptance；如果本 PR 删除/拆出 AscendC 生产路径，则 Torch Host 测试结构本身可视为充分。

## 4. ST 触发判断

| 生产代码变更 | 受影响训练场景 | 为什么需要 ST | 现有代表测试 | 判断 | 合入前置条件 |
| --- | --- | --- | --- | --- | --- |
| 默认 V4.1 Engram Torch Host + EP/FSDP + FullAC | 2-rank EP2/FSDP2 | 涉及真实 NPU model forward/backward、CPU/NPU transfer、FSDP ignored param、SparseAdam、DCP | `dsv41_engram_torch_ep2_fsdp2_resume` | 复用/新增充分 | 保持该 case |
| sparse table replica over eFSDP | 4-rank, EP2 + DP shard4 | 2-rank case没有同 shard replica；必须验证 replica sum；R11 已确认该 sum 与 FSDP no-divide/global-token loss 标度一致 | `dsv41_engram_torch_ep2_fsdp4_resume` | 新增充分 | 保持 <=4 NPU；R1/R2 重构后仍须触发相同 replica group |
| disable Engram 保持既有 V4.1 golden | 2-rank | 默认启用特性改变旧配置，需要守住原轨迹 | `dsv41_golden_2p_ep2_fsdp2` + `--no-engram-enabled` | 调整充分 | README 说明该 case 的定位 |
| AscendC Fetch/FetchGrad override | 2-rank EP2/FSDP2 | 新 custom op、HCCL communicator、真实 ElasticBuffer、Host storage registration 都不是 CPU fake 能证明 | `dsv41_engram_ascendc_ep2_fsdp2_resume`（仅显式 suite） | **需要真实验收** | 运行并记录 operator-pack 版本；或拆出本 PR |
| CP>1 / TP / PP / model compile | PR 明确不支持 | 不属于本 PR 支持范围 | 无 | 延后 | 文档明确，不新增组合 |

## 5. 测试格式/隔离审查

| 位置 | 类型 | 代码阅读结果 | 修改建议 |
| --- | --- | --- | --- |
| `tests/unit_tests/models/deepseek_v41/test_engram*.py` | 文件位置 | CPU/Gloo UT 均位于对应模型目录，worker 文件不会被 pytest 当 testcase 收集 | 通过 |
| `tests/unit_tests/override/deepseek_v41/test_engram_ascendc.py` | 文件位置 | override contract 位于 override 对应目录，Fake 明确，不冒充真实 NPU | 通过 |
| `tests/integration_tests/deepseek_v41.py` + `run_tests.py` | integration runner 收集 | Torch Engram cases 进入 default `models`；AscendC case 进入显式 `engram_ascendc` suite | 通过；README 需同步 |
| `OverrideDefinitions.requires_engram_ops` | 测试结构 | runner 不消费，仅被测试自身断言 | 删除或接入真实 gating，见 R7 |
| `_build_engram_case(use_golden=False, env_vars=GOLDEN_ENV)` | 命名/测试入口 | 元数据说 non-golden，但实际 env 选择 V4.1 golden/reference 主干 operator path | 让元数据与实际命令一致，见 R7 |
| Gloo subprocess tests | 状态隔离 | timeout/failure 无统一子进程组清理 | 按 R8 增加 cleanup helper |

## 6. 4-NPU ST 事实表

| 测试 | 模型/配置 | 并行数值 | 替换实现/融合算子 | 编译模式 | NPU 数 | 启用与完成检查 | golden | 执行阶段 |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| `dsv41_golden_2p_ep2_fsdp2` | V4.1 debug, Engram disabled | EP2 / DP-shard2 / CP1 / TP1 / PP1 | RoPE workaround + existing virtual optimizer | eager | 2 | 30-step TensorBoard step/loss exact golden | 是 | default `models` |
| `dsv41_engram_torch_ep2_fsdp2_resume` | V4.1 debug + Torch Host Engram | EP2 / DP-shard2 / CP1 / TP1 / PP1 | Torch Host lookup + Host SparseAdam | eager + FullAC | 2 | phase0 steps 1-4；phase1 3-4；resume loss + grad_norm exact dynamic compare | 否 | default `models` |
| `dsv41_engram_torch_ep2_fsdp4_resume` | 同上 + same-shard replicas | EP2 / DP-shard4 / CP1 / TP1 / PP1 | Torch Host lookup | eager + FullAC | 4 | 同上，额外触发 sparse replica reduction | 否 | default `models` |
| `dsv41_engram_ascendc_ep2_fsdp2_resume` | V4.1 debug + AscendC HostOffload | EP2 / DP-shard2 / CP1 / TP1 / PP1 | `host_offload` -> CANN Fetch/FetchGrad | eager + FullAC | 2 | case 定义包含两阶段 resume；本次 review 未执行，PR 自述未完成新算子包 NPU 验收 | 否 | 显式 `engram_ascendc` suite |

### 6.1 V4.1 模型投影

| 并行方式 | 参考/默认 Torch eager | AscendC eager | 编译 |
| --- | --- | --- | --- |
| 2P EP2/FSDP2 | `dsv41_engram_torch_ep2_fsdp2_resume` | `dsv41_engram_ascendc_ep2_fsdp2_resume`（显式 suite，需真实验收） | 不支持（PR scope） |
| 4P EP2/FSDP4 | `dsv41_engram_torch_ep2_fsdp4_resume` | 无需重复；2P 已覆盖算子、4P 默认路径覆盖 replica 语义 | 不支持 |
| CP>1 | 不支持 | 不支持 | 不支持 |
| TP/PP | 不支持 | 不支持 | 不支持 |

## 7. ST 不能证明的内容

| 项目 | 当前状态 | 说明 |
| --- | --- | --- |
| 正式 384M-row/table 的容量/长期性能 | 未由 debug ST 证明 | debug table 缩小；正式 Host 内存、A2A/CPU lookup 吞吐需要性能/容量评估，不应从 4-step ST 外推 |
| 跨 EP degree checkpoint reshaping | 明确不支持 | 当前 key 带 EP shard suffix，只承诺同 EP 拓扑恢复 |
| HF import/export with Engram | 明确拒绝 | state-dict adapter 主动抛错；应在用户文档写清 |
| AscendC operator-pack compatibility | 未证明 | CPU fake 不能证明真实 HCCL/ElasticBuffer；需要 R5 的 capability/version + NPU acceptance |
| CP>1、TP、PP、model compile | 不在范围 | 不应由当前 ST 宣称支持 |

## 8. 架构与 clean-code 专项结论

| 维度 | 结论 | 说明 |
| --- | --- | --- |
| `patches/torchtitan` 定位 | 通过 | 本 PR 没有把 NPU/model-specific Engram 放到 patch 目录；R1/R2 建议若必须动上游，只有“通用 hook / ignored_params”这种可上游的小接口才适合临时 patch。 |
| Extensions 解耦 | **不通过** | R1 永久 monkey patch 上游 global；应使用显式 hook。 |
| 优化器分层约定 | **基本通过** | R10：`HostSparseOptimizersContainer` 放在 `extensions/components/optimizer.py`，继承/复用上游 `OptimizersContainer` 与 `ParamGroupConfig`，没有复制 mixed-optimizer 主框架；`_materialize_missing_adam_state()` 是 fixed v0.3.0 partial-state 缺口的局部兼容补丁。明确越界点仍是 R1 的进程级 global clip monkey patch。 |
| eFSDP 同分片 sparse-grad 语义 | **通过** | R11：table 只沿 EP 切 shard，manual reduction 只沿 `dp_replicate/efsdp` 求和并排除 EP；Host weight 被 FSDP ignored，不会 double-reduce；global-token loss + FSDP no-divide 下应 sum 而不是 average。 |
| Override 边界 | 基本通过 | AscendC 特有 table 在 `override/deepseek_v41/engram`，通过显式 `override.imports` 选择，不隐式替换默认 Torch 路径。R5 的 operator-pack 私有 contract 仍需整改。 |
| 上游升级鲁棒性 | **不通过** | R2 复制 `apply_fsdp_to_decoder.__code__/__globals__`；另有 `DeviceMesh._flatten()` 私有 API。 |
| 单一训练入口 | **不通过** | 默认正式 flavor 额外依赖手工生成 token map；应并入现有 asset/训练入口。 |
| 配置职责 | **不通过** | 双 `engram_enabled` + Model 修改 optimizer；见 R3。 |
| 防御式校验 | 需精简 | backend 边界的 shape/capacity 检查合理，但相同 EP divisibility/spec invariant 在多个层级重复。保留最靠近配置入口和外部 backend 的必要检查即可。 |
| 文档刷新 | **不通过** | README/测试矩阵没有 Engram case；PR 自身 checklist 也未勾文档。 |

## 9. 逐文件审查清单（35/35）

| 变更文件 | Review 结果 |
| --- | --- |
| `examples/deepseek_v41/debug/deepseek_v41_flash_8p_cpt_4k_a3.sh` | optimizer group 调整与默认 Engram 对齐；但未处理 token-map 前置条件，见 R4。 |
| `requirements.txt` | Torch/TorchTitan pin 未变；新增 `tokenizers==0.22.2` 只为离线 map/tooling 的理由不足，随 R4 调整依赖层级。 |
| `scripts/generate_engram_token_map.py` | normalization pipeline/确定性实现清楚；问题是它当前成为默认正式训练的隐藏前置步骤，见 R4。 |
| `tests/integration_tests/__init__.py` | 新 `requires_engram_ops` 是 dead metadata，见 R7。 |
| `tests/integration_tests/deepseek_v41.py` | 2P/4P Torch resume case 设计合理；4P case 会真实进入 same-EP-shard replica reduction，和 R11 的 4-rank Gloo oracle共同守护 eFSDP 语义；AscendC suite 静态注册正确；metadata 与实际 golden env 有偏差，见 R5/R7。 |
| `tests/integration_tests/run_tests.py` | suite 注册正确；没有消费 `requires_engram_ops`，见 R7。 |
| `tests/unit_tests/models/deepseek_v41/engram_host_worker.py` | 真 2-rank Gloo worker，生产调用真实，作为 helper 位置/命名正确。 |
| `tests/unit_tests/models/deepseek_v41/engram_replica_grad_worker.py` | 真 4-rank `efsdp × ep` Gloo mesh；直接断言同 EP shard 的 eFSDP replicas 求和、不同 EP shard 不混合、无命中 replica 仍参加 collective，给 R11 提供了独立于 ST resume 的数值依据；launcher cleanup 见 R8。 |
| `tests/unit_tests/models/deepseek_v41/test_engram.py` | hash/gate/boundary/registry 独立 oracle 较完整，无新增 blocker。 |
| `tests/unit_tests/models/deepseek_v41/test_engram_host.py` | Host lookup/SparseAdam/Gloo coverage 合理；子进程 cleanup 见 R8。 |
| `tests/unit_tests/models/deepseek_v41/test_engram_replica_grad.py` | 真实 4-rank mesh 是 R11 的必要覆盖，验证两个 replica 对同一 EP shard 获得相同 summed sparse gradient；timeout cleanup 见 R8。 |
| `tests/unit_tests/models/deepseek_v41/test_engram_sparse_grad.py` | clip/replica/checkpoint suffix 语义覆盖充分；其中 mesh-axis UT 显式验证选择 `dp_replicate/efsdp` 且排除 `ep`，与 R11 生产语义一致；R1 重构后需把 clip 测试目标从 global patch 改为实例 hook。 |
| `tests/unit_tests/models/deepseek_v41/test_state_dict_adapter.py` | 关闭 Engram 后保留既有 HF adapter contract，合理。 |
| `tests/unit_tests/models/deepseek_v41/test_training_contract.py` | 真实 registry + model forward + FullAC 的层级正确；覆盖 selected/non-selected layer 与 image mask。 |
| `tests/unit_tests/override/deepseek_v41/test_engram_ascendc.py` | CPU fake wrapper contract 细；不能替代真实 operator-pack NPU acceptance，见 R5。 |
| `tests/unit_tests/tooling/engram/test_token_map.py` | 使用真实 committed tokenizer，能锁定 map 生成/加载契约；不解决训练入口两阶段问题。 |
| `tests/unit_tests/tooling/integration_tests/test_case_definitions.py` | suite/模型隔离断言有价值；当前也在固化 dead `requires_engram_ops`，随 R7 简化。 |
| `torchtitan_npu/extensions/components/optimizer.py` | **R10：优化器分层基本符合。** 文件位置与职责正确，`HostSparseOptimizersContainer` 继承上游 `OptimizersContainer`、复用 `ParamGroupConfig/_build_param_groups/super().step()/zero_grad()/state_dict()`，没有复制混合优化器主框架。`_materialize_missing_adam_state()` 是 fixed v0.3.0 `init_optim_state()` 对 partial state 直接 no-op 的局部补丁，仍在 optimizer/checkpoint state 边界，但手写 Adam/SparseAdam state schema 应优先上游化。真正的架构 blocker 仍是 R1：该文件越过实例边界永久 monkey patch 全局 clip。 |
| `torchtitan_npu/models/deepseek_v41/__init__.py` | 仅导出 `EngramArgs`，无问题。 |
| `torchtitan_npu/models/deepseek_v41/block.py` | Engram 插在 attention residual 保存之前，与 PR 声明/HF 对齐目标一致。 |
| `torchtitan_npu/models/deepseek_v41/config.py` | 正式/debug geometry 集中定义合理；`engram_enabled` source-of-truth 问题在 R3。 |
| `torchtitan_npu/models/deepseek_v41/config_registry.py` | SparseAdam table group 继续用上游 `ParamGroupConfig`，并选择 `HostSparseOptimizersContainer.Config` 承接 mixed optimizer，符合 R10 的分层方向；新增 model-specific Trainer Config 与双状态问题仍见 R3。 |
| `torchtitan_npu/models/deepseek_v41/engram.py` | hash/gate/model reference 实现边界清楚；token-map 外部 asset 入口见 R4。 |
| `torchtitan_npu/models/deepseek_v41/engram_config.py` | geometry -> table/gate config 组织合理；若继续收敛校验，可减少与下层重复 invariant。 |
| `torchtitan_npu/models/deepseek_v41/engram_host.py` | **R11：eFSDP replica gradient 同步语义正确。** Host table 仅按 EP row-shard，`wire_sparse_grad_replicas()` 在 full-DTensor/SPMD 下只选 `DataParallelMeshDims.replicate/shard`（`dp_replicate/efsdp`）并排除 `ep`，partial-DTensor 下传入 mesh 本身已不含 EP；`reduce_sparse_gradient_across_replicas()` 对变长 sparse rows all-gather 后 coalesce 求和，包含 empty replica。Host weight 被 FSDP ignored，所以该 manual sum 不会与 FSDP double-count；global-token loss + FSDP no-divide 下不除 replica size 是正确的。global registration/R1、私有 mesh `_flatten`/R9 仍需处理。 |
| `torchtitan_npu/models/deepseek_v41/engram_lookup.py` | Torch EP All-to-All forward/backward 边界清楚，有真实 Gloo oracle。 |
| `torchtitan_npu/models/deepseek_v41/model.py` | image mask 修正与 Engram path 合理；跨层修改 optimizer + hidden token-map precondition 见 R3/R4。 |
| `torchtitan_npu/models/deepseek_v41/model_registry.py` | attach selected layers、ModelSpec 仍是单一入口；无额外 model flavor/script 膨胀。 |
| `torchtitan_npu/models/deepseek_v41/parallelize.py` | Host table 在 FSDP 外管理的需求合理；`_shard_engram_tables()` 将 Host weight 加入 ignored set，并用 sparse EDP mesh wiring replicas，这正是 R11 不 double-count 且能补齐 eFSDP DP reduction 的前提；`FunctionType` adapter 本身仍不可接受，见 R2。 |
| `torchtitan_npu/models/deepseek_v41/sharding.py` | Engram dense boundary/table state sharding 与现有 V4.1 config 结构一致，无单独 blocker。 |
| `torchtitan_npu/models/deepseek_v41/state_dict_adapter.py` | 明确拒绝 Engram HF conversion，与 PR support scope 一致；需要文档同步。 |
| `torchtitan_npu/ops/ascendc/deepep.py` | 仅补 union return typing cast，不改变计算路径；无问题。 |
| `torchtitan_npu/ops/ascendc/engram.py` | custom-op 边界/opaque handle/fake shape 设计符合 NPU op wrapper 定位；真实 pack 语义仍需 R5 验收。 |
| `torchtitan_npu/override/deepseek_v41/engram/__init__.py` | 显式 `override` 替换 Host table Config，模型/NPU 特有代码没有进入 patches；通过。 |
| `torchtitan_npu/override/deepseek_v41/engram/ascendc.py` | NPU-specific 实现位置正确；私有 operator-pack capability probe 和未验收状态见 R5。 |

## 10. 固定上游调用链核对

| 项目 | 核对结果 |
| --- | --- |
| TorchTitan 版本 | `requirements.txt` 固定 `torchtitan==0.3.0`；`.ci/setup_torchtitan.sh` / smoke/unit CI 从版本推导并 checkout `v0.3.0`。 |
| FSDP helper | 固定上游 `v0.3.0` 的 `apply_fsdp_to_decoder()` 确实没有 `ignored_params` 参数，因此 R2 的需求是真实缺口；但解决方式应是通用上游接口，不是 `FunctionType` 克隆。 |
| Gradient clip | 固定上游 Trainer 在 `train_step()` 中直接调用 `dist_utils.clip_grad_norm_`，解释了作者为何做 R1；同样说明需要一个真正的上游/Trainer extension point。 |
| Optimizer container/state init | 固定上游 `OptimizersContainer` 明确支持通过继承扩展 `step/zero_grad/state_dict/load_state_dict`，并提供 `ParamGroupConfig` mixed-optimizer 架构；本 PR 的 Host container 正是在该骨架上增量实现。上游 `init_optim_state()` 又在 `optim.state` 任意非空时直接返回，证明 R10 中 partial-state materialization 的需求真实存在，但更合适的长期修复点是上游通用 helper，而不是继续扩展手写 state schema。 |
| Optimizer hooks | 上游 `OptimizersContainer` 已支持 optimizer step hook，但它发生在 grad clip 之后，不能直接替代“把 sparse grad 纳入 global clip”的需求；不能用现有 step hook 假装 R1 已解决。 |
| eFSDP sparse replica/scaling | 固定上游 `resolve_sparse_fsdp_mesh()` 使用 `dp_replicate × efsdp × ep` sparse storage mesh，并仅把 `dp_replicate/efsdp` 标为 DP axes；`apply_fsdp_to_decoder()` 最后调用 `disable_fsdp_gradient_division()`。Trainer 反传的 loss 是 `local SUM / global_valid_tokens`，因此 R11 的 Host sparse replica reduction 应与 dense FSDP 一样做 sum 而不是 average。 |
| patches 目录 | 本 PR 无 patch 变更；没有 NPU/model-specific code 误入 `patches/torchtitan`。 |

## 11. 合入前置条件（按优先级）

| 顺序 | 合入前必须完成的修改/确认 | 验证方式 |
| ---: | --- | --- |
| 1 | 删除 R1 的进程级 `clip_grad_norm_` monkey patch，改成显式、实例级、可上游的 grad-clip 扩展点。**这是 R10 中优化器分层唯一明确的不符合点**；整改时应继续保留 `HostSparseOptimizersContainer -> OptimizersContainer` 的继承/复用关系，不要把 sparse optimizer/clip 逻辑反向散落到 model config 或其它模型模块。 | UT 增加“构造/销毁 Engram trainer 不改变另一个普通 Trainer 的 clip callable/行为”；现有 sparse norm 数值 oracle 保留；静态确认 optimizer 扩展仍位于 `extensions/components/optimizer.py` 并复用 `ParamGroupConfig`。 |
| 2 | 删除 R2 的 `FunctionType(__code__/__globals__)` 适配，提供上游可接受的 `ignored_params`/filter hook。整改时必须保持 R11 的关键前提：Host table weight 继续完全排除在 FSDP gradient reduction 外，且 replica mesh 仍来自 sparse DP axes。 | fixed v0.3.0 patch 或升级后的公开签名上直接断言 Host table weight 被 ignored，其余 FSDP 行为复用上游 helper；4-rank replica UT 继续通过其静态调用链。 |
| 3 | 收敛 R3 配置 source of truth，Model 不再修改 optimizer config | CLI `--no-engram-enabled` 测试仍应断言 layers 无 Engram、optimizer 无 SparseAdam group、用户自定义 dense optimizer group 不被误删 |
| 4 | 解决 R4：正式默认入口无需隐藏的手工 token-map 预处理；同步依赖层级 | 从普通 `${hf_assets_path}/tokenizer.json` 启动 config/build 能获得 token map；example 不需要额外未声明命令 |
| 5 | 对 R5 做选择：拆出 AscendC，或稳定 capability API + 真实 2P NPU acceptance | 显式 `engram_ascendc` suite forward/backward + 4-step/step2 resume；记录 operator-pack 兼容版本 |
| 6 | 更新 `tests/integration_tests/README.md`、支持范围和 token-map/checkpoint 文档 | 文档矩阵与 `build_*_test_list` / runner suite 一致 |
| 7 | 清理 R7/R8/R9 | 静态 review：无 dead metadata、subprocess failure 有完整 cleanup、无无效 compile flag/不必要私有 API |
| 8 | **R10 无新增架构 blocker。** 保留现有基于上游 mixed-optimizer 的容器扩展方式；`_materialize_missing_adam_state()` 在当前 fixed v0.3.0 下可作为局部兼容层，但应记录/推进上游 `init_optim_state()` 对 partial state 的通用修复，避免长期依赖手写 Adam state schema。 | 现有 `test_engram_optimizer_materializes_state_for_unused_parameters` 继续直接断言 unused parameter 的 `step=0`、`exp_avg/exp_avg_sq=0`；若上游 helper 修复后，删除本地 materialization 并让同一测试走上游能力。 |
| 9 | **R11 不要求改变现有求和数学。** R1/R2 重构后必须保持“backward 中 dense FSDP reduction 完成 -> Host sparse grad 沿 `dp_replicate/efsdp` 求和且排除 EP -> global norm/clip -> SparseAdam step”的时序，并保持 no `/ replica_size`。 | 保留 `test_replica_group_excludes_the_expert_parallel_axis`、真实 4-rank `engram_replica_grad_worker`（same-shard sum / EP isolation / empty replica）以及 4P Torch resume case；若未来上游重新启用 FSDP gradient division 或改变 loss normalization，必须同步重新审视该 scaling contract。 |

---

### Review boundary

本次只做静态代码与测试设计审查，遵守仓内 `developer-tests-review` 规则，没有执行 pytest、integration、pre-commit 或 NPU 训练。因此本文不会把 PR 描述中的“已通过”执行结果当成本次 review 自己验证过的事实；这些结果仅作为作者提供的背景。