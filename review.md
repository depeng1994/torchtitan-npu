# PR #19 / GitCode !864 Maintainer Review

## 1. Review 范围与结论

| 项目 | 结论 |
| --- | --- |
| PR | `deepseek v4.1 vision instruction tuning` |
| Review 基线 | `master@3f24b35ed83be17c012207400984bf1917cd05f8` |
| Review 代码 | `pr_864@29d1cad23ca5ba1a952feaa3e8a295736c7dcd6b` |
| 变更规模 | 4 个全新增文件，+727 / -0 |
| PR 目标 | 为 DeepSeek V4.1 增加图文/纯文本 SFT 数据链路，通过 dataloader override 切换，不修改模型主体与 CPT 路径 |
| Maintainer 总体结论 | **暂不建议按当前形态合入，Request changes。** SFT 产品语义本身成立，但当前实现同时引入第二套数据生命周期、第二个训练 shell 入口、SFT 专属 resize 分叉以及一套对官方 `encoding.py` 内部流程的复制。应先显著删减/收敛架构，再补测试。 |
| 测试维度结论 | **补充测试后合入**（仅测试维度）；当前 PR 没有任何 UT/ST 变更。 |
| 测试执行 | **未执行（仅静态审查）**。按仓内 `.agents/skills/developer-tests-review` 的 review 流程，不执行 testcase。 |
| patch / extension 边界 | 本 PR 未修改 `patches/` 或 `extensions/`，没有把 NPU/模型限定代码塞入 patch；这一点符合仓库边界。 |
| 原 PR 行间意见 | “不要加 meta 头”在当前 squash 版本中已满足：4 个新文件均使用仓内 BSD-style header，不再作为问题保留。 |

## 2. 主要 Review 意见

| ID | 级别 | 代码位置 | 问题点 | 影响 / 证据 | 建议修改方案 |
| --- | --- | --- | --- | --- | --- |
| R1 | **Blocker / 架构** | `examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_sft_4k_a3.sh:7-19` | **新增 SFT wrapper shell 没有独立产品语义，应删除。** 现有 A3 CPT launcher 已允许在末尾追加 override 与标准 CLI；本 PR 的真正切换点已经是 `vision_language_dataloader.sft`，不需要再创造一个训练入口。 | wrapper 只做 3 件事：追加 SFT override、重复 `DATASET_PATH`、把 checkpoint 改成 enable/save。这样把“数据模式=SFT”和“checkpoint 策略”错误耦合；而且只为 A3 建 wrapper，后续 A5/其它拓扑会自然复制更多脚本。当前 `examples/deepseek_v4_1/readme.md` 已明确“末尾 CLI 覆盖脚本默认值”。 | **直接删除该文件。** 文档给一条基于现有 `deepseek_v4_1_flash_8p_cpt_4k_a3.sh` 的命令：追加 `torchtitan_npu.override.deepseek_v4_1.vision_language_dataloader.sft` 与 `--dataloader.dataset-path ...`。checkpoint 是否启用继续由用户显式 CLI 决定，不由 SFT 入口隐式改变。 |
| R2 | **Blocker / 解耦** | `torchtitan_npu/models/deepseek_v4_1/vision_language_encoder.py:20-161`，尤其 `:47-61, 88-123` | **动态加载任意版本的官方 `encoding.py`，但又在本仓复制它的消息预处理、drop-thinking、tool merge/sort、逐 message render 流程来推导 assistant span。** 这是对模型资产内部实现的双写，而不是稳定扩展点。 | PR 最后用 `rendered_prompt != prompt` 作为运行时一致性断言；一旦官方 encoding 升级，本仓不会向前兼容，而是训练直接崩。该风险已经可见：截至 2026-09-20 的官方 V4.1 `encoding.py` 中 `_drop_thinking_messages` 的 keep roles 已包含 `direct_search_results`，PR 自己复制的集合没有该角色。官方代码见 [DeepSeek-V4.1 encoding.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/encoding/encoding.py)。 | **不要长期维护第二份 encoding pipeline。** 首选推动官方 encoding 暴露“rendered assistant spans / supervision mask”之类稳定接口，本仓仅做薄 adapter；短期若必须兼容当前资产，也应把依赖缩到最小、显式校验所需 API/资产 revision，并避免复制 `_drop_thinking_messages` 等内部语义。不能把“最终字符串相等，否则 RuntimeError”当成解耦方案。 |
| R3 | **High / 架构删减** | `vision_language_dataset.py:49-98, 101-138, 420-440` | **为 SFT 重新实现 JSON 流式解析、HF source 构造、DP sharding、epoch 循环、Stateful save/load，形成第二套数据生命周期。** 新需求真正独立的语义是“chat schema 归一化 + assistant supervision + 图文交错”，不是另一套 sharding/checkpoint 框架。 | 固定基线是 TorchTitan `v0.3.0`（`requirements.txt` + `.ci/setup_torchtitan.sh`）。该版本 `HuggingFaceMultiModalDataset` 已拥有 `load_dataset`、`split_dataset_by_node`、HF state 恢复和 packer state；仓内现有 V4.1 CPT loader 也已经专门修过跨 epoch restore。上游参考：[mm_datasets.py](https://github.com/pytorch/torchtitan/blob/v0.3.0/torchtitan/hf_datasets/multimodal/mm_datasets.py)。另外固定依赖允许 `datasets>=3.6.0`，其内置 JSON loader 已处理 JSON/JSONL：[datasets 3.6 json.py](https://github.com/huggingface/datasets/blob/3.6.0/src/datasets/packaged_modules/json/json.py)。 | 先证明哪些上游/仓内生命周期确实无法复用，再只保留缺失的最小层。至少应：1）删除能由 HF loader 覆盖的 `_json_rows`；若“大型顶层 JSON array 真正流式读取”是独立需求，只把它隔离成 source adapter；2）复用/下沉现有 V4.1 loader 的 shard/epoch/state 逻辑，不再让 SFT 独立维护一套；3）SFT 文件聚焦 schema normalization、assistant mask 和 image/text sample assembly。 |
| R4 | **High / 单一协议** | `vision_language_dataset.py:35-46`；对照 `torchtitan_npu/models/deepseek_v4_1/vision_data.py::ImagePatchProcessor.target_grid` | **同一个 V4.1 模型出现 CPT 和 SFT 两套 resize 规则。** 新子类唯一差异是在 minimum-pixel 放大后先 `int()`，但它只修 SFT，不修共享 `ImagePatchProcessor`。 | 官方 V4.1 当前 `inference/image_processor.py::plan_image_grid` 确实先对放大后的 width/height 做 `int`，见 [official image_processor.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/image_processor.py)。因此这里不是 SFT 特性，而是**共享图像协议**应统一。现有 `test_target_grid_fixed_oracle` 的样例恰好没有击中 rounding 差异；例如原始 `height=50,width=53` 时，两条实现会产生不同 patch grid。 | **删除 `DeepSeekV41SFTImagePatchProcessor`。** 如果官方语义应以当前 V4.1 为准，就直接修共享 `ImagePatchProcessor.target_grid`，让 CPT/SFT 共用；补一个能区分 float 与 int 顺序的固定 oracle，避免再分叉。 |
| R5 | **High / 测试** | PR 全部生产改动；`tests/unit_tests/**`、`tests/integration_tests/**` 无 diff | **+727 行新的数据/编码/状态语义没有任何仓内 UT 或注册 ST。** PR 描述中的 8-die 拟合、held-out loss、greedy generation 和 DCP 手工验证不能替代仓内可复现测试入口。 | 现有 `tests/unit_tests/models/deepseek_v4_1/test_dataloader.py` 只覆盖 CPT/CC12M 路径；`tests/integration_tests/run_tests.py::build_models_test_list` 没有 DeepSeek V4.1 case，默认 CI 的 `.ci/smoke_test.sh --test_suite models` 因而不会进入新增 SFT override。 | 架构收敛后补最小 UT + 1 个最小 NPU ST，详见第 5、6 节。不要为每种 JSON 方言/每个并行度复制 ST；格式与 mask 用 CPU UT，真实训练入口只需一个能触发新路径的最小 case。 |
| R6 | **High / 文档一致性** | `examples/deepseek_v4_1/readme.md`（PR 未修改） | **现有文档与 PR 新行为直接冲突，而不只是“少了一段说明”。** 当前 readme 仍写 V4.1 数据入口是 CC12M caption prediction、直接复用 `HuggingFaceMultiModalDataset`，没有 SFT schema/override/限制说明。PR checklist 也明确未勾选文档更新。 | 用户按当前文档无法知道 JSON/JSONL/Parquet schema、`image_root`、assistant-only supervision、图片来源限制、`num_workers=0`、超长样本直接报错等新契约。 | 删除 R1 wrapper 后，在同一个 `examples/deepseek_v4_1/readme.md` 增加“SFT”小节：只给现有 launcher + override + CLI；明确支持的数据 schema、图像 source 范围、loss mask、seq_len 超限策略、worker 限制和 checkpoint 仍由标准 CLI 控制。 |
| R7 | **Medium / 配置单一来源** | `vision_language_dataset.py:310-321`；`override/deepseek_v4_1/vision_language_dataloader.py:25-47` | CLI 已经提供 `thinking_mode/drop_thinking/add_default_bos_token/reasoning_effort`，但 dataset row 又可用同名字段逐样本覆盖；训练策略出现两个来源。另有 `conversation` paired、`query/response` 等格式分支并未在 PR 描述中声明。 | 一份数据中偶然出现这些 key 就会静默改变模板/监督语义；这违背“训练入口和配置尽量单一、显式”的仓库方向，也扩大了未测试接口面。 | 默认只保留 CLI/override 作为训练 policy 的唯一来源。若确实要兼容官方 case-level `thinking_mode/reasoning_effort`，只保留有真实数据集需求且官方明确支持的字段，文档写清优先级；其它逐行 policy 与未声明 schema 分支先删除，等有独立用户语义再加。 |
| R8 | **Medium / 用户契约** | `vision_language_dataset.py:331-369` | PR 宣称兼容 OpenAI-style messages，但 `image_url` 最终只接受本地 path、data URL、bytes/PIL；任何含 `://` 的普通 URL 都被拒绝。 | “兼容 OpenAI-style messages”容易被理解为标准 `image_url: {url: ...}` 可用；而官方 V4.1 image processor 当前支持 data URL、HTTP(S) 与本地路径。这里可以选择更保守的本地策略，但必须是显式契约。 | **不建议为了“兼容”在训练 worker 中随意加网络下载。** 更简单的方案是文档把支持范围写成“OpenAI message schema + local/data URI image source”，并对 HTTP(S) 给出明确错误；若远端 URL 是真实产品需求，再设计缓存/重试/安全边界。 |
| R9 | **Medium / 配置失效** | `vision_language_dataset.py:447-484`；`override/.../vision_language_dataloader.py:34-46` | SFT Config 继承 `ParallelAwareDataloader.Config`，但构造 `ParallelAwareDataloader` 时只传 `batch_size`；`pin_memory` 等继承字段并未透传。同时 override 无条件把 `num_workers=0`，用户若显式传过其它值会被静默覆盖，后面的 “`config.num_workers != 0`” 检查实际上永远看不到原值。 | 打印出来的 CLI 配置与实际 DataLoader 行为可能不同；仓库开发规范要求用户配置被静默跳过时至少 warning。 | 如果 SFT 目前只支持 `num_workers=0`，在 override 处对用户非零配置**显式报错/告警**，不要静默改写；支持的 `ParallelAwareDataloader` 参数应透传，不支持的字段不要暴露成看似有效的 CLI。 |
| R10 | **Medium / 无必要抽象** | `vision_language_dataset.py:443-456`；`vision_language_encoder.py:164-177` | `_sft_messages` 只转调 static method；`sample_processor` 是 tyro suppress 的隐藏 callable 配置，但 PR 没有第二实现；`DeepSeekV41VisionLanguageEncoderConfig` 主要用于把同一组 override 参数再包装一层后 build。 | 这些都增加了 config/extension surface，却没有独立用户语义。按本仓“新增 abstraction 默认不需要存在”的原则，应先删后证。 | `_message_parts` 可直接做模块级 normalization function；删除当前无复用点的 `_sft_messages/sample_processor` 插槽。encoder 参数只保留一个配置来源；若嵌套 Config 不能证明有独立 build/reuse 价值，直接并入 SFT dataloader 配置或用最小不可变参数对象。 |

## 3. 对每个新增文件/抽象的“删减者视角”结论

| 新增项 | 是否有独立用户语义 | Review 结论 | 目标形态 |
| --- | --- | --- | --- |
| `deepseek_v4_1_flash_8p_sft_4k_a3.sh` | 否；只是已有 A3 launcher + override/CLI | **删除** | 保持单一 A3/A5 launcher，SFT 仅通过 override + CLI 选择 |
| `DeepSeekV41SFTImagePatchProcessor` | 否；差异属于 V4.1 共用图像协议 | **删除** | 修共享 `ImagePatchProcessor` |
| `_json_rows` | 未证明；HF 固定依赖已有 JSON loader | **默认删除；如大 JSON array 流式需求成立则只保留 source adapter** | 不承担 sharding/state/epoch |
| `DeepSeekV41VisionLanguageDataset` | “SFT sample 语义”成立，但“第二套数据生命周期”不成立 | **大幅收缩** | 只保留 schema → message → tokens/images/labels 的新语义，复用既有 shard/state 生命周期 |
| `_message_parts` | LLaVA/OpenAI/Alpaca normalization 有用户语义 | **保留最小必要分支，改成普通函数** | 未声明的 paired/query-response 等分支先删 |
| `DeepSeekV41VisionLanguageEncoder` | assistant-only supervision 有用户语义 | **保留薄 adapter，不接受官方 pipeline fork** | 上游/官方 encoding 提供 spans/mask，仓内只做 token/image layout 连接 |
| `DeepSeekV41VisionLanguageEncoderConfig` | 目前主要是参数搬运 | **待删除/合并** | CLI/override 是唯一训练 policy 来源 |
| `sample_processor` suppressed callable Config | 否；没有第二实现、没有用户入口 | **删除** | 直接调用 normalization function |
| `vision_language_dataloader.sft` override | **有**；这是用户切换 CPT/SFT 的单一入口 | **保留** | 继续显式 `@override`，但减少派生字段并禁止静默覆盖用户配置 |

## 4. 已认可的设计点

| 位置 | 结论 |
| --- | --- |
| `torchtitan_npu/override/deepseek_v4_1/vision_language_dataloader.py` | 用显式 `@override(target=DeepSeekV41DataLoader.Config, fqns=["dataloader"], exact=True)` 切换数据组件，方向正确；SFT 是模型限定能力，放在 model + override，而不是 patch，符合仓库边界。 |
| `vision_language_dataset.py` 对 `build_image_token_layout` 的调用 | 复用了已有 V4.1 image token layout，而不是复制 image span/token type 规则，这部分应继续保留复用。 |
| PR 不修改模型主体 | SFT 不应为了数据格式给模型 forward 加 `if sft` 分支；当前没有这么做，符合职责边界。 |
| PR 描述中的手工训练 | 能证明“这版代码在作者环境可拟合数据”，可作为开发验证材料；但不能替代第 5、6 节要求的仓内回归测试。 |

## 5. UT 静态审查（developer-tests-review）

### 5.1 语义变换与独立 oracle

| 正向功能 | 生产路径 / 应观察结果 | 当前直接检查 | 独立 oracle | 状态 |
| --- | --- | --- | --- | --- |
| 多种 SFT row 归一化为 V4.1 messages | row schema → normalization → role/content/image 顺序稳定 | 无 | 手写最小输入/期望 messages；不调用待测 normalizer 生成 expected | **未覆盖** |
| 官方 chat prompt → assistant-only token mask | official encoding → token ids/mask；prompt/user/image token 不监督，assistant 内容与 EOS 监督 | 无 | 对一个最小 2-turn case 手工给出监督 token 区间，并与官方 prompt 固定结果对照 | **未覆盖** |
| 图文交错位置与 image feature index | message image 顺序 → image grid/layout → `image_feature_indices` 与模型 consumer 一致 | 现有 CPT UT 只覆盖 CPT producer，不进入新 SFT producer | 共享 `build_image_token_layout` 固定小图 oracle + 实际 SFT producer 输出 | **未覆盖** |
| text-only SFT | 无真实图片时仍满足分布式 vision graph 输入契约，labels 只监督 assistant | 无 | 固定 token/label/valid mask/虚拟 image shape | **未覆盖** |
| overlength / padding / label shift | `seq_len` 内 pad；target `t` 写到 `label[t-1]`；超长明确失败 | 无 | 手写短序列的 input/labels；边界长度 = seq_len 与 seq_len+1 | **未覆盖** |
| DP shard 与 resume | rank 间样本不重叠；保存后恢复从相同下一条样本继续 | 现有 CPT `test_upstream_webdataset_dp_shards` / `test_packed_loader_resume_pending_samples` 不进入新 Dataset | 小 JSONL fixture，比较 rank 集合和恢复前后后续 N 条样本 | **未覆盖** |
| SFT override 真正替换 dataloader config | 真实 `override.imports`/override factory 后选中 SFT Config，参数落到 consumer | 无 | 检查最终 Config 类型与显式参数，不只直接调用 helper | **未覆盖** |
| V4.1 resize 官方顺序 | shared processor 与官方 `plan_image_grid` 同语义 | 现有 `test_target_grid_fixed_oracle` 没有覆盖 int-rounding 差异 | 增加至少一个 rounding-sensitive 固定尺寸 oracle | **部分覆盖** |

### 5.2 最小 UT 合入前置条件

| 建议测试位置 | 最小 testcase | 必须断言 | 为什么足够 |
| --- | --- | --- | --- |
| `tests/unit_tests/models/deepseek_v4_1/test_dataloader.py` 或同目录新的 `test_vision_language_data.py` | `test_sft_interleaved_sample_has_assistant_only_targets` | 使用真实 SFT producer；固定一条 user-text → image → text → assistant；断言 tokens/types/image indices/valid_tokens/labels，尤其 prompt/image 为 -100、assistant+EOS 参与监督 | 一条 case 同时保护最核心 producer→model 输入契约，不需要为每个 schema 复制同类断言 |
| 同文件 | `test_sft_schema_normalization` 参数化 LLaVA/OpenAI/Alpaca（只覆盖最终决定保留的 schema） | role/content/image 顺序的手写 expected | schema 是纯 CPU 语义，应放 UT 而不是 NPU ST |
| 同文件 | `test_sft_loader_dp_shard_and_resume` | 2-rank CPU source 集合不重叠；state restore 后连续 N 条样本逐字段一致 | 直接保护 PR 自己新增的 Stateful/sharding 契约；如果按 R3 删除自研生命周期，则应尽量复用现有 lifecycle 测试而不是新增重复测试 |
| `tests/unit_tests/models/deepseek_v4_1/test_dataloader.py` | 扩展 `test_target_grid_fixed_oracle` | 加一个 int-rounding 敏感尺寸，expected 来自官方公式固定值 | 防止 R4 再次分叉 |
| `tests/unit_tests/override/deepseek_v4_1/test_vision_language_dataloader.py` | `test_sft_override_selects_vision_language_dataloader` | 从真实 override 配置入口检查最终 dataloader Config、dataset path、thinking policy、worker 限制 | 证明 CLI/override 真正选中新实现，而不是只证明 factory 可直接调用 |
| `tests/unit_tests/models/deepseek_v4_1/test_vision_language_encoder.py` | `test_encoder_assistant_mask_matches_official_prompt` | 使用固定 V4.1 encoding asset/fixture；断言官方 prompt 不被本仓重写、assistant token span 和 EOS mask 有独立 expected | 直接保护最脆弱的 R2 seam |

## 6. NPU ST 静态审查

### 6.1 ST 触发判断

| 生产代码改动 | 受影响训练场景 | 为什么需要 ST | 现有代表测试 | 判断 | 合入前置条件 |
| --- | --- | --- | --- | --- | --- |
| 新增 SFT dataloader override + 图文/纯文本 batch | DeepSeek V4.1 从标准 Trainer 入口选择 `vision_language_dataloader.sft`，进入 vision forward、loss、backward、optimizer | CPU UT 不能证明真实 NPU 模型/override/vision tower/训练循环连接成功 | `tests/integration_tests` 当前没有 DeepSeek V4.1 case；README 的 A3/A5 手工入口不计入 ST | **未覆盖** | 在现有 integration runner 增加 1 个最小 V4.1 SFT case，不新建 shell/runner |
| 新增 DataLoader Stateful 恢复 | 完整 checkpoint 恢复时 loader 要从正确 sample 继续 | 若该能力作为产品承诺，需要确认 DCP 对新 state shape/序列化可消费 | 现有 V4 checkpoint case 不进入新 SFT loader | **部分覆盖 / 可由 UT + 通用 DCP 能力承担** | 先用 CPU state round-trip 保护新 Dataset 语义；除非 PR 明确承诺 SFT checkpoint-resume 数值等价，否则本 PR 不必复制一个 8P checkpoint suite |

### 6.2 建议的最小 NPU case

| 测试 | 模型/配置 | 并行数值 | 替换实现/override | 编译模式 | NPU 数 | 启用与完成检查 | golden | 执行阶段 |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| **拟新增** `dsv4_1_sft_debug_1rank` | `deepseek_v4_1_debugmodel_multimodal` | DP1 / EP1 / TP1 / CP1 / PP1 | 显式启用 `vision_language_dataloader.sft`；其余尽量 reference，减少融合噪声 | eager | 1 | 使用仓内小 JSONL + 小图片，至少 2 step 依次触发图文与纯文本 sample；检查实际选中 SFT Config、训练正常完成、loss finite | 不要求 | 默认 `models` suite |

### 6.3 模型投影

| 并行方式 | reference/eager + SFT | 融合/eager + SFT | compile + SFT |
| --- | --- | --- | --- |
| 1 rank | **缺口：拟新增 `dsv4_1_sft_debug_1rank`** | 本 PR 未改变融合算子；无需为了 SFT 再复制一条 | 当前 V4.1 文档声明不支持 compile，本 PR 不新增 |
| EP/FSDP 多卡 | PR 手工验证过 8P，但不在 integration runner；数据链路没有独立的多卡分支语义时不额外复制 ST | 同左 | 不支持/不适用 |

## 7. 测试格式审查

| 位置 | 类型 | 代码阅读结果 | 修改建议 |
| --- | --- | --- | --- |
| PR diff | 文件位置 | PR 没有新增或修改任何测试文件，因此没有可审查的 testcase 格式对象 | 新 UT 按生产路径镜像放入 `tests/unit_tests/models/deepseek_v4_1/` 与 `tests/unit_tests/override/deepseek_v4_1/`；NPU case 放 `tests/integration_tests/deepseek_v4_1.py` 并由 `run_tests.py` 正常注册 |
| `tests/integration_tests/run_tests.py` / `.ci/smoke_test.sh` | 测试入口 | 默认 `models` suite 目前静态不包含 DeepSeek V4.1 SFT | 只在现有 runner 的 model list 中注册最小 case；**不要**为本 PR 修改 `.ci/smoke_test.sh` 增加一次性命令 |

## 8. 文档/行为一致性检查

| 文档或 PR 声明 | 代码事实 | 结论 / 修改 |
| --- | --- | --- |
| “复用同目录 8P CPT 启动脚本，通过 dataloader override 切换到 SFT” | 实际又新增了 SFT wrapper shell，并顺便改变 checkpoint policy | **不一致。** 删除 wrapper，真正落实“同一 launcher + override/CLI”。 |
| “支持 JSON、JSONL、Parquet” | 有自研 JSON array parser + HF Parquet；没有文档说明大 JSON、错误处理、resume 边界 | 在 readme 明确格式；优先复用 HF loader，只有真实缺口才保留自研 source adapter |
| “兼容 OpenAI-style messages” | OpenAI `image_url` 中 HTTP(S) URL 会被拒绝 | 文档缩窄为 local/data URI，或另立真实远端 URL 需求；不要含糊称完全兼容 |
| “仅监督 assistant 文本和 EOS” | 代码按 assistant render spans 做 mask，但没有独立 UT 证明 tool/thinking/image/role token 边界 | 文档保留该承诺前，必须有 assistant mask 固定 oracle |
| 当前 V4.1 readme：“数据入口是 CC12M caption prediction，复用 upstream MM dataset” | PR 新增完全不同的 SFT data stack | **文档已过期。** 合入前必须更新同一 readme |
| 当前 V4.1 readme：“不保留专用手动冒烟 suite，真实训练走 A3/A5” | 新 SFT 是新的生产数据入口，但 integration runner 无 case | 对 SFT 至少注册 1 个最小 1-rank case；不需要恢复历史 8P 手工 suite |

## 9. ST 不能证明的内容

| 项目 | 本 PR 描述现有证据 | 仍需边界 |
| --- | --- | --- |
| 多格式 schema 正确性 | 作者手工数据拟合成功 | 应由 UT 的独立 expected 证明，不用 NPU ST 穷举 |
| assistant-only loss mask 精确性 | loss/accuracy 最终很好 | 拟合成功不能证明 prompt/image token 没被误监督；必须直接检查 labels/mask |
| tokenizer asset 升级兼容性 | 当前作者资产运行成功 | R2 的内部 pipeline copy 需要架构消除；单次 ST 不能证明未来 encoding 版本兼容 |
| checkpoint 精确 resume | 描述称 step-64 DCP 可加载 | “能加载”不等于 sample cursor 精确恢复；CPU state round-trip 至少要逐样本比较；若承诺完整 resume 数值等价再增加两阶段 ST |
| 8P 性能/吞吐 | 训练能跑 | `num_workers=0` 的数据吞吐没有 benchmark；本 review 不把“能跑”当性能验收 |

## 10. 建议修改顺序

| 顺序 | 动作 | 预期删减/收敛结果 |
| ---: | --- | --- |
| 1 | 删除 SFT wrapper shell，用现有 A3/A5 launcher + override/CLI | 去掉一个永久维护入口，消除 checkpoint 隐式策略 |
| 2 | 把 official V4.1 resize 的 int-rounding 修到共享 `ImagePatchProcessor`，删除 SFT processor 子类 | V4.1 图像协议恢复单一实现 |
| 3 | 收缩数据层：先复用 HF/TT shard/state 生命周期，再只保留 SFT normalization + supervision | 删除自研 JSON/shard/epoch/checkpoint 重复代码 |
| 4 | 重做 encoder seam：不复制官方完整 preprocessing/render pipeline；推进/使用稳定 assistant-span API | 消除最易随 tokenizer asset 升级 break 的双写 |
| 5 | 删除 hidden `sample_processor` 等无第二实现的扩展点，收敛 config 来源 | 减少 Config/helper/wrapper |
| 6 | 更新 `examples/deepseek_v4_1/readme.md` | 文档与单一入口一致 |
| 7 | 按第 5 节补最小 UT，按第 6 节补 1 个 1-rank NPU ST | 用最小测试保护真实新增语义，而不是为当前复杂实现“补覆盖率” |

---

**最终 Maintainer 意见：** SFT 功能方向可以接受，override 作为切换入口也正确；但当前实现的新增复杂度明显高于真实产品语义所需。合入前应先完成 R1-R5 的架构收敛，尤其删除额外 shell、统一 image resize、复用既有 data lifecycle、解除对官方 encoding 内部实现的复制，再基于收敛后的代码补最小 UT/ST 与文档。当前版本不建议直接合入。
