### V4.1 独立 reference 训练基线

V4.1 的模型、图像路由、压缩 attention、metadata 和并行化均由 `torchtitan_npu/models/deepseek_v41` 持有，不依赖 V4 模型或其专属 override。

当前支持 **FSDP + EP、TP1 / CP1 / PP1、eager 执行**，保留 FullAC 与图文输入。算子选择按硬件矩阵：A3 默认启用四项融合（RMSNorm/RoPE/MoE grouped GEMM/mHC post），A5 默认额外启用 sparse attention；`USE_GOLDEN=1` 切回纯 reference。不支持量化、ngram、MTP、GraphTrainer；不支持的 TP、CP、PP 和 compile 配置在入口拒绝。

单机 8 卡、40 层、16 专家的入口：

```sh
bash examples/deepseek_v41/debug/deepseek_v41_flash_8p_cpt_4k_a3.sh
STEPS=5 bash examples/deepseek_v41/debug/deepseek_v41_flash_8p_cpt_4k_a3.sh
bash examples/deepseek_v41/debug/deepseek_v41_flash_8p_cpt_4k_a5.sh
```

A5 入口经 `ENABLE_A5_FUSION`（默认开，`=0` 可关闭）一并启用 sparse attention 与 mHC Sinkhorn 融合；CPU 亲和性可通过 `CPU_AFFINITY_CONF` 覆盖；`USE_GOLDEN=1` 始终优先于融合开关。A5（Ascend950PR）8 卡已完成 50 步对照验证：sparse 组 max |Δloss| 5.7e-3、全部启用组（+Sinkhorn）7.6e-3，均随机无漂移；步时中位 13.80s→3.58s（快 74.1%），显存省约 3 GiB（记录见 PR 822 描述与 FUSION50 验证报告 §2.2）。

操作符选择沿用 dsv4 模式：**launcher 默认启用 FUSION50 已验收的融合栈**（`rms_norm.ascendc` + `rope.ascendc` + `moe.ascendc` grouped GEMM + `mhc.asc_hc_post`；RoPE/MoE 在已含 RMSNorm 的组合上未引入可观测轨迹差异，RMSNorm/mHC Post 偏差 ≤4.8e-3 随机无漂移，已评审接受）。`USE_GOLDEN=1` 切回纯 reference 路径（与 FUSION50 B0 基线逐位等价，用于 A/B 对照）。CI 的 golden 2p 锚不受影响（case 显式设置 golden env）。

CI 使用 `dsv41_golden_2p_ep2_fsdp2`：2 卡 FSDP2/EP2、30 步，逐步精确比较固定 loss 锚。8 卡形状用于手动 A/B。两种并行形状各自比较对应参照，不混用锚。

入口默认关闭 checkpoint；需要保存/加载时显式配置。图像与 tokenizer 的内容保持现有测试资源；V4.1 tokenizer 环境变量为 `DSV41_TOKENIZER_PATH`，兼容旧 `DSV4_TOKENIZER_PATH` 的处理集中在配置入口。

### 真实 CC12M 数据入口（图像条件 caption 预测）

`deepseek_v41_flash_40layers_16experts_cc12m` 是唯一的真实图文数据配方：模型形状与并行配置沿用合成配方，数据入口更换为固定清单（warmup=2、总步数与合成配方不同，以 CC12M 入口为准）。序列固定为 `BOS + 完整 V4.1 图片协议 + caption + EOS`，caption 与 EOS 参与监督，BOS、图片协议和 padding 不参与。准备与运行共用同一个 `assemble_caption_sequence` 组装函数，长度过滤、监督和统计不会漂移；DP 分片按 `manifest[rank::world]`，构造时校验 world/rank 合法且每个 rank 分片非空。

数据准备（一次性；本地 WebDataset tar → 固定 8K 子集，seed=42，完整解码验证坏图、按图片 hash 去重、只提取选中图片并复核 hash，输出 manifest.jsonl + meta.json 含全部校验值；`--revision` 记录下载来源，本地自产 tar 不传）：

```sh
python examples/deepseek_v41/prepare_cc12m.py \
    --tars /data/p00465316/fused/datasets/cc12m/tars/cc12m-train-0000.tar \
           /data/p00465316/fused/datasets/cc12m/tars/cc12m-train-0001.tar \
    --revision 796118f2eabdb9984f23f7f15d1e74d388612fc6 \
    --tokenizer /data/p00465316/fused/dsv41_tokenizer \
    --output-dir /data/p00465316/fused/datasets/cc12m/subset_8k \
    --count 8000 --seq-len 512 --seed 42
```

数据集构造时会读取 meta.json 校验 manifest SHA256、tokenizer 文件 hash 与图片处理参数；manifest/tokenizer/预处理参数被改动会显式失败；同路径图片文件被替换不会触发该校验，正式 A/B 比较须对比 digest 输出的 pixel_sha 字段。A/B 预检（各 rank 前若干步的样本 ID 与 input/labels/图片预处理 hash；`--num-batches` 是优化器步数，梯度累积时传 `--global-batch-size`）：

```sh
python -m torchtitan_npu.models.deepseek_v41.cc12m_loader digest \
    --manifest /data/p00465316/fused/datasets/cc12m/subset_8k/manifest.jsonl \
    --data-dir /data/p00465316/fused/datasets/cc12m/subset_8k \
    --tokenizer /data/p00465316/fused/dsv41_tokenizer \
    --world 8 --num-batches 3
```

8 卡训练（单入口：基脚本 `deepseek_v41_flash_8p_cpt_4k_a3.sh`，默认即 CC12M 40 层；默认 40 步，`STEPS=<n>` 是唯一运行长度入口——它同步设置 training.steps、total-steps 和 warmup=2，直接传 `--training.steps` 等 CLI 参数会被拒绝）：

```sh
bash examples/deepseek_v41/debug/deepseek_v41_flash_8p_cpt_4k_a3.sh
STEPS=30 bash examples/deepseek_v41/debug/deepseek_v41_flash_8p_cpt_4k_a3.sh
```

合成 vision fixture（Golden 路径）随时可用 `CONFIG=deepseek_v41_flash_40layers_16experts_vision` 切回，其原有默认（STEPS=40、warmup 25、默认 attention chunk）不受 CC12M 接线影响。CC12M 入口自动固定 `TTNPU_DSA_ATTN_CHUNK=128`（40 层反向 workspace 在默认 chunk 256 下 OOM，已实测验证；128 下 3 步 rc=0、73.5% 显存）与 `TASK_QUEUE_ENABLE=1`。注意 CC12M 配方 warmup=2 是自有值（合成配方为 25），两条曲线不要混在同一对比里。数据路径用 `CC12M_MANIFEST_PATH` / `CC12M_DATA_DIR` / `CC12M_TOKENIZER_PATH` 覆盖。数据集内的循环语义是显式的：固定对齐配方必须满足 `步数 × 全局batch ≤ 清单样本数`（digest 子命令按 `--global-batch-size` 做消费量预检；8K 清单 / GBS8 下 40 步仅需 320 条）。

### 相同初始权重（生成与加载）

baseline 与融合组要从同一份显式权重出发时，先生成再加载（dcp 格式，model-only，fp32 ~116GB）：

```sh
# 生成：跑 3 步并在结束时保存 model-only 权重
STEPS=3 bash examples/deepseek_v41/debug/deepseek_v41_flash_8p_cpt_4k_a3.sh \
    --checkpoint.enable \
    --checkpoint.folder /data/p00465316/fused/weights/cc12m_3step

# 加载同一份权重起跑（--checkpoint.load-only 禁用后续保存）
bash examples/deepseek_v41/debug/deepseek_v41_flash_8p_cpt_4k_a3.sh \
    --checkpoint.enable \
    --checkpoint.initial-load-path /data/p00465316/fused/weights/cc12m_3step/step-3 \
    --checkpoint.initial-load-model-only \
    --checkpoint.load-only
```

两个要点：`--checkpoint.enable` 是加载生效的前提（不传时 CheckpointManager 未激活，会静默 fresh start，日志中只有一条 warning）；`initial-load-path` 指向 `step-N` 子目录。已验证：同一权重两次加载运行逐位一致，且与 fresh 初始化的轨迹明确不同（加载真实生效）。HF safetensors 导出（`--checkpoint.last-save-in-hf`）在共享宿主机上受锁页内存限制（8 rank 并发 d2h 触发 rtsMallocHost 失败，与机器上其他 NPU 任务的锁页占用相关），dcp 格式等价可用；确需 HF 格式时在锁页空闲期重试。

契约测试见 `tests/unit_tests/models/deepseek_v41/test_cc12m_loader.py`：可移植契约测试使用仓内 mini tokenizer（组装/监督/单次 shift/图片块/padding/分片/坏样本/预算不一致显式失败），真实 V4.1 tokenizer 兼容性是单列的集成测试（未部署时仅该项 skip）。

### 融合算子逐项说明（已默认使能，可按项显式对照）

四个融合 override 已随 launcher 默认启用（`USE_GOLDEN=1` 全部关闭做对照）。逐项语义：

| override | 替换范围 | FUSION50 结论 |
|---|---|---|
| `rms_norm.ascendc` | V4.1 RMSNorm（文本/compressor/视觉塔；融合调用前 FP32 提升，输出恢复原 dtype） | 偏差 ≤4.8e-3 随机无漂移，通过 |
| `rope.ascendc` | V4.1 旋转乘法 | 在含 RMSNorm 组合上未引入可观测轨迹差异 |
| `moe.ascendc` | routed experts grouped GEMM（保留路由/dispatch/FP32 route weight） | 在含 RMSNorm+RoPE 组合上未引入可观测轨迹差异 |
| `mhc.asc_hc_post` | mHC post 变换 | 偏差 ≤4.1e-3 随机无漂移，通过 |

不改变模型参数名称、FSDP/EP 或 checkpoint 格式。默认 30 步精确 Golden 看护不变。

融合输出、输入/权重梯度与训练轨迹的误差需要与同配置 reference 对照验收；仅运行成功不代表数值或性能验收完成。

### RoPE 适配补充说明

在原有 `--override.imports` 中增加 `torchtitan_npu.override.deepseek_v41.rope.ascendc`，只切换旋转计算；RoPE 缓存、位置生成与 CSA2 编排不变。可与 RMSNorm override 独立启用或组合。

文本 attention 使用 interleave，compressor/indexer 保留原 complex 参考顺序，视觉使用 half-rotation。融合输入与 cos/sin 保持 FP32，输出恢复原 dtype，不截取 batch 的第一行位置，也不原地修改 partial RoPE 输入。新增旋转模块无参数，不增加 checkpoint state。

CPU 测试覆盖旋转及逆旋转的输出/梯度、partial 非连续输入、batch 位置、融合参数传递与配置隔离；CPU mock 不代表 NPU kernel 验证通过。

数值验收须先确认 reference 30 步 loss/grad_norm 不变，再做融合算子前反向与两个训练组合的 A/B；不能通过改写 reference Golden 消除差异。



### Muon 优化器注意事项

`--optimizer.name Muon` 启用 DistMuon 时，`materialize()` 会整体替换 `param_groups` 为 `[DistMuon(pattern), AdamW(.*)]`，launcher 传入的 `--optimizer.param-groups.0.*` 参数静默失效。学习率等超参请使用顶层字段：`--optimizer.lr`、`--optimizer.weight-decay`、`--optimizer.muon-momentum` 等。
