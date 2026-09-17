### V4.1 独立 reference 训练基线

V4.1 的模型、图像路由、压缩 attention、metadata 和并行化均由 `torchtitan_npu/models/deepseek_v4_1` 持有，不依赖 V4 模型或其专属 override。

当前只支持 **FSDP + EP、TP1 / CP1 / PP1、eager/reference**，保留 FullAC 与图文输入。不支持量化、ngram、MTP、GraphTrainer 或融合 attention；不支持的 TP、CP、PP 和 compile 配置在入口拒绝。

单机 8 卡、40 层、16 专家的入口：

```sh
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh --training.steps 5
```

V4.1 固定运行 eager/reference 算子路径：AscendC 融合稀疏 attention 目前仍不接受 ratio-1 的 shared/global KV 契约。投影与专家的参考算术，以及 Attention Gym 的 eager `selected_attention` 稀疏内核（`attn-gym==0.0.9`，见 `CompressedSparseInnerAttention2`）都是模型默认实现，只保留通用 RoPE workaround 与 virtual optimizer override，不再加载 V4 sparse attention 或 V4.1 Golden MoE 类替换。

CI 使用 `dsv41_debugmodel_2p_ep2_fsdp2`：2 卡 FSDP2/EP2 的 reference 路径真实训练执行；它是 smoke run，末步 loss 不保证 run-to-run 复现（deterministic 与 seed 只在 `check_loss=True` 时启用），需要稳定数值时显式传 `--debug.deterministic --debug.seed=42`。V4.1 与上游 torchtitan 的该模型一致，不保留专门 loss 锚，也不做逐值比较；8 卡形状用 `run_train.sh` 手动回归。

indexer 蒸馏损失由 `IndexerKLLoss` 实现（上游默认 `coeff=0.01`）：每层只要消费了 selection 就挂一个损失，教师用该层自身 attention 的完整 softmax 分母（窗口 + 压缩条目 + sink）重建，按压缩切片的边际质量加权；梯度经 `_AuxLossInjection` 注入，只训练 indexer 自身参数，训练指标为 `indexer_kl_loss/mean`。

入口默认关闭 checkpoint；需要保存/加载时显式配置。图像与 tokenizer 的内容保持现有测试资源；V4.1 tokenizer 环境变量为 `DSV41_TOKENIZER_PATH`，兼容旧 `DSV4_TOKENIZER_PATH` 的处理集中在配置入口。
