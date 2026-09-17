### V4.1 独立 reference 训练基线

V4.1 的模型、图像路由、压缩 attention、metadata 和并行化均由 `torchtitan_npu/models/deepseek_v41` 持有，不依赖 V4 模型或其专属 override。

当前只支持 **FSDP + EP、TP1 / CP1 / PP1、eager/reference**，保留 FullAC 与图文输入。不支持量化、ngram、MTP、GraphTrainer 或融合 attention；不支持的 TP、CP、PP 配置在入口拒绝；compile（aot_eager / inductor）已支持，默认关闭、按需通过 `--compile.*` 参数启用。

单机 8 卡、40 层、16 专家的入口：

```sh
bash examples/deepseek_v41/debug/deepseek_v41_flash_8p_cpt_4k.sh
bash examples/deepseek_v41/debug/deepseek_v41_flash_8p_cpt_4k.sh --training.steps 5
```

`USE_GOLDEN=1` 是唯一支持的模式，launcher 默认启用；显式设置为 0 会报错。reference sparse attention 和专家算术已经是模型默认实现，只保留通用 RoPE workaround 与 virtual optimizer override，不再加载 V4 sparse attention 或 V4.1 Golden MoE 类替换。

CI 使用 `dsv41_golden_2p_ep2_fsdp2`：2 卡 FSDP2/EP2、30 步，逐步精确比较固定 loss 锚。8 卡形状用于手动 A/B。两种并行形状各自比较对应参照，不混用锚。

入口默认关闭 checkpoint；需要保存/加载时显式配置。图像与 tokenizer 的内容保持现有测试资源；V4.1 tokenizer 环境变量为 `DSV41_TOKENIZER_PATH`，兼容旧 `DSV4_TOKENIZER_PATH` 的处理集中在配置入口。
