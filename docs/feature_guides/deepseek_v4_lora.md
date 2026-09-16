# DeepSeek-V4 LoRA 微调

`torchtitan_npu/models/deepseek_v4/lora_config.py` 在现有模型配置上启用 dense、batched 和 routed-expert LoRA，并保存 adapter 与训练状态。rank、alpha 和目标模块在该配置中设置；暂不支持量化基座。

`model_registry(..., converters=[DeepSeekV4LoRAConverter.Config(...)])` 自动在模型并行化后、optimizer 创建前冻结非 LoRA 参数，并关闭 MoE load-balancing hook，保留基座 routing bias。无需使用示例专用 wrapper。部分 adapter 训练时，在该 parallelize callback 返回后冻结相应 adapter（例如全部 `lora_a`），仅训练剩余参数。

```sh
MODULE=torchtitan_npu.models.deepseek_v4.lora_config CONFIG=deepseek_v4_lora NGPU=8 \
  bash scripts/run_train.sh --hf-assets-path /path/to/tokenizer \
  --checkpoint.initial-load-path /path/to/base-weights --checkpoint.initial-load-in-hf \
  --checkpoint.interval 10 --dump-folder /path/to/run
```

`checkpoint.periodic_save_adapter_only=True` 时，周期 checkpoint 使用 native DCP，包含 adapter、routing buffer、optimizer、scheduler、dataloader 和 step，不重复保存冻结的基座。恢复时保持相同基座、训练总步数和运行目录；用 `--checkpoint.load-step 10` 指定恢复点。

最后一步导出 PEFT 的 `adapter_model.safetensors` 和 `adapter_config.json`，仅包含 adapter 权重。训练续跑使用周期 DCP；若最后一步也需保存训练状态，设置 `--checkpoint.no-last-save-in-peft --checkpoint.no-last-save-model-only`。启用 expert adapter 时，PEFT 导出要求 dense 和 expert 的 rank 相同。启用 checkpoint 后，MTP adapter、rank 不兼容和不支持的 PEFT 目标在 checkpoint 初始化时拒绝，而不是训练结束后才失败。`last_save_in_hf=True` 不合并 LoRA 增量，因此本 checkpoint manager 禁止该选项；请使用 PEFT 或 native DCP。

## 加载 PEFT adapter

导出格式对应 Transformers 5.17.0 的 DeepSeek-V4 模型和 PEFT 0.20.0。安装可选依赖：

```sh
pip install -e '.[peft]'
```

导出使用 `target_parameters`，保留 batched projection 的分组前向计算，并将 routed expert 的 gate/up adapter 转成融合参数格式。加载时显式指定与训练相同的基座：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("/path/to/hf-base")
model = PeftModel.from_pretrained(base, "/path/to/run/checkpoint/step-100")
```

基座路径须为 Transformers 可加载的模型目录；训练用 native DCP 路径不能直接用于此处。Transformers 5.17.0 的 DeepSeek-V4 不加载 MTP adapter；训练 MTP adapter 时使用 native checkpoint，或设置 `include_mtp=False` 后导出 PEFT。

## 验证

CPU 消费者测试在实际 Transformers 模型上加载 dense、batched 和 routed-expert adapter，并与直接合并权重的模型比较 logits：

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest \
  tests/unit_tests/models/deepseek_v4/test_lora.py -k peft_export_loads
```

该测试需要上述可选依赖。NPU 用例复用现有 integration runner，以 2 步 A/B 训练验证 native checkpoint 恢复和最终 PEFT 导出；从第 1 步恢复后精确比较第 2 步的 loss 与 grad_norm。冻结 A、仅训练 B 由 CPU 单元测试覆盖：

```sh
python -m tests.integration_tests.run_tests /tmp/lora-ab \
  --test_suite models --test_name dsv4_lora_1rank --ngpu 1 --no-parallel
```
