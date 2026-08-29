## 单机最简命令：
1. 默认单机1P裁剪模型 + USE_GOLDEN
```sh
bash examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh
USE_GOLDEN=1 bash examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh
```
2.默认单机8P等效裁剪模型 + Profiling
```sh
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh \
  --profiler.enable-profiling \
  --training.steps 10
```
3. 首次导出 Hugging Face Checkpoint
```sh
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh \
  --checkpoint.folder ./export_ckpt \
  --training.steps 1 \
  --debug.seed 42 \
  --debug.deterministic \
  --checkpoint.enable \
  --checkpoint.no-load-only \
  --checkpoint.last-save-in-hf
```
4. 加载 Debug Hugging Face Checkpoint 训练
```sh
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh \
  --training.steps 1 \
  --debug.seed 42 \
  --debug.deterministic \
  --checkpoint.enable \
  --checkpoint.initial-load-path ./export_ckpt/checkpoint/step-1 \
  --checkpoint.initial-load-in-hf
```

## 多机模型命令：
1. A3 DeepSeek V4 flash full‑model 4k序列训练
```sh
NODE_IPS=192.168.1.10,192.168.1.11,192.168.1.12,192.168.1.13,192.168.1.14,192.168.1.15,192.168.1.16,192.168.1.17 \
bash examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a3.sh \
  --checkpoint.initial-load-path /path/to/model_ckpt \
  --training.steps 500
```
2. A5 DeepSeek V4 flash bf16 full‑model 4k序列 performance
```sh
NODE_IPS="192.168.1.10,192.168.1.11,192.168.1.12,192.168.1.13,192.168.1.14,192.168.1.15,192.168.1.16,192.168.1.17,\
192.168.1.18,192.168.1.19,192.168.1.20,192.168.1.21,192.168.1.22,192.168.1.23,192.168.1.24,192.168.1.25" \
COMPILE_BACKEND=aot_eager \
bash examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a5.sh \
  --debug.moe-force-load-balance \
  --training.steps 20
```
3. A5 DeepSeek V4 flash mxfp8 full‑model 4k序列 performance
```sh
NODE_IPS="192.168.1.10,192.168.1.11,192.168.1.12,192.168.1.13,192.168.1.14,192.168.1.15,192.168.1.16,192.168.1.17,\
192.168.1.18,192.168.1.19,192.168.1.20,192.168.1.21,192.168.1.22,192.168.1.23,192.168.1.24,192.168.1.25" \
COMPILE_BACKEND=aot_eager \
bash examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a5.sh \
  --debug.moe-force-load-balance \
  --extension.quantization.enable-quantized-training \
  --extension.quantization.recipe all_block_fp8 \
  --training.steps 20
```
4. A5 DeepSeek V4 flash bf16 full‑model 64k序列 performance
```sh
NODE_IPS="192.168.1.10,192.168.1.11,192.168.1.12,192.168.1.13,192.168.1.14,192.168.1.15,192.168.1.16,192.168.1.17,\
192.168.1.18,192.168.1.19,192.168.1.20,192.168.1.21,192.168.1.22,192.168.1.23,192.168.1.24,192.168.1.25" \
EP=128 CP=8 DP_SHARD=16 SEQ_LEN=65536 GBS=128 \
COMPILE_BACKEND=aot_eager \
bash examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a5.sh \
  --debug.moe-force-load-balance \
  --training.steps 20
```
5. A5 DeepSeek V4 flash mxfp8 full‑model 64k序列 performance
```sh
NODE_IPS="192.168.1.10,192.168.1.11,192.168.1.12,192.168.1.13,192.168.1.14,192.168.1.15,192.168.1.16,192.168.1.17,\
192.168.1.18,192.168.1.19,192.168.1.20,192.168.1.21,192.168.1.22,192.168.1.23,192.168.1.24,192.168.1.25" \
EP=128 CP=8 DP_SHARD=16 SEQ_LEN=65536 GBS=128 \
COMPILE_BACKEND=aot_eager \
bash examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a5.sh \
  --debug.moe-force-load-balance \
  --extension.quantization.enable-quantized-training \
  --extension.quantization.recipe all_block_fp8 \
  --training.steps 20
```
6. A5 DeepSeek V4 flash bf16 full‑model 1M序列 performance
```sh
NODE_IPS="192.168.1.10,192.168.1.11,192.168.1.12,192.168.1.13,192.168.1.14,192.168.1.15,192.168.1.16,192.168.1.17,\
192.168.1.18,192.168.1.19,192.168.1.20,192.168.1.21,192.168.1.22,192.168.1.23,192.168.1.24,192.168.1.25" \
COMPILE_BACKEND=aot_eager \
bash examples/deepseek_v4/deepseek_v4_flash_cpt_1024k_a5.sh \
  --debug.moe-force-load-balance \
  --training.steps 20
```
7. A5 DeepSeek V4 flash mxfp8 full‑model 1M序列 performance
```sh
NODE_IPS="192.168.1.10,192.168.1.11,192.168.1.12,192.168.1.13,192.168.1.14,192.168.1.15,192.168.1.16,192.168.1.17,\
192.168.1.18,192.168.1.19,192.168.1.20,192.168.1.21,192.168.1.22,192.168.1.23,192.168.1.24,192.168.1.25" \
COMPILE_BACKEND=aot_eager \
bash examples/deepseek_v4/deepseek_v4_flash_cpt_1024k_a5.sh \
  --debug.moe-force-load-balance \
  --extension.quantization.enable-quantized-training \
  --extension.quantization.recipe all_block_fp8 \
  --training.steps 20
```
8. A5 DeepSeek V4 Pro bf16 裁剪模型 4k序列 performance
```sh
NODE_IPS=192.168.1.10,192.168.1.11,192.168.1.12,192.168.1.13 \
COMPILE_BACKEND=aot_eager \
bash examples/deepseek_v4/debug/deepseek_v4_pro_32p_cpt_4k_a5.sh \
  --debug.moe-force-load-balance \
  --training.steps 20
```
9. A5 DeepSeek V4 Pro mxfp8 裁剪模型 4k序列 performance
```sh
NODE_IPS=192.168.1.10,192.168.1.11,192.168.1.12,192.168.1.13 \
COMPILE_BACKEND=aot_eager \
bash examples/deepseek_v4/debug/deepseek_v4_pro_32p_cpt_4k_a5.sh \
  --debug.moe-force-load-balance \
  --extension.quantization.enable-quantized-training \
  --extension.quantization.recipe all_block_fp8 \
  --training.steps 20
```
10. A5 DeepSeek V4 Pro bf16 裁剪模型 128k序列 performance
```sh
NODE_IPS=192.168.1.10,192.168.1.11,192.168.1.12,192.168.1.13 \
EP=32 CP=32 DP_SHARD=1 SEQ_LEN=131072 GBS=8 \
COMPILE_BACKEND=aot_eager \
bash examples/deepseek_v4/debug/deepseek_v4_pro_32p_cpt_4k_a5.sh \
  --debug.moe-force-load-balance \
  --training.steps 20
```
11. A5 DeepSeek V4 Pro mxfp8 裁剪模型 128k序列 performance
```sh
NODE_IPS=192.168.1.10,192.168.1.11,192.168.1.12,192.168.1.13 \
EP=32 CP=32 DP_SHARD=1 SEQ_LEN=131072 GBS=8 \
COMPILE_BACKEND=aot_eager \
bash examples/deepseek_v4/debug/deepseek_v4_pro_32p_cpt_4k_a5.sh \
  --debug.moe-force-load-balance \
  --extension.quantization.enable-quantized-training \
  --extension.quantization.recipe all_block_fp8 \
  --training.steps 20
```
