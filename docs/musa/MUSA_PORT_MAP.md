# MUSA / S4000 端口映射（qfn NVFP4 正确性）

> 目标不是“把 CUDA 全量搬过去”，而是找出**最小正确性路径**。

## 1. 分层映射

| 层 | 当前（CUDA/vLLM） | MUSA 目标 | 优先级 |
|---|---|---|---|
| vLLM 平台 | `vllm` CUDA platform | `vllm-musa` 的 `MUSAPlatform` / `MTGPUWorker` | P0 |
| 模型 | 官方 `Qwen/Qwen3.8-Flash-Next-FP8` | qwen4_exp 支持；镜像/补丁只作参考 | P0 |
| attention | GDN/QSA/mamba | vllm-musa 的 MUSA attention/Mamba patch | P0 |
| KV cache | CUDA paged KV | vLLM-MUSA KV / Mamba pool | P0 |
| 官方 FP8 专家 | FP8 | MUSA FP8 / qwen 官方路径 | P0 |
| NVFP4 专家（FP8 后立即转） | W4A16 g16 | MUSA dequant / Model-Optimizer-musa | P1 |
| FP8_PB_WO | blockwise 128 weight-only | MUSA FP8 linear / 自定义 | P0 |
| CPU 专家 | `xiaotu_moe` AVX512/AVX2 | 同一 x86 `.so`（海光） | P0 |
| GPU prefill | 我们的 Triton tile ring | **先关闭**；正确性不走 GPU prefill | P1 |
| PLE | host/PLE offload | host 查表；必要时 NVMe | P1 |
| MTP | DSpark/MTP | **先关** | P2 |
| CUDA Graph | `FULL_DECODE_ONLY` | 先 eager，不开 graph | P2 |
| 通信 | NCCL | MCCL / MT-DeepEP | P2 |
| 监控 | nvidia-smi | mthreads-ml-py / `musaInfo` | P2 |

## 2. 最小路径

```text
vllm-musa v0.32（平台 + qwen4_exp + attention）
  + 官方 Qwen3.8-Flash-Next FP8 权重加载
  + CPU 全部专家（xiaotu_moe）
  + PLE host
  + MTP off
  + eager
  + 8K / batch 1
```

## 3. 关键判断

1. **能否用 vllm-musa 的 qfn 镜像做 oracle？**
   - 能：只采集输出/行为作参考，**不作为运行路径或交付依赖**；
   - 不能：只取 qwen4_exp/attention 补丁，接到我们树；
   - 若镜像只支持官方 BF16/FP8，不能当 NVFP4 的 oracle；BF16 本身不在路线内。

2. **NVFP4 是否有现成 MUSA kernel？**
   - 先查 `Model-Optimizer-musa`、vllm-musa 的 quant patch；
   - 没有就 CPU golden + 自写 dequant；不要先写高性能 kernel。

3. **我们的 CPU 引擎能否在海光上编？**
   - C++/AVX2 路径优先；
   - AVX512 现场确认；
   - 只要求 correctness，不要求快。

4. **GPU prefill 是否必须？**
   - 不需要；qfn correctness 走 CPU 全专家 + GPU attention；
   - 关闭 `gpu_prefill`，减少变量。

## 4. 明确不 port

- Marlin/CUTLASS CUDA kernel；
- CUDA Graph 捕获；
- NCCL 自定义 all-reduce；
- 3×A100 全驻留；
- DSpark/MTP 变长验证。
