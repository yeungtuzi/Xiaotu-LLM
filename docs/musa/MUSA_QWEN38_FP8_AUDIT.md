# 官方 Qwen3.8-Flash-Next-FP8 权重审计（2026-10-21，离线）

> 来源：`Qwen/Qwen3.8-Flash-Next-FP8`，Hugging Face revision 以实际下载为准。  
> 方法：只读 `config.json` + `model.safetensors.index.json`；不加载权重、不需要 GPU。  
> 复现：`python scripts/musa/audit_qwen38_fp8.py <model_dir>`

## 1. 基本结构

| 项 | 值 |
|---|---|
| 架构 | `Qwen4ExpForConditionalGeneration` |
| 层数 | 48 |
| routed experts | 512，top-10 |
| MoE intermediate | 640 |
| hidden | 2560 |
| 上下文 | 262144 |
| MTP | 1 层 |
| checkpoint 总大小 | **185.502 GB** |
| 张量条目 | **152,089** |
| 量化 | FP8（`quant_method=fp8`） |

## 2. 张量分类

| 类别 | 张量数 | 说明 |
|---|---:|---|
| experts | 147,456 | 48 × 512 × 3 投影 × (weight + scale_inv) |
| MTP | 3,101 | `mtp.*` |
| attention | 732 | attn / linear_attn |
| PLE | 138 | `layers.1.ple.*`、ngram embedding shards |
| embed / lm_head | 5 | — |
| other | 657 | norm/router/shared/mHC 等 |

关键 PLE 名称：

```text
model.language_model.layers.1.ple.conv1d.weight
model.language_model.layers.1.ple.key_proj.weight
model.language_model.layers.1.ple.norm_conv.weight
model.language_model.layers.1.ple.norm_key.weight
model.language_model.layers.1.ple.norm_query.weight
model.language_model.layers.1.ple.ple_embedding.layer_multipliers
model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight
...
```

## 3. 与 NVFP4 的区别

| | 官方 FP8 | 社区 NVFP4 |
|---|---|---|
| 大小 | 185.50 GB | 128.95 GB |
| experts | FP8 + scale_inv | NVFP4 W4A16 g16 |
| PLE | 原表（~51.2GB） | 原表 / 可 host |
| 量化 kernel | FP8 GEMM | dequant/NVFP4 |
| 我们的 CPU 引擎 | **已有 qfn官方 FP8 路径** | 需补 NVFP4 g16 |
| MUSA 第一目标 | ✅ | 后续 |

## 4. 对 MUSA 移植的意义

- **CPU 侧最省事**：qfn官方 FP8 的 CPU 专家路径已经在 CUDA 线上验证（B1055），可复用；
- **GPU 侧是主工作**：MUSA 需要 qwen4_exp 的 attention/GDN/QSA/mamba + FP8 linear；
- **PLE 在 host**：官方 FP8 PLE 仍是 51.2GB 量级；256GB host 可放下；
- **无 BF16 阶段**：官方 FP8 直接接 NVFP4。

## 5. 第一轮验收

```text
官方 FP8 qfn
  + CPU 全部专家
  + MUSA attention/KV
  + PLE host
  + MTP off
  + 8K / batch 1
  + eager
```

与现有 CUDA qfn官方 FP8 的固定 prompt 输出对拍；不追性能。

## 6. 尚未确认

1. ~~vllm-musa `v0.28.0-dev` 是否包含 `qwen4_exp` 完整模型代码~~  
   ⇒ **已确认：vLLM 0.32 才包含 `vllm/models/qwen4_exp/`；0.28/0.29 没有。MUSA 目标改为 vllm-musa `v0.32.0-dev`。**
2. qfn 专用镜像的 qwen4_exp 补丁是否可以提取到我们树（只作参考，不作依赖）；
3. MUSA 上 GDN/QSA/mamba 的实际可用性；
4. FP8 linear/MoE 在 S4000 上的 kernel 覆盖。
