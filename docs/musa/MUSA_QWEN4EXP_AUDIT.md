# qwen4_exp / MUSA 兼容性审计（2026-10-21）

> 参考源码：vLLM v0.32.0 `vllm/models/qwen4_exp/`（本地只读副本：`/root/lvllm/tmp/refs/vllm-0.32-qwen4`）  
> 目标：先官方 FP8 qfn，后 NVFP4；不追性能。

## 1. 结论

**qwen4_exp 目前只声明支持 CUDA 和 ROCm，没有 MUSA 分支。**

`vllm/models/qwen4_exp/__init__.py`：

```python
if current_platform.is_xpu() or current_platform.is_tpu():
    raise NotImplementedError("Qwen4Exp currently supports CUDA and ROCm only")
if current_platform.is_rocm():
    from .amd.model import ...
else:
    from .nvidia.model import ...
```

MUSA 上：

- `is_xpu()` / `is_tpu()` = False；
- `is_rocm()` = False；
- 于是走 **`nvidia` 路径**，不是 AMD/Triton 路径；
- nvidia 路径含 CuTe DSL、CUDA 专用 ops，和 MUSA 不匹配。

⇒ **必须给 qwen4_exp 增加 MUSA 分支，优先选 AMD/Triton 路径。**

## 2. 两条现有路径

| 路径 | 依赖 | 对 MUSA 的可用性 |
|---|---|---|
| `nvidia/` | CuTe DSL、CUDA ops、`cute_dsl` | ❌ 不可直接用 |
| `amd/` | Triton、`torch.ops.vllm.*`、部分 `tl.extra.cuda` | ⚠️ 最接近，但仍有 CUDA 专用 PDL/GDC 调用 |

AMD 路径里仍看到：

```python
tl.extra.cuda.gdc_wait()
tl.extra.cuda.gdc_launch_dependents()
launch_pdl=current_platform.is_arch_support_pdl()
```

这些在 MUSA Triton 上未必存在，需要：

- `tl.extra.cuda.*` 用平台判断/空操作包起来；
- `launch_pdl` 在 MUSA 上关闭；
- `current_platform.is_rocm()` 分支改为“AMD or MUSA”。

## 3. 其他依赖

- `qwen_gdn_linear_attn`、`MambaHybridModelState`、`MambaSpec`：MUSA 的 mamba 支持需单独验；
- `Qwen4ExpPLELayer` / `ngram_embedding`：PLE host；正确性优先；
- `Qwen4ExpMTP`：第一轮关闭；
- `low_latency_gemm`：第一轮可关闭；
- `Qwen4ExpQSAAttention`：QSA/Triton 是主风险；
- `Qwen3_5`/`Qwen3_next`/`Qwen3_vl` 的复用层：确认 MUSA 上可用。

## 4. 候选补丁（未验证）

`patches/musa/0001-qwen4exp-musa-select-triton.patch`：

- 在 `qwen4_exp/__init__.py` 里把 MUSA 也导向 `amd` 路径；
- 仅作为起点，不保证 AMD/Triton 路径在 MUSA 上能跑；
- 后续还要处理 `tl.extra.cuda.gdc_*` / PDL；已补 `0002`（hc.py）、`0003`（PLE）、`0004`（QSA cache）三个 PDL candidate patch。

静态检查：`python scripts/musa/check_qwen4exp_musa.py <vllm_root>`；当前 v0.32 报告 `is_musa_branch=0`、`tl_extra_cuda=29`、`pdl=61`。

## 5. 第一轮验证顺序

```text
1. 用 MUSA 平台 + 小 dense 模型，确认 vllm-musa 基础栈
2. 只 import qwen4_exp，确认平台分支选到 amd
3. 单层 QSA/attention，和 CPU golden 对拍
4. PLE host + 单层 MoE
5. 8K 端到端，MTP off
```

## 6. 未决问题

- MUSA Triton 是否支持 `tl.extra.cuda.gdc_*`；
- MUSA 上 `torch.ops.vllm.*` 的注册情况；
- MUSA 的 mamba/GDN 支持；
- vllm-musa v0.32 是否已有 qwen4_exp 的 MUSA 私有补丁（当前公开分支未见）。
