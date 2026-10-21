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

## 2.1 torchada 的两个关键兼容点

已核对 torchada 源码（`/root/lvllm/data/refs/musa/torchada`）：

| 兼容点 | 行为 | 对 qwen4_exp 的影响 |
|---|---|---|
| `Tensor.is_cuda` | torchada 把 MUSA tensor 的 `is_cuda` 补成 True（`src/torchada/_patch.py:1774`） | qwen4 的 `if not q.is_cuda: return` 不会误跳过 QSA/PLE kernel |
| `tl.extra.cuda` | torchada 映射到 `tl.extra.musa` | GDC/PDL 符号可能可用，但 MUSA PDL 语义未验证 |

因此：

- `q.is_cuda` 检查在 MUSA 上**不是**主要阻塞；
- 我们的 `0002`–`0004` 强制 `launch_pdl=False` 是**保守正确性选择**，不是唯一方案；
- 若 S4000 实测 PDL 可用，可再放开；第一轮先关。

`current_platform.is_cuda()` 仍为 False，所以 QSA 的 top-k 会走
`ops.top_k_per_row_decode`；vllm-musa 已注册该 op，但要在设备上确认。

---

## 2.2 插件 PLE offload 的 MUSA 适配

`vllm_xiaotu_moe/ple_offload.py` 原本只找 `qwen4_exp.nvidia.*`，并用
`torch.cuda.is_available()` 判断设备。MUSA 上会：

- 找不到 AMD PLE 类（MUSA 走 `amd` 分派）；
- `torch.cuda.is_available()` 可能为 False，导致小 scale 不回设备。

已改为：

- MUSA 时优先尝试 `vllm.models.qwen4_exp.amd.ple_layer`；
- 量化方法类从 `common.ngram_embedding` 兜底；
- 用 `current_platform.is_available()` / `current_platform.device_type`
  替代硬编码 `torch.cuda`。

状态：**代码已改，未上机验证**。

---

## 2.3 CPU 引擎的 CUDA runtime 依赖（关键）

我们的 CPU expert 引擎原生 `cpu_decode` 在 C++ 里直接使用：

```text
cudaMemcpyAsync / cudaStreamSynchronize / cudaStreamIsCapturing
```

MUSA 没有 CUDA runtime，所以 **CPU 引擎不能直接走 `cpu_decode`**。

已在 `vllm_xiaotu_moe/mixed_experts.py` 增加 MUSA 正确性回退：

```text
hidden/ids/weights → CPU
engine.cpu_prefill(...)          # stream-free, 只做 host forward_many
out_cpu → MUSA device
```

特点：

- 不依赖 CUDA runtime / CUDA graph；
- 单 rank，EP 关闭；
- 比原生 `cpu_decode` 慢，但正确性优先；
- 只在 `current_platform.is_musa()` 时启用，CUDA 路径不变。

这是 MUSA 上能跑通 CPU 专家的关键改动之一。

---

## 2.4 预编译 .so 的 CUDA 依赖（MUSA 硬阻塞）

本仓库自带的 CPU 引擎 `.so` 当前链接：

```text
libcudart.so.13
libcuda.so.1
```

在普通 x86 开发机上：

```text
ldd xiaotu_moe/build/_xiaotu_moe_C_avx2...so
→ libcudart.so.13 => not found
→ libcuda.so.1 => not found
```

MUSA 没有 CUDA runtime / CUDA driver，因此：

> **直接 `import xiaotu_moe` 会在 dlopen 阶段失败，stream-free 回退也来不及生效。**

已实现 CPU-only 构建开关：

1. `binding.cpp` 增加 `XIAOTU_MOE_NO_CUDA` 条件编译；
2. 去掉 `CpuDecodeState` / `cpu_decode` / `copy_hostbuf_*` /
   `prealloc_decode_buffers`；
3. 只保留 `cpu_prefill`（host-only）；
4. `scripts/build_engine_variants.sh` 支持 `NO_CUDA=1`；
5. 插件 MUSA 回退只调用 `cpu_prefill`。

已验证（本机）：

```text
NO_CUDA=1 ONLY=scalar ... build_engine_variants.sh
→ _xiaotu_moe_C_scalar.cpython-313...so
→ ldd: no cudart / no libcuda
→ import ok
```

这是比“Python 层设备判断”更底层的阻塞；preflight 仍会检查 `.so` 依赖。

---

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
