"""Long-prefill GPU MoE path for the vllm-xtu-moe hybrid plugin.

Design (mirrors ktransformers' ``KT_GPU_PREFILL_TOKEN_THRESHOLD`` and the fork's
``LVLLM_GPU_PREFILL_MIN_BATCH_SIZE``): when a layer's prefill token count is >=
``VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS``, compute that layer's routed MoE on the
GPU by streaming the layer's raw MXFP4 weights H2D per layer, then freeing them,
overlapping the next layer's H2D with the current layer's compute.

Why raw MXFP4 and not Marlin: ``marlin_utils_fp4`` only supports NVFP4
group-size-16, not MXFP4 block-size-32, so vLLM's Marlin MoE cannot be reused.
We dequantize fp4 (e2m1 nibbles) with per-(groupN=1, groupK=32) e8m0 block scales
in-kernel — the same MXFP4 math as the CPU engine (moe_v2_packed4.hpp).

Data layout (per layer, matching the checkpoint / CPU engine params):
  w13 [E, 2I, H//2] u8: low nibble = even k, high nibble = odd k; gate rows
      [0,I) then up rows [I,2I).  N=2I=4096, K=H=4096.
  s13 [E, 2I, H//32] u8 e8m0 (dequant 2^(byte-127)) per (n, k//32).
  w2  [E, H, I//2] u8 fp4 (N=H=4096, K=I=2048).
  s2  [E, H, I//32] u8 e8m0.
  out[t,:] = sum_s topk_weights[t,s] * f_{topk_ids[t,s]}(x[t,:]);
      f = W2^T * ( SiLU(x@W13_gate^T) .* (x@W13_up^T) ).

Per layer we stream the raw weights H2D (from cached pinned host copies => real
DMA), then transpose the fp4 bytes (and scales) once to K-major on the GPU via
``transpose().contiguous()`` (a cheap device copy at ~1 TB/s). The grouped GEMM
kernels then load ``b`` directly as ``[BK, BN]`` K-major, so no in-loop trans —
this removes the biggest in-kernel overhead. Only the raw ~3.4 GiB (bytes) is
in VRAM per layer.
"""

import os
import sys
import threading
import time

import torch

import triton
import triton.language as tl

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
# Same table as a Python tuple, for the in-kernel 16-entry LUT (see
# _dequant_e2m1_lut). Indices follow the e2m1 encoding: bit3 sign, bits2:1 exp,
# bit0 mantissa.
_E2M1_CONST = tuple(_E2M1.tolist())
_LUT_CACHE: dict = {}


def _e2m1_table(device) -> torch.Tensor:
    """16-entry fp32 e2m1 table on ``device`` (for the gather-based dequant)."""
    t = _LUT_CACHE.get(device)
    if t is None:
        t = _E2M1.to(device).contiguous()
        _LUT_CACHE[device] = t
    return t


def gpu_prefill_min_tokens() -> int:
    """Current threshold.

    ``XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE`` (if set) wins over the env var: with
    TP>1 the model runs in worker processes that do not see the client's
    ``os.environ`` changes, so a file is the only way to switch the threshold at
    runtime (used by the measurement harness).
    """
    # 注意:变量名必须同时接受带/不带 VLLM_ 前缀两种写法 —— `scripts/tune_serve.sh`
    # 导出的是 VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE,而这里原先只读
    # XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE ⇒ **阈值文件从未生效**,导致所有"运行时切换
    # 阈值"的 A/B(2026-09-11 预填充对比)实际上都走了同一条 GPU 路径。
    f = os.environ.get("XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE") or \
        os.environ.get("VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE")
    if f:
        try:
            with open(f) as fh:
                return int(fh.read().strip())
        except (OSError, ValueError):
            pass
    v = os.environ.get("VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS", "0")
    try:
        return int(v)
    except ValueError:
        return 0


@triton.jit
def _dequant_e2m1_nibble(nib):
    nib = nib.to(tl.int32)
    sign = (nib >> 3) & 1
    ex = (nib >> 1) & 3
    man = nib & 1
    val_n = tl.math.exp2(ex.to(tl.float32) - 1.0) * (1.0 + 0.5 * man.to(tl.float32))
    val_s = 0.5 * man.to(tl.float32)
    val = tl.where(ex == 0, val_s, val_n)
    return tl.where(sign == 1, -val, val)


@triton.jit
def _dequant_e2m1_gather(nib, lut_ptr):
    """e2m1 dequant via a 16-entry fp32 table in global memory (L1-resident).

    Replaces ~10 int/float ops per element with one 4-byte gather from a 64-byte
    table; only worth it if the dequant ALU work is the kernel's bottleneck.
    """
    return tl.load(lut_ptr + nib.to(tl.int32))


@triton.jit
def _e4m3_byte_to_f32(b):
    """**E4M3 尺度字节** → f32（NVFP4 用）。

    为什么不用 `tl.float8e4nv` 的 bitcast：MUSA 上的 Triton 对 float8 dtype 的
    支持面不确定（厂商清单 §18），而这段**纯整数位运算**与我们
    `_decode_flat_kernel` 里已验证过的解码完全同式 ⇒ 零新依赖。
    e4m3 布局：bit7 符号 / bits6-3 指数(偏置 7) / bits2-0 尾数；无 inf，
    subnormal（指数 0）按 1.953125e-03 线性解。
    """
    i = b.to(tl.int32)
    ex = (i >> 3) & 0xF
    m = i & 0x7
    bits = ((ex + 120) << 23) | (m << 20)
    v = tl.where(ex == 0, m.to(tl.float32) * 1.953125e-03,
                 bits.to(tl.float32, bitcast=True))
    return tl.where(((i >> 7) & 1) == 1, -v, v)


@triton.jit
def _dequant_e2m1_lut(nib):
    """Same e2m1 dequant as ``_dequant_e2m1_nibble`` via a 16-entry register LUT.

    The nibble is a 4-bit index into {0,±0.5,±1,±1.5,±2,±3,±4,±6}; comparing the
    index against every table entry and selecting is cheaper than the arithmetic
    form (exp2 + several int ops) on SM80.
    """
    nib = nib.to(tl.int32)
    t0 = tl.zeros_like(nib).to(tl.float32)
    for i in tl.static_range(16):
        t0 = tl.where(nib == i, _E2M1_CONST[i], t0)
    return t0


@triton.jit
def _gate_up_kernel_split(
    x_ptr, x_stride, tok_ptr,
    seg_ptr,
    w13t_ptr, w13t_row_stride, s13t_ptr, s13t_row_stride,
    inter_ptr, inter_ld,
    W13_E, S13_E, ts_ptr,
    H: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NS: tl.constexpr,
    LUT: tl.constexpr,
    GKS: tl.constexpr = 32,        # ⭐ NVFP4 用 16；默认 32 = MXFP4 现状
    SC_E4M3: tl.constexpr = 0,     # ⭐ 1 = 尺度是 E4M3 字节（NVFP4）；0 = e8m0（MXFP4）
    TS: tl.constexpr = 1.0,        # ⭐ 每张量的 F32 二级尺度（NVFP4 的 weight_scale_2）
    HAS_TS: tl.constexpr = 0,    # ⭐ 1 = 每专家二级尺度从 ts_ptr[pid_e] 取（NVFP4）
):
    """gate_up without the per-k-block join+trans+reshape.

    A packed byte holds k=2j (low nibble) and k=2j+1 (high nibble). Instead of
    interleaving them back into a [BK, BN] tile with a register shuffle, we split
    the *activation* tile into its even/odd k columns and do two dots. The scale
    (block 32 along k) is identical for both halves after a 16x repeat, so no
    extra loads are needed.
    """
    pid_e = tl.program_id(0)
    pid_n = tl.program_id(1)
    base = tl.load(seg_ptr + pid_e)
    end = tl.load(seg_ptr + pid_e + 1)
    if end <= base:
        return
    n = pid_n * BN + tl.arange(0, BN)
    # int64: the expert base offset is pid_e * stride(0), and for V4.1 the
    # K-major w13/w2 tensors exceed 2^31 bytes (4.53 GB / 2.26 GB), so
    # `pid_e * W13_E` OVERFLOWS int32. For w2 that wraps negative and faults
    # (illegal memory access); for w13 it wraps POSITIVE (383*11796480 =
    # 4,518,051,840 -> 223,084,544), so it silently reads the WRONG expert.
    # Triton types a Python int arg by its value, so the cast must be here.
    w_off = pid_e.to(tl.int64) * W13_E
    s_off = pid_e.to(tl.int64) * S13_E
    mt = 0
    while mt * BM < (end - base):
        g_rows = base + mt * BM + tl.arange(0, BM)
        rmask = g_rows < end
        toks = tl.load(tok_ptr + g_rows, mask=rmask, other=0)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for kk in tl.range(0, H, BK, num_stages=NS):
            a = tl.load(
                x_ptr + toks[:, None] * x_stride + (kk + tl.arange(0, BK))[None, :],
                mask=rmask[:, None], other=0.0)
            a_e, a_o = tl.split(tl.reshape(a, (BM, BK // 2, 2)))
            b_lo = tl.load(
                w13t_ptr + w_off + ((kk >> 1) + tl.arange(0, BK // 2))[:, None] * w13t_row_stride
                + n[None, :])
            if LUT:
                w_e = _dequant_e2m1_lut(b_lo & 0x0F)
                w_o = _dequant_e2m1_lut((b_lo >> 4) & 0x0F)
            else:
                w_e = _dequant_e2m1_nibble(b_lo & 0x0F)
                w_o = _dequant_e2m1_nibble((b_lo >> 4) & 0x0F)
            scale = tl.load(
                s13t_ptr + s_off + ((kk // GKS) + tl.arange(0, BK // GKS))[:, None] * s13t_row_stride
                + n[None, :])
            if HAS_TS:
                _tsv = tl.load(ts_ptr + pid_e)   # ⭐ 每专家的 fp32 二级尺度(weight_scale_2)
            else:
                _tsv = TS
            if SC_E4M3:
                sc = _e4m3_byte_to_f32(scale) * _tsv
            else:
                sc = tl.exp2(scale.to(tl.float32) - 127.0) * _tsv
            # k = 2j+e shares the block-32 scale with k = 2j, and j>>4 selects
            # the block, so the even and odd halves use the SAME [BK//2, BN]
            # scale tile (each block-32 row repeats 16 times).
            sc_h = tl.reshape(
                tl.broadcast_to(sc[:, None, :], (BK // GKS, GKS // 2, BN)), (BK // 2, BN))
            b_e = (w_e * sc_h).to(tl.bfloat16)
            b_o = (w_o * sc_h).to(tl.bfloat16)
            acc = tl.dot(a_e, b_e, acc)
            acc = tl.dot(a_o, b_o, acc)
        tl.store(
            inter_ptr + g_rows[:, None] * inter_ld + n[None, :],
            acc.to(tl.bfloat16), mask=rmask[:, None])
        mt += 1


@triton.jit
def _gate_up_kernel(
    x_ptr, x_stride, tok_ptr,
    seg_ptr,
    w13t_ptr, w13t_row_stride, s13t_ptr, s13t_row_stride,  # [E, H//2, 2I] / [E, H//32, 2I]
    inter_ptr, inter_ld,
    W13_E, S13_E, lut_ptr, ts_ptr,
    H: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NS: tl.constexpr,
    LUT: tl.constexpr,
    GKS: tl.constexpr = 32,        # ⭐ NVFP4 用 16；默认 32 = MXFP4 现状
    SC_E4M3: tl.constexpr = 0,     # ⭐ 1 = 尺度是 E4M3 字节（NVFP4）；0 = e8m0
    TS: tl.constexpr = 1.0,        # ⭐ 每张量 F32 二级尺度（weight_scale_2）
    HAS_TS: tl.constexpr = 0,    # ⭐ 1 = 每专家二级尺度从 ts_ptr[pid_e] 取（NVFP4）
):
    pid_e = tl.program_id(0)
    pid_n = tl.program_id(1)
    base = tl.load(seg_ptr + pid_e)
    end = tl.load(seg_ptr + pid_e + 1)
    if end <= base:
        return
    n = pid_n * BN + tl.arange(0, BN)
    # int64: the expert base offset is pid_e * stride(0), and for V4.1 the
    # K-major w13/w2 tensors exceed 2^31 bytes (4.53 GB / 2.26 GB), so
    # `pid_e * W13_E` OVERFLOWS int32. For w2 that wraps negative and faults
    # (illegal memory access); for w13 it wraps POSITIVE (383*11796480 =
    # 4,518,051,840 -> 223,084,544), so it silently reads the WRONG expert.
    # Triton types a Python int arg by its value, so the cast must be here.
    w_off = pid_e.to(tl.int64) * W13_E
    s_off = pid_e.to(tl.int64) * S13_E
    mt = 0
    while mt * BM < (end - base):
        g_rows = base + mt * BM + tl.arange(0, BM)
        rmask = g_rows < end
        toks = tl.load(tok_ptr + g_rows, mask=rmask, other=0)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for kk in tl.range(0, H, BK, num_stages=NS):
            a = tl.load(
                x_ptr + toks[:, None] * x_stride + (kk + tl.arange(0, BK))[None, :],
                mask=rmask[:, None], other=0.0)
            b_lo = tl.load(
                w13t_ptr + w_off + ((kk >> 1) + tl.arange(0, BK // 2))[:, None] * w13t_row_stride
                + n[None, :])                                        # [BK//2, BN]
            nib_even = b_lo & 0x0F
            nib_odd = (b_lo >> 4) & 0x0F
            if LUT:
                w_e = _dequant_e2m1_gather(nib_even, lut_ptr)
                w_o = _dequant_e2m1_gather(nib_odd, lut_ptr)
            else:
                w_e = _dequant_e2m1_nibble(nib_even)
                w_o = _dequant_e2m1_nibble(nib_odd)
            # ⭐ A78-9（2026-10-26）：`tl.trans` 对 **3 维**张量在本机 Triton 上直接
            #   报 `permute dims must have the same length as input shape`；而且即使能编，
            #   `trans` 是**整体反序** (2,BN,BK//2) ⇒ 语义也不对。
            #   正确：join 得 (BK//2, BN, 2)，要按 k=2j+e 交错 ⇒ permute (0,2,1) 再 reshape ✓
            b_val = tl.reshape(tl.permute(tl.join(w_e, w_o), (0, 2, 1)), (BK, BN))  # [BK, BN]
            scale = tl.load(
                s13t_ptr + s_off + ((kk // GKS) + tl.arange(0, BK // GKS))[:, None] * s13t_row_stride
                + n[None, :])                                        # [BK//32, BN]
            if HAS_TS:
                _tsv = tl.load(ts_ptr + pid_e)   # ⭐ 每专家的 fp32 二级尺度(weight_scale_2)
            else:
                _tsv = TS
            if SC_E4M3:
                sc = _e4m3_byte_to_f32(scale) * _tsv
            else:
                sc = tl.exp2(scale.to(tl.float32) - 127.0) * _tsv
            sc = tl.reshape(tl.broadcast_to(sc[:, None, :], (BK // GKS, GKS, BN)), (BK, BN))
            b = (b_val * sc).to(tl.bfloat16)                         # [BK, BN]
            acc = tl.dot(a, b, acc)
        tl.store(
            inter_ptr + g_rows[:, None] * inter_ld + n[None, :],
            acc.to(tl.bfloat16), mask=rmask[:, None])
        mt += 1


@triton.jit
def _down_kernel(
    inter_ptr, inter_ld, tok_ptr, wts_ptr,
    seg_ptr,
    w2t_ptr, w2t_row_stride, s2t_ptr, s2t_row_stride,  # [E, I//2, H] / [E, I//32, H]
    out_ptr, out_ld,
    W2_E, S2_E, lut_ptr, ts_ptr,
    H: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    NS: tl.constexpr,
    LUT: tl.constexpr,
    LIMIT: tl.constexpr = 0.0,
    GKS: tl.constexpr = 32,        # ⭐ NVFP4 用 16；默认 32 = MXFP4 现状
    SC_E4M3: tl.constexpr = 0,     # ⭐ 1 = 尺度是 E4M3 字节（NVFP4）
    TS: tl.constexpr = 1.0,        # ⭐ 每张量 F32 二级尺度（weight_scale_2）
    HAS_TS: tl.constexpr = 0,    # ⭐ 1 = 每专家二级尺度从 ts_ptr[pid_e] 取（NVFP4）
):
    pid_e = tl.program_id(0)
    pid_h = tl.program_id(1)
    base = tl.load(seg_ptr + pid_e)
    end = tl.load(seg_ptr + pid_e + 1)
    if end <= base:
        return
    h = pid_h * BH + tl.arange(0, BH)
    # int64: see _gate_up_kernel -- V4.1's w2 is 2.26 GB, so pid_e * W2_E
    # overflows int32 and faults at high expert ids.
    w_off = pid_e.to(tl.int64) * W2_E
    s_off = pid_e.to(tl.int64) * S2_E
    mt = 0
    while mt * BM < (end - base):
        g_rows = base + mt * BM + tl.arange(0, BM)
        rmask = g_rows < end
        toks = tl.load(tok_ptr + g_rows, mask=rmask, other=0)
        wts = tl.load(wts_ptr + g_rows, mask=rmask, other=0.0).to(tl.float32)
        acc = tl.zeros((BM, BH), dtype=tl.float32)
        for kk in tl.range(0, I, BK, num_stages=NS):
            gate = tl.load(
                inter_ptr + g_rows[:, None] * inter_ld + (kk + tl.arange(0, BK))[None, :],
                mask=rmask[:, None], other=0.0).to(tl.float32)
            up = tl.load(
                inter_ptr + g_rows[:, None] * inter_ld + (I + kk + tl.arange(0, BK))[None, :],
                mask=rmask[:, None], other=0.0).to(tl.float32)
            if LIMIT > 0.0:
                gate = tl.minimum(gate, LIMIT)
                up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
            sig = 1.0 / (1.0 + tl.math.exp(-gate))
            act = (gate * sig) * up
            b_lo = tl.load(
                w2t_ptr + w_off + ((kk >> 1) + tl.arange(0, BK // 2))[:, None] * w2t_row_stride
                + h[None, :])                                        # [BK//2, BH]
            nib_even = b_lo & 0x0F
            nib_odd = (b_lo >> 4) & 0x0F
            if LUT:
                w_e = _dequant_e2m1_gather(nib_even, lut_ptr)
                w_o = _dequant_e2m1_gather(nib_odd, lut_ptr)
            else:
                w_e = _dequant_e2m1_nibble(nib_even)
                w_o = _dequant_e2m1_nibble(nib_odd)
            # A78-9：同上，用 permute(0,2,1) 而不是 3 维的 tl.trans
            b_val = tl.reshape(tl.permute(tl.join(w_e, w_o), (0, 2, 1)), (BK, BH))  # [BK, BH]
            scale = tl.load(
                s2t_ptr + s_off + ((kk // GKS) + tl.arange(0, BK // GKS))[:, None] * s2t_row_stride
                + h[None, :])                                        # [BK//32, BH]
            if HAS_TS:
                _tsv = tl.load(ts_ptr + pid_e)   # ⭐ 每专家的 fp32 二级尺度(weight_scale_2)
            else:
                _tsv = TS
            if SC_E4M3:
                sc = _e4m3_byte_to_f32(scale) * _tsv
            else:
                sc = tl.exp2(scale.to(tl.float32) - 127.0) * _tsv
            sc = tl.reshape(tl.broadcast_to(sc[:, None, :], (BK // GKS, GKS, BH)), (BK, BH))
            b = (b_val * sc).to(tl.bfloat16)                         # [BK, BH]
            acc = tl.dot(act.to(tl.bfloat16), b, acc)
        # sem="relaxed": the accumulations into ``out`` are the only accesses to
        # that buffer in this kernel and the caller reads it after the launch,
        # so we do not need acq_rel ordering. Measured 198.2 -> 189.8 ms/layer
        # (T=16K); a full "no atomic" restructure (per-slot buffer + reduction)
        # would reach 177.4 ms but is a larger change.
        tl.atomic_add(
            out_ptr + toks[:, None] * out_ld + h[None, :],
            acc * wts[:, None], mask=rmask[:, None], sem="relaxed")
        mt += 1


@triton.jit
def _down_kernel_split(
    inter_ptr, inter_ld, tok_ptr, wts_ptr,
    seg_ptr,
    w2t_ptr, w2t_row_stride, s2t_ptr, s2t_row_stride,
    out_ptr, out_ld,
    W2_E, S2_E, ts_ptr,
    H: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    NS: tl.constexpr,
    LUT: tl.constexpr,
    LIMIT: tl.constexpr = 0.0,
    GKS: tl.constexpr = 32,        # ⭐ NVFP4 用 16；默认 32 = MXFP4 现状
    SC_E4M3: tl.constexpr = 0,     # ⭐ 1 = 尺度是 E4M3 字节（NVFP4）；0 = e8m0
    TS: tl.constexpr = 1.0,        # ⭐ 每张量 F32 二级尺度（weight_scale_2）
    HAS_TS: tl.constexpr = 0,    # ⭐ 1 = 每专家二级尺度从 ts_ptr[pid_e] 取（NVFP4）
):
    """down projection, even/odd-k split (see _gate_up_kernel_split)."""
    pid_e = tl.program_id(0)
    pid_h = tl.program_id(1)
    base = tl.load(seg_ptr + pid_e)
    end = tl.load(seg_ptr + pid_e + 1)
    if end <= base:
        return
    h = pid_h * BH + tl.arange(0, BH)
    # int64: see _gate_up_kernel -- V4.1's w2 is 2.26 GB, so pid_e * W2_E
    # overflows int32 and faults at high expert ids.
    w_off = pid_e.to(tl.int64) * W2_E
    s_off = pid_e.to(tl.int64) * S2_E
    mt = 0
    while mt * BM < (end - base):
        g_rows = base + mt * BM + tl.arange(0, BM)
        rmask = g_rows < end
        toks = tl.load(tok_ptr + g_rows, mask=rmask, other=0)
        wts = tl.load(wts_ptr + g_rows, mask=rmask, other=0.0).to(tl.float32)
        acc = tl.zeros((BM, BH), dtype=tl.float32)
        for kk in tl.range(0, I, BK, num_stages=NS):
            gate = tl.load(
                inter_ptr + g_rows[:, None] * inter_ld + (kk + tl.arange(0, BK))[None, :],
                mask=rmask[:, None], other=0.0).to(tl.float32)
            up = tl.load(
                inter_ptr + g_rows[:, None] * inter_ld + (I + kk + tl.arange(0, BK))[None, :],
                mask=rmask[:, None], other=0.0).to(tl.float32)
            if LIMIT > 0.0:
                gate = tl.minimum(gate, LIMIT)
                up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
            sig = 1.0 / (1.0 + tl.math.exp(-gate))
            act = (gate * sig) * up
            act_e, act_o = tl.split(tl.reshape(act, (BM, BK // 2, 2)))
            b_lo = tl.load(
                w2t_ptr + w_off + ((kk >> 1) + tl.arange(0, BK // 2))[:, None] * w2t_row_stride
                + h[None, :])
            if LUT:
                w_e = _dequant_e2m1_lut(b_lo & 0x0F)
                w_o = _dequant_e2m1_lut((b_lo >> 4) & 0x0F)
            else:
                w_e = _dequant_e2m1_nibble(b_lo & 0x0F)
                w_o = _dequant_e2m1_nibble((b_lo >> 4) & 0x0F)
            scale = tl.load(
                s2t_ptr + s_off + ((kk // GKS) + tl.arange(0, BK // GKS))[:, None] * s2t_row_stride
                + h[None, :])
            if HAS_TS:
                _tsv = tl.load(ts_ptr + pid_e)   # ⭐ 每专家的 fp32 二级尺度(weight_scale_2)
            else:
                _tsv = TS
            if SC_E4M3:
                sc = _e4m3_byte_to_f32(scale) * _tsv
            else:
                sc = tl.exp2(scale.to(tl.float32) - 127.0) * _tsv
            sc_h = tl.reshape(
                tl.broadcast_to(sc[:, None, :], (BK // GKS, GKS // 2, BH)), (BK // 2, BH))
            b_e = (w_e * sc_h).to(tl.bfloat16)
            b_o = (w_o * sc_h).to(tl.bfloat16)
            acc = tl.dot(act_e.to(tl.bfloat16), b_e, acc)
            acc = tl.dot(act_o.to(tl.bfloat16), b_o, acc)
        # sem="relaxed": the accumulations into ``out`` are the only accesses to
        # that buffer in this kernel and the caller reads it after the launch,
        # so we do not need acq_rel ordering. Measured 198.2 -> 189.8 ms/layer
        # (T=16K); a full "no atomic" restructure (per-slot buffer + reduction)
        # would reach 177.4 ms but is a larger change.
        tl.atomic_add(
            out_ptr + toks[:, None] * out_ld + h[None, :],
            acc * wts[:, None], mask=rmask[:, None], sem="relaxed")
        mt += 1


# 【§598】按形状复用 GPU MoE 的**中间缓冲**(`inter`)与输出(`out`)。
# 背景:每层新分配 inter=(T*K, 2I)(qlen=13816 时 382 MB)会让 `alloc/reserved`
# 逐层上涨 ~0.42 GiB,40 层就是 ~17 GiB —— 直接把 free 压到预填充阈值之下。
# `inter` 是 `torch.empty` 语义(只读被写过的段)⇒ 复用完全等价。
_BUF_MOE: dict = {}
_BOUND_BUF_LOGGED: dict = {}


_LAST_REPORT = [0.0]
_LAST_EMPTY_CACHE = [0.0]   # 【2026-10-27】`XIAOTU_EMPTY_CACHE_EVERY_S` 的限频游标

# ⭐【2026-10-27 · 软/硬上限的配套】按"最近使用步号"做**选择性**淘汰。
# 与 empty_cache 的区别：只丢**空闲 >= max_idle_steps 步**的条目，正在用的一个不动。
_STEP = [0]
_LAST_USE: dict = {}


def bump_step() -> None:
    """每个模型步进一次（由 GPU MoE 入口调用）。"""
    _STEP[0] += 1


def _touch(key) -> None:
    _LAST_USE[key] = _STEP[0]


def evict_idle(max_idle_steps: int = 32) -> int:
    """⚠️【2026-10-27 默认关闭】审计发现：`_SEG_BUF` 等缓冲的设备地址**被烘进 CUDA Graph**，
    而淘汰判据是"Python 侧最近使用"——图重放**不经过 Python** ⇒ 只被图消费的缓冲会被判空闲、
    被释放、随后被别的张量复用 ⇒ 图重放写已释放内存 ⇒ `cudaErrorIllegalAddress`（显存可以完全健康）。
    这类 bug 症状与本次故障完全吻合，因此在把"图引用"纳入判据之前，**淘汰默认不启用**
    （`XIAOTU_MEM_EVICT=1` 才开）。
    """
    if os.environ.get("XIAOTU_MEM_EVICT", "0") != "1":
        return 0
    """淘汰"本次用不到"的缓存条目，返回释放的字节数。

    ⚠️ 必须先 `synchronize()`：这些缓冲可能刚被**侧流**写入，Python 侧释放后
    caching allocator 会把内存再发出去 ⇒ 侧流仍写 = use-after-free（历史事故形态）。
    """
    freed = 0
    torch.cuda.synchronize()
    # ⛔ `_SEG_BUF` 永不可淘汰：它的设计目的就是"捕获期零分配" ⇒ 地址必然被烘进 CUDA Graph。
    for cache in (_BUF_CACHE, _BUF_MOE):
        for k in list(cache.keys()):
            last = _LAST_USE.get(k, None)
            if last is None or _STEP[0] - int(last) <= int(max_idle_steps):
                continue                      # 未知(=可能在用) 或 仍在窗口内 ⇒ 不动
            v = cache.pop(k, None)
            _LAST_USE.pop(k, None)
            for t in (v if isinstance(v, (list, tuple)) else [v]):
                if isinstance(t, torch.Tensor):
                    freed += int(t.numel()) * int(t.element_size())
    return freed



_CLEANER = {"thread": None, "lock": threading.Lock()}


def _cleaner_loop(period_s: float) -> None:
    """后台清理线程：周期把空闲段还给驱动 + 驱动显存守卫。

    与 forward 路径解耦的**唯一理由**：`empty_cache()`（`cudaFree`）会隐式同步设备，
    在计算线程上执行会阻塞后续 kernel 提交（最坏 = 一个 prefill chunk 的时间）。
    在这个线程里同步，GPU 仍在为计算线程干活，摊销损失 ≈ T_free / 间隔。
    """
    while True:
        time.sleep(max(5.0, float(period_s)))
        try:
            if capture_in_progress():
                continue                  # CUDA Graph 捕获期间绝不能动 allocator
            t0 = time.perf_counter()
            for _i in range(torch.cuda.device_count()):
                try:
                    # ⚠️ `empty_cache()` 不接受参数；传 device 会 TypeError 且被吞 ⇒ 静默失效
                    with torch.cuda.device(_i):
                        torch.cuda.empty_cache()
                except Exception as e:    # noqa: BLE001
                    print(f"[xtu-vram] empty_cache(cuda:{_i}) 失败: "
                          f"{type(e).__name__} {e}", flush=True)
            ms = (time.perf_counter() - t0) * 1000.0
            dev = torch.device("cuda", torch.cuda.current_device())
            free, total = torch.cuda.mem_get_info(dev)
            print(f"[xtu-vram] empty_cache 已执行（耗 {ms:.1f} ms）"
                  f"⇒ free={int(free) / 2**30:.3f} GiB", flush=True)
        except Exception:                 # noqa: BLE001
            pass


def _start_cleaner_thread() -> None:
    with _CLEANER["lock"]:
        if _CLEANER["thread"] is not None:
            return
        period = float(os.environ.get("XIAOTU_EMPTY_CACHE_EVERY_S", "300") or 300)
        if period <= 0:
            _CLEANER["thread"] = False    # 显式关闭
            return
        t = threading.Thread(target=_cleaner_loop, args=(period,),
                             name="xtu-mem-cleaner", daemon=True)
        t.start()
        _CLEANER["thread"] = t
        print(f"[xtu-vram] 显存清理线程已启动（周期 {period:.1f} s，独立于 forward 路径）",
              flush=True)


def maybe_report(period_s: float = 60.0, tag: str = "steady") -> None:
    """每 `period_s` 秒打一次持久缓冲总账（`XIAOTU_VRAM_REPORT=0` 可关）。

    判据：与启动那次 `startup-before-kv` 的 total_bytes **完全相等**即为"不涨"。
    """
    if os.environ.get("XIAOTU_VRAM_REPORT", "1") != "1":
        return
    now = time.monotonic()
    if now - _LAST_REPORT[0] < period_s:
        return
    _LAST_REPORT[0] = now
    bump_step()
    vram_report(tag)
    if tag == "steady":                    # 启动期不动手（权重加载时 reserved 天然高）
        # ⭐【2026-10-27 修正】守卫**必须由这条 60 s 遥测驱动**：实测把 tick 只挂在
        #   清理线程上时，线程首轮 sleep 未到 ⇒ 守卫完全不工作（free 已 50 MiB 仍 guard=0）。
        #   tick 里的硬限检查只是 `mem_get_info`（微秒级、不同步），可以安全放在这条路径上。
        try:
            from . import mem_guard
            mem_guard.tick(sys.modules[__name__], tag=tag)
        except Exception:                  # noqa: BLE001
            pass
        # ⭐【2026-10-27 · 用户要求】清理**不在 forward 路径上做**：`cudaFree` 会隐式同步设备，
        # 在计算线程上执行最坏可阻塞一个 32 s 的 prefill chunk（实测 prompt 吞吐 255 tok/s）。
        # ⇒ 改为独立后台线程：它自己等 sync，计算线程照常发起 kernel。
        _start_cleaner_thread()


def persistent_bytes() -> int:
    """**我们自己的**持久缓存总量（`evict_idle` 的可淘汰范围就是这个集合）。

    用途：回答"为什么 evict 释放 0 MiB 却有 11 GiB 碎片" —— 那 11 GiB 属于
    torch 段里**别处**（vLLM 的 workspace 等）的张量，不在本集合内。
    """
    total = 0
    for cache in (_BUF_CACHE, _SEG_BUF, _BUF_MOE, _TILE_RINGS, _LANDING_BUF, _SLOTS):
        stack = list(cache.values())
        while stack:
            v = stack.pop()
            if isinstance(v, torch.Tensor):
                total += int(v.numel()) * int(v.element_size())
            elif isinstance(v, dict):
                stack.extend(v.values())
            elif isinstance(v, (list, tuple)):
                stack.extend(v)
            else:
                b = getattr(v, "bufs", None)
                if b:
                    stack.extend(b)
    return total


def vram_report(tag: str = "") -> dict:
    """把所有**持久缓冲**按类别加总并打印 —— "不随时间涨"的机械判据。

    判据：启动打一次（`startup-before-kv`）、跑一段流量后再打，
    **两组 total_bytes 必须完全相等**；任何上涨都是回归。
    """
    def _tensors(cache):
        out = []
        stack = list(cache.values())
        while stack:
            v = stack.pop()
            if isinstance(v, torch.Tensor):
                out.append(v)
            elif isinstance(v, dict):
                stack.extend(v.values())
            elif isinstance(v, (list, tuple)):
                stack.extend(v)
            else:
                b = getattr(v, "bufs", None)
                if b:
                    stack.extend(b)
        return out

    groups = {
        "moe_reuse(_BUF_MOE)": _BUF_MOE,
        "seg(_SEG_BUF)": _SEG_BUF,
        "dma(_BUF_CACHE)": _BUF_CACHE,
        "tile_ring(_TILE_RINGS)": _TILE_RINGS,
        "landing(_LANDING_BUF)": _LANDING_BUF,
        "slots(_SLOTS)": _SLOTS,
    }
    total = 0
    per = {}
    parts = []
    for name, cache in groups.items():
        ts = _tensors(cache)
        nb = sum(int(t.numel()) * int(t.element_size()) for t in ts)
        total += nb
        per[name] = nb
        parts.append(f"{name}={nb / 2**30:.3f}GiB/{len(ts)}块")
    print(f"[xtu-vram] {tag or 'report'}: 持久缓冲合计 {total / 2**30:.3f} GiB | "
          + " ".join(parts), flush=True)
    # ⭐【2026-10-27 · 用户质疑"为什么涨"】上面只统计**插件自有**缓冲 ⇒ 它"平"**不等于**
    # 整卡"平"。必须把 allocator 三项分开报，才能机械区分两种完全不同的"涨"：
    #   · allocated 涨 ⇒ 真有对象没释放（泄漏），要找持有者
    #   · reserved 涨、allocated 平 ⇒ 缓存/碎片（`expandable_segments:False` 下 reserved
    #     只增不减，块尺寸种类越多涨得越多）⇒ 修法是固定分配模式或换 expandable，不是"留余量"
    cuda_stats = {}
    try:
        for _i in range(torch.cuda.device_count()):
            _d = torch.device(f"cuda:{_i}")
            _a = int(torch.cuda.memory_allocated(_d))
            _r = int(torch.cuda.memory_reserved(_d))
            _m = int(torch.cuda.max_memory_allocated(_d))
            if _a == 0 and _r == 0:
                # 该进程没有 CUDA 上下文（如 API server）⇒ 不打，免噪声
                continue
            _free, _tot = torch.cuda.mem_get_info(_d)
            cuda_stats[f"cuda:{_i}"] = {
                "allocated": _a, "reserved": _r, "max_alloc": _m,
                "frag": _r - _a, "free": int(_free), "total": int(_tot),
            }
            print(f"[xtu-vram] {tag or 'report'} cuda:{_i} "
                  f"allocated={_a / 2**30:.3f} reserved={_r / 2**30:.3f} "
                  f"max_alloc={_m / 2**30:.3f} frag={(_r - _a) / 2**30:.3f} "
                  f"free={int(_free) / 2**30:.3f} total={int(_tot) / 2**30:.3f} GiB",
                  flush=True)
    except Exception as _e:                       # noqa: BLE001
        print(f"[xtu-vram] {tag or 'report'} cuda 统计失败: "
              f"{type(_e).__name__} {_e}", flush=True)
    # ⭐ 逐次分配的离线快照（`XIAOTU_MEM_TRACE=1` 时按限频落盘）：
    # 两份快照相减 ⇒ 按**调用栈**聚合差值 ⇒ 直接指出"哪个栈在长"。
    try:
        from . import mem_trace as _mt
        _mt.maybe_dump(tag or "periodic")
    except Exception:                             # noqa: BLE001
        pass
    return {"total_bytes": total, "groups": per, "cuda": cuda_stats}


def _reuse_moe(key, shape, dtype, device):
    """按**定长**复用；行数按 2 的幂向上取整 ⇒ 尺寸类有限，**永不随 qlen 累积**。

    ⚠️ 背景（2026-10-26 审计）：旧实现把**整个 shape**（含行维 `A = T*K`）放进 key ⇒
    每个新的 chunk 长度都新建一块、模块级 dict 无淘汰 ⇒ 显存随流量"只增不减"。
    新实现：key 只含**列维**，行维按 2 的幂向上取整（MBT=8192/K=8 ⇒ 收敛到 65536），
    大于当前容量才扩容一次。`XIAOTU_MOE_BOUND_BUF=0` 回退旧行为。
    """
    if os.environ.get("XIAOTU_MOE_BOUND_BUF", "1") != "1":
        k = (str(device), key, tuple(int(v) for v in shape), dtype)
        t = _BUF_MOE.get(k)
        if t is None or tuple(t.shape) != tuple(int(v) for v in shape):
            _retire(t)
            t = torch.empty(tuple(int(v) for v in shape), dtype=dtype, device=device)
            _BUF_MOE[k] = t
        return t
    rows = int(shape[0])
    tail = tuple(int(v) for v in shape[1:])
    k = (str(device), key, tail, dtype)
    t = _BUF_MOE.get(k)
    if t is None or int(t.shape[0]) < rows:
        cap = max(rows, int(t.shape[0]) if t is not None else 0)
        cap = 1 << max(0, (cap - 1).bit_length())   # 2 的幂
        _retire(t)
        t = torch.empty((cap,) + tail, dtype=dtype, device=device)
        _BUF_MOE[k] = t
        if not _BOUND_BUF_LOGGED.get(k):
            _BOUND_BUF_LOGGED[k] = True
            print(f"[xtu-bound] {key}: 定长缓冲 {cap}x{tail} "
                  f"({t.numel() * t.element_size() / 2**20:.0f} MiB, 按 2 的幂上行、只增到上限)",
                  flush=True)
    return t[:rows]


# ⛔ 扩容时**绝不 free 旧缓冲**：CUDA graph 可能仍烘焙着旧指针，
# 提前释放 ⇒ 图重放时 cudaErrorIllegalAddress（2026-10-28 r93 死因嫌疑）。
# 与 pinned 缓冲同理：进退休区，交给 GC 在进程结束时统一回收。
_RETIRED_BUF = []


def _retire(t):
    """把即将被替换的缓冲放进退休区（保留最近 16 代），并记录当前流。"""
    if t is None:
        return
    try:
        t.record_stream(torch.cuda.current_stream(t.device))
    except Exception:
        pass
    _RETIRED_BUF.append(t)
    while len(_RETIRED_BUF) > 16:
        _RETIRED_BUF.pop(0)


_SEG_BUF = {}


def _seg_buf(name, shape, dtype, device, init=None):
    """segmentation 用的**持久缓冲**（捕获期零分配）。"""
    k = (str(device), name, tuple(int(v) for v in shape), dtype)
    t = _SEG_BUF.get(k)
    if t is None:
        t = torch.empty(tuple(int(v) for v in shape), dtype=dtype, device=device)
        if init is not None:
            init(t)
        _SEG_BUF[k] = t
    _touch(k)
    return t


def _build_segmentation_reuse(topk_ids, topk_weights, num_experts, device):
    """`_build_segmentation` 的**零分配**版本（`XIAOTU_MOE_SEG_REUSE=1` 启用）。

    与逐次分配版**语义完全一致**（垃圾桶桶、稳定排序、scatter_add 计数），
    但所有中间张量都来自持久缓存、所有写入都是 in-place ⇒ 捕获期不产生任何分配。
    目的：排除"捕获期分配/释放池内存"这一类可疑点（HANDOFF §74）。
    """
    T, K = topk_ids.shape
    A = T * K
    E = int(num_experts)
    ids_v = topk_ids.reshape(-1)
    wts_v = topk_weights.reshape(-1)

    ids_c = _seg_buf("ids_c", (A,), torch.int32, device)
    ids_c.copy_(ids_v)                                   # copy_ 自带类型转换
    bad = _seg_buf("bad", (A,), torch.bool, device)
    bad2 = _seg_buf("bad2", (A,), torch.bool, device)
    torch.lt(ids_c, 0, out=bad)
    torch.ge(ids_c, E, out=bad2)
    bad.logical_or_(bad2)
    ids_c.masked_fill_(bad, E)                            # 垃圾桶桶 = E

    vals = _seg_buf("vals", (A,), torch.int32, device)
    order = _seg_buf("order", (A,), torch.int64, device)
    torch.sort(ids_c, stable=True, out=(vals, order))     # 输出到缓存

    tok_ids = _seg_buf(
        "tok_ids", (A,), torch.int64, device,
        init=lambda t: t.copy_(torch.arange(T, device=device).repeat_interleave(K)))
    sorted_tok = _seg_buf("sorted_tok", (A,), torch.int64, device)
    torch.index_select(tok_ids, 0, order, out=sorted_tok)

    wts_tmp = _seg_buf("wts_tmp", (A,), torch.float32, device)
    wts_tmp.copy_(wts_v)                                  # copy_ 自带类型转换
    sorted_wts = _seg_buf("sorted_wts", (A,), torch.float32, device)
    torch.index_select(wts_tmp, 0, order, out=sorted_wts)

    idx64 = _seg_buf("idx64", (A,), torch.int64, device)
    idx64.copy_(ids_c)
    ones = _seg_buf("ones", (A,), torch.int32, device, init=lambda t: t.fill_(1))
    counts = _seg_buf("counts", (E + 1,), torch.int32, device)
    counts.zero_()
    counts.scatter_add_(0, idx64, ones)
    seg_start = _seg_buf("seg_start", (E + 1,), torch.int32, device)
    seg_start[0].zero_()
    torch.cumsum(counts[:E], 0, out=seg_start[1:])
    return sorted_tok, sorted_wts, seg_start, A


def _build_segmentation(topk_ids, topk_weights, num_experts, device):
    """把 [T,K] 路由压成按专家分段的排序列表(供两个 Triton 内核按段遍历)。

    **必须是形状静态的**:CUDA graph 捕获期间不允许出现数据相关的形状
    (`ids[ok]` 这种布尔掩码索引会触发 `cudaErrorStreamCaptureUnsupported`,
    见 dev-docs/report/tuning/TRIED_AND_REVERTED.md R14)。所以无效/不属于本 rank 的
    id 不删除,而是**归入垃圾桶专家桶 `num_experts`**(它排在最后,内核 grid=E
    不会读它)⇒ 与"过滤掉"完全等价。

    返回的 ``A`` 固定为 ``T*K``(内核只按段边界取行,``A`` 只用来定缓冲大小;
    注意 ``inter`` 是按 **token id** 索引的,所以行数本来就要 ≥ T)。
    """
    if os.environ.get("XIAOTU_MOE_SEG_REUSE", "0") == "1":
        return _build_segmentation_reuse(topk_ids, topk_weights, num_experts, device)
    T, K = topk_ids.shape
    ids = topk_ids.to(torch.int32).reshape(-1)
    wts = topk_weights.to(torch.float32).reshape(-1)
    tok_ids = torch.arange(T, device=device).repeat_interleave(K)
    # 垃圾桶桶:任何 <0 / ≥E 的 id 都映射到 E(固定形状,无掩码索引)
    bad = (ids < 0) | (ids >= num_experts)
    ids_c = torch.where(
        bad, torch.full((), num_experts, dtype=torch.int32, device=device), ids)
    order = torch.argsort(ids_c, stable=True)
    sorted_tok = tok_ids[order]
    sorted_wts = wts[order]
    # scatter_add 而不是 bincount:bincount 会从数据里求 max 来决定输出长度(可能同步);
    # scatter_add 的输出长度固定为 E+1。取前 E 个桶(垃圾桶段排在最后,内核不读)。
    ones = torch.ones_like(ids_c, dtype=torch.int32)
    counts = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    counts.scatter_add_(0, ids_c.long(), ones)
    counts = counts[:num_experts]
    seg_start = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(counts, dim=0)]
    )
    return sorted_tok, sorted_wts, seg_start, T * K


_GP_TIMING = os.environ.get("XIAOTU_GP_TIMING") == "1"
_GP_H2D = {"n": 0, "ms": 0.0}


def _gp_h2d(ms: float) -> None:
    st = _GP_H2D
    st["n"] += 1
    st["ms"] += ms
    if st["n"] % 40 == 0:
        print(f"[gp-h2d] n={st['n']} per_layer={st['ms'] / st['n']:.1f}ms", flush=True)
        st.update(n=0, ms=0.0)


_GP_STAT = {"n": 0, "seg": 0.0, "rest": 0.0, "tok": 0}
_GP_EVERY = 40


def _gp_add(T: int, A: int, seg_s: float, rest_s: float) -> None:
    """GPU 预填充路径的**主机侧**分段计时(env XIAOTU_GP_TIMING=1)。

    seg  = _build_segmentation(argsort/scatter_add/arange 等)耗时
    rest = 其后到内核 launch 之间的主机耗时(out/inter 分配、zeros 等)
    两者都是**串在关键路径上的主机时间**,用来定位"服务里每层多出的 ~31ms"。
    """
    st = _GP_STAT
    st["n"] += 1
    st["seg"] += seg_s
    st["rest"] += rest_s
    st["tok"] += T
    if st["n"] % _GP_EVERY == 0:
        n = st["n"]
        print(f"[gp-timing] n={n} Tavg={st['tok'] / n:.0f} "
              f"seg={st['seg'] / n * 1e3:.2f}ms rest={st['rest'] / n * 1e3:.2f}ms "
              f"host_total={(st['seg'] + st['rest']) / n * 1e3:.2f}ms", flush=True)
        st.update(n=0, seg=0.0, rest=0.0, tok=0)


_CAPTURE_DEPTH = 0
_CAPTURE_GUARD = threading.Lock()
_CAPTURE_WATCH_INSTALLED = False
_CAP_FREE = []


def capture_in_progress() -> bool:
    """True while a CUDA graph capture is running in this process."""
    return _CAPTURE_DEPTH > 0


def wait_no_capture(timeout_s: float = 120.0) -> None:
    """Block until no CUDA graph capture is in flight (best effort, bounded).

    Used by the background pinned prebuild: any CUDA call it makes
    (cudaHostRegister / pin_memory) during capture both fails and **invalidates
    the capture**, which aborts engine startup.
    """
    import time as _time

    end = _time.monotonic() + timeout_s
    while capture_in_progress() and _time.monotonic() < end:
        _time.sleep(0.05)


def wait_capture_done(quiet_s: float = 10.0, grace_s: float = 300.0,
                      poll_s: float = 0.1) -> bool:
    """Wait until CUDA-graph capture is finished **and has stayed quiet**.

    Rationale (2026-09-11, three failed startups): vLLM builds its graphs ~100 s
    after the first big warm-up forward — which is exactly the forward that
    triggers the pinned prebuild. A `cudaHostRegister` already inside the driver
    when capture begins cannot be recalled, and it both invalidates the capture
    (`cudaErrorStreamCaptureInvalidated` -> EngineCore init fails) and has been
    seen to segfault inside libcuda (`cuMemHostRegister_v2` -> SIGSEGV).

    Timeline measured on this box (deliver13 log): warm-up forward -> +100 s ->
    "Breakable CUDA graph enabled" -> main capture -> DSpark speculator capture.
    A fixed grace period therefore cannot work; we wait for the capture state to
    be *quiet* (no capture for `quiet_s`), and if this process never captures
    (``EAGER=1``) we give up after `grace_s` and proceed.
    """
    import time as _time

    t0 = _time.monotonic()
    last_end = 0.0
    seen = False
    while _time.monotonic() - t0 < grace_s:
        if capture_in_progress():
            seen = True
            while capture_in_progress() and _time.monotonic() - t0 < grace_s + 600.0:
                _time.sleep(poll_s)
            last_end = _time.monotonic()
            continue
        if seen and (_time.monotonic() - last_end) >= quiet_s:
            return True
        _time.sleep(poll_s)
    return not capture_in_progress()


def _install_capture_watch() -> None:
    """Count CUDA graph captures process-wide by wrapping CUDAGraph begin/end.

    There is no CUDA API to ask "is any stream in this context capturing?", and
    the capturing stream lives on another thread, so we track it ourselves. The
    wrapper is installed once and is a no-op otherwise.
    """
    global _CAPTURE_WATCH_INSTALLED
    if _CAPTURE_WATCH_INSTALLED:
        return
    try:
        import torch
        cls = torch.cuda.CUDAGraph
        if getattr(cls, "_xiaotu_capture_watched", False):
            _CAPTURE_WATCH_INSTALLED = True
            return
        _begin, _end = cls.capture_begin, cls.capture_end

        def capture_begin(self, *a, **k):
            global _CAPTURE_DEPTH
            with _CAPTURE_GUARD:
                _CAPTURE_DEPTH += 1
            if os.environ.get("XIAOTU_CAP_MEM") == "1":
                # 记录"每张图吃多少显存"（vLLM 在 MUSA 上跳过了 profile_cudagraph_memory，
                # 所以我们自己量；只读 mem_get_info，零副作用）
                try:
                    _CAP_FREE.append(torch.musa.mem_get_info()[0] / 2**20)
                except Exception:  # noqa: BLE001
                    _CAP_FREE.append(-1.0)
            try:
                return _begin(self, *a, **k)
            except BaseException:
                with _CAPTURE_GUARD:
                    _CAPTURE_DEPTH = max(0, _CAPTURE_DEPTH - 1)
                raise

        def capture_end(self, *a, **k):
            global _CAPTURE_DEPTH
            try:
                return _end(self, *a, **k)
            finally:
                if os.environ.get("XIAOTU_CAP_MEM") == "1" and _CAP_FREE:
                    try:
                        free_now = torch.musa.mem_get_info()[0] / 2**20
                        before = _CAP_FREE.pop()
                        print("[cap-mem] graph#%d 期间显存减少 %.0f MiB（capture 前空闲 %.0f MiB）"
                              % (len(_CAP_FREE) + 1, before - free_now, before), flush=True)
                    except Exception:  # noqa: BLE001
                        pass
                with _CAPTURE_GUARD:
                    _CAPTURE_DEPTH = max(0, _CAPTURE_DEPTH - 1)

        cls.capture_begin = capture_begin
        cls.capture_end = capture_end
        cls._xiaotu_capture_watched = True
        _CAPTURE_WATCH_INSTALLED = True
    except Exception:  # noqa: BLE001 - torch/driver variants: degrade to no-op
        pass


_install_capture_watch()


def _register_host(t: torch.Tensor) -> bool:
    """Page-lock ``t`` **in place** (no copy) via cudaHostRegister.

    ``Tensor.pin_memory()`` allocates a new page-locked buffer and copies into it
    at ~1.6 GB/s (measured); cudaHostRegister locks the pages we already have at
    ~5.8 GB/s (measured) and keeps the same H2D throughput, so the one-time
    K-major cache build drops from ~114 s to ~45 s (and with the parallel
    prebuild below, to ~15-25 s).
    """
    if t.is_pinned():
        return True
    try:
        rt = torch.cuda.cudart()
        err = rt.cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
        return int(err) == 0 and t.is_pinned()
    except Exception:  # noqa: BLE001 - driver/cudart missing
        return False


_PIN_LOCKS: dict[tuple, threading.Lock] = {}
_PIN_LOCKS_GUARD = threading.Lock()


def _key_lock(key: tuple) -> threading.Lock:
    with _PIN_LOCKS_GUARD:
        lk = _PIN_LOCKS.get(key)
        if lk is None:
            lk = threading.Lock()
            _PIN_LOCKS[key] = lk
        return lk


def _pin_key(t: torch.Tensor, tag: str):
    """缓存键 + 源 storage。

    键用**源 storage 地址**(对同一参数的切片是稳定的,如 ``param.data[a:b]``);
    同时持有源 storage 的引用,避免地址被无关分配复用后命中过期条目。
    """
    stor = t.untyped_storage()
    return (stor.data_ptr(), t.storage_offset(), tuple(t.shape), t.dtype, tag), stor


def _ensure_pinned(key: tuple, ent: tuple) -> torch.Tensor:
    """确保缓存项已锁页(**就地** cudaHostRegister,失败才退回 pin_memory)。"""
    c = ent[1]
    if c.is_pinned():
        return c
    if not _register_host(c):
        c = c.pin_memory()
        ent = (ent[0], c)
        _PIN_CACHE[key] = ent
    return c


def _pinned(t: torch.Tensor) -> torch.Tensor:
    key, stor = _pin_key(t, "plain")
    ent = _PIN_CACHE.get(key)
    if ent is not None:
        return _ensure_pinned(key, ent)
    with _key_lock(key):
        ent = _PIN_CACHE.get(key)
        if ent is None:
            ent = (stor, t.contiguous())
            _PIN_CACHE[key] = ent
        return _ensure_pinned(key, ent)


_PIN_CACHE: dict[tuple, tuple] = {}


def _kmajor_bytes(t, dst=None):
    """[E, A, B] u8 -> [E, B, A] u8(字节转置;nibble 仍留在字节内)。`dst` 给了就写进去。

    【§596】**默认改用分块 Triton kernel**:实测 `transpose(1,2).contiguous()`
    对这种 uint8 逐字节转置只有 **~90 GB/s**(4 个张量合计 42.8 ms/层),
    而 128×128 分块版有 **~600 GB/s**(合计 6.5 ms/层),快 5-7× 且逐位相同。
    `XIAOTU_GPF_TT=0` 可退回旧实现(出问题时的一键回滚)。
    """
    try:
        from vllm_xiaotu_moe.byte_transpose import ktranspose_bytes
        return ktranspose_bytes(t, dst)
    except Exception:  # noqa: BLE001
        r = t.transpose(1, 2).contiguous()
        if dst is not None:
            dst.copy_(r)
            return dst
        return r


def _kmajor_cached(t: torch.Tensor):
    """纯 CPU 阶段:做 K-major 转置并放入缓存(**不锁页**)。

    转置是普通 CPU 拷贝,任何时刻都安全(包括 CUDA graph 捕获期间);只有
    ``cudaHostRegister`` 不能在捕获窗口里调用(见 TRIED_AND_REVERTED R4)。
    把两件事拆开,后台就能在 warmup 期间先把转置做完。
    """
    key, stor = _pin_key(t, "kmajor")
    ent = _PIN_CACHE.get(key)
    if ent is not None:
        return key, ent
    with _key_lock(key):
        ent = _PIN_CACHE.get(key)
        if ent is None:
            ent = (stor, _kmajor_bytes(t))
            _PIN_CACHE[key] = ent
    return key, ent


def _pinned_kmajor(t: torch.Tensor) -> torch.Tensor:
    """Pin the K-major view of ``t`` (host-side transpose, cached).

    Doing the byte transpose once on the host (at cache-build time) instead of
    once per layer on the GPU removes a ~3.4 GiB device read+write per layer and
    lets the H2D land directly in the layout the kernels index.

    The cache is keyed on ``t``'s **own** storage (a long-lived model param), not
    on the transposed temporary: keying on the temporary would miss on every
    call and re-pin 3.19 GiB per layer (~2.6 s) while also leaking pinned
    entries until the host OOMs.
    """
    key, ent = _kmajor_cached(t)
    return _ensure_pinned(key, ent)


# --------------------------------------------------------------------------
# 显存预算与"优雅放弃"(用户 2026-09-15 指示)
#
# 逐层流式要在**设备上**再放一份当前层的权重(raw + K-major,约 2x 单层专家大小;
# V4.1@TP=1 单层 6.72 GiB ⇒ 峰值 ~14 GiB)。这块内存与 vLLM 的 KV cache 竞争,
# 而 vLLM 是在 **profile run** 之后按"峰值显存"给 KV 定容的。若不挡住,
# profile 期间走 GPU 路径会把峰值抬高,直接得到
#   `Available KV cache memory: -1.57 GiB` ⇒ **整个服务起不来**。
#
# 所以两层保护:
#   1. profile run 期间**绝不**走 GPU 路径(否则定容就崩);
#   2. 运行时先做一次**预检**:腾不出 staging 就留在 CPU,并打印一次明确提示 ——
#      "慢但能跑"优于"起不来"。用户的选择:TP>=2(每 rank staging 减半)、
#      换更大显存的卡、或调低 --gpu-memory-utilization 给它留地方。
# --------------------------------------------------------------------------
_IN_PROFILE_RUN = False
# 更宽的窗口:从进程起来到 KV cache 初始化完成之前都算 **startup**
# (profile run / CUDA graph 捕获 / warmup 都落在里面)。实测只挡 `profile_run`
# **不够** —— 峰值显存的测量发生在它外面,`Available KV cache memory` 仍然变成
# 负数(NOTES §460.1)。用 "_initialize_kv_caches 返回" 作为 startup 结束的锚点,
# 它与"KV cache 已定容"是同一件事,比去找具体的 profile 调用稳。
_IN_STARTUP = [True]
# ⭐ 2026-10-09:`Worker.compile_or_warm_up_model` 的执行窗口 = **warmup + CUDA graph 捕获** ⇒
#   该窗口内**必须挡住** GPU 预填(侧流预填与捕获交错 ⇒ replay 时 `cuGraphLaunch` SIGSEGV,实测)✗
#   而 **profile_run 期必须放行**(只有真分配 staging,vLLM 的账本才含它)✓
#   ⇒ 两者必须分开:本 flag 只覆盖 warmup/捕获,不覆盖 profile ✓
_IN_WARMUP = [False]

# ⭐ 2026-10-20:activation 占位必须在 **KV 定容之前** 真正分配。
#   用户原则:"只指定上下文长度,不指定 KV 大小;我们的内存申请排在 KV 之前,
#   让 KV 来适应我们"。原先 `activation_reserve_bytes()` 只进 GPU prefill 的
#   *预检*,没进 vLLM 的 `determine_available_memory` ⇒ KV 把 free 吃满,
#   长 prefill 的 512 MiB `q_padded` 在谷底找不到连续块(C=4 OOM, BUG A68)。
#   这里在 profile 期持有一个 **3 GiB 活张量** ⇒ vLLM 测到的 free 少 3 GiB
#   ⇒ KV 池自动缩小;KV 分配完成后再释放,块留在 torch cache 里给 q_padded 复用。
_ACT_RESERVE = None


def _reserve_activation_for_kv() -> None:
    """Allocate the activation reserve **before** vLLM sizes the KV pool."""
    global _ACT_RESERVE
    if _ACT_RESERVE is not None:
        return
    n = activation_reserve_bytes()
    if n <= 0:
        return
    try:
        _ACT_RESERVE = torch.empty(int(n), dtype=torch.uint8, device="cuda")
        print(f"[vllm-xtu-moe] ⭐ profile 期 activation 占位 "
              f"{n / 2**30:.2f} GiB(在 KV 定容前分配 ⇒ KV 自动让出这块)✓",
              flush=True)
    except Exception as exc:  # noqa: BLE001
        _ACT_RESERVE = None
        print(f"[vllm-xtu-moe] ⚠️ activation 占位失败,本次 KV 可能吃满: "
              f"{type(exc).__name__}: {exc}", flush=True)


def _release_activation_after_kv() -> None:
    """Drop the reserve **after** KV caches are sized; block stays in torch cache."""
    global _ACT_RESERVE
    if _ACT_RESERVE is not None:
        _ACT_RESERVE = None
        print("[vllm-xtu-moe] ⭐ KV 已定容 ⇒ 释放 activation 占位(块留给 q_padded 复用)✓",
              flush=True)



def in_profile_run() -> bool:
    """True during startup (profiling/capture/warmup) -- GPU prefill must not run."""
    return _IN_PROFILE_RUN or _IN_STARTUP[0]


def in_warmup_or_capture() -> bool:
    """⭐ True **只在** `Worker.compile_or_warm_up_model` 执行期间(= warmup + 图捕获)✓

    为什么要与 `in_profile_run()` 分开(2026-10-09 实测事故):
      * **`profile_run` 期要【允许】GPU 预填** —— 只有 staging 真的被分配,vLLM 的
        `total_consumed` / `non_kv_cache_memory` 才会把它算进去(SPEC_mbt_budget 改动②)✓
      * **warmup + CUDA graph 捕获期必须【挡住】** ✗ —— 否则侧流 GPU 预填会与图捕获交错
        ⇒ 捕获被污染 ⇒ replay 时 `cuGraphLaunch` **SIGSEGV**:
        实测 2026-10-09 17:35,两个 rank 同构栈
        `at::cuda::CUDAGraph::replay → cudaGraphLaunch → cuGraphLaunch`,服务整体下线 ✗
      * ⚠️ `torch.cuda.is_current_stream_capturing()` **不够** —— 它只在捕获调用之内为真,
        挡不住 warmup 里"捕获前后"的那些 forward ✗
    """
    return bool(_IN_WARMUP[0])


# ══════════════════════════════════════════════════════════════════════════════
# ⭐⭐ 2026-10-09(SPEC_mbt_budget 改动② 的**正解**):让 staging **进 vLLM 的账本**,
#    但**不让 GPU 预填在 profile 期实跑** ✓
#
# 实测教训(见 dev-docs/report/tuning/vllm_vram_experiment_2026-10-09.md §7/§11):
#   * 让预填在 `profile_run` 期**实跑**(无论 warmup 是否挡住)⇒ **两次**都在首个长请求后
#     `cuGraphLaunch` SIGSEGV(CUDA graph 捕获被侧流活动污染)✗
#   * 而**只占位、不跑内核**:同一次 profile 里,「已消耗显存」照样 +1.9 GiB
#     ⇒ vLLM 的 `total_consumed`/`non_kv_cache_memory` 记上它 ⇒ KV 池按"剩下的"定容 ✓
#   * 占位在 profile 结束**立刻释放** ⇒ 那块显存留给随后分配的 KV,预算分毫不差 ✓
# ══════════════════════════════════════════════════════════════════════════════
_PROF_RESERVE: list = []


def reserve_staging_for_profile(need_bytes: int, device=None) -> bool:
    """占位分配 `need_bytes`(单层 staging 峰值),**不跑任何内核** ✓ 返回是否成功。"""
    if int(need_bytes) <= 0:
        return False
    if device is None:
        try:
            device = torch.device("cuda", torch.cuda.current_device())
        except Exception:  # noqa: BLE001
            return False
    try:
        t = torch.empty(int(need_bytes), dtype=torch.uint8, device=device)
    except Exception as exc:  # noqa: BLE001
        print(f"[vllm-xtu-moe] ⚠️ profile 期 staging 占位分配失败"
              f"({int(need_bytes) / 2**30:.2f} GiB):{type(exc).__name__}: {exc}"
              f" —— 照常继续(只是本次 KV 会按'没有 staging'定容)", flush=True)
        return False
    _PROF_RESERVE.append(t)
    print(f"[vllm-xtu-moe] ⭐ profile 期 staging 占位 {int(need_bytes) / 2**30:.2f} GiB"
          f"(计入 vLLM 的 total_consumed ⇒ KV 按剩下的定容;不跑任何预填内核)✓", flush=True)
    return True


def release_staging_reserve() -> None:
    """profile 结束立刻释放占位 ⇒ 那块显存留给随后分配的 KV / 真正的 staging ✓

    ⭐ 2026-10-25（A69/A71）：释放前后各打一次 free，便于**机械验证**
    「free 回升 ≈ 占位大小」⇒ 证明释放真的生效、且 empty_cache 也做了 ✓
    """
    try:
        import torch as _t
        _p = _t.cuda.current_device()
        _f0, _tot = _t.cuda.mem_get_info(_p)
        print("[xtu-kv-diag] 释放 staging 占位: free %.2f GiB / total %.2f GiB"
              % (_f0 / 2**30, _tot / 2**30), flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"[xtu-kv-diag] 释放前取显存失败(不掩盖): {type(_e).__name__}: {_e}", flush=True)
    if _PROF_RESERVE:
        _PROF_RESERVE.clear()
        # ⭐ A69：`clear()` 只是丢掉活引用，段仍留在 torch 缓存分配器里 ⇒
        #   记账以为已归还，实际没还 ⇒ **必须** empty_cache 才真正还给驱动 ✓
        try:
            import torch as _t2
            _t2.cuda.empty_cache()
            _f1, _tot1 = _t2.cuda.mem_get_info(_t2.cuda.current_device())
            print("[xtu-kv-diag] 释放 staging 占位: free %.2f GiB（已 empty_cache）"
                  % (_f1 / 2**30,), flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[xtu-kv-diag] 释放后 empty_cache 失败(不掩盖): "
                  f"{type(_e).__name__}: {_e}", flush=True)
        print("[vllm-xtu-moe] profile 结束 ⇒ 释放 staging 占位(留给 KV)✓", flush=True)


def _reserve_staging_from_model(runner) -> None:
    """取各 MoE 层的 staging 需求,**取最大值**(单层峰值,不是求和)✓

    ⭐⭐ 2026-10-09 实测修正:`runner.model.modules()` **扫不到**我们的 MoE 模块
    (V2 runner 的 `.model` 不是那棵树 ⇒ 实测"扫到 0 个")✗
    ⇒ 改为走**插件自己的层注册表** `mixed_experts._GP_LAYERS`(引擎构建时填的),
       它与 vLLM 的模型树无关 ✓;模型树只作兜底 ✓
    """
    need = 0
    _asked = 0
    cands = []
    try:
        from vllm_xiaotu_moe import mixed_experts as _mx
        # ⭐⭐ 【item④】`_GP_PROFILE_LAYERS` 在**权重加载期**(`process_weights_after_loading`)
        #    就登记 ⇒ 它在 profile 期**非空**;而 `_GP_LAYERS` 只在引擎构建时(第一次
        #    forward)才填 ⇒ profile 期**是空的** ⇒ 只问后者会"扫到 0 个"、占位 = 0 ✗
        cands.extend(list(getattr(_mx, "_GP_PROFILE_LAYERS", [])))
        cands.extend(list(getattr(_mx, "_GP_LAYERS", {}).values()))
    except Exception:  # noqa: BLE001
        pass
    model = getattr(runner, "model", None)
    if model is not None:
        try:
            cands.extend(list(model.modules()))
        except Exception:  # noqa: BLE001
            pass
    seen = set()
    _best = None
    for m in cands:
        if id(m) in seen:
            continue
        seen.add(id(m))
        f = getattr(m, "staging_need_bytes", None)
        if not callable(f):
            continue
        _asked += 1
        try:
            v = int(f())
        except Exception:  # noqa: BLE001
            continue
        if v > need:
            need, _best = v, m
    # ⭐ 把**产生最大需求那个模块**的几何与 ns 打出来:ns 是除数,偏小⇒多算(抢 KV)、
    #    偏大⇒少算(首次长请求 OOM)⇒ 它必须可见,而不是静默 ✗
    _info = getattr(_best, "_prof_info", None) if _best is not None else None
    _extra = ""
    if isinstance(_info, dict) and _info:
        _src = "引擎" if _info.get("engine") else "权重形状(引擎尚未构建)"
        _extra = (f" [E={_info.get('E')} H={_info.get('H')} I={_info.get('I')}"
                  f" ns={_info.get('ns')} group_k={_info.get('group_k')} 几何来源={_src}]")
    print(f"[vllm-xtu-moe] profile 期 staging 记账:扫到 {_asked} 个可问 staging 的模块,"
          f" max need = {need / 2**30:.2f} GiB{_extra}", flush=True)
    if need <= 0:
        print("[vllm-xtu-moe] ⚠️ profile 期 staging 记账:**扫到 0** ⇒ 本次 KV 会按"
              "『没有 staging』定容 ⇒ 首次长请求可能 OOM ✗(须查 _GP_PROFILE_LAYERS 是否为空)",
              flush=True)
        return
    # ⭐⭐【B512】优先让**产生最大需求那个模块**去分配**持久真缓冲** ✓
    #    ⛔ 不能用替身代替:替身在 KV 定容测点之后**立刻被释放** ⇒ 那块显存被 KV 拿走
    #       ⇒ 运行期再分配同一批缓冲 ⇒ **必然 OOM**(B444 原话 + B510/B512 实测)✗
    #    ✅ 真缓冲 = 运行时复用的**同一批命名缓冲** ⇒ vLLM 测得到、且运行期缓存命中 ✓
    _real_ok = False
    _pref = getattr(_best, "prealloc_staging", None) if _best is not None else None
    if callable(_pref):
        try:
            _real_ok = bool(_pref())
        except Exception as _e:  # noqa: BLE001
            print(f"[vllm-xtu-moe] ⛔ 真缓冲预分配抛异常(不掩盖):"
                  f" {type(_e).__name__}: {_e}", flush=True)
            _real_ok = False
    if _real_ok:
        print(f"[vllm-xtu-moe] ⭐ profile 期**真缓冲**已分配(不跑任何内核)"
              f" ~{need / 2**30:.2f} GiB{_extra} ⇒ 进 vLLM 的 total_consumed ⇒ KV 据实定容 ✓",
              flush=True)
        return
    # 回落:后端没有真分配接口(非 FP8),或分配失败。**必须出声** ✗
    print("[vllm-xtu-moe] ⚠️ **回落**到替身占位(后端无真分配接口或真分配失败)"
          "⇒ 替身在 KV 定容后即被释放 ⇒ 运行期可能 OOM ✗", flush=True)
    reserve_staging_for_profile(need)


# ══════════════════════════════════════════════════════════════════════════════
# 长上下文激活记账（R34 老账 · 2026-10-26）
#
# 问题（实测，`dev-docs/INCIDENT_2026-10-26_XID31.md` §15.1/§18）：
#   vLLM 只用 `MBT=8192` 的一批 profile_run 去测 `peak activation` ⇒ 本模型测得 **1.88 GiB**，
#   而定容 `max_model_len=524288` 的服务在**长上下文 decode** 时的真实工作区远大于此。
#   后果：KV 池按"只留 1.88 GiB 激活"定容 ⇒ 稳态把设备填到 97%，
#   之后任何一次稍大的临时分配都会失败（r41 4 次 / mm16 10 次），
#   而"空出来了也不归还"（93 个空闲样本实测 +6.4 GiB 常驻）⇒ **跑十几小时后变成 Xid 31**。
#
# 本改动 = 把这份"没被计入的激活"在**定 KV 之前**真实占住：
#   `determine_available_memory` 的入口分配、`orig_dam()` 返回后释放
#   ⇒ vLLM 的 `after_profile.free_memory` 据实少这么多 ⇒ **KV 自动让出这块** ✓
#   ⇒ 饱和从"运行 14 小时后暴露"提前到"启动那 10 分钟就暴露"（R34 的要求）。
#
# ⚠️ 默认 **0.0 = 关闭**：这个数值必须**先量再定**，不能拍。
#   标定配方见 `dev-docs/INCIDENT_2026-10-26_XID31.md` §22；
#   粗略参考：268k 上下文实例的峰值比"启动后实测占用"高约 5.2 GiB（含 allocator 常驻，
#   故是**上界**；真实"活激活"介于 vLLM 的 1.88 与 5.2 之间）。
#   打开方式：`XIAOTU_GP_ACT_RESERVE_LC_GIB=3.0`（示例值，务必用标定结果替换）。
# ⚠️ 打开后 KV 池会**等比变小**：代价是并发/上下文容量，收益是"不再静默贴顶"。
#   这正是 R34 要的诚实账 —— 不够就在启动期说清楚，而不是运行中崩。
# ══════════════════════════════════════════════════════════════════════════════

def _lc_activation_gib() -> float:
    """要额外计入的长上下文激活量（GiB）；0 或负数 = 关闭。"""
    try:
        return float(os.environ.get("XIAOTU_GP_ACT_RESERVE_LC_GIB", "0") or 0)
    except (TypeError, ValueError):
        return 0.0


def _reserve_lc_activation():
    """在 profile 测点之前占住长上下文激活；返回持有的张量（或 None）。"""
    gib = _lc_activation_gib()
    if gib <= 0:
        return None
    n = int(gib * 2**30)
    try:
        t = torch.empty(n, dtype=torch.uint8, device="cuda")
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"[vllm-xtu-moe] ⭐ 长上下文激活记账: 在 KV 定容前占住 **{gib:.2f} GiB**"
              f"(设备剩 {free_b / 2**30:.2f}/{total_b / 2**30:.2f} GiB)"
              f" ⇒ 本次 KV 池会相应变小，账本据此诚实 ✓", flush=True)
        return t
    except Exception as _e:  # noqa: BLE001
        # 出声，不静默：静默退化会让"以为开了"变成谎报（B512 的教训）
        print(f"[vllm-xtu-moe] ⛔ 长上下文激活记账**失败**(不掩盖) ⇒ 本次 KV 仍会按"
              f"『只留 vLLM 自测激活』定容，贴顶风险照旧:"
              f" {type(_e).__name__}: {_e}", flush=True)
        return None


def _release_lc_activation(t) -> None:
    """KV 已据实定容 ⇒ 释放占位（块留在 torch cache 里，正好给长上下文工作区复用）✓"""
    if t is None:
        return
    del t
    print("[vllm-xtu-moe] ⭐ 长上下文激活占位已释放(KV 已按含该占位的口径定容)"
          " ⇒ 这块显存留给真正的长上下文工作区 ✓", flush=True)


def install_profile_guard() -> list[str]:
    """把 `profile_run` 暴露给我们(用来**设 `_IN_PROFILE_RUN`**,禁止 profile 期跑预填内核)✓
    ⭐ 2026-10-15【B441】:占位**不再**挂在这里(会漏记)⇒ 见 `GPUWorker.determine_available_memory` ✓

    ⭐⭐ 2026-10-09 修(R38 复审后实测发现的真 bug):本构建的 worker 用的是 **V2** runner
    (`vllm/v1/worker/gpu_worker.py:551` → `vllm.v1.worker.gpu.model_runner.GPUModelRunner`),
    而本函数原先**只打 V1**(`vllm.v1.worker.gpu_model_runner.GPUModelRunner`)⇒
    **shim 一直是死的**:`_IN_PROFILE_RUN` 从不为真、占位也从未发生 ✗
    ⇒ 现在**两个模块都打**(各自 `_xtu_shim` 幂等)✓
    """
    global _IN_PROFILE_RUN
    import functools
    import importlib

    applied: list[str] = []
    for mod_name in ("vllm.v1.worker.gpu_model_runner",     # V1(旧)
                     "vllm.v1.worker.gpu.model_runner"):    # ⭐ V2(本构建实际在用)
        try:
            _m = importlib.import_module(mod_name)
        except Exception:  # noqa: BLE001
            continue
        _cls = getattr(_m, "GPUModelRunner", None)
        if _cls is None:
            continue
        orig = getattr(_cls, "profile_run", None)
        if orig is None or getattr(orig, "_xtu_shim", False):
            continue

        def _make(orig_fn):
            @functools.wraps(orig_fn)
            def profile_run(self, *a, **kw):
                global _IN_PROFILE_RUN
                _IN_PROFILE_RUN = True
                try:
                    # ⛔【B441 修】占位**不能**放在这里!实测(真模型启动,OOM 崩溃):
                    #   vLLM 是在 `profile_run` **返回之后**才测 `after_profile.free_memory`
                    #   ⇒ 在本函数的 finally 里释放 ⇒ 占位**必然漏记** ⇒ KV 把 0.9 用满
                    #   ⇒ 随后真正的 tile 环要 800 MiB 时只剩 475 MiB ⇒ **OOM 崩溃** ✗
                    #   ⇒ 已挪到下面包住 `GPUWorker.determine_available_memory` ✓
                    return orig_fn(self, *a, **kw)
                finally:
                    _IN_PROFILE_RUN = False
            profile_run._xtu_shim = True  # type: ignore[attr-defined]
            return profile_run

        _cls.profile_run = _make(orig)
        applied.append(f"{mod_name.rsplit('.', 1)[-1]}.GPUModelRunner.profile_run")
    if not applied:
        return []

    # ⭐⭐【B441 修 · 关键】占位必须**活到 vLLM 测完 KV 预算**那一刻 ✓
    #   vLLM `GPUWorker.determine_available_memory()` 的真实结构(读码,mainline 树):
    #       with memory_profiling(self.init_snapshot, ...) as profile_result:
    #           self.model_runner.profile_run(...)
    #       free_gpu_memory = profile_result.after_profile.free_memory   # ← 测点在这
    #   ⇒ 旧实现把 `release_staging_reserve()` 放在 `profile_run` 的 finally 里
    #     ⇒ **释放发生在测点之前** ⇒ 占位必然漏记 ⇒ KV 把 `gpu_util` 用满
    #     ⇒ 随后真正的 tile 环分配只剩 475 MiB ⇒ **OOM 崩溃**(B441 实测)✗
    #   ⇒ 正确位置 = 包住**整个** `determine_available_memory`:
    #     占位是 category 2 的【活引用】⇒ `memory_profiling` 里的 `empty_cache()`
    #     **不会**回收它(empty_cache 只回收 cache 块,不动活张量)⇒ 必被计入
    #     `after_profile` ⇒ KV 预算**恰好少 need** ⇒ 释放后那块显存正好留给真正的环 ✓
    #   ⚠️⚠️【B442 实测踩坑】该模块里的类叫 **`Worker`**(`gpu_worker.py:187`)✗,
    #      **不是** `GPUWorker` —— 我第一版写 `from ... import GPUWorker` ⇒ ImportError
    #      被下面那个 `except` **静默吞掉** ⇒ shim 根本没装;而日志只显示
    #      `mainline shims applied (32)`(**没有 +1**)⇒ 白跑一次启动才发现 ✗
    #      ⇒ ① 类名改为 `Worker`(带回退)② **装不上必须出声**,不许静默 ✗
    _gpf_worker = None
    for _wn in ("Worker", "GPUWorker"):
        try:
            _gpf_worker = getattr(
                __import__("vllm.v1.worker.gpu_worker", fromlist=[_wn]), _wn)
            break
        except Exception:  # noqa: BLE001
            continue
    if _gpf_worker is None:
        print("[vllm-xtu-moe] ⚠️ profile 占位:**装不上** gpu_worker shim(Worker 找不到)"
              "⇒ KV 预算**不会**为 staging 留量 ✗", flush=True)
    else:
        orig_dam = getattr(_gpf_worker, "determine_available_memory", None)
        if orig_dam is None or getattr(orig_dam, "_xtu_shim", False):
            print("[vllm-xtu-moe] ⚠️ profile 占位:`determine_available_memory` 缺失或已装"
                  "⇒ 本次**未**新装 ✗", flush=True)
        else:
            @functools.wraps(orig_dam)
            def determine_available_memory(self, *a, **kw):
                # ⚠️ 参数是 **Worker**,而 `_reserve_staging_from_model` 要的是
                #    **runner**(它读 `runner.model` 作兜底)⇒ 必须取 `.model_runner` ✓
                # ⭐⭐【B512 · 2026-10-15 修】恢复调用,但**不再放替身** ✓
                #   * 本 hook 是**唯一被证明会执行**的记账落点:类名 `Worker`
                #     (`gpu_worker.py:187`)、方法体 `600→770`、KV 日志 `:721` 在其内 ✓
                #   * ⛔ 原先"由 `mixed_experts` 在 profile 窗口内分配真缓冲"的落点
                #     (`mixed_experts.py:1771`)被套在 `elif _gpu_pf:` 里,而 profile 期
                #     `_gpu_pf ≡ False` ⇒ **结构性不可达** ⇒ 记账**从未发生**(B508/B510)✗
                #   * ✅ 现由 `_reserve_staging_from_model` 调**各层的 `prealloc_staging()`**
                #     分配**持久真缓冲** ⇒ vLLM 测得到 ⇒ KV 据实定容 ✓
                #   * ⛔ 失败**必须出声**:静默退化正是 B510 那次事故的形态 ✗
                try:
                    _reserve_activation_for_kv()
                except Exception as _e:  # noqa: BLE001
                    print("[vllm-xtu-moe] ⛔ activation 占位失败(不掩盖): "
                          f"{type(_e).__name__}: {_e}", flush=True)
                try:
                    _reserve_staging_from_model(getattr(self, "model_runner", None))
                except Exception as _e:  # noqa: BLE001
                    print("[vllm-xtu-moe] ⛔ profile 期 staging 记账失败(不掩盖)"
                          f" ⇒ 本次 KV 可能按『没有 staging』定容:"
                          f" {type(_e).__name__}: {_e}", flush=True)
                # ⭐【2026-10-27 修正】预算基线的正确位置 = **预分配之后、KV 定容之前**
                #   （上一版打在预分配前 ⇒ 恒为 0.000 GiB，等于没有判据 ✗）
                try:
                    vram_report("startup-before-kv")
                except Exception as _e:  # noqa: BLE001
                    print(f"[xtu-vram] 基线报告失败(不掩盖): {type(_e).__name__}: {_e}", flush=True)
                # 【MUSA A71】staging 占位必须在 KV 定容前释放（否则运行期多占 1.76 GiB）
                # 【主线 R34】长上下文激活记账在 orig_dam 期间占住，让 KV 据实少算同样多
                _lc_res = _reserve_lc_activation()
                try:
                    return orig_dam(self, *a, **kw)
                finally:
                    try:
                        release_staging_reserve()
                    except Exception as _e:  # noqa: BLE001
                        print("[vllm-xtu-moe] ⛔ 释放 staging 占位失败(不掩盖): "
                              f"{type(_e).__name__}: {_e}", flush=True)
                    _release_lc_activation(_lc_res)

            determine_available_memory._xtu_shim = True  # type: ignore[attr-defined]
            _gpf_worker.determine_available_memory = determine_available_memory
            applied.append(f"{_gpf_worker.__module__}.{_gpf_worker.__name__}"
                           ".determine_available_memory")
            # ⛔【B512 修 · R63-3】旧文案声称"占位已挂到…(预算恰好少 need)"✗,
            #   而 B444 已把该处的替身删空 ⇒ 那是一句**谎报**,曾把判断带偏两轮 ✗
            #   ⇒ 新文案只描述**能力**,真实结果由 `_reserve_staging_from_model` 的
            #     运行时锚点(带 need/E/H/I 数值)负责报出 ✓
            print("[vllm-xtu-moe] ⭐ profile 期 staging 记账钩子已挂到 "
                  f"{_gpf_worker.__name__}.determine_available_memory"
                  "(将由各层 prealloc_staging() 分配**持久真缓冲**;真实结果见"
                  "『profile 期 staging 记账:… max need = …』那行)✓", flush=True)

    # 关键的一半:KV cache 定容完成 = startup 结束。
    try:
        from vllm.v1.engine.core import EngineCore
    except Exception:  # noqa: BLE001
        return applied
    orig_kv = getattr(EngineCore, "_initialize_kv_caches", None)
    if orig_kv is None or getattr(orig_kv, "_xtu_shim", False):
        return applied

    @functools.wraps(orig_kv)
    def _initialize_kv_caches(self, *a, **kw):
        try:
            return orig_kv(self, *a, **kw)
        finally:
            _IN_STARTUP[0] = False
            print("[vllm-xtu-moe] startup finished (KV cache sized) -> "
                  "GPU prefill is now allowed subject to the VRAM preflight",
                  flush=True)

    _initialize_kv_caches._xtu_shim = True  # type: ignore[attr-defined]
    EngineCore._initialize_kv_caches = _initialize_kv_caches
    applied.append("EngineCore._initialize_kv_caches")

    # 【§589 修】**必须同时给每个 worker 进程装一个锚点!**
    # `_IN_STARTUP` 是**进程级**标志,而门控 `_gpu_pf` 是在 **worker 进程**里求值的;
    # `EngineCore._initialize_kv_caches` 只在 EngineCore 进程执行(实测日志里那行
    # "startup finished" 只有 EngineCore 打过)⇒ **worker 里 `_IN_STARTUP[0]` 永远是 True**
    # ⇒ `in_profile_run()` 恒真 ⇒ **GPU 预填充从来没真正生效过**(4K 输入 TTFT 15.6 s,
    # 40 层 × 382 ms 全是 CPU MoE)。正确的 per-worker 锚点是
    # `Worker.compile_or_warm_up_model`:它在 KV 定容之后、图捕获完成时返回,正好是
    # "该 worker 的 startup 结束"(且期间保持禁用,才能不污染 profile 的显存测量与捕获)。
    try:
        from vllm.v1.worker.gpu_worker import Worker
    except Exception:  # noqa: BLE001
        return applied
    orig_warm = getattr(Worker, "compile_or_warm_up_model", None)
    if orig_warm is not None and not getattr(orig_warm, "_xtu_shim", False):
        @functools.wraps(orig_warm)
        def compile_or_warm_up_model(self, *a, **kw):
            # ⭐ 2026-10-20:KV 已定容 ⇒ 现在释放 activation 占位(在 warmup/图捕获前),
            #   块留在 torch cache 里供长 prefill 的 q_padded 复用。
            _release_activation_after_kv()
            # ⭐ 2026-10-09:本窗口 = **warmup + CUDA graph 捕获** ⇒ 期间挡住 GPU 预填 ✗
            #   (实测:侧流预填与捕获交错 ⇒ replay 时 `cuGraphLaunch` SIGSEGV)✓
            _IN_WARMUP[0] = True
            try:
                return orig_warm(self, *a, **kw)
            finally:
                _IN_WARMUP[0] = False
                # ⛔【B444】这里**不再释放任何"占位"** ✗ —— 真缓冲由 mixed_experts
                #   在 profile 窗口内分配并**常驻**(vLLM 已据实给 KV 定容)✓
                if _IN_STARTUP[0]:
                    _IN_STARTUP[0] = False
                    print("[vllm-xtu-moe] worker startup finished (KV sized + graphs "
                          "captured) -> GPU prefill allowed subject to VRAM preflight",
                          flush=True)

        compile_or_warm_up_model._xtu_shim = True  # type: ignore[attr-defined]
        Worker.compile_or_warm_up_model = compile_or_warm_up_model
        applied.append("Worker.compile_or_warm_up_model(per-worker startup anchor)")
    return applied


def staging_tile_experts(n_experts: int) -> int:
    """【微 ping/pong】每个 staging tile 覆盖多少专家。

    ⭐⭐ 2026-10-15 用户裁定:「**所有 GPU prefill 链路都要走 tile 形态(微 ping/pong)**,
    不然显存肯定不够;而且**之前明确要求去掉之前的按层缓冲机制**」✓
    ⇒ 本函数**永不返回 `E`(整层)** —— 按层缓冲那条路**已不存在** ✗

      * **默认(env 未设 / `0` / `>= E`)** ⇒ **48**（2026-10-27 用户裁定；
        原为 `max(1, E//3)`。实测 tile 128/96/72/48 的 H2D 带宽平坦 ⇒ 更小不亏，显存更省 ✓）
      * `XIAOTU_GPF_STAGE_TILE_E=k` ⇒ 用 k,但**强制夹到 `[1, E-1]`**
        ⇒ ⛔ `=0` **不再能**退回整层(按层路径已删除 ✓)

    ⭐⭐ **记账(`staging_bytes` / `fp8_staging_bytes`)与分配(槽形状)必须共用本函数** ——
    这是"预检说够、实际 OOM"的唯一防线(§600/§601 两次事故都是口径不一致造成的)✗
    ⭐ 现在两者**都只可能是 tile 形态** ⇒ 结构上不可能再分叉 ✓
    """
    E = int(n_experts)
    if E <= 1:
        return max(1, E)
    try:
        v = int(os.environ.get("XIAOTU_GPF_STAGE_TILE_E", "0") or 0)
    except (TypeError, ValueError):
        v = 0
    if v <= 0 or v >= E:
        # ⭐【2026-10-27 · 用户裁定】默认值统一为 **48**（原 `max(1, E//3)`：E=384→128、
        #   E=288→96）。实测依据（H2D 探针）：tile=128/96/72/48 的 H2D 带宽
        #   26.83/26.82/26.81/**26.79** GB/s（差 0.15%，平坦）⇒ 更小的 tile **不付性能代价**，
        #   而显存线性少占（v41 实测环 2.24 → **0.84 GiB**，KV 池多出 1.57 GiB）。
        #   ⇒ 不再依赖 `serve_v41.sh` 传 env；插件默认即 48（GLM/E=288 同样受益）。
        v = 48
    return min(max(1, int(v)), E - 1)


def stage_tiles(n_experts: int) -> list:
    """`[(e0, e1), ...]` —— 按 tile 切专家维。

    ⭐【2026-10-17 注释更正】原句「整层时恰好返回 `[(0, E)]` ⇒ 调用方循环一次,
    行为与改动前等价」**已过期** ✗ —— 见 `staging_tile_experts` 的文档串:
    2026-10-15 用户裁定去掉按层缓冲后,该函数**永不返回 `E`** ✓
    ⇒ 本函数是它的**纯派生**(`t = staging_tile_experts(E)` ✓),
      `E >= 2` 时 `t <= E-1` ⇒ **至少 2 个 tile**,**不可能**再返回 `[(0, E)]` ✓
    """
    E = int(n_experts)
    t = staging_tile_experts(E)
    if t <= 0:
        return []
    return [(e0, min(e0 + t, E)) for e0 in range(0, E, t)]


def staging_bytes(n_experts: int, hidden: int, inter: int, group_k: int = 32,
                  ns: int = 2) -> int:
    """Peak device bytes one layer's streaming path needs.

    `kmajor_from_engine_shards` now fills the K-major target directly and finishes
    w13 (including freeing its shard buffers) before allocating w2, so the peak is
        max(w13_kmajor + w13_node_shard,
            w13_kmajor + w2_kmajor + w2_node_shard) + 2 * scales
    rather than the old raw+K-major double (which was ~2x dense = 13.4 GiB).
    """
    E, H, I = int(n_experts), int(hidden), int(inter)
    gk = int(group_k) if int(group_k) > 0 else 1
    n = max(1, int(ns))
    w13d = E * (2 * I) * (H // 2)
    w2d = E * H * (I // 2)
    s13d = E * (2 * I) * (H // gk)
    s2d = E * H * (I // gk)
    # 【§600 修口径】不是只有 ping/pong 两槽!一次 staging 的**真实峰值**是:
    #   槽 2×(w13+w2+s13+s2) + raw(w13+w2,组装目标) + DMA 暂存(N 张量各一份)
    # 原式只算第一项(6.72 GiB),于是 preflight 说 8.4 GiB 够、实际要 ~11 GiB ⇒
    # "预检通过但跑起来 OOM"。这里把三项都算上(暂存已改为每张量一份)。
    # 【微 ping/pong】`tile < E` 时槽与 DMA 暂存都按 tile/E 缩小;`tile == E` ⇒ 与上式逐字相同 ✓
    _sum = w13d + w2d + s13d + s2d
    _tile = staging_tile_experts(E)
    if _tile >= E:
        slots = 2 * _sum
    else:
        slots = 2 * (_tile / E) * _sum
    raw = w13d + w2d
    # 暂存现在**每张量一份**(见 `_dma_hostbuf`),所以是"一个 node 的量"而不是全部 node 之和
    if _tile >= E:
        tmp = _sum / n
    else:
        tmp = (_tile / E) * _sum / n
    # 【方案 B】逐 node 直填 K-major 时**不再分配 raw 规范布局缓冲** ⇒ 预检口径必须同步,
    # 否则"改了却没省"会被误读(⚠️ 实测踩过:拿这条公式的数当验收判据是错的)。
    if os.environ.get("XIAOTU_GPF_KMADAPTIVE", "1") != "0":
        return slots + tmp
    return slots + raw + tmp


def activation_reserve_bytes() -> int:
    """VRAM the streaming path must **leave for vLLM's own activation peak**.

    【§601,2026-09-19 生产事故】staging 缓冲是**进程级持久**的(`_buf`/`_BUF_CACHE` 按名字
    缓存,永不释放),而 vLLM 是拿"weights + 2.9 GiB 激活峰 + CUDA graph"去**定 KV 大小**的:
    一次性吃掉 staging 之后,那份激活峰就没了 —— 于是长 prefill 的 attention/KDA 工作区
    必然 OOM。实测(8070 生产服务,util 0.90,GLM-5.3-Flash,28,553-token prompt):
    `GPU prefill` 的预检放行、4 GiB 级 staging 分配成功,随后 `chunk_kda_with_fused_gate`
    申请 52 MiB 失败,GPU 0/1 各只剩 45 MiB 空闲(PyTorch 已分配 37.90 GiB、另有
    695 MiB reserved-unallocated),**服务整进程崩溃**。

    ⇒ 预检必须比"staging 峰值 × margin"更严:还要额外留出这份激活预算。默认 3.0 GiB
    (与启动日志里 `peak activation: 2.9 GiB` 同量级),可用
    `XIAOTU_GP_ACT_RESERVE_GIB` 覆盖;置 0 恢复旧行为(只比 staging,不建议)。
    """
    try:
        # ⭐【2026-10-26 · 审计后重定】默认 3.0 → **1.0 GiB**。
        #   理由（INCIDENT_2026-10-26 §31–§32）：
        #     * 旧的 3.0 是在"运行期缓冲无界增长"前提下拍的；现在 `_BUF_MOE`/`_BUF_CACHE`
        #       已按 2 的幂定长，运行期**固定**开销 ≈ inter 512 MiB + dma ~0.1 + 余量
        #       ⇒ 1.0 GiB 足够覆盖，且不再需要为"不可预测的增长"付 2 GiB。
        #     * vLLM **自己的**激活峰在它的公式里已单独扣除（`transient_peak_headroom`），
        #       这里再留 3.0 属于**重复记账** ⇒ KV 池被白扣 2 GiB。
        #   KV 影响：4.14 → 3.14 GiB ≈ 2.37×512K（仍需满足 2×512K ✓）。
        # ⭐⭐【2026-10-27 · 用户裁定】"预算准确 + 运行期不涨" 之后**不需要预留**。
        #   预算准确 = 运行期定长缓冲已在 KV 定容前真分配（`mixed_experts.prealloc_staging`
        #   的 MXFP4 分支），vLLM 自己测得进 `total_consumed`；
        #   运行期不涨 = `_BUF_MOE`/`_BUF_CACHE` 已按 2 的幂定长（见 `_reuse_moe`/`_reuse`）。
        #   ⇒ 默认 0：不再用固定常数猜，也不再重复扣 vLLM 已单独扣除的 activation 峰。
        #   仍可用 env 覆盖（例如把长上下文实测激活峰再加回来）。
        # 默认 **0**（用户 2026-10-27 裁定）：inter + tile 环(含 raw 落点) 全部在
        # profile 期真分配并进账 ⇒ 运行期**不该再有新分配**；真涨了就是 bug，让它立刻崩出来。
        # 旧的 0.25 只是"落点未并入环"时的保险，现在不需要了。
        # 原文：默认 0.25 GiB（不是 3.0）：inter + tile 环已在 profile 期真分配并进账，
        # 这 0.25 只覆盖"DMA 暂存首次分配 + 碎片抖动"——它是一次性、有界的
        # （每 (which,slot) 一块，2 的幂定长）。首次真实流量后可用 `[xtu-vram]` 复核，
        # 若实测 DMA 总量已能并入 profile 预分配，则可降到 0。
        gib = float(os.environ.get("XIAOTU_GP_ACT_RESERVE_GIB", "0.0"))
    except (TypeError, ValueError):
        gib = 3.0
    return int(max(0.0, gib) * 2**30)


def required_bytes(need_bytes: int, margin: float = 1.10,
                   reserve_bytes: int | None = None) -> int:
    """Free VRAM the preflight demands for `need_bytes` of staging."""
    if reserve_bytes is None:
        reserve_bytes = activation_reserve_bytes()
    return int(int(need_bytes) * margin) + int(reserve_bytes)


def fits_device(need_bytes: int, device, margin: float = 1.10,
                reserve_bytes: int | None = None):
    """(ok, free_bytes). 把 staging 的**真实峰值**与**空闲**显存比。

    【§600】margin 从 1.25 收到 **1.10**:`need_bytes` 现在是诚实口径
    (槽 + raw + 单份暂存 = ~10.8 GiB,原来只有槽的 6.72),再乘 1.25 会把
    "实测能跑"的配置(如 free=12.95)误判为不够。1.10 覆盖碎片抖动。

    【§601】但 1.10 仍然**不够**:它只覆盖"分配 staging 的瞬间够",没覆盖"分配完之后
    还要跑完这次 forward"。持久 staging 会把 vLLM 预留的激活峰吃掉 ⇒ 再加一份显式的
    {@link activation_reserve_bytes}(默认 3.0 GiB)。预检不通过时优雅退回 CPU 预填充
    (慢但绝不会崩服务),这也是为什么这里的口径要偏保守。
    """
    try:
        free, _total = torch.cuda.mem_get_info(torch.device(device))
    except Exception:  # noqa: BLE001
        return True, -1        # cannot tell -> let it try (still guarded by try/except)
    return (free >= required_bytes(need_bytes, margin, reserve_bytes)), int(free)


# 复用中间缓冲:实测一次性函数 789 ms/层,而各段相加只有 424 ms —— 差额 ~365 ms
# 全是**每层重新分配**那些中间张量(w13/w2 的 node 缓冲 + raw + K-major,约 13 GB)
# 造成的 allocator churn(NOTES §466)。这些缓冲在同一 stream 上是安全可复用的:
# 下一层的 copy_ 排在前一层 kernel 之后,stream 会保证顺序。
_BUF_CACHE: dict = {}

# 方案 B(逐 node 直填 K-major)是否已经报过到 —— 只打一次,作为"路径真的走了"的正向证据
_KMADAPTIVE_LOGGED = False


def _reuse(key, shape, device):
    """可复用的 uint8 缓冲；**定长**（字节数按 2 的幂上行，只增到上限）。

    ⚠️ 旧实现把 `shape`（含字节数）放进 key ⇒ 每种 DMA 尺寸都新建一块且永不淘汰。
    新实现键只含 key/device，容量按 2 的幂上行，超过才扩容一次。
    `XIAOTU_MOE_BOUND_BUF=0` 回退旧行为。
    """
    if os.environ.get("XIAOTU_MOE_BOUND_BUF", "1") != "1":
        k = (str(device), int(shape[0]) if len(shape) == 1 else -1, tuple(shape))
        t = _BUF_CACHE.get(k)
        if t is None or t.shape != tuple(shape) or t.device != torch.device(device):
            _retire(t)
            t = torch.empty(tuple(shape), dtype=torch.uint8, device=device)
            _BUF_CACHE[k] = t
        return t
    n = int(shape[0]) if len(shape) == 1 else int(__import__("math").prod(shape))
    k = (str(device), str(key))
    t = _BUF_CACHE.get(k)
    if t is None or int(t.numel()) < n:
        cap = 1 << max(0, (max(n, int(t.numel()) if t is not None else 0) - 1).bit_length())
        _retire(t)
        t = torch.empty((cap,), dtype=torch.uint8, device=device)
        _BUF_CACHE[k] = t
        if not _BOUND_BUF_LOGGED.get(k):
            _BOUND_BUF_LOGGED[k] = True
            print(f"[xtu-bound] dma({key}): 定长缓冲 {cap / 2**20:.1f} MiB", flush=True)
    _touch(k)
    return t[:n]


def _dma_hostbuf(engine, which: int, node: int, nbytes: int, device) -> torch.Tensor:
    """DMA one of the ENGINE's own host buffers to a fresh device tensor."""
    if nbytes <= 0:
        raise RuntimeError(f"gpu-prefill: engine buffer which={which} node={node} is empty")
    # 【§600】暂存缓冲**每张量一份**(不再每 node 一份)。同一条 stream 上
    # "DMA n → copy n → DMA n+1" 是严格有序的 ⇒ 复用安全;而每 node 一份会让
    # ns 份暂存同时常驻(TP=2 共 3.59 GiB),把真正的 staging 峰值从 ~11 GiB 抬到 ~14 GiB。
    t = _LANDING.get((str(device), int(which)))
    if t is None or int(t.numel()) < int(nbytes):
        # 没预分配(未走 tile 环路径) ⇒ 回退到定长缓存（仍是 2 的幂、一次性）
        t = _reuse(("dma", int(which)), (int(nbytes),), device)
    else:
        t = t[: int(nbytes)]
    stream = torch.cuda.current_stream(device).cuda_stream
    got = engine.copy_hostbuf_to_device(int(which), int(node), t.data_ptr(), stream)
    if int(got) != int(nbytes):
        raise RuntimeError(
            f"gpu-prefill: shard DMA failed (which={which} node={node}: "
            f"copied {got}/{nbytes} bytes)"
        )
    return t


def _dma_hostbuf_range(engine, which: int, node: int, device,
                       off: int, size: int, slot: int = 0) -> torch.Tensor:
    """【微 ping/pong】只搬 node 缓冲里 `[off, off+size)` 这一段(而不是整段)。

    ⭐⭐ 为什么**必须**这样 —— 2026-10-09 实测教训(§18):
    `copy_hostbuf_to_device(which, node, dst, stream)` **只按 (which, node)**,每次都搬
    **整段 node**(w13 = 566 MB)。我按 tile 调它 4 次 ⇒ **同一个整段被搬了 4 遍**
    ⇒ **PCIe 流量 ×4** —— 与"少搬字节"完全相反 ✗(实测 prefill **+72.5%**)✗

    这里改用 `copy_hostbuf_to_device_2d` 的 `src_off`;取 `dpitch=spitch=width`、`height=1`
    ⇒ 它退化成**一次普通的、带偏移的连续拷贝** ⇒ 字节数与整层路径**完全相同** ✓
    (四个张量的宿主 node 缓冲都是**专家维在最外层且连续** ⇒ 专家区间 == 连续字节区间 ✓)
    """
    if int(size) <= 0:
        raise RuntimeError(f"gpu-prefill: ranged DMA size<=0 (which={which} node={node})")
    fn = getattr(engine, "copy_hostbuf_to_device_2d", None)
    if fn is None:
        return None                      # 引擎不支持 ⇒ 调用方回退整段路径
    # ⭐⭐⭐【B995 修】中间缓冲按【槽号】隔离✗
    #   原键 `("dma_tile", which)` 只有 which(0=w13/1=w2)，没有槽号
    #   ⇒ `binding.cpp:1115` 作者实测原话：【one staging buffer per weight
    #   block serializes every DMA in the layer behind the previous copy】
    #   ⇒ 槽 s+1 的 H2D 必须等槽 s 的转置读完才能写，双缓冲形同虚设✗
    #   现在按 slot 隔离 ⇒ 各槽各自一块 ⇒ DMA 可真并发✓
    t = _reuse(("dma_tile", int(which), int(slot)), (int(size),), device)
    stream = torch.cuda.current_stream(device).cuda_stream
    got = fn(int(which), int(node), int(off), t.data_ptr(),
             int(size), int(size), int(size), 1, stream)
    if int(got) != int(size):
        raise RuntimeError(
            f"gpu-prefill: ranged shard DMA failed (which={which} node={node} "
            f"off={off}: copied {got}/{size} bytes)")
    return t


_STAGE_ACC = {"n": 0, "dma": 0.0, "asm": 0.0, "tr": 0.0}

_STAGE_EV = {"pend": []}   # 【B142】待收割的 (key, start_ev, end_ev)

def _stage_begin():
    """相**开始**处记一个 event,返回值交给下一次 `_stage_mark` 当 `t0`。

    【B142 口径修正】旧实现靠 `perf_counter` + 每个子相 `torch.cuda.synchronize()`,
    而 `synchronize()` 等的是**设备上所有工作**(含与之并发的 prefill 计算),
    于是把 dma 量成 **537 ms/层**,而同一批数据用 CUDA event 只有 **167–206 ms/层**
    (`[fp8-asm]`),**差 ~3×**。现在改为:**不 sync**,每相记 event,靠
    `event.query()` 在后续调用里收割已完成的那一段 —— 既不污染被测对象,
    读数也与 `[fp8-asm]` 同口径,两者可互相交叉验证。
    """
    if os.environ.get("XIAOTU_GPF_STAGE") != "1":
        import time as _t
        return _t.perf_counter()
    try:
        e = torch.cuda.Event(enable_timing=True)
        e.record(torch.cuda.current_stream())
        return e
    except Exception:  # noqa: BLE001
        import time as _t
        return _t.perf_counter()


def _stage_mark(t0, key: str):
    """相**结束**处记 event,并在 `key == "tr"`(一层结束)时收割已完成的分相。

    与 `_stage_begin` 配对;`t0` 是上一相/本相开始的 event(诊断关闭时是 float)。
    返回本相的 event,供下一次调用当 `t0` —— 调用点因此**无需改动**。
    """
    if os.environ.get("XIAOTU_GPF_STAGE") != "1":
        import time as _t
        return _t.perf_counter()
    try:
        e = torch.cuda.Event(enable_timing=True)
        e.record(torch.cuda.current_stream())
        if isinstance(t0, torch.cuda.Event):
            _STAGE_EV["pend"].append((key, t0, e))
        if key == "tr":
            keep = []
            for k, a, b in _STAGE_EV["pend"]:
                try:
                    if not (a.query() and b.query()):
                        keep.append((k, a, b))
                        continue
                    _STAGE_ACC[k] = _STAGE_ACC.get(k, 0.0) + a.elapsed_time(b)
                except Exception:  # noqa: BLE001
                    pass
            _STAGE_EV["pend"] = keep[-8:]
            _STAGE_ACC["n"] += 1
            n = _STAGE_ACC["n"]
            if n % 40 == 0:
                d, a_, r = _STAGE_ACC["dma"], _STAGE_ACC["asm"], _STAGE_ACC["tr"]
                print(f"[gpf-stage] n={n} dma={d/40:.1f}ms asm={a_/40:.1f}ms "
                      f"tr={r/40:.1f}ms total={(d+a_+r)/40:.1f}ms/层 "
                      f"(CUDA event,**无 sync**)", flush=True)
                _STAGE_ACC.update({"dma": 0.0, "asm": 0.0, "tr": 0.0})
        return e
    except Exception:  # noqa: BLE001
        import time as _t
        return _t.perf_counter()


def kmajor_from_engine_shards_v2(engine, device, hidden: int, inter: int,
                                 n_experts: int, group_k: int):
    """【§595 A/B】同 `kmajor_from_engine_shards`,但把三个阶段**分批**做:

        D) 先把所有节点的 DMA **一次性全发出去**(同一 stream,背靠背) → sync
        B) 再做全部跨步组装(纯 D2D)                                          → sync
        C) 最后做 4 个 K-major 转置                                            → sync

    为什么要这样:原实现是 **per-node `DMA → copy_` 交替**,每个 `copy_` 都排在
    自己那次 DMA 之后,于是 **PCIe 在每次 D2D 期间空转**,而且三段耗时混在一起、
    无法归因(§594 只能量到 asm 总量)。分批后既能让 DMA 连续占满链路,也能把
    "DMA = 多少 / 组装 = 多少 / 转置 = 多少" 分开报出来。
    """
    import time as _time
    import torch as _torch

    geo = engine.shard_geometry()
    ns = int(geo["ns"])
    if ns < 2 or not geo["w13_node_bytes"] or not geo["w2_node_bytes"]:
        return None
    _pin_engine_hostbufs(engine)
    H, I, E = int(hidden), int(inter), int(n_experts)
    rb13 = H // 2
    rb2 = I // 2
    gk = int(group_k) if int(group_k) > 0 else 1
    dbg = os.environ.get("XIAOTU_GPF_STAGE") == "1"
    # 【goal③】`XIAOTU_GPF_EVT=1`：用 CUDA event 计时，**只在最后同步一次**。
    # 旧写法每个子相都 `cuda.synchronize()`，会把上一层未完成的 asm/tr 算进本层 dma
    # （这就是实测 dma 334 vs 599 ms/层 漂移的根源），且把并行度打掉。
    evt = os.environ.get("XIAOTU_GPF_EVT") == "1"
    _e = None
    if evt:
        _e = [_torch.cuda.Event(enable_timing=True) for _ in range(4)]
        _e[0].record()

    # ---- D) 所有 DMA 背靠背发出(不夹 D2D)----------------------------------
    t0 = _time.perf_counter()
    b13 = [_dma_hostbuf(engine, 0, n, int(geo["w13_node_bytes"]), device) for n in range(ns)]
    b2 = [_dma_hostbuf(engine, 1, n, int(geo["w2_node_bytes"]), device) for n in range(ns)]
    bs13 = _dma_hostbuf(engine, 2, 0, int(geo["w13_scale_bytes"]), device)
    bs2 = _dma_hostbuf(engine, 3, 0, int(geo["w2_scale_bytes"]), device)
    if evt:
        _e[1].record()
    elif dbg:
        _torch.cuda.synchronize()
    t_d = _time.perf_counter() - t0

    # ---- B) 组装(纯 D2D,源是连续的 node 缓冲,目标是跨步行区间)----------
    t1 = _time.perf_counter()
    w13_raw = _reuse(("raw13",), (E, 2 * I, rb13), device)
    c13 = int(geo["w13_cbytes"])
    cr13 = int(geo["w13_crows"])
    for n in range(ns):
        blk = b13[n].view(E, 2, c13)
        c0 = n * cr13
        w13_raw[:, c0:c0 + cr13, :].copy_(blk[:, 0, :].reshape(E, cr13, rb13))
        w13_raw[:, I + c0:I + c0 + cr13, :].copy_(blk[:, 1, :].reshape(E, cr13, rb13))
    w2_raw = _reuse(("raw2",), (E, H, rb2), device)
    c2 = int(geo["w2_cbytes"])
    cr2 = int(geo["w2_crows"])
    for n in range(ns):
        c0 = n * cr2
        w2_raw[:, c0:c0 + cr2, :].copy_(b2[n].view(E, cr2, rb2))
    s13_raw = bs13.view(E, 2 * I, H // gk)
    s2_raw = bs2.view(E, H, I // gk)
    if evt:
        _e[2].record()
    elif dbg:
        _torch.cuda.synchronize()
    t_b = _time.perf_counter() - t1

    # ---- C) K-major 转置 ---------------------------------------------------
    t2 = _time.perf_counter()
    out = (_kmajor_bytes(w13_raw), _kmajor_bytes(s13_raw),
           _kmajor_bytes(w2_raw), _kmajor_bytes(s2_raw))
    if evt:
        _e[3].record()
    elif dbg:
        _torch.cuda.synchronize()
    t_c = _time.perf_counter() - t2

    if evt:
        _torch.cuda.synchronize()   # 全路径只在这同步一次
        t_d = _e[0].elapsed_time(_e[1])
        t_b = _e[1].elapsed_time(_e[2])
        t_c = _e[2].elapsed_time(_e[3])
    if dbg or evt:
        a = _STAGE_ACC
        a["n"] += 1
        a["dma"] += t_d * 1e3
        a["asm"] += t_b * 1e3
        a["tr"] += t_c * 1e3
        if a["n"] % 40 == 0:
            print(f"[gpf-v2] n={a['n']} dma={a['dma']/40:.1f}ms 组装={a['asm']/40:.1f}ms "
                  f"转置={a['tr']/40:.1f}ms total={(a['dma']+a['asm']+a['tr'])/40:.1f}ms/层",
                  flush=True)
            a.update({"dma": 0.0, "asm": 0.0, "tr": 0.0})
    return out


_PIN_DONE = {"n": -1}


def _pin_engine_hostbufs(engine) -> None:
    """【§612】首次走 GPU 预填充时,把引擎自有的 host 分片**就地锁页一次**。

    为什么:这些分片是 `mmap`/`numa_alloc_onnode` 的 **pageable** 内存,实测 DMA 只有
    16-21 GB/s;而同一机器上**锁页**拷贝是 **26.85 GB/s**(微基准 5.74 s/chunk)⇒
    每 chunk 固定成本 8.9 s → ~5.4 s。**不额外占内存**(只是锁住已有页,不复制)。
    幂等;失败只告警不抛(继续用 pageable)。`XIAOTU_GPF_PIN=0` 可关。
    """
    if _PIN_DONE["n"] >= 0 or os.environ.get("XIAOTU_GPF_PIN", "1") == "0":
        return
    try:
        n = int(engine.pin_hostbufs())
        _PIN_DONE["n"] = n
        print(f"[vllm-xtu-moe] GPU 预填充:引擎 host 分片锁页 {n} 个缓冲"
              f"(DMA 16-21 → 26.85 GB/s)", flush=True)
    except Exception as exc:  # noqa: BLE001
        _PIN_DONE["n"] = -1
        print(f"[vllm-xtu-moe] ⚠️ host 分片锁页不可用(继续 pageable): "
              f"{type(exc).__name__}: {exc}", flush=True)


def kmajor_from_engine_shards_noraw(engine, device, hidden: int, inter: int,
                                    n_experts: int, group_k: int, dst=None,
                                    e0: int = 0, e1: "int | None" = None):
    """【§459 / 方案 B】逐 NUMA node 直接转置进 K-major —— **不需要 raw 规范布局缓冲**。

    与 `kmajor_from_engine_shards` 的区别:后者先把各 node 分片重组进设备上的
    ``w13_raw``/``w2_raw``(canonical ``[E,2I,H/2]`` / ``[E,H,I/2]``,共 **3.16 GiB/rank**),
    再做一次**连续** K-major 转置;本函数**跳过重组**,把每个 node 的块直接转置写进
    ``dst`` 的对应**列区间**(见 `byte_transpose.ktranspose_into`)。

    几何(node n,``cr`` = rowsplit):gate 行 → ``dst[0][:, :, n·cr:(n+1)·cr)``;
    up 行 → ``dst[0][:, :, I+n·cr : I+(n+1)·cr)``;w2 → ``dst[2][:, :, n·cr2:(n+1)·cr2)``。
    因 ``rowsplit`` 是 64 的整数倍,每个 BN=64 块整体落在一个 node 内(§459)。

    ⚠️ 我一度以为"strided 写是稀疏的 ⇒ 会比旧路径慢" —— **实测是错的**:旧路径自己那次
    `copy_` 重组就是一遍 strided 搬运,新路径省掉它 ⇒ **两个维度都赚**。离线同层 A/B:
    **四个张量逐字节相同**、**364.91 → 341.31 ms(−23.6 ms/层)**、**峰值 24.92 → 20.70 GiB**;
    服务级:预检 staging **10.73 → 7.56 GiB**、16k prefill **510.3 → 527.0 tok/s**。

    ⇒ **默认开启**;`XIAOTU_GPF_KMADAPTIVE=0` 退回旧路径(行为逐字不变)。
    """
    from vllm_xiaotu_moe.byte_transpose import ktranspose_into

    # 正向证据:路径真的被走了(⚠️ env 桥在 setdefault 模式下对"已在环境中且相等"的键
    # **不打印**,所以"日志里没有这个键"≠"没生效";必须由路径自己吭一声)。
    global _KMADAPTIVE_LOGGED
    if not _KMADAPTIVE_LOGGED:
        _KMADAPTIVE_LOGGED = True
        print("[vllm-xtu-moe] kmajor 装配:逐 NUMA node 直填 K-major"
              "(方案 B,无 raw 缓冲)✓", flush=True)

    geo = engine.shard_geometry()
    ns = int(geo["ns"])
    if ns < 2 or not geo["w13_node_bytes"] or not geo["w2_node_bytes"]:
        return None
    H, I, E = int(hidden), int(inter), int(n_experts)
    rb13, rb2 = H // 2, I // 2
    gk = int(group_k) if int(group_k) > 0 else 1
    # 【微 ping/pong】只填 [e0, e1) 这个专家区间 ⇒ `dst` 可以是 **tile 形状**(Et 行)。
    # 为什么切片是合法的:引擎宿主分片的**专家维是 dim 0**(见上面的 `buf.view(E, 2, c13)`),
    # 所以 `[_e0:_e1]` 是连续子块,后续 view/转置全部按 Et 形状走,与整层路径逐字同构 ✓
    # 默认 (0, None) ⇒ Et == E ⇒ **与改动前完全等价** ✓
    _e0 = max(0, int(e0))
    _e1 = E if e1 is None else min(int(e1), E)
    Et = max(0, _e1 - _e0)
    if Et <= 0:
        return None
    # 【2026-10-16 goal-A1 · 废除"按层缓冲"】此处原有 `if dst is None:` ⇒ 每次新建
    # (Et, rb13, 2I)/(Et, H/gk, 2I)/(Et, rb2, H)/(Et, I/gk, H) 四块**整层**缓冲。
    # 那就是用户 2026-10-15 明令废除的"按层缓冲(太消耗显存)"✗ ⇒ **已删除** ✓
    # ⭐ 现在 `dst` 必须由调用方(tile 环)提供 —— 环是唯一形态 ✓
    # 为什么安全:`mixed_experts.py` 里唯一的 `dst=None` 整层兜底已同批删除 ✓;
    #   实测 `logs/glm53_b550.log` 中 `gp-tile` 2 次、整层标记 0 次 ⇒ 运行期从不走整层 ✓
    if dst is None:
        raise RuntimeError(
            "[vllm-xtu-moe] ⛔ kmajor_from_engine_shards 收到 dst=None:"
            "按层缓冲已于 2026-10-16 废除(goal-A1),必须由 tile 环提供 dst ✗"
            " —— 若在此看到本行,说明有调用方仍在走【整层】路径,请改走 tile 环 ✓"
        )
    c13, cr13 = int(geo["w13_cbytes"]), int(geo["w13_crows"])
    c2, cr2 = int(geo["w2_cbytes"]), int(geo["w2_crows"])
    # 【2026-10-04 goal-A 诊断】把 `dma`/`tr` 两相接**实际生效路径**上。
    # WHY:`mixed_experts.py` 打出的 `asm=` 是 **sync 口径** —— 计时末 `torch.cuda.synchronize()`
    # 等的是**设备上全部工作(含与之并发的 attention/GEMM)**,所以那 250 ms/层是被污染的 ✗
    # (本文件 `_stage_begin` 的注释记着同一坑:sync 口径 537 ms vs **event 口径 167–206 ms**,差 ~3×)
    # ⇒ 只有 event 口径才能裁决"装配卡在 H2D 还是卡在转置" ✓
    # 关闭 `XIAOTU_GPF_STAGE` 时 `_stage_begin` 只返回 `perf_counter()`(不建事件)✓ 零开销 ✓
    # ⚠️ 暂存仍是"每张量一份"(_dma_hostbuf 内部 _reuse),同一 stream 上 DMA/copy 有序 ⇒ 复用安全
    # 【微 ping/pong】tile 模式:只搬该专家区间的字节 —— §18 那个 ×4 流量就是在这里修的
    _tiled = Et < E
    _per13 = 2 * c13                       # 每专家 w13 node 字节
    _per2 = c2                             # 每专家 w2 node 字节
    _s13_per = 2 * I * (H // gk)           # 每专家 w13 scale 字节
    _s2_per = H * (I // gk)                # 每专家 w2 scale 字节
    for n in range(ns):
        _t = _stage_begin()
        if _tiled:
            buf = _dma_hostbuf_range(engine, 0, n, device, _e0 * _per13, Et * _per13)
            if buf is None:
                return None
            blk = buf.view(Et, 2, c13)
        else:
            buf = _dma_hostbuf(engine, 0, n, int(geo["w13_node_bytes"]), device)
            blk = buf.view(E, 2, c13)
        _stage_mark(_t, "dma")
        c0 = n * cr13
        # gate 块 -> K-major 的列 [c0, c0+cr13);up 块 -> 列 [I+c0, I+c0+cr13)
        _t = _stage_begin()
        ktranspose_into(blk[:, 0, :].reshape(Et, cr13, rb13),
                        dst[0][:, :, c0:c0 + cr13])
        ktranspose_into(blk[:, 1, :].reshape(Et, cr13, rb13),
                        dst[0][:, :, I + c0:I + c0 + cr13])
        _stage_mark(_t, "asm")   # ⚠️ 不能用 "tr":tr 会**收割并复位**累加器 ⇒ 每 node 复位一次 ✗
        del buf, blk
    for n in range(ns):
        _t = _stage_begin()
        if _tiled:
            buf = _dma_hostbuf_range(engine, 1, n, device, _e0 * _per2, Et * _per2)
            if buf is None:
                return None
            w2blk = buf.view(Et, cr2, rb2)
        else:
            buf = _dma_hostbuf(engine, 1, n, int(geo["w2_node_bytes"]), device)
            w2blk = buf.view(E, cr2, rb2)
        _stage_mark(_t, "dma")
        c0 = n * cr2
        _t = _stage_begin()
        ktranspose_into(w2blk, dst[2][:, :, c0:c0 + cr2])
        _stage_mark(_t, "asm")
        del buf
    # scales 只有 ~0.13 / 0.07 GiB,且引擎给的是一整块(非按 node)⇒ 沿用原来的连续路径
    _t = _stage_begin()
    if _tiled:
        _b13 = _dma_hostbuf_range(engine, 2, 0, device, _e0 * _s13_per, Et * _s13_per)
        _b2 = _dma_hostbuf_range(engine, 3, 0, device, _e0 * _s2_per, Et * _s2_per)
        if _b13 is None or _b2 is None:
            return None
        s13_raw = _b13.view(Et, 2 * I, H // gk)
        s2_raw = _b2.view(Et, H, I // gk)
    else:
        s13_raw = _dma_hostbuf(engine, 2, 0, int(geo["w13_scale_bytes"]), device) \
            .view(E, 2 * I, H // gk)
        s2_raw = _dma_hostbuf(engine, 3, 0, int(geo["w2_scale_bytes"]), device) \
            .view(E, H, I // gk)
    _stage_mark(_t, "dma")
    _t = _stage_begin()
    _kmajor_bytes(s13_raw, dst[1])
    _kmajor_bytes(s2_raw, dst[3])
    _stage_mark(_t, "tr")   # key == "tr" ⇒ 收割并打印本层已完成的 dma/tr 两相 ✓
    return tuple(dst)


def kmajor_from_engine_shards(engine, device, hidden: int, inter: int,
                              n_experts: int, group_k: int, dst=None,
                              e0: int = 0, e1: "int | None" = None):
    """K-major device weights built from the engine's OWN host buffers.

    WHY (dev-docs/report/tuning/NOTES.md §459): the GPU prefill path used to stream the
    checkpoint source tensors, which forced ``XIAOTU_RELEASE_SOURCE=0`` and cost an
    extra ~269 GiB of host memory (measured: EngineCore hit 1073 GB RSS before
    finishing the load, with ~3 GB free per NUMA node). But the engine already owns
    a complete copy of the expert weights -- as per-NUMA-node *compact* shards that
    together are exactly one dense copy, laid out per expert as
    ``[gate cbytes][up cbytes]`` with node n holding gate rows
    ``[n*crows,(n+1)*crows)`` and up rows ``[I+n*crows, I+(n+1)*crows)``
    (see ``shard_fill_w13``/``shard_fill_w2``) -- plus its own copy of the scales.

    So we DMA those and reassemble the canonical layout on the DEVICE. The
    reassembly is just ``view``+``cat``: node n's gate block IS the contiguous row
    range ``[n*crows,(n+1)*crows)`` of the canonical ``[E, I, H/2]`` gate matrix,
    so concatenating the nodes along dim 1 reproduces it exactly.

    Returns ``(w13_t, s13_t, w2_t, s2_t)`` already K-major (ready for a
    ``PrefetchSlot``), or ``None`` when the engine has no shards (e.g. NOSHARD) --
    the caller then falls back to the source tensors.
    """
    # 【2026-10-04 goal-A 修】把引擎的 host 分片**就地锁页一次**。
    # 为什么必须在这里:默认走的是下面的 `noraw`/kmadaptive 分支
    # (`XIAOTU_GPF_KMADAPTIVE` 默认 "1" ⇒ 直接 return),而**那条路径没有这个调用** ✗
    # ⇒ 装配一直在 **pageable** 内存上跑:实测生产 worker 的 `VmLck` 全程 **0.0 GiB**,
    # 装配速率 **15.4 GB/s**,而作者在本文件里实测 pinned 为 **26.85 GB/s**
    # (原话:"每 chunk 固定成本 **8.9 s → ~5.4 s**")⇒ 白丢 ~1.65× 的预填吞吐 ✗
    # 证据:锁页的成功/失败消息在当前生产与我所有测试实例的日志里都是 **0 行**,
    # 但在旧日志(`abB2…abB10.log`)里有 `[pin] host 分片锁页: 10/10 个缓冲成功` ✓
    # 本调用幂等、失败只告警(自动退回 pageable),且**已有 env 门控** `XIAOTU_GPF_PIN=0`
    # ⇒ A/B 不需要新增开关 ✓
    _pin_engine_hostbufs(engine)
    # 【方案 B 开关】默认 **0 = 走下面这条旧路径**(行为逐字不变);=1 才逐 node 直填、
    # 省掉 raw 缓冲。取舍见 kmajor_from_engine_shards_noraw 的文档。
    if os.environ.get("XIAOTU_GPF_KMADAPTIVE", "1") != "0":
        return kmajor_from_engine_shards_noraw(
            engine, device, hidden, inter, n_experts, group_k, dst, e0, e1)
    if int(e0) > 0 or (e1 is not None and int(e1) < int(n_experts)):
        # 【微 ping/pong】旧 raw 路径的 `w13_raw`/`w2_raw` 是**按整层 E 复用**的共享缓冲
        # (`_reuse(("raw13",), (E, 2*I, rb13), …)`)⇒ 它**只支持整层**。
        # 请求了专家区间就明确返回 None(调用方据此回退),**绝不允许半填** ✗
        return None
    geo = engine.shard_geometry()
    ns = int(geo["ns"])
    if ns < 2 or not geo["w13_node_bytes"] or not geo["w2_node_bytes"]:
        return None
    H, I, E = int(hidden), int(inter), int(n_experts)
    rb13 = H // 2
    rb2 = I // 2
    gk = int(group_k) if int(group_k) > 0 else 1

    # Assemble by copy_ into the RAW target's row ranges: both sides are contiguous
    # runs of `cbytes` bytes, so each node's part is a plain contiguous copy.
    # (Measured: `view`+`cat` of the strided gate/up views then one transpose cost
    # **894 ms/layer**, because cat of strided views falls back to a slow kernel;
    # this form keeps the copies contiguous. See NOTES §466.)
    import time as _time
    _a0 = None
    if os.environ.get("XIAOTU_GPF_STAGE") == "1":
        import torch as _t0
        _a0 = _t0.cuda.memory_allocated(device)
    _t_dma = _stage_begin()
    w13_raw = _reuse(("raw13",), (E, 2 * I, rb13), device)
    c13 = int(geo["w13_cbytes"])
    cr13 = int(geo["w13_crows"])
    for n in range(ns):
        buf = _dma_hostbuf(engine, 0, n, int(geo["w13_node_bytes"]), device)
        blk = buf.view(E, 2, c13)
        c0 = n * cr13
        w13_raw[:, c0:c0 + cr13, :].copy_(blk[:, 0, :].reshape(E, cr13, rb13))
        w13_raw[:, I + c0:I + c0 + cr13, :].copy_(blk[:, 1, :].reshape(E, cr13, rb13))
        del buf, blk

    w2_raw = _reuse(("raw2",), (E, H, rb2), device)
    c2 = int(geo["w2_cbytes"])
    cr2 = int(geo["w2_crows"])
    for n in range(ns):
        buf = _dma_hostbuf(engine, 1, n, int(geo["w2_node_bytes"]), device)
        c0 = n * cr2
        w2_raw[:, c0:c0 + cr2, :].copy_(buf.view(E, cr2, rb2))
        del buf

    s13_raw = _dma_hostbuf(engine, 2, 0, int(geo["w13_scale_bytes"]), device) \
        .view(E, 2 * I, H // gk)
    s2_raw = _dma_hostbuf(engine, 3, 0, int(geo["w2_scale_bytes"]), device) \
        .view(E, H, I // gk)
    # [§594] 到此为止 = "DMA 发射 + 跨步组装" 都排在同一条流上;下面 sync 一次把它们
    # 一起结掉,所以先记 dma,再单独量转置(转置只在 GPU 上,不含 DMA)。
    _t_dma_end = _stage_mark(_t_dma, "dma")
    # K-major outputs are returned FRESH: `_kmajor_bytes` is a fast contiguous
    # transpose (73 ms total), whereas `reused.copy_(t.transpose(1,2))` is a
    # strided read and cost ~126 ms extra (NOTES §466). Reuse is worth it for the
    # big INTERMEDIATES (node DMA buffers + raw, ~13 GB of churn); not here.
    if dst is not None:
        # 【§597】写进调用方给的**复用缓冲**(环形槽),避免每层新分配 3.589 GiB ——
        # 实测每层新分配会让 `reserved` 单调上涨、free 跌破预填充阈值,
        # 于是第 8 层起逐层 DISABLED、退化成比纯 CPU 还慢的混合模式(76.5 s/13.8K)。
        _kmajor_bytes(w13_raw, dst[0])
        _kmajor_bytes(s13_raw, dst[1])
        _kmajor_bytes(w2_raw, dst[2])
        _kmajor_bytes(s2_raw, dst[3])
        _stage_mark(_t_dma_end, "tr")
        if _a0 is not None:
            import torch as _t2
            _d = (_t2.cuda.memory_allocated(device) - _a0) / 2**20
            print(f"[gpf-delta] staging_alloc={_d:+.1f} MiB (dst 复用路径)", flush=True)
        return tuple(dst)
    res = (_kmajor_bytes(w13_raw), _kmajor_bytes(s13_raw),
           _kmajor_bytes(w2_raw), _kmajor_bytes(s2_raw))
    _stage_mark(_t_dma_end, "tr")
    if _a0 is not None:
        import torch as _t1
        _d = (_t1.cuda.memory_allocated(device) - _a0) / 2**20
        print(f"[gpf-delta] staging_alloc={_d:+.1f} MiB", flush=True)
    return res




def prebuild_pinned_kmajor(tensors: list, workers: int = 6) -> None:
    """两阶段预建 K-major 缓存(见 dev-docs/report/tuning/TRIED_AND_REVERTED.md R4)。

    阶段 1(**纯 CPU**):并行做字节转置并放进缓存(不锁页)。任何时刻都安全,
      包括 CUDA graph 捕获期间,所以它能在 warmup 期间就跑起来 —— 这也是启动
      速度的关键:否则服务路径会单线程地逐层转置+锁页,启动要多花好几分钟。
    阶段 2(**CUDA**):等捕获静止(``wait_capture_done``)后并行 ``cudaHostRegister``。
      捕获窗口里调用它会作废捕获(`cudaErrorStreamCaptureInvalidated`,启动失败),
      甚至让 libcuda segfault(`cuMemHostRegister_v2` → SIGSEGV)。
    """
    from concurrent.futures import ThreadPoolExecutor

    def _transpose(quad):
        for t in quad:
            if t is not None:
                try:
                    _kmajor_cached(t)
                except Exception as e:  # noqa: BLE001
                    print(f"[xiaotu] prebuild transpose failed: {type(e).__name__}: {e}",
                          flush=True)

    def _register(quad):
        for t in quad:
            if t is not None:
                try:
                    wait_no_capture()
                    _pinned_kmajor(t)
                except Exception as e:  # noqa: BLE001
                    print(f"[xiaotu] prebuild pin failed: {type(e).__name__}: {e}",
                          flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_transpose, tensors))
    if not wait_capture_done():
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_register, tensors))


# --------------------------------------------------------------------------
# Prefetch-ahead: overlap layer L+1's H2D with layer L's compute.
#
# A small ring of device slots per (device, shape) is filled by an async copy on
# a dedicated side stream; the compute stream only waits on the slot's ready
# event. Reuse of a slot is guarded by the consumer's ``busy`` event so a copy
# never overwrites bytes the GPU is still reading.
# --------------------------------------------------------------------------
class PrefetchSlot:
    __slots__ = ("bufs", "ready", "busy", "t0", "t1")

    def __init__(self):
        self.bufs = None
        self.ready = None
        self.busy = None
        self.t0 = None   # XIAOTU_GP_TIMING: H2D 起止事件(量服务里真实传输时间)
        self.t1 = None

    def alloc(self, tensors, device):
        self.alloc_shapes(tuple(tuple(t.shape) for t in tensors), device)

    def alloc_shapes(self, shapes, device):
        shapes = tuple(tuple(int(v) for v in sh) for sh in shapes)
        if self.bufs is None or tuple(tuple(b.shape) for b in self.bufs) != shapes:
            self.bufs = tuple(
                torch.empty(s, dtype=torch.uint8, device=device) for s in shapes
            )


def slot_for_shapes(tensors, device, nslots: int = 0):
    """按**形状**取/建全局环形槽(所有层共享同一 key),并 alloc 出复用缓冲。

    与 `prefetch_layer` 共用 `_SLOTS`。**注意**:`prefetch_layer` 的"≥2 槽"下限是因为
    它在**侧流上为下一层预取**,单槽会覆盖正在用的权重(正确性 bug);
    而 **engine-shards 的同步路径** staging 与 GEMM 都在**同一条流**上严格有序
    ⇒ **单槽是安全的**,且能省 3.59 GiB(§601)。
    返回的 `slot.bufs` 可直接当 `kmajor_from_engine_shards(..., dst=slot.bufs)` 的目标。
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    shapes = tuple(tuple(t.shape) if hasattr(t, "shape") else tuple(t) for t in tensors)
    key = (dev.index,) + shapes
    slots = _SLOTS.get(key)
    if slots is None:
        if nslots and int(nslots) > 0:
            n = max(1, int(nslots))
        else:
            n = max(2, int(os.environ.get("XIAOTU_MOE_PREFETCH_SLOTS", "2") or 2))
        slots = [PrefetchSlot() for _ in range(n)]
        _SLOTS[key] = slots
    i = _RING.get(key, 0)
    _RING[key] = i + 1
    slot = slots[i % len(slots)]
    slot.alloc_shapes(shapes, dev)
    return slot


_SLOTS: dict[tuple, list] = {}
_RING: dict[tuple, int] = {}
_PREFETCH_STREAMS: dict[int, torch.cuda.Stream] = {}
_PREFETCH_DISABLED = False


def _prefetch_stream(device: torch.device) -> torch.cuda.Stream:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    s = _PREFETCH_STREAMS.get(idx)
    if s is None:
        s = torch.cuda.Stream(device=device)
        _PREFETCH_STREAMS[idx] = s
    return s


def prefetch_layer(w13, s13, w2, s2, device):
    """Issue the async H2D for one layer's K-major raw weights; return a slot.

    ``w13/s13/w2/s2`` must already be the K-major host (pinned) tensors, i.e.
    ``_pinned_kmajor(...)`` of the layer's checkpoint tensors (optionally an
    expert-parallel slice of them).

    Returns ``None`` when overlap is not possible (weights already on device, or
    the device is too full to hold the ping-pong buffers) — the caller then
    falls back to the synchronous per-layer H2D path.
    """
    global _PREFETCH_DISABLED
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if _PREFETCH_DISABLED or w13.device == dev:
        return None
    key = (dev.index, tuple(w13.shape), tuple(s13.shape),
           tuple(w2.shape), tuple(s2.shape))
    slots = _SLOTS.get(key)
    if slots is None:
        # 环深度可配(默认 2 = 一层的预取窗口)。更深 = 给 H2D 更多提前量,代价是
        # VRAM:每个槽 ≈ 一层权重(TP=2 每 rank 1.7 GB,TP=1 3.4 GB)。
        # 【下限必须是 2,不能用 1 槽省显存】实测(/tmp/test_1slot.py,2026-09-11):
        # 只有 1 槽时,为 L+1 发起的 H2D 会覆盖 L 正在用的那份权重 ⇒ 本层被算成
        # 下一层的权重(err_vs_L=17.5 vs err_vs_L+1=0.096),是**正确性 bug**。
        # 想省这 ~3GB 请改 XIAOTU_MOE_RESIDENT_BUDGET_GB(少放常驻层),不要动这里。
        nslots = max(2, int(os.environ.get("XIAOTU_MOE_PREFETCH_SLOTS", "2") or 2))
        slots = [PrefetchSlot() for _ in range(nslots)]
        _SLOTS[key] = slots
    i = _RING.get(key, 0)
    _RING[key] = i + 1
    slot = slots[i % len(slots)]
    try:
        slot.alloc((w13, s13, w2, s2), dev)
    except RuntimeError as e:  # includes torch.cuda.OutOfMemoryError
        if "out of memory" not in str(e).lower():
            raise
        # 【2026-10-27】OOM 现场取证（不改变控制流：取证后仍走原来的降级路径）
        try:
            from . import mem_guard
            mem_guard.on_oom(sys.modules[__name__], e, where="prefetch_slot.alloc")
        except Exception:                     # noqa: BLE001
            pass
        # vLLM sized the KV cache first; there is no room for ping-pong buffers.
        # Drop the (now useless) slots and disable overlap for this process.
        _SLOTS.pop(key, None)
        _PREFETCH_DISABLED = True
        print("[xiaotu] GPU prefill: no VRAM for ping-pong slots, "
              "falling back to synchronous H2D (no overlap)", flush=True)
        return None
    st = _prefetch_stream(dev)
    if slot.busy is not None:
        st.wait_event(slot.busy)
    timing = os.environ.get("XIAOTU_GP_TIMING") == "1"
    if timing:
        slot.t0 = torch.cuda.Event(enable_timing=True)
        slot.t1 = torch.cuda.Event(enable_timing=True)
        slot.t0.record(st)
    with torch.cuda.stream(st):
        for buf, src in zip(slot.bufs, (w13, s13, w2, s2)):
            buf.copy_(src, non_blocking=True)
        slot.ready = torch.cuda.Event()
        slot.ready.record(st)
        if timing:
            slot.t1.record(st)
    return slot


def gpu_moe_layer(
    x, topk_ids, topk_weights,
    w13, s13, w2, s2,
    H: int, I: int, K: int,
    *,
    device: torch.device,
    slot: "PrefetchSlot | None" = None,
    out_dtype=None,
    swiglu_limit: float = 0.0,
    group_k: int = 32,
    scale_kind: "str | None" = None,   # "e4m3"(NVFP4) | "e8m0"(MXFP4)；None=按 dtype 推断
    s13_ts=None,
    s2_ts=None,
    out=None,
):
    """One layer's routed MoE on GPU (K-major raw bytes, in-kernel fp4 dequant).

    Streams this layer's raw MXFP4 weights H2D once (from cached pinned copies,
    a real DMA), then transposes the bytes to K-major on the GPU so the grouped
    GEMM kernels load ``b`` as ``[BK, BN]`` with no in-loop transpose.

    If ``slot`` is given it is a device buffer already filled (or being filled)
    by ``prefetch_layer``: we only wait for its ready event, so this layer's H2D
    overlaps the previous layer's kernels.
    """
    T = x.shape[0]
    E = w13.shape[0]
    device = torch.device(device) if not isinstance(device, torch.device) else device
    _t0 = time.perf_counter() if _GP_TIMING else 0.0
    # ⚠️ 顺序要紧：**先按原 dtype 判 kind，再换成 uint8 视图** ——
    #   反过来会让下面的 `_dt` 看到 uint8 而误判成 e8m0（实测：inter≈1e-20、out=0，
    #   2026-10-26 离线单测抓到的第一个假阴性）
    _dt = getattr(s13, "dtype", None)
    _kind = scale_kind or ("e4m3" if _dt == torch.float8_e4m3fn else "e8m0")
    if _dt == torch.float8_e4m3fn:
        s13 = s13.view(torch.uint8)      # fp8 ⇒ uint8 视图（字节不变，内核按字节解码）
    if getattr(s2, "dtype", None) == torch.float8_e4m3fn:
        s2 = s2.view(torch.uint8)
    if slot is not None:
        # 【不要在图捕获期间 wait_event】常驻层的 slot.ready 是在**捕获外**
        # (构建时)record 的,图内等待它会触发 cudaErrorStreamCaptureIsolation
        # ("dependency created on uncaptured work in another stream"),整个捕获作废。
        # 常驻槽位的数据在构建时就已写完(阻塞 H2D + 之后早已完成的事件),图内
        # 不需要这个等待;非捕获时仍然保留(预取路径靠它保证顺序)。
        # 见 dev-docs/report/tuning/TRIED_AND_REVERTED.md R14。
        if not torch.cuda.is_current_stream_capturing():
            torch.cuda.current_stream(device).wait_event(slot.ready)
        w13_t, s13_t, w2_t, s2_t = slot.bufs
    elif w13.device == device:
        w13_t = _kmajor_bytes(w13); s13_t = _kmajor_bytes(s13)
        w2_t = _kmajor_bytes(w2); s2_t = _kmajor_bytes(s2)
    else:
        w13_t = _kmajor_bytes(_pinned(w13).to(device, non_blocking=True))
        s13_t = _kmajor_bytes(_pinned(s13).to(device, non_blocking=True))
        w2_t = _kmajor_bytes(_pinned(w2).to(device, non_blocking=True))
        s2_t = _kmajor_bytes(_pinned(s2).to(device, non_blocking=True))

    # ⭐ `XIAOTU_MOE_STEPTRACE=1`：捕获期挂死定位（分点打印，最后一条 = 挂在那之前的阶段）
    _tr = os.environ.get("XIAOTU_MOE_STEPTRACE", "0") == "1"

    def _trp(tag):
        if _tr:
            print(f"[step-trace] {tag} T={T} E={E} A={A if 'A' in dir() else '?'} "
                  f"capture={torch.cuda.is_current_stream_capturing()}", flush=True)

    _trp("A: 进入 gpu_moe_layer")
    tok, wts, seg_start, A = _build_segmentation(topk_ids, topk_weights, E, device)
    _t1 = time.perf_counter() if _GP_TIMING else 0.0
    _trp("B: segmentation 完成")
    if out is None:
        out = torch.zeros((T, H), dtype=out_dtype or torch.bfloat16, device=device)
    else:
        # ⭐ 调用方给**持久缓冲**（地址稳定）⇒ 图捕获的 eager-break 段可以安全复用
        out.zero_()
    _trp("C: out 分配完成")
    if T == 0 or K == 0:
        # 形状判空(不是数据判空):A 现在是固定的 T*K,空批之外不会为 0。
        # 全部 id 无效时它们都落进垃圾桶段,每个专家的段为空 ⇒ 内核不写 out。
        return out
    # 【inter 需要 A=T*K 行,不是 T 行】gate_up 内核用**排序位置** g_rows 索引
    # inter(`inter_ptr + g_rows*inter_ld`),只有写 out 时才换成 token id。
    # 曾经按 T 行分配 ⇒ 越界写,cudaErrorIllegalAddress(第 19 轮实测)。
    # 想省这块分配只能改成"排序后行数=有效项数"或持久缓冲,不能再动行数语义。
    # 【§598】复用:见 `_reuse_moe` 的注释。`XIAOTU_GPF_REUSE=0` 可回退到每层新分配。
    if os.environ.get("XIAOTU_GPF_REUSE", "1") == "1" and slot is None:
        inter = _reuse_moe("inter", (A, 2 * I), torch.bfloat16, device)
    elif slot is not None:
        # PIN 分流(slot):垃圾桶(-1)行 kernel 不写 ⇒ 环/empty 残留(inf/nan)
        # 会以 0×inf=NaN 进入 partial ⇒ slot 路径专属 zeros 并禁用复用环。
        inter = torch.zeros((A, 2 * I), dtype=torch.bfloat16, device=device)
    else:
        inter = torch.empty((A, 2 * I), dtype=torch.bfloat16, device=device)
    if os.environ.get("XIAOTU_GPF_STAGE") == "1":
        import torch as _tp
        print(f"[gpf-ptr] out_ptr={out.data_ptr()} inter_ptr={inter.data_ptr()} "
              f"alloc={_tp.cuda.memory_allocated(device)/2**20:.1f}MiB "
              f"reserved={_tp.cuda.memory_reserved(device)/2**20:.1f}MiB", flush=True)

    W13_E = w13_t.stride(0)
    S13_E = s13_t.stride(0)
    W2_E = w2_t.stride(0)
    S2_E = s2_t.stride(0)
    BM = int(os.environ.get("XIAOTU_GPU_PREFILL_BM", "64"))
    BN = int(os.environ.get("XIAOTU_GPU_PREFILL_BN", "64"))
    BK = int(os.environ.get("XIAOTU_GPU_PREFILL_BK", "64"))
    BH = int(os.environ.get("XIAOTU_GPU_PREFILL_BH", "64"))
    NS = int(os.environ.get("XIAOTU_GPU_PREFILL_STAGES", "2"))
    # Kernel variant: "split" avoids the per-k-block join+trans+reshape by
    # splitting the activation tile into even/odd k columns and doing two dots.
    _variant = os.environ.get("XIAOTU_GPU_PREFILL_KERNEL", "base")
    _lut = os.environ.get("XIAOTU_GPU_PREFILL_LUT", "0") == "1"
    NW = int(os.environ.get("XIAOTU_GPU_PREFILL_WARPS", "4"))
    lut_t = _e2m1_table(device)
    # ⭐⭐ NVFP4（A78-4，2026-10-26）：`GKS`/`SC_E4M3`/`TS` 这三个 constexpr 一直存在
    #   却**从来没被传过** ⇒ 默认 e8m0 + GKS=32 ⇒ 喂 e4m3 组尺度会静默错数 ✗
    #   现在按尺度 dtype 自动选：fp8_e4m3 = NVFP4(组 16) / uint8 = MXFP4(组 32, e8m0) ✓
    #   `s13_ts`/`s2_ts` 非空 ⇒ 每专家一个 fp32 二级尺度(weight_scale_2)，经 ts_ptr 传内核。
    # ⚠️ 内核要的是**原始字节**（`_e4m3_byte_to_f32` 做位运算）⇒ 传进去的尺度必须是
    #   uint8；fp8 张量会让 `tl.load` 得到 fp8 类型、位运算直接编译失败
    #   （实测 `cannot cast fp8e4nv[...] to int32`，2026-10-26）。
    #   而**真模型里 NVFP4 的尺度参数本来就是 uint8**（e4m3 的位模式）⇒ 无法用 dtype 区分
    #   NVFP4/MXFP4 ⇒ 加显式 `scale_kind`（None 时按 dtype 推断，兼容旧调用）。
    _sc_e4m3 = 1 if _kind == "e4m3" else 0
    _has_ts = s13_ts is not None
    _ts13 = s13_ts if _has_ts else s13_t
    _ts2 = s2_ts if _has_ts else s2_t
    _nvkw = dict(GKS=int(group_k), SC_E4M3=_sc_e4m3, HAS_TS=1 if _has_ts else 0)
    if _variant == "split":
        _gate_up_kernel_split[(E, triton.cdiv(2 * I, BN))](
            x, x.stride(0), tok, seg_start,
            w13_t, w13_t.stride(1), s13_t, s13_t.stride(1),
            inter, inter.stride(0), W13_E, S13_E,
            H=H, BM=BM, BN=BN, BK=BK, NS=NS, LUT=_lut, num_warps=NW,
            ts_ptr=_ts13, **_nvkw,
        )
        _down_kernel_split[(E, triton.cdiv(H, BH))](
            inter, inter.stride(0), tok, wts, seg_start,
            w2_t, w2_t.stride(1), s2_t, s2_t.stride(1),
            out, out.stride(0), W2_E, S2_E,
            H=H, I=I, BM=BM, BH=BH, BK=BK, NS=NS, LUT=_lut,
            LIMIT=float(swiglu_limit), num_warps=NW,
            ts_ptr=_ts2, **_nvkw,
        )
    else:
        if os.environ.get("XIAOTU_MOE_BISECT_SEG") == "1":
            # 只做 segmentation（不 launch 任何 MoE 内核）——二分定位用
            _trp("D: 内核 launch 完成")
            return out
        _gate_up_kernel[(E, triton.cdiv(2 * I, BN))](
            x, x.stride(0), tok, seg_start,
            w13_t, w13_t.stride(1), s13_t, s13_t.stride(1),
            inter, inter.stride(0), W13_E, S13_E, lut_t,
            H=H, BM=BM, BN=BN, BK=BK, NS=NS, LUT=_lut, num_warps=NW,
            ts_ptr=_ts13, **_nvkw,
        )
        if os.environ.get("XIAOTU_MOE_BISECT_UP") == "1":
            # 只做 gate_up（跳过 down + 原子加）——二分定位用
            _trp("D: 内核 launch 完成")
            return out
        _down_kernel[(E, triton.cdiv(H, BH))](
            inter, inter.stride(0), tok, wts, seg_start,
            w2_t, w2_t.stride(1), s2_t, s2_t.stride(1),
            out, out.stride(0), W2_E, S2_E, lut_t,
            H=H, I=I, BM=BM, BH=BH, BK=BK, NS=NS, LUT=_lut,
            LIMIT=float(swiglu_limit), num_warps=NW,
            ts_ptr=_ts2, **_nvkw,
        )
    if os.environ.get("XIAOTU_GPF_STAGE") == "1":
        # 诊断：内核到底有没有写数据（inter=gate_up 输出、out=最终输出）
        print(f"[gpf-stage] T={T} A={A} variant={_variant} E={E} I={I} H={H} "
              f"GKS={group_k} e4m3={_sc_e4m3} has_ts={_has_ts} "
              f"inter.max={float(inter.abs().max()) if T else 0.0} "
              f"out.max={float(out.abs().max()) if T else 0.0}", flush=True)
    if _GP_TIMING:
        _gp_add(T, A, _t1 - _t0, time.perf_counter() - _t1)
        if slot is not None and getattr(slot, "t0", None) is not None:
            try:   # H2D 传输时间(事件已完成:本层已在等 ready)
                _gp_h2d(slot.t0.elapsed_time(slot.t1))
            except Exception:  # noqa: BLE001
                pass
    if slot is not None:
        # Mark the slot free only after these kernels have actually finished.
        # ⭐ 2026-10-20:旧实现每次调用都 new 一个 Event;PIN 路径每层每 forward 调一次
        #   ⇒ 长时间压测累计大量 CUDA event(与 as_slot 同族泄漏)。复用同一个 event。
        # ⛔【捕获期不要 record event】MUSA 上在**捕获中的流**里 record 事件，
        #   会让 `cudaStreamEndCapture` 返回 invalid argument（实测：关掉 MoE 断点后
        #   两 rank 都捕获成功；开着就有一侧失败）。事件只用于"层用完可以复用"的调度，
        #   常驻层在捕获后本来就不走预取 ⇒ 捕获期跳过它，语义不变 ✓
        if not torch.cuda.is_current_stream_capturing():
            if getattr(slot, "busy", None) is None:
                slot.busy = torch.cuda.Event()
            slot.busy.record(torch.cuda.current_stream(device))
    _trp("D: 内核 launch 完成")
    return out


def _moe_tile_launch(x, tok, wts, seg_local, w13_t, s13_t, w2_t, s2_t,
                     inter_view, out, lut_t, H, I, *, BM, BN, BK, BH, NS,
                     variant, NW, LUT):
    """按【专家 tile】发一次 gate_up + down。

    ⭐ 为什么可以只传视图(内核**零改动**):`_gate_up_kernel`/`_down_kernel` 的
    `pid_e` 只通过 `seg_ptr` 与专家 stride(`W*_E`)寻址,而 `inter`/`tok`/`wts`
    全部按**全局排序行号** `g_rows` 索引 —— 而按专家切分后,一个 tile 拥有的行
    正好是连续区间 `[seg_start[e0], seg_start[e1])`
    ⇒ 把 `tok/wts/inter` **先切好视图**、把 `seg` 平移到 0 起点,
    内核看到的局部 `g_rows`(0-based)与视图基址相加后 == 全局行 ✓
    `out` 在内核里是 `tl.atomic_add` ⇒ 多次 launch 自然累加 ✓
    """
    E_t = int(w13_t.shape[0])
    W13_E, S13_E = w13_t.stride(0), s13_t.stride(0)
    W2_E, S2_E = w2_t.stride(0), s2_t.stride(0)
    if variant == "split":
        _gate_up_kernel_split[(E_t, triton.cdiv(2 * I, BN))](
            x, x.stride(0), tok, seg_local,
            w13_t, w13_t.stride(1), s13_t, s13_t.stride(1),
            inter_view, inter_view.stride(0), W13_E, S13_E,
            H=H, BM=BM, BN=BN, BK=BK, NS=NS, LUT=LUT, num_warps=NW,
            ts_ptr=s13_t,
        )
        _down_kernel_split[(E_t, triton.cdiv(H, BH))](
            inter_view, inter_view.stride(0), tok, wts, seg_local,
            w2_t, w2_t.stride(1), s2_t, s2_t.stride(1),
            out, out.stride(0), W2_E, S2_E,
            H=H, I=I, BM=BM, BH=BH, BK=BK, NS=NS, LUT=LUT, num_warps=NW,
            ts_ptr=s2_t,
        )
    else:
        _gate_up_kernel[(E_t, triton.cdiv(2 * I, BN))](
            x, x.stride(0), tok, seg_local,
            w13_t, w13_t.stride(1), s13_t, s13_t.stride(1),
            inter_view, inter_view.stride(0), W13_E, S13_E, lut_t,
            H=H, BM=BM, BN=BN, BK=BK, NS=NS, LUT=LUT, num_warps=NW,
            ts_ptr=s13_t,
        )
        _down_kernel[(E_t, triton.cdiv(H, BH))](
            inter_view, inter_view.stride(0), tok, wts, seg_local,
            w2_t, w2_t.stride(1), s2_t, s2_t.stride(1),
            out, out.stride(0), W2_E, S2_E, lut_t,
            H=H, I=I, BM=BM, BH=BH, BK=BK, NS=NS, LUT=LUT, num_warps=NW,
            ts_ptr=s2_t,
        )


def gpu_moe_layer_tiled(x, topk_ids, topk_weights, w13, s13, w2, s2,
                        H: int, I: int, K: int, *, device,
                        tile_e: int = 0, out: "torch.Tensor | None" = None):
    """【微 ping/pong】把一个整层的 routed MoE 按【专家 tile】分几次算完。

    与 `gpu_moe_layer` 的唯一区别是**粒度**:后者要求整层的 4 份 K-major 权重
    同时驻留设备(现行 = 2 槽 × 整层 ⇒ 生产 preflight 7.56 GiB/rank),本函数
    允许每次只让 `tile_e` 个专家驻留 ⇒ 槽可缩小到 `tile_e/E`。

    ⚠️ 本函数**只负责计算**;权重的分块 DMA 由调用方按同样的 tile 边界准备
    (`kmajor_from_engine_shards(..., e0=, e1=)`),这样搬运与计算可以用同一个
    环深度做流水,保持"下一 tile 的 H2D 与本 tile 的内核重叠"。

    `tile_e<=0 或 >=E` ⇒ 退化成**整层一次算完**(与 `gpu_moe_layer` 等价),
    便于先做数值等价性验证再开流水。
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    T = x.shape[0]
    E = int(w13.shape[0])
    G = int(tile_e) if 0 < int(tile_e) < E else E

    tok_all, wts_all, seg_start, A = _build_segmentation(
        topk_ids, topk_weights, E, device)
    if out is None:
        out = torch.zeros((T, H), dtype=torch.bfloat16, device=device)
    if T == 0 or K == 0:
        return out

    if os.environ.get("XIAOTU_GPF_REUSE", "1") == "1":
        inter = _reuse_moe("inter", (A, 2 * I), torch.bfloat16, device)
    else:
        inter = torch.empty((A, 2 * I), dtype=torch.bfloat16, device=device)

    BM = int(os.environ.get("XIAOTU_GPU_PREFILL_BM", "64"))
    BN = int(os.environ.get("XIAOTU_GPU_PREFILL_BN", "64"))
    BK = int(os.environ.get("XIAOTU_GPU_PREFILL_BK", "64"))
    BH = int(os.environ.get("XIAOTU_GPU_PREFILL_BH", "64"))
    NS = int(os.environ.get("XIAOTU_GPU_PREFILL_STAGES", "2"))
    variant = os.environ.get("XIAOTU_GPU_PREFILL_KERNEL", "base")
    LUT = os.environ.get("XIAOTU_GPU_PREFILL_LUT", "0") == "1"
    NW = int(os.environ.get("XIAOTU_GPU_PREFILL_WARPS", "4"))
    lut_t = _e2m1_table(device)

    for e0 in range(0, E, G):
        e1 = min(e0 + G, E)
        row0, row1 = int(seg_start[e0].item()), int(seg_start[e1].item())
        if row1 <= row0:
            continue                                  # 本 tile 无路由 ⇒ 跳过(省一次 launch)
        seg_local = (seg_start[e0:e1 + 1] - row0).contiguous()
        tok_t = tok_all[row0:row1]
        wts_t = wts_all[row0:row1]
        # ⭐ 视图切片:`inter` 按全局行索引 ⇒ 视图基址偏移后局部 g_rows 仍指向同一行 ✓
        inter_view = inter[row0:row1]
        _moe_tile_launch(
            x, tok_t, wts_t, seg_local,
            w13[e0:e1], s13[e0:e1], w2[e0:e1], s2[e0:e1],
            inter_view, out, lut_t, int(H), int(I),
            BM=BM, BN=BN, BK=BK, BH=BH, NS=NS,
            variant=variant, NW=NW, LUT=LUT,
        )
    return out


_TILE_RINGS: dict = {}
_LANDING: dict = {}
_LANDING_BUF: dict = {}


def prealloc_tile_ring(device, E, H, I, group_k=32, nslots: int = 2) -> int:
    """在 KV 定容**之前**把 tile 环真分配出来（返回字节数）。

    ⭐【2026-10-27】tile 环原来"首次真实预填才分配" ⇒ 0.84 GiB 落在 vLLM 账外，
    稳态 free 被它吃掉。现在 profile 期就分配 ⇒ 进 `total_consumed` ⇒ KV 据实让位。
    形状与运行时 `_stage_tiles` 内**同一表达式**（tile/gk/rb13/rb2），不得另写一套。
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    tile = staging_tile_experts(int(E))
    if not tile:
        return 0
    gk = int(group_k) if int(group_k) > 0 else 1
    rb13, rb2 = int(H) // 2, int(I) // 2
    shapes = ((tile, rb13, 2 * int(I)), (tile, int(H) // gk, 2 * int(I)),
              (tile, rb2, int(H)), (tile, int(I) // gk, int(H)))
    # ⚠️【2026-10-27 修正 · 实测教训】落点**不能**塞进同一个 `_tile_ring` 调用里！
    #   我上一版 `nslots+1=3` 去分配，而运行时用的是 `XIAOTU_GPF_STAGE_NSLOTS`(默认 **2**)
    #   ⇒ 两者的缓存键不同 ⇒ 运行期**又分配了一整环**：实测 `tile_ring=2.101 GiB/20 块`
    #   （期望 1.26/12），凭空多出 0.84 GiB 账外增长 ✗。
    #   现在：环按**运行时的 nslots** 预分配（键必须逐字一致），落点单独一块。
    if nslots is None:
        try:
            nslots = int(os.environ.get("XIAOTU_GPF_STAGE_NSLOTS", "2") or 2)
        except (TypeError, ValueError):
            nslots = 2
    ring = _tile_ring(device, shapes, max(2, int(nslots)))
    # 落点：大小 = 一个槽的字节数；`which` 顺序与 `_dma_hostbuf` 调用点一致
    # (0=w13, 1=w2, 2=w13_scale, 3=w2_scale)；node 分片比槽大时 `_dma_hostbuf` 自动回退。
    _lk = (str(device), "landing", shapes)
    land = _LANDING_BUF.get(_lk)
    if land is None:
        _n = sum(int(t.numel()) for t in ring[0])
        land = torch.empty(_n, dtype=torch.uint8, device=device)
        _LANDING_BUF[_lk] = land
    _off = 0
    _views = {}
    for _i in range(4):
        _n2 = int(ring[0][_i].numel())
        _views[_i] = land[_off:_off + _n2]
        _off += _n2
    for _w, _i in {0: 0, 1: 2, 2: 1, 3: 3}.items():
        _LANDING[(str(device), _w)] = _views[_i]
    return (sum(int(t.numel()) * int(t.element_size()) for slot in ring for t in slot)
            + int(land.numel()))


def _tile_ring(device, shapes, nslots):
    """⭐ 【微 ping/pong】**进程级复用**的 tile 环(与 `_SLOTS`/`PrefetchSlot` 同一条纪律)。

    为什么**必须**复用、**不能**在每次层调用里现分配 —— 两条独立理由:
      1. `mixed_experts.py:1831-1833` 已归档同一条教训:每层新分配 ~3.6 GiB 会造成
         **不可回收的碎片**(`reserved` 单调上涨);
      2. ⭐ **更危险的是跨流生命周期**:本环由**侧流**写入,却按**主流**登记到 caching
         allocator ⇒ 函数返回、Python 释放之后,allocator 可能把这块内存**再发出去**,
         而侧流的写入**还没跑完** ⇒ **use-after-free ⇒ `cudaErrorIllegalAddress`** ✗
         —— 这正是 2026-10-09 上机那次崩溃的形态(非法访问出现在**转置相**,
         且**早于任何 tile 内核**,见 `REEVAL_STAGING_TILING` §15.2)✓
    复用之后环**永不释放** ⇒ 该竞态**结构上不可能**发生 ✓
    而且这与 `staging_bytes()` 的口径一致 —— 它算的本来就是**常驻**量 ✓
    """
    key = (str(device), tuple(tuple(int(v) for v in s) for s in shapes), int(nslots))
    r = _TILE_RINGS.get(key)
    if r is None:
        r = [tuple(torch.empty(s, dtype=torch.uint8, device=device) for s in shapes)
             for _ in range(int(nslots))]
        _TILE_RINGS[key] = r
        per = sum(int(s[0]) * int(s[1]) * int(s[2]) for s in shapes)
        print(f"[gp-tile] 分配 tile 环(进程级复用,不再每次层调用现分配):"
              f"tile={shapes[0][0]} nslots={nslots} 每槽={per / 2**20:.0f} MiB "
              f"合计={per * int(nslots) / 2**30:.2f} GiB", flush=True)
    return r


def gpu_moe_layer_from_engine_tiled(engine, x, topk_ids, topk_weights,
                                    H: int, I: int, E: int, K: int, *, device,
                                    group_k: int = 32, nslots: int = 2, out=None):
    """【微 ping/pong】整层 routed MoE:按专家 tile 做 **DMA ‖ 计算** 流水(权重取自引擎宿主分片)。

    与现行机制的关系(重要):
      * **现行**:`mixed_experts.py` 用 **2 个【整层】槽**做**跨层** ping/pong
        (下一层 DMA ‖ 本层计算)⇒ 驻留 = 2 × 整层 = 6.72 GiB/rank(生产 preflight 7.56)
      * **本条**:同一个重叠思想**下沉到层内**,槽只装 `tile_e` 个专家
        ⇒ 驻留 ∝ `tile_e/E`(tile=96/384 ⇒ 1.89 GiB/rank)
      * 环深 `nslots >= 2` 是**正确性下限** —— 单槽会被"下一个 tile 的 DMA"覆盖掉
        本 tile 正在算的权重(实测过:1 槽 ⇒ 本层被算成下一层的权重,`prefetch_layer`
        的注释与 2026-09-11 的 `/tmp/test_1slot.py` 都记着这条)✗

    调度(每个 tile 只与**同一条侧流**交互 ⇒ `_dma_hostbuf` 的复用仍然安全 ✓):

        issue(t)   : 侧流上 wait(该槽的上一个 free 事件) → kmajor_from_engine_shards(…, e0,e1)
                     → record ready
        compute(t) : 主流 wait(ready[t]) → `_moe_tile_launch`(共享 out,原子累加)
                     → record free[t]

    `tile_e<=0/>=E` ⇒ 只有 1 个 tile,等价于整层一次算完(便于先做数值对拍)✓
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    T = x.shape[0]
    E = int(E)
    tiles = stage_tiles(E)
    if not tiles:
        return None
    tile = staging_tile_experts(E)
    gk = int(group_k) if int(group_k) > 0 else 1
    rb13, rb2 = int(H) // 2, int(I) // 2
    nslots = max(2, int(nslots))

    # 环:每个槽按 **最大 tile** 形状分配;最后一个(较小的)tile 用 `[:Et]` 视图 ✓
    shapes = ((tile, rb13, 2 * int(I)), (tile, int(H) // gk, 2 * int(I)),
              (tile, rb2, int(H)), (tile, int(I) // gk, int(H)))
    ring = _tile_ring(device, shapes, nslots)
    free = [None] * nslots
    ready = [None] * nslots

    tok_all, wts_all, seg_start, A = _build_segmentation(
        topk_ids, topk_weights, E, device)
    if out is None:
        out = torch.zeros((T, H), dtype=torch.bfloat16, device=device)
    if T == 0 or K == 0:
        return out
    if os.environ.get("XIAOTU_GPF_REUSE", "1") == "1":
        inter = _reuse_moe("inter", (A, 2 * int(I)), torch.bfloat16, device)
    else:
        inter = torch.empty((A, 2 * int(I)), dtype=torch.bfloat16, device=device)

    BM = int(os.environ.get("XIAOTU_GPU_PREFILL_BM", "64"))
    BN = int(os.environ.get("XIAOTU_GPU_PREFILL_BN", "64"))
    BK = int(os.environ.get("XIAOTU_GPU_PREFILL_BK", "64"))
    BH = int(os.environ.get("XIAOTU_GPU_PREFILL_BH", "64"))
    NS = int(os.environ.get("XIAOTU_GPU_PREFILL_STAGES", "2"))
    variant = os.environ.get("XIAOTU_GPU_PREFILL_KERNEL", "base")
    LUT = os.environ.get("XIAOTU_GPU_PREFILL_LUT", "0") == "1"
    NW = int(os.environ.get("XIAOTU_GPU_PREFILL_WARPS", "4"))
    lut_t = _e2m1_table(device)

    cur = torch.cuda.current_stream(device)
    side = _prefetch_stream(device)

    def _issue(i: int) -> bool:
        s = i % nslots
        e0, e1 = tiles[i]
        if free[s] is not None:
            side.wait_event(free[s])          # 该槽上次的使用者(主流上的内核)必须先完成
        with torch.cuda.stream(side):
            km = kmajor_from_engine_shards(
                engine, device, int(H), int(I), E, gk,
                dst=tuple(b[:e1 - e0] for b in ring[s]), e0=e0, e1=e1)
            ev = torch.cuda.Event()
            ev.record(side)
        if km is None:
            return False
        ready[s] = ev
        return True

    if not _issue(0):
        return None

    # ⭐ 每层**只做一次**设备→主机同步取段边界(原来是每 tile 一次 `.item()`
    #   ⇒ 4 tile × 40 层 × 4 块 ≈ **640 次同步**,整条流水被反复打断)✓
    _bounds = seg_start.tolist()

    for i, (e0, e1) in enumerate(tiles):
        s = i % nslots
        if ready[s] is None:                  # 该槽的 DMA 没排上(引擎无分片)⇒ 整体放弃
            return None
        cur.wait_event(ready[s])
        row0, row1 = int(_bounds[e0]), int(_bounds[e1])
        if row1 > row0:
            seg_local = (seg_start[e0:e1 + 1] - row0).contiguous()
            b = ring[s]
            _moe_tile_launch(
                x, tok_all[row0:row1], wts_all[row0:row1], seg_local,
                b[0][:e1 - e0], b[1][:e1 - e0], b[2][:e1 - e0], b[3][:e1 - e0],
                inter[row0:row1], out, lut_t, int(H), int(I),
                BM=BM, BN=BN, BK=BK, BH=BH, NS=NS,
                variant=variant, NW=NW, LUT=LUT,
            )
        fe = torch.cuda.Event()
        fe.record(cur)
        free[s] = fe
        ready[s] = None
        if i + 1 < len(tiles):
            if not _issue(i + 1):
                return None
    return out


def torch_reference_layer(
    x, topk_ids, topk_weights, w13, s13, w2, s2, H: int, I: int,
) -> torch.Tensor:
    """Pure-torch reference of the SAME MXFP4 layout for golden testing."""
    T = x.shape[0]
    K = topk_ids.shape[1]
    dev = x.device
    E = w13.shape[0]
    vals = _E2M1.to(dev)

    def exp_dequant(e):
        lo13 = (w13[e, :, :] & 0x0F).to(torch.int64)
        hi13 = ((w13[e, :, :] >> 4) & 0x0F).to(torch.int64)
        w13e = torch.stack([vals[lo13], vals[hi13]], dim=-1).reshape(2 * I, H)
        sc13 = torch.exp2(s13[e].float() - 127.0).repeat_interleave(32, dim=1)[..., :H]
        W13e = (w13e * sc13).to(x.dtype)
        lo2 = (w2[e, :, :] & 0x0F).to(torch.int64)
        hi2 = ((w2[e, :, :] >> 4) & 0x0F).to(torch.int64)
        w2e = torch.stack([vals[lo2], vals[hi2]], dim=-1).reshape(H, I)
        sc2 = torch.exp2(s2[e].float() - 127.0).repeat_interleave(32, dim=1)[..., :I]
        W2e = (w2e * sc2).to(x.dtype)
        return W13e, W2e

    cache = {}
    out = torch.zeros(T, H, dtype=x.dtype, device=dev)
    for t in range(T):
        for ssl in range(K):
            e = int(topk_ids[t, ssl])
            if e not in cache:
                cache[e] = exp_dequant(e)
            W13e, W2e = cache[e]
            gate = x[t] @ W13e[:I].t()
            up = x[t] @ W13e[I:].t()
            act = torch.nn.functional.silu(gate) * up
            out[t] += topk_weights[t, ssl] * (act @ W2e.t())
    return out
