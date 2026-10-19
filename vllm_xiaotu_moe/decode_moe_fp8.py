"""Decode 专用 FP8 MoE 内核(pair 中心,M=1)。

为什么单独写(2026-10-19):
  * `gpu_prefill_fp8.gpu_moe_layer_fp8` 是**prefill 内核**:grid=(E, 2I/bn),每个 (expert, 列块) 一个 block,
    且 tile 按大 M 设计。decode 时 M=1、活跃专家 ~10 ⇒ ① 512×12 个 block 绝大多数空转;
    ② 64 行 tile 只用了 1 行 ⇒ 算力空转 ~64×。实测 40 pairs 要 0.9 ms/层,而理论只要 ~12 µs。
  * 本模块把 grid 改为 **P = T×K(≤40)**:一个 block 负责一个 (token, expert) 对的完整 gemv,
    M=1、逐 128 列/行块向量化 ⇒ 发射数与活跃 pair 成正比,不再随 E 放大。

数值口径与 prefill 完全一致:e4m3 单字节 + **fp32 block-128** 尺度;
gate 在 `[0, I)`、up 在 `[I, 2I)`(与 `gpu_prefill_fp8` 的布局同源)。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm_xiaotu_moe.gpu_prefill_fp8 import _e4m3_uint8_to_f32


@triton.jit
def _gate_up_pairs_kernel(
    x_ptr, x_ld, ids_ptr, wts_ptr,
    w13_ptr, s13_ptr, acc_ptr,
    P, K, HS: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
    W13_E: tl.constexpr, S13_E: tl.constexpr,
    BH: tl.constexpr, BJ: tl.constexpr, LIMIT: tl.constexpr,
):
    p = tl.program_id(0)
    jb = tl.program_id(1)
    hs = tl.program_id(2)
    e = tl.load(ids_ptr + p)
    if e < 0:
        return
    t = p // K
    HCH: tl.constexpr = H // HS
    h_lo = hs * HCH
    if True:
        j0 = jb * BJ
        j = j0 + tl.arange(0, BJ)
        acc_g = tl.zeros((BJ,), dtype=tl.float32)
        acc_u = tl.zeros((BJ,), dtype=tl.float32)
        for h0 in range(h_lo, h_lo + HCH, BH):
            h = h0 + tl.arange(0, BH)
            x = tl.load(x_ptr + t * x_ld + h).to(tl.float32)
            hb = h0 // 128
            jb = j0 // 128
            # gate: w13[e, h, j];up: w13[e, h, I + j]
            wg = tl.load(w13_ptr + e * W13_E + h[:, None] * (2 * I) + j[None, :])
            wu = tl.load(w13_ptr + e * W13_E + h[:, None] * (2 * I) + (I + j)[None, :])
            sg = tl.load(s13_ptr + e * S13_E + hb * (2 * I // 128) + jb).to(tl.float32)
            su = tl.load(s13_ptr + e * S13_E + hb * (2 * I // 128) + (I // 128 + jb)).to(tl.float32)
            gw = _e4m3_uint8_to_f32(wg) * sg
            uw = _e4m3_uint8_to_f32(wu) * su
            acc_g += tl.sum(x[:, None] * gw, axis=0)
            acc_u += tl.sum(x[:, None] * uw, axis=0)
        tl.atomic_add(acc_ptr + p * (2 * I) + j, acc_g)
        tl.atomic_add(acc_ptr + p * (2 * I) + I + j, acc_u)


@triton.jit
def _act_pairs_kernel(acc_ptr, act_ptr, P, I: tl.constexpr,
                      BI: tl.constexpr, LIMIT: tl.constexpr):
    p = tl.program_id(0)
    ib = tl.program_id(1)
    i = ib * BI + tl.arange(0, BI)
    g = tl.load(acc_ptr + p * (2 * I) + i)
    u = tl.load(acc_ptr + p * (2 * I) + I + i)
    if LIMIT > 0.0:
        g = tl.minimum(g, LIMIT)
        u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
    a = (g / (1.0 + tl.exp(-g))) * u
    tl.store(act_ptr + p * I + i, a)


@triton.jit
def _down_pairs_kernel(
    act_ptr, ids_ptr, wts_ptr, w2_ptr, s2_ptr, out_ptr,
    out_ld, P, K, H: tl.constexpr, I: tl.constexpr,
    W2_E: tl.constexpr, S2_E: tl.constexpr,
    BH: tl.constexpr, BI: tl.constexpr,
):
    p = tl.program_id(0)
    e = tl.load(ids_ptr + p)
    if e < 0:
        return
    t = p // K
    w = tl.load(wts_ptr + p).to(tl.float32)
    act_off = p * I
    h0 = tl.program_id(1) * BH
    i0 = tl.program_id(2) * BI
    if True:
        h = h0 + tl.arange(0, BH)
        i = i0 + tl.arange(0, BI)
        a = tl.load(act_ptr + act_off + i).to(tl.float32)
        wb = tl.load(w2_ptr + e * W2_E + i[:, None] * H + h[None, :])
        sc = tl.load(s2_ptr + e * S2_E + (i0 // 128) * (H // 128) + (h0 // 128)).to(tl.float32)
        acc = tl.sum(a[:, None] * (_e4m3_uint8_to_f32(wb) * sc), axis=0)
        tl.atomic_add(out_ptr + t * out_ld + h, acc * w)


def gpu_moe_decode_fp8(x, topk_ids, topk_weights, w13t, s13t, w2t, s2t,
                       H: int, I: int, K: int, *, device,
                       swiglu_limit: float = 0.0,
                       BH: int = 128, BJ: int = 128, BI: int = 128):
    """decode 版:grid=P=T×K。返回 fp32 [T,H]。"""
    T = int(x.shape[0])
    if s13t.dtype == torch.uint8:
        s13t = s13t.view(torch.float32)
    if s2t.dtype == torch.uint8:
        s2t = s2t.view(torch.float32)
    ids = topk_ids.to(torch.int32).reshape(-1)
    wts = topk_weights.to(torch.float32).reshape(-1)
    P = int(ids.numel())
    out = torch.zeros((T, H), dtype=torch.float32, device=device)
    if P == 0:
        return out
    acc = torch.zeros((P, 2 * I), dtype=torch.float32, device=device)
    act = torch.empty((P, I), dtype=torch.float32, device=device)
    HS = 4 if H % (4 * BH) == 0 else 1
    _gate_up_pairs_kernel[(P, (I + BJ - 1) // BJ, HS)](
        x, int(x.stride(0)), ids, wts,
        w13t, s13t, acc,
        P, K, HS=HS, H=H, I=I,
        W13_E=int(w13t.stride(0)), S13_E=int(s13t.stride(0)),
        BH=BH, BJ=BJ, LIMIT=float(swiglu_limit),
        num_warps=4,
    )
    _act_pairs_kernel[(P, (I + BI - 1) // BI)](
        acc, act, P, I=I, BI=BI, LIMIT=float(swiglu_limit), num_warps=4)
    _down_pairs_kernel[(P, (H + BH - 1) // BH, (I + BI - 1) // BI)](
        act, ids, wts, w2t, s2t, out,
        int(out.stride(0)), P, K, H=H, I=I,
        W2_E=int(w2t.stride(0)), S2_E=int(s2t.stride(0)),
        BH=BH, BI=BI, num_warps=8,
    )
    return out
