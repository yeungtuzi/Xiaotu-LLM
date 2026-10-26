"""挎斗 N-major GEMV（MXFP4，qlen=1）：只算命中 slot，按权重求和。

plane 布局（由 K-major `[S,K,N]` 转置而来）：
  w13_t [S, 2I, H//2], s13_t [S, 2I, H//32]
  w2_t  [S, H, I//2],  s2_t  [S, H, I//32]
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _deq(nib):
    nib = nib.to(tl.int32)
    sign = (nib >> 3) & 1
    ex = (nib >> 1) & 3
    man = nib & 1
    val_n = ((ex + 126) << 23).to(tl.float32, bitcast=True) * (1.0 + 0.5 * man.to(tl.float32))
    val = tl.where(ex == 0, 0.5 * man.to(tl.float32), val_n)
    return tl.where(sign == 1, -val, val)


@triton.jit
def _gateup(x_ptr, w13_ptr, s13_ptr, slot_ptr, part_ptr,
            H2: tl.constexpr, N2: tl.constexpr, G2: tl.constexpr,
            BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    pn = tl.program_id(1)
    slot = tl.load(slot_ptr + pid)
    n = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k in range(0, H2 * 2, BK):
        b = tl.load(w13_ptr + slot * N2 * H2 + n[:, None] * H2 + (k // 2 + tl.arange(0, BK // 2))[None, :])
        lo = (b & 0x0F).to(tl.int32)
        hi = ((b >> 4) & 0x0F).to(tl.int32)
        w = _deq(tl.interleave(lo, hi))
        sg = tl.load(s13_ptr + slot * N2 * G2 + n[:, None] * G2 + (k // 32 + tl.arange(0, BK // 32))[None, :]).to(tl.float32)
        sc = tl.exp2(sg - 127.0)
        w = tl.reshape(tl.reshape(w, (BN, BK // 32, 32)) * sc[:, :, None], (BN, BK))
        x = tl.load(x_ptr + k + tl.arange(0, BK)).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(part_ptr + pid * N2 + n, acc)


@triton.jit
def _act(part_ptr, inter_ptr, I: tl.constexpr, N2: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    pn = tl.program_id(1)
    i = pn * BN + tl.arange(0, BN)
    g = tl.load(part_ptr + pid * N2 + i)
    u = tl.load(part_ptr + pid * N2 + I + i)
    tl.store(inter_ptr + pid * I + i, (g * tl.sigmoid(g) * u).to(tl.bfloat16))


@triton.jit
def _down(inter_ptr, w2_ptr, s2_ptr, slot_ptr, part_out_ptr,
          H: tl.constexpr, I2: tl.constexpr, G2: tl.constexpr,
          BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    pn = tl.program_id(1)
    slot = tl.load(slot_ptr + pid)
    n = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for i in range(0, I2 * 2, BK):
        b = tl.load(w2_ptr + slot * H * I2 + n[:, None] * I2 + (i // 2 + tl.arange(0, BK // 2))[None, :])
        lo = (b & 0x0F).to(tl.int32)
        hi = ((b >> 4) & 0x0F).to(tl.int32)
        w = _deq(tl.interleave(lo, hi))
        sg = tl.load(s2_ptr + slot * H * G2 + n[:, None] * G2 + (i // 32 + tl.arange(0, BK // 32))[None, :]).to(tl.float32)
        sc = tl.exp2(sg - 127.0)
        w = tl.reshape(tl.reshape(w, (BN, BK // 32, 32)) * sc[:, :, None], (BN, BK))
        x = tl.load(inter_ptr + pid * (I2 * 2) + i + tl.arange(0, BK)).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(part_out_ptr + pid * H + n, acc)


@triton.jit
def _wsum(part_out_ptr, w_ptr, out_ptr, H: tl.constexpr, BN: tl.constexpr, NP: tl.constexpr):
    pn = tl.program_id(0)
    n = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for p in range(NP):
        acc += tl.load(part_out_ptr + p * H + n) * tl.load(w_ptr + p)
    tl.store(out_ptr + n, acc)


def transpose_planes(pool):
    """K-major [S,K,N] → N-major [S,N,K]；返回 (w13t,s13t,w2t,s2t)。"""
    p = pool.planes
    return (p[0].transpose(1, 2).contiguous(), p[1].transpose(1, 2).contiguous(),
            p[2].transpose(1, 2).contiguous(), p[3].transpose(1, 2).contiguous())


def forward(x: torch.Tensor, slot_ids: torch.Tensor, weights: torch.Tensor,
            planes_t, H: int, I: int, GK: int,
            BN: int = 4, BK: int = 256, bufs=None) -> torch.Tensor:
    """x [H] bf16, slot_ids [n] int32, weights [n] fp32 → out [H] fp32。

    bufs=(part, inter, part_out, out) 可复用（外部按 max_hits 预分配）。
    """
    n = int(slot_ids.numel())
    dev = x.device
    w13t, s13t, w2t, s2t = planes_t
    N2, H2, G2 = 2 * I, H // 2, H // 32
    if bufs is None:
        part = torch.empty(n, N2, dtype=torch.float32, device=dev)
        inter = torch.empty(n, I, dtype=torch.bfloat16, device=dev)
        part_out = torch.empty(n, H, dtype=torch.float32, device=dev)
        out = torch.empty(H, dtype=torch.float32, device=dev)
    else:
        part, inter, part_out, out = bufs
        part, inter, part_out = part[:n], inter[:n], part_out[:n]
    _gateup[(n, triton.cdiv(N2, BN))](x, w13t, s13t, slot_ids, part,
                                     H2=H2, N2=N2, G2=G2, BN=BN, BK=BK, num_warps=4)
    _act[(n, triton.cdiv(I, 128))](part, inter, I=I, N2=N2, BN=128, num_warps=4)
    _down[(n, triton.cdiv(H, BN))](inter, w2t, s2t, slot_ids, part_out,
                                   H=H, I2=I // 2, G2=I // 32, BN=BN, BK=BK, num_warps=4)
    _wsum[(triton.cdiv(H, 128),)](part_out, weights, out, H=H, BN=128, NP=n, num_warps=4)
    return out
@triton.jit
def _gateup_t(x_ptr, w13_ptr, s13_ptr, slot_ptr, part_ptr,
              H2: tl.constexpr, N2: tl.constexpr, G2: tl.constexpr,
              BN: tl.constexpr, BK: tl.constexpr, KPAIR: tl.constexpr):
    pid = tl.program_id(0); pn = tl.program_id(1)
    slot = tl.load(slot_ptr + pid)
    t = pid // KPAIR
    n = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k in range(0, H2 * 2, BK):
        b = tl.load(w13_ptr + slot * N2 * H2 + n[:, None] * H2 + (k // 2 + tl.arange(0, BK // 2))[None, :])
        lo = (b & 0x0F).to(tl.int32); hi = ((b >> 4) & 0x0F).to(tl.int32)
        w = _deq(tl.interleave(lo, hi))
        sg = tl.load(s13_ptr + slot * N2 * G2 + n[:, None] * G2 + (k // 32 + tl.arange(0, BK // 32))[None, :]).to(tl.float32)
        sc = tl.exp2(sg - 127.0)
        w = tl.reshape(tl.reshape(w, (BN, BK // 32, 32)) * sc[:, :, None], (BN, BK))
        x = tl.load(x_ptr + t * (H2 * 2) + k + tl.arange(0, BK)).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(part_ptr + pid * N2 + n, acc)

@triton.jit
def _wsum_t(part_out_ptr, w_ptr, out_ptr, H: tl.constexpr, BN: tl.constexpr, NP: tl.constexpr):
    t = tl.program_id(0); pn = tl.program_id(1)
    n = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for p in range(NP):
        acc += tl.load(part_out_ptr + (t * NP + p) * H + n) * tl.load(w_ptr + t * NP + p)
    tl.store(out_ptr + t * H + n, acc)


def forward_batch(x, slot_ids, weights, planes_t, H, I, GK, KPAIR, BN=4, BK=256, bufs=None):
    T=int(x.shape[0]); n=T*KPAIR; dev=x.device
    w13t,s13t,w2t,s2t=planes_t
    N2,H2,G2=2*I,H//2,H//32
    if bufs is None:
        part=torch.empty(n,N2,dtype=torch.float32,device=dev)
        inter=torch.empty(n,I,dtype=torch.bfloat16,device=dev)
        part_out=torch.empty(n,H,dtype=torch.float32,device=dev)
        out=torch.empty(T,H,dtype=torch.float32,device=dev)
    else:
        part,inter,part_out,out,_=bufs
        part,inter,part_out=part[:n],inter[:n],part_out[:n]; out=out[:T]
    _gateup_t[(n, triton.cdiv(N2,BN))](x,w13t,s13t,slot_ids,part,H2=H2,N2=N2,G2=G2,BN=BN,BK=BK,KPAIR=KPAIR,num_warps=4)
    _act[(n, triton.cdiv(I,128))](part,inter,I=I,N2=N2,BN=128,num_warps=4)
    _down[(n, triton.cdiv(H,BN))](inter,w2t,s2t,slot_ids,part_out,H=H,I2=I//2,G2=I//32,BN=BN,BK=BK,num_warps=4)
    _wsum_t[(T, triton.cdiv(H,128))](part_out,weights,out,H=H,BN=128,NP=KPAIR,num_warps=4)
    return out
