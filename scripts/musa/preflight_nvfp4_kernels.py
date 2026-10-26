#!/usr/bin/env python3
"""NVFP4 GPU(Triton) MoE 路径的**离线数值单测** —— 不加载模型，几秒钟出结果。

为什么需要它
------------
`gpu_moe_layer` 的 NVFP4 接线有三处极易错、且错了**不会报错**（只出垃圾数）：
  1. `SC_E4M3=1`（尺度是 1 字节 e4m3，不是 e8m0）
  2. `GKS=16`（NVFP4 组大小；MXFP4 是 32）
  3. `ts_ptr`（每专家 fp32 `weight_scale_2`，经 `HAS_TS=1`）
而验证它通常要跑一次 8 分钟的模型加载 + 预热 ⇒ 试错成本极高。
本脚本用**小形状 + torch 参考实现**在几秒内把这三处钉死。

用法（远端 GPU 机器）
---------------------
    /root/venvs/vllm310/bin/python scripts/musa/preflight_nvfp4_kernels.py
    ... --variant split          # 测另一个内核变体（base/split）
    ... --T 64                   # 更大的 token 数（默认 8；不影响正确性）

判据：打印的相对误差应 ≈ 1e-2（bf16 累加）以内；退出码 0 = 通过。
"""

from __future__ import annotations

import argparse
import os
import sys


def _pack_fp4(vals):
    """把 [.., k] 的 fp4 值（-6..6，E2M1 网格）打包成 [.., k//2] uint8（低半 k=2j，高半 k=2j+1）。"""
    import torch

    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    a = vals.abs()
    # 注意：`a[:, :, None]` 在 3 维张量上是 (E,2I,1,H)，会广播错维 ⇒ 用 unsqueeze(-1)
    idx = (a.unsqueeze(-1) - grid).abs().argmin(dim=-1).to(torch.uint8)
    sign = (vals < 0).to(torch.uint8) << 3
    code = (idx | sign).to(torch.uint8)
    lo, hi = code[..., 0::2], code[..., 1::2]
    return (lo | (hi << 4)).contiguous()


def _unpack_fp4(packed):
    """[.., k//2] uint8 -> [.., k] float32（与内核的 LUT 语义一致）。"""
    import torch

    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    lo = (packed & 0x0F).long()
    hi = ((packed >> 4) & 0x0F).long()
    out = torch.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=torch.float32)
    out[..., 0::2] = lut[lo]
    out[..., 1::2] = lut[hi]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--I", type=int, default=64)
    ap.add_argument("--E", type=int, default=4)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--gks", type=int, default=16)
    ap.add_argument("--variant", default="base", help="base | split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trivial", action="store_true", help="权重/尺度/ts 全 1，用来二分定位")
    args = ap.parse_args()

    import torch

    os.environ.setdefault("XIAOTU_GPU_PREFILL_KERNEL", args.variant)
    from vllm_xiaotu_moe.gpu_prefill import gpu_moe_layer

    torch.manual_seed(args.seed)
    dev = torch.device("cuda", torch.cuda.current_device())
    H, I, E, K, GKS = args.H, args.I, args.E, args.K, args.gks
    assert H % GKS == 0 and I % GKS == 0 and H % 2 == 0 and I % 2 == 0

    def rnd(shape, scale=1.0):
        return (torch.randn(*shape) * scale).to(torch.float32)

    # ── 权重（fp4 值，打包成字节）与尺度（e4m3 字节 + 每专家二级尺度）──────────
    if args.trivial:
        w13_f = torch.full((E, 2 * I, H // 2), 0x02, dtype=torch.uint8)   # 全部 +1.0
        w2_f = torch.full((E, H, I // 2), 0x02, dtype=torch.uint8)
    else:
        w13_f = _pack_fp4(rnd((E, 2 * I, H)))      # [E, 2I, H/2]
        w2_f = _pack_fp4(rnd((E, H, I)))           # [E, H, I/2]
    if args.trivial:
        s13_f = torch.full((E, 2 * I, H // GKS), 1.0).to(torch.float8_e4m3fn)
        s2_f = torch.full((E, H, I // GKS), 1.0).to(torch.float8_e4m3fn)
        ts13 = torch.ones(E)
        ts2 = torch.ones(E)
    else:
        s13_f = rnd((E, 2 * I, H // GKS), 0.5).to(torch.float8_e4m3fn)   # 1 字节/组
        s2_f = rnd((E, H, I // GKS), 0.5).to(torch.float8_e4m3fn)
        ts13 = rnd((E,), 0.2).abs() + 0.1          # 每专家 fp32 二级尺度
        ts2 = rnd((E,), 0.2).abs() + 0.1

    # ── 参考实现（纯 torch，逐位解码同一批字节）─────────────────────────────
    w13 = _unpack_fp4(w13_f)                       # [E, 2I, H]
    w2 = _unpack_fp4(w2_f)                         # [E, H, I]
    s13 = s13_f.to(torch.float32)                  # [E, 2I, H/GKS]
    s2 = s2_f.to(torch.float32)
    d13 = w13 * s13.repeat_interleave(GKS, dim=2) * ts13.view(E, 1, 1)
    d2 = w2 * s2.repeat_interleave(GKS, dim=2) * ts2.view(E, 1, 1)

    x = rnd((args.T, H)).to(torch.bfloat16)
    logits = rnd((args.T, E))
    wts, ids = torch.topk(logits.softmax(-1), K, dim=-1)
    wts = wts.to(torch.float32)

    ref = torch.zeros((args.T, H), dtype=torch.float32)
    xf = x.to(torch.float32)
    for t in range(args.T):
        acc = torch.zeros(H, dtype=torch.float32)
        for j in range(K):
            e = int(ids[t, j])
            gate = xf[t] @ d13[e, :I, :].T
            up = xf[t] @ d13[e, I:, :].T
            h = torch.nn.functional.silu(gate) * up
            acc += float(wts[t, j]) * (h @ d2[e].T)
        ref[t] = acc

    # 诊断 1：K-major 字节转置是否真的搬了数据（若返回全 0 ⇒ 内核读到全 0 ⇒ 输出恒 0）
    from vllm_xiaotu_moe.gpu_prefill import _kmajor_bytes
    _w13t = _kmajor_bytes(w13_f.to(dev))
    _s13t = _kmajor_bytes(s13_f.view(torch.uint8).to(dev))
    print(f"[dbg] w13_t.max={int(_w13t.max())} s13_t.max={int(_s13t.max())} "
          f"w13_t.shape={tuple(_w13t.shape)} s13_t.shape={tuple(_s13t.shape)}")

    # 诊断：分段是否为空（空段 ⇒ 内核直接 return，输出恒 0）
    from vllm_xiaotu_moe.gpu_prefill import _build_segmentation
    _tok, _wts, _seg, _A = _build_segmentation(ids.to(torch.int32).to(dev),
                                               wts.to(dev), E, dev)
    print(f"[dbg] seg_start={_seg.tolist()} A={int(_A)} ids={ids.tolist()}")

    out = gpu_moe_layer(
        x.to(dev), ids.to(torch.int32).to(dev), wts.to(dev),
        w13_f.to(dev), s13_f.to(dev), w2_f.to(dev), s2_f.to(dev),
        H=H, I=I, K=K, device=dev,
        group_k=GKS, s13_ts=ts13.to(dev), s2_ts=ts2.to(dev),
        # ⚠️ MUSA 的 Triton **不支持 bf16 的 atomic_add**（实测 `atomic_add does not support bf16`）
        #    ⇒ 输出必须 fp32（FP8 路径一直如此，见 gpu_prefill_fp8.gpu_moe_layer 的注释）
        out_dtype=torch.float32,
    )
    got = out.to(torch.float32).cpu()

    denom = ref.abs().mean().clamp_min(1e-6)
    err = (got - ref).abs().mean() / denom
    print(f"[preflight] variant={args.variant} T={args.T} H={H} I={I} E={E} K={K} GKS={GKS}")
    print(f"[preflight] ref|mean|={ref.abs().mean():.4e} got|mean|={got.abs().mean():.4e}")
    print(f"[preflight] NaN 个数 = {int(torch.isnan(got).sum())} / {got.numel()}")
    print(f"[preflight] 相对误差 = {float(err):.4e}")
    ok = float(err) < 5e-2
    print("[preflight]", "✅ 数值一致" if ok else "⛔ 数值不一致（查 SC_E4M3/GKS/ts_ptr 三处接线）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
