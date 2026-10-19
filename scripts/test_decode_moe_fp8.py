#!/usr/bin/env python3
"""离线对拍 + 计时:`decode_moe_fp8.gpu_moe_decode_fp8` vs CPU 反量化参考。

用法:
  python scripts/test_decode_moe_fp8.py            # 小规模对拍 + E=512 计时
不启动服务、不占 GPU 以外的资源。
"""
import sys
import time

import torch

from vllm_xiaotu_moe.decode_moe_fp8 import gpu_moe_decode_fp8
from vllm_xiaotu_moe.gpu_prefill_fp8 import gpu_moe_layer_fp8


def deq(u8, scale, blk=128):
    """u8 [E, A, n] -> f32;scale [E, A/128, n/128](A、n 两个维度都按 128 分块)。"""
    f = u8.view(torch.float8_e4m3fn).to(torch.float32)
    E, A, n = f.shape
    f = f.reshape(E, A // blk, blk, n // blk, blk)
    f = f * scale[:, :, None, :, None]
    return f.reshape(E, A, n)


def ref_moe(x, ids, wts, w13, s13, w2, s2, H, I, K):
    """x[T,H] bf16(->f32),ids[T,K],wts[T,K] -> [T,H] f32。"""
    T = x.shape[0]
    g13 = deq(w13, s13).cpu()    # [E,H,2I]
    g2 = deq(w2, s2).cpu()       # [E,I,H]
    out = torch.zeros(T, H, dtype=torch.float32)
    xf = x.to(torch.float32).cpu()
    for t in range(T):
        for k in range(K):
            e = int(ids[t, k])
            if e < 0:
                continue
            g = xf[t] @ g13[e, :, :I]
            u = xf[t] @ g13[e, :, I:]
            a = torch.nn.functional.silu(g) * u
            out[t] += float(wts[t, k].cpu()) * (a @ g2[e])
    return out


def mk(E, H, I):
    dev = "cuda:0"
    w13 = (torch.randn((E, H, 2 * I), device=dev) * 0.3).to(
        torch.float8_e4m3fn).view(torch.uint8)
    s13 = torch.rand((E, H // 128, (2 * I) // 128), dtype=torch.float32, device=dev) * 0.05
    w2 = (torch.randn((E, I, H), device=dev) * 0.3).to(
        torch.float8_e4m3fn).view(torch.uint8)
    s2 = torch.rand((E, I // 128, H // 128), dtype=torch.float32, device=dev) * 0.05
    return w13, s13, w2, s2


def check():
    torch.manual_seed(0)
    E, H, I, T, K = 16, 256, 128, 4, 10
    dev = "cuda:0"
    w13, s13, w2, s2 = mk(E, H, I)
    x = torch.randn(T, H, dtype=torch.bfloat16, device=dev) * 0.1
    ids = torch.randint(0, E, (T, K), dtype=torch.int32, device=dev)
    wts = torch.rand(T, K, device=dev) * 0.2
    out = gpu_moe_decode_fp8(x, ids, wts, w13, s13, w2, s2, H, I, K, device=dev)
    ref = ref_moe(x, ids, wts, w13, s13, w2, s2, H, I, K)
    got = out.cpu()
    if not torch.isfinite(got).all():
        print(f"[check] ⛔ 输出含 NaN/Inf: {(~torch.isfinite(got)).sum().item()}")
        return float("inf")
    rel = (got - ref).norm() / ref.norm().clamp_min(1e-9)
    print(f"[check] T={T} K={K} E={E} H={H} I={I}  rms_rel={float(rel):.3e}")
    return float(rel)


def bench():
    E, H, I, K = 512, 2560, 384, 10
    dev = "cuda:0"
    w13, s13, w2, s2 = mk(E, H, I)
    for T in (1, 4, 16):
        x = torch.randn(T, H, dtype=torch.bfloat16, device=dev) * 0.1
        ids = torch.randint(0, E, (T, K), dtype=torch.int32, device=dev)
        wts = torch.rand(T, K, device=dev) * 0.2
        for _ in range(10):
            gpu_moe_decode_fp8(x, ids, wts, w13, s13, w2, s2, H, I, K, device=dev)
        torch.cuda.synchronize()
        t0 = time.time()
        n = 200
        for _ in range(n):
            gpu_moe_decode_fp8(x, ids, wts, w13, s13, w2, s2, H, I, K, device=dev)
        torch.cuda.synchronize()
        pairs = (time.time() - t0) / n * 1000
        # 旧 prefill 内核对照(同输入)
        for _ in range(5):
            gpu_moe_layer_fp8(x, ids, wts, w13, s13, w2, s2, H, I, K, device=dev)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            gpu_moe_layer_fp8(x, ids, wts, w13, s13, w2, s2, H, I, K, device=dev)
        torch.cuda.synchronize()
        old = (time.time() - t0) / n * 1000
        print(f"[bench] T={T:2d} pairs={T*K:3d}  新(pair): {pairs:.3f} ms/层   "
              f"旧(prefill grid=E): {old:.3f} ms/层   加速 {old/pairs:.1f}×")


if __name__ == "__main__":
    rel = check()
    if rel > 5e-2:
        print(f"⛔ 对拍失败 rms_rel={rel:.3e}")
        sys.exit(1)
    print("✓ 对拍通过")
    bench()
