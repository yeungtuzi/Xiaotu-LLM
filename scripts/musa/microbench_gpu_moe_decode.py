#!/usr/bin/env python3
"""单层 GPU MoE 的 **decode（T=1）微基准**：把"每步时间"拆成"算子 vs 框架"。

背景（2026-10-26）：48 层**全 GPU 驻留**跑通后，decode 仍只有 1.7 tok/s（578 ms/步），
而同一次运行的 prefill 是 51 ms/token；CPU 引擎路径在相近上下文是 271 ms/步。
⇒ 必须回答：**这 578 ms 里，我们自己的 MoE 内核占多少？**

做法：真实几何（E=256 / H=2560 / I=640 / K=10 / groupK=16），
把一层的 NVFP4 权重按引擎约定打包 → `_pinned_kmajor` → `PrefetchSlot`（=常驻路径），
然后在 T=1 上反复调用 `gpu_moe_layer(..., slot=slot)`，用 CUDA 事件计时。

判据：
* 若 `48 × ms_per_call ≪ 578 ms` ⇒ 时间在**框架**（eager 每步上千次发射）⇒ 该开 CUDA 图
* 若 `48 × ms_per_call ≈ 578 ms` ⇒ 时间在**我们的内核**（T=1 时权重带宽/发射密度）

用法：
    /root/venvs/vllm310/bin/python scripts/musa/microbench_gpu_moe_decode.py
    ... --T 1 --iters 100 --layers 48
"""

from __future__ import annotations

import argparse
import os
import sys


def _pack_fp4(vals):
    """[E, M, K] fp32 → [E, M, K/2] uint8（两个 fp4 挤一字节，低半字节在前）。"""
    import torch

    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32)
    v = vals.reshape(*vals.shape[:-1], vals.shape[-1] // 2, 2)
    d = (v.unsqueeze(-1) - table.view(1, 1, 1, 1, -1)).abs()
    idx = d.argmin(-1).to(torch.uint8)
    return (idx[..., 0] | (idx[..., 1] << 4)).contiguous()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--H", type=int, default=2560)
    ap.add_argument("--I", type=int, default=640)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--T", type=int, default=1)
    ap.add_argument("--gks", type=int, default=16)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--layers", type=int, default=48,
                    help="按多少层折算整步时间（本机 48 层）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    # ⚠️ 顺序/写法都很讲究（实测 2026-10-26 三次失败）：
    #   * `torch.cuda.current_device()` 在**没有 vllm 平台初始化**时会报
    #     `torch._C has no attribute _cuda_getDevice` ⇒ 这里**不查** current_device，
    #     直接用 `torch.device("cuda")`（下游 `PrefetchSlot.alloc` 只需设备对象）✓
    #   * 反过来，`import gpu_prefill`（→ vllm → deep_ep）必须在 **torch 已导入之后**，
    #     且不要再手工 `import torch_musa`（它会和 vllm 的平台 patch 打架）✓
    sys.path.insert(0, "/root/lvllm")
    from vllm_xiaotu_moe.gpu_prefill import (PrefetchSlot, _pinned_kmajor,
                                             gpu_moe_layer)

    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    H, I, E, K, GKS = args.H, args.I, args.E, args.K, args.gks

    def rnd(shape, scale=1.0):
        return (torch.randn(*shape) * scale).float()

    w13_f = _pack_fp4(rnd((E, 2 * I, H)))
    w2_f = _pack_fp4(rnd((E, H, I)))
    s13_f = rnd((E, 2 * I, H // GKS), 0.5).to(torch.float8_e4m3fn)
    s2_f = rnd((E, H, I // GKS), 0.5).to(torch.float8_e4m3fn)
    ts13 = rnd((E,), 0.2).abs() + 0.1
    ts2 = rnd((E,), 0.2).abs() + 0.1
    # ⚠️ `_pinned_kmajor` 只能吃**主机**张量（`pin_memory` 对 musa 张量直接报
    #    `cannot pin 'musaByteType'`）⇒ slot 用 CPU 源构建，内核实参另存设备副本
    src = (_pinned_kmajor(w13_f), _pinned_kmajor(s13_f.view(torch.uint8)),
           _pinned_kmajor(w2_f), _pinned_kmajor(s2_f.view(torch.uint8)))
    w13_f, s13_f = w13_f.to(dev), s13_f.to(dev)
    w2_f, s2_f = w2_f.to(dev), s2_f.to(dev)
    slot = PrefetchSlot()
    slot.alloc(src, dev)
    for buf, s_ in zip(slot.bufs, src):
        buf.copy_(s_)
    slot.ready = torch.cuda.Event()
    slot.ready.record()
    print(f"[micro] 常驻 slot: {sum(b.numel() for b in slot.bufs) / 2**30:.3f} GiB "
          f"(E={E} H={H} I={I} K={K} GKS={GKS})", flush=True)

    x = rnd((args.T, H)).to(torch.bfloat16).to(dev)
    ids = torch.stack([torch.randperm(E, device=dev)[:K]
                       for _ in range(args.T)]).to(torch.int32)
    wts = torch.full((args.T, K), 1.0 / K, dtype=torch.float32, device=dev)

    def _call():
        return gpu_moe_layer(
            x, ids, wts, w13_f, s13_f, w2_f, s2_f,
            H=H, I=I, K=K, device=dev, slot=slot,
            group_k=GKS, s13_ts=ts13.to(dev), s2_ts=ts2.to(dev),
            out_dtype=torch.float32,   # MUSA Triton 不支持 bf16 原子加（A78-10）
        )

    for _ in range(args.warmup):
        _call()
    torch.cuda.synchronize()

    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(args.iters):
        _call()
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / args.iters
    proj = ms * args.layers
    print(f"[micro] T={args.T} iters={args.iters}: {ms:.3f} ms/层")
    print(f"[micro] 折算 {args.layers} 层整步 ≈ {proj:.1f} ms "
          f"⇒ {1000.0 / proj:.1f} tok/s（若时间全在 MoE）")
    print(f"[micro] 对照实测 reside decode = 578.03 ms/步 ⇒ MoE 占比 "
          f"{100.0 * proj / 578.03:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
