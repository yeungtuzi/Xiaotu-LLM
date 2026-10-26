#!/usr/bin/env python3
"""NVFP4 **CPU 引擎**（xiaotu_moe.MOE_NVFP4 + cpu_prefill）的离线数值单测。

为什么需要
----------
每次 e2e 跑一轮要 ~9 分钟（加载 24 GiB 权重 + 建 96 个引擎）。而 CPU 引擎的 NVFP4
接线里有三处**错了也不会报错**（只出垃圾数）：
  1. 组尺度必须是 **fp32**（NVFP4 的 e4m3 1 字节尺度会被当 e8m0 解 ⇒ 数值塌）
  2. `weight_scale_2` 必须**折进**组尺度（引擎只吃每专家一个全局尺度；传 0 ⇒ 用 1.0）
  3. 尺度布局 `[E, N, K/16]`、权重 `[E, N, K/2]`（nibble 沿 k 打包）
本脚本用**小形状**把这三处钉死，几秒钟出结果。

用法（远端，纯 CPU，不占 GPU）
------------------------------
    /root/venvs/vllm310/bin/python scripts/musa/preflight_nvfp4_cpu_engine.py
    ... --T 64 --E 8          # 换规模（K=top_k 也可调）

判据：相对误差 < 5e-2（bf16 激活/累加）⇒ 退出码 0。
"""

from __future__ import annotations

import argparse
import sys

import torch


def _unpack_fp4(packed):
    """[.., k//2] uint8 -> [.., k] float32（与内核 LUT 语义一致）。"""
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
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import xiaotu_moe

    torch.manual_seed(args.seed)
    H, I, E, K, GKS, T = args.H, args.I, args.E, args.K, args.gks, args.T
    assert H % GKS == 0 and I % GKS == 0 and H % 2 == 0 and I % 2 == 0

    def rnd(shape, sc=1.0):
        return (torch.randn(*shape) * sc).to(torch.float32)

    # ── 权重（fp4 打包字节，nibble 沿 k）+ 组尺度（**fp32**，已折叠 scale_2）─────
    def pack(vals):
        grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        idx = (vals.unsqueeze(-1) - grid).abs().argmin(dim=-1).to(torch.uint8)
        code = idx | ((vals < 0).to(torch.uint8) << 3)
        return (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()

    w13_f = pack(rnd((E, 2 * I, H)))                 # [E, 2I, H/2]
    w2_f = pack(rnd((E, H, I)))                      # [E, H, I/2]
    s13_q = rnd((E, 2 * I, H // GKS), 0.5).to(torch.float8_e4m3fn).to(torch.float32)
    s2_q = rnd((E, H, I // GKS), 0.5).to(torch.float8_e4m3fn).to(torch.float32)
    ts13 = rnd((E,), 0.2).abs() + 0.1
    ts2 = rnd((E,), 0.2).abs() + 0.1
    # 【路线 A·折叠】scale_2 乘进组尺度（fp32），引擎侧传 gs=0
    s13 = (s13_q * ts13.view(E, 1, 1)).contiguous()
    s2 = (s2_q * ts2.view(E, 1, 1)).contiguous()

    # ── 参考实现 ────────────────────────────────────────────────────────────
    d13 = _unpack_fp4(w13_f) * s13.repeat_interleave(GKS, dim=2)
    d2 = _unpack_fp4(w2_f) * s2.repeat_interleave(GKS, dim=2)

    x = rnd((T, H)).to(torch.bfloat16)
    wts, ids = torch.topk(rnd((T, E)).softmax(-1), K, dim=-1)
    wts = wts.to(torch.float32).contiguous()
    ids = ids.to(torch.int32).contiguous()
    xf = x.to(torch.float32)
    ref = torch.zeros((T, H), dtype=torch.float32)
    for t in range(T):
        acc = torch.zeros(H)
        for j in range(K):
            e = int(ids[t, j])
            gate = xf[t] @ d13[e, :I, :].T
            up = xf[t] @ d13[e, I:, :].T
            acc += float(wts[t, j]) * ((torch.nn.functional.silu(gate) * up) @ d2[e].T)
        ref[t] = acc

    # ── 建引擎（**只能吃主机指针**；喂 torch 张量给 binding 会 SIGSEGV）────────
    cfg = xiaotu_moe.MOEConfigV2()
    cfg.num_processes = 1
    cfg.process_id = 0
    cfg.gpu_id = 0
    cfg.has_gate_proj = True
    cfg.expert_num = E
    cfg.top_k = K
    cfg.hidden_size = H
    cfg.intermediate_size = I
    cfg.max_batch_size = max(64, T + 8)
    cfg.max_num_seqs = 64
    cfg.stride = 32
    cfg.group_min_len = 10
    cfg.group_max_len = max(4096, T) + 128
    cfg.groupN = 1
    cfg.groupK = GKS
    cfg.activation_type = 0
    cfg.use_gpu_prefill = False
    cfg.int8_activation = 0

    for nm, t in (("w13", w13_f), ("w2", w2_f), ("s13", s13), ("s2", s2)):
        assert t.device.type == "cpu" and t.is_contiguous(), nm

    eng = xiaotu_moe.MOE_NVFP4(
        cfg,
        w13_f.data_ptr(), w2_f.data_ptr(),
        s13.data_ptr(), s2.data_ptr(),
        0, 0,
    )
    out = torch.zeros((T, H), dtype=torch.float32)
    xh = x.contiguous()
    eng.cpu_prefill(
        T, K,
        ids.data_ptr(), wts.data_ptr(),
        xh.data_ptr(), out.data_ptr(),
    )

    denom = ref.abs().mean().clamp_min(1e-6)
    err = (out - ref).abs().mean() / denom
    print(f"[cpu-preflight] T={T} H={H} I={I} E={E} K={K} GKS={GKS}")
    print(f"[cpu-preflight] ref|mean|={ref.abs().mean():.4e} got|mean|={out.abs().mean():.4e}")
    print(f"[cpu-preflight] NaN={int(torch.isnan(out).sum())}/{out.numel()} "
          f"相对误差 = {float(err):.4e}")
    ok = float(err) < 5e-2
    print("[cpu-preflight]", "✅ CPU 引擎数值一致" if ok
          else "⛔ 不一致（查 fp32 组尺度 / scale_2 折叠 / 布局）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
