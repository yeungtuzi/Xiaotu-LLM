#!/usr/bin/env python
"""离线判决:FP8 池的 swap_in/swap_out 是否真的可用(不依赖服务)。

背景:生产日志里 `[xtu-pin-adapt]` 恒 0,而 `maybe_adapt` 的异常被 `except: pass` 吞掉
⇒ 必须离线把 `swap_in` 单独叫一次,看它抛什么/装进去的字节对不对。

判据:
  ① swap_in 返回槽号、slotmap[新专家] == n_local + 槽号;
  ② 换入后 GPU 该槽的输出 ≈ CPU 对同一专家的输出(rms_rel < 2e-2)。
用法:CUDA_VISIBLE_DEVICES=0 XIAOTU_MOE_THREADS=8 python scripts/test_fp8_pool_swap.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import xiaotu_moe  # noqa: E402
from vllm_xiaotu_moe import expert_pool as xp  # noqa: E402
from vllm_xiaotu_moe.gpu_prefill_fp8 import gpu_moe_layer_fp8  # noqa: E402

BLOCK, LIMIT = 128, 10.0


def main() -> int:
    E, H, I, T, K, NP = 16, 2560, 384, 8, 2, 1
    dev = torch.device("cuda:0")
    g = np.random.default_rng(3)

    def q(shape):
        u8 = (torch.randn(shape, generator=torch.Generator().manual_seed(1)) * 0.1) \
            .to(torch.float8_e4m3fn).view(torch.uint8).numpy().copy()
        u8[(u8 == 0x7F) | (u8 == 0xFF)] = 0
        return u8

    w13, w2 = q((E, 2 * I, H)), q((E, H, I))
    s13 = (10.0 ** g.uniform(-0.5, 0.5, size=(E, 2 * I // BLOCK, H // BLOCK))).astype(np.float32)
    s2 = (10.0 ** g.uniform(-0.5, 0.5, size=(E, H // BLOCK, I // BLOCK))).astype(np.float32)
    cfg = xiaotu_moe.MOEConfigV2()
    for k, v in dict(num_processes=NP, process_id=0, gpu_id=0, has_gate_proj=True,
                     expert_num=E, top_k=K, hidden_size=H, intermediate_size=I,
                     max_batch_size=256, max_num_seqs=64, stride=BLOCK,
                     group_min_len=10, group_max_len=4096, groupN=BLOCK, groupK=BLOCK,
                     activation_type=1, swiglu_limit=LIMIT, swiglu_alpha=1.0,
                     swiglu_beta=0.0).items():
        setattr(cfg, k, v)
    eng = xiaotu_moe.MOE_FP8(cfg, w13, w2, s13, s2, 0, 0)

    # 参照:专家 8 单独走 CPU
    ids = np.full((T, K), 8, dtype=np.uint32)
    wts = g.uniform(0.05, 0.2, size=(T, K)).astype(np.float32)
    xf = (g.standard_normal((T, H)) * 0.5).astype(np.float32)
    x_u16 = (xf.view(np.uint32) >> 16).astype(np.uint16)
    cpu_out = np.zeros((T, H), dtype=np.float32)
    eng.cpu_prefill(T, K, ids, wts, x_u16, cpu_out)
    ref = torch.from_numpy(cpu_out)

    x_b = torch.from_numpy(x_u16).to(dev).view(torch.bfloat16).reshape(T, H)
    pinned = list(range(8))
    pool = xp.LayerPool(eng, dev, H, I, BLOCK, E, pinned, layer_idx=0,
                        layout="fp8", topk=K)
    print("[build] nslots=", pool.nslots, "slotmap[8]=", int(pool.slotmap[8]))
    got = pool.swap_in(None, 8, 0)   # ⚠️ gp 只在 mxfp4 用;fp8 走 _fp8_kmajor_tiled
    print("[swap_in] ->", got, "expert_of[0]=", pool.expert_of[0],
          "slotmap[8]=", int(pool.slotmap[8]))
    torch.cuda.synchronize()
    # 用槽 0 的 plane 算专家 8(part id = 槽号 0)
    ids_t = torch.zeros((T, K), dtype=torch.int32, device=dev)
    wts_t = torch.from_numpy(wts).to(dev)
    out = gpu_moe_layer_fp8(x_b, ids_t, wts_t, *pool.planes, H, I, K,
                            device=dev, swiglu_limit=LIMIT)
    torch.cuda.synchronize()
    rms = ref.pow(2).mean().sqrt().item()
    rel = ((out.float().cpu() - ref).pow(2).mean().sqrt() / (rms + 1e-12)).item()
    print(f"[verify] rms_rel={rel:.3e} ref_rms={rms:.4f} -> "
          f"{'✅ swap_in 可用' if rel < 2e-2 else '⛔ swap_in 装错'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
