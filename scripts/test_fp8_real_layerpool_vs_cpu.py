#!/usr/bin/env python
"""B917 判决:用【真 `expert_pool.LayerPool`】测 —— 这才是生产路径。

⭐ 为什么必须用它
  B914 我手填 planes 得到 `rms_rel=8.06e+02`,但 B916 证明**真环 API 是 4.4e-3**
  ⇒ 806 只说明"我的手填用法错",⛔ **不等于池错** ✗
  ⇒ 唯一有效判据 = ⭐ **直接实例化 `LayerPool`**,用它的 `as_slot()` planes 去算 ✓

口径:NP=1(消除 EP 全局/本地歧义);`pinned` = 全部本地专家 ⇒ 无 miss,
      池路径应能【完整复刻】CPU ⇒ 与 `cpu_out` 直接可比 ✓
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import xiaotu_moe  # noqa: E402
from vllm_xiaotu_moe import expert_pool as xp  # noqa: E402
from vllm_xiaotu_moe.gpu_prefill_fp8 import gpu_moe_layer_fp8  # noqa: E402

BLOCK = 128
LIMIT = 10.0


def q(shape):
    u8 = (torch.randn(shape, generator=torch.Generator().manual_seed(1)) * 0.1) \
        .to(torch.float8_e4m3fn).view(torch.uint8).numpy().copy()
    u8[(u8 == 0x7F) | (u8 == 0xFF)] = 0
    return u8


def rel(x, y):
    rms = y.pow(2).mean().sqrt().item()
    return ((x - y).pow(2).mean().sqrt() / (rms + 1e-12)).item(), (x - y).abs().max().item(), rms


def main() -> int:
    E, H, I, T, K, NP = (16, 2048, 1024, 64, 16, 1)
    dev = torch.device("cuda:0")
    g = np.random.default_rng(0)
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
    e_shard = E // NP
    print("=" * 76)
    print(f"E={E} H={H} I={I} T={T} K={K} NP={NP}  分片专家={e_shard}")

    xf = (g.standard_normal((T, H)) * 0.5).astype(np.float32)
    x_u16 = (xf.view(np.uint32) >> 16).astype(np.uint16)
    ids = g.integers(0, e_shard, size=(T, K)).astype(np.uint32)
    wts = g.uniform(0.05, 0.2, size=(T, K)).astype(np.float32)
    cpu_out = np.zeros((T, H), dtype=np.float32)
    eng.cpu_prefill(T, K, ids, wts, x_u16, cpu_out)
    a = torch.from_numpy(cpu_out)
    x_b = torch.from_numpy(x_u16).to(dev).view(torch.bfloat16).reshape(T, H)
    ids_t = torch.from_numpy(ids).to(dev).to(torch.int32)
    wts_t = torch.from_numpy(wts).to(dev)

    # ⭐ 真 LayerPool:pinned = 全部本地专家(无 miss)
    pinned = list(range(e_shard))
    print(f"【建池】pinned={pinned} n_local 参数候选:")
    for n_local_tag, n_local in (("分片数 e_shard", e_shard), ("全体数 E", E)):
        try:
            pool = xp.LayerPool(eng, dev, H, I, BLOCK, n_local, pinned,
                                layer_idx=0, layout="fp8", topk=K)
            slot = pool.as_slot()
            planes = slot.bufs if hasattr(slot, "bufs") else slot[0]
            print(f"  ✅ [{n_local_tag}] n_local={n_local}: planes="
                  f"{[tuple(p.shape) for p in planes]}")
            # ⭐ 池路径:part_ids 用本地 id(全部命中)
            out = gpu_moe_layer_fp8(x_b, ids_t, wts_t, *planes, H, I, K,
                                    device=dev, swiglu_limit=LIMIT)
            torch.cuda.synchronize()
            r = rel(out.float().cpu(), a)
            flag = "✅" if r[0] < 2e-2 else "⛔"
            print(f"      {flag} 池路径: rms_rel={r[0]:.3e} max_abs={r[1]:.3e} "
                  f"ref_rms={r[2]:.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"  ⛔ [{n_local_tag}] n_local={n_local}: {type(e).__name__}: {str(e)[:120]}")
    print("=" * 76)
    print("【判据】若某一档 ≈1e-2 ⇒ ⭐ 池的装载是【对的】,0% 真因在别处;")
    print("        若两档都 O(1) ⇒ ⭐ 池的装载路径确凿破损 ★")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
