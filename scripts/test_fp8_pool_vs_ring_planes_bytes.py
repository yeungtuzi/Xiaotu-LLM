#!/usr/bin/env python
"""B918:逐平面【字节比对】环 vs 池 —— 定位第一处差异。

B917 已定案:真 `LayerPool` 路径 `rms_rel=8.196e+02`(NP=1),而真环 API 是 4.4e-3。
⇒ 用【同一个 `_fp8_kmajor_tiled`】分别填「环形状的新鲜缓冲」与「池的 planes」,
   再逐平面对比 ⇒ 第一处不同的平面就是坏掉的那个量 ✓
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
from vllm_xiaotu_moe.gpu_prefill_fp8 import _fp8_kmajor_tiled  # noqa: E402

BLOCK = 128
NAMES = ("w13t", "s13t", "w2t", "s2t")


def q(shape):
    u8 = (torch.randn(shape, generator=torch.Generator().manual_seed(1)) * 0.1) \
        .to(torch.float8_e4m3fn).view(torch.uint8).numpy().copy()
    u8[(u8 == 0x7F) | (u8 == 0xFF)] = 0
    return u8


def main() -> int:
    E, H, I, NP = 16, 2048, 1024, 1
    dev = torch.device("cuda:0")
    g = np.random.default_rng(0)
    w13, w2 = q((E, 2 * I, H)), q((E, H, I))
    s13 = (10.0 ** g.uniform(-0.5, 0.5, size=(E, 2 * I // BLOCK, H // BLOCK))).astype(np.float32)
    s2 = (10.0 ** g.uniform(-0.5, 0.5, size=(E, H // BLOCK, I // BLOCK))).astype(np.float32)
    cfg = xiaotu_moe.MOEConfigV2()
    for k, v in dict(num_processes=NP, process_id=0, gpu_id=0, has_gate_proj=True,
                     expert_num=E, top_k=16, hidden_size=H, intermediate_size=I,
                     max_batch_size=256, max_num_seqs=64, stride=BLOCK,
                     group_min_len=10, group_max_len=4096, groupN=BLOCK, groupK=BLOCK,
                     activation_type=1, swiglu_limit=10.0, swiglu_alpha=1.0,
                     swiglu_beta=0.0).items():
        setattr(cfg, k, v)
    eng = xiaotu_moe.MOE_FP8(cfg, w13, w2, s13, s2, 0, 0)
    print("=" * 78)

    # ---------- 参照:环形状的新鲜缓冲,一次性填 [0, E) ----------
    ring = (
        torch.zeros((E, H, 2 * I), dtype=torch.uint8, device=dev),
        torch.zeros((E, H // BLOCK, (2 * I) // BLOCK * 4), dtype=torch.uint8, device=dev),
        torch.zeros((E, I, H), dtype=torch.uint8, device=dev),
        torch.zeros((E, I // BLOCK, H // BLOCK * 4), dtype=torch.uint8, device=dev),
    )
    _fp8_kmajor_tiled(eng, dev, H, I, E, 0, E, tuple(b[:E] for b in ring))
    torch.cuda.synchronize()
    print("【参照】环形状新鲜缓冲填 [0,E) 完成")

    # ---------- 被测:真 LayerPool ----------
    pool = xp.LayerPool(eng, dev, H, I, BLOCK, E, list(range(E)),
                        layer_idx=0, layout="fp8", topk=16)
    torch.cuda.synchronize()
    planes = pool.planes

    # ---------- 逐平面比对 ----------
    print("\n【逐平面对比】环 vs 池(池 planes 的第 e 行 vs 环的第 e 行)")
    for name, rp, pp in zip(NAMES, ring, planes):
        if rp.dtype == torch.uint8 and pp.dtype == torch.uint8:
            same_all = torch.equal(rp, pp)
            diff = (rp != pp)
            nd = int(diff.sum())
            print(f"  {name:5s} shape={tuple(rp.shape)} dtype={rp.dtype} "
                  f"完全相同={same_all} 不同字节={nd}/{rp.numel()}")
            if nd:
                rows = diff.any(dim=tuple(range(1, diff.dim()))).nonzero().flatten()
                print(f"        不同的行(专家)= {rows[:12].tolist()}"
                      f"{' …' if rows.numel() > 12 else ''}  共 {rows.numel()} 行")
        else:
            print(f"  {name:5s} dtype 不同: 环={rp.dtype} 池={pp.dtype}")

    # ---------- 尺度单独看(池是 uint8 字节槽,需 view) ----------
    print("\n【尺度视图对比】")
    for name, rp, pp in (("s13t", ring[1], planes[1]), ("s2t", ring[3], planes[3])):
        rv = rp.view(torch.float32) if rp.dtype == torch.uint8 else rp
        pv = pp.view(torch.float32) if pp.dtype == torch.uint8 else pp
        print(f"  {name}: 环{tuple(rv.shape)} vs 池{tuple(pv.shape)} "
              f"完全相同={torch.equal(rv, pv)}")
        if not torch.equal(rv, pv):
            d = (rv - pv).abs()
            print(f"        最大差={d.max().item():.6e}  环均值={rv.abs().mean().item():.6e} "
                  f"池均值={pv.abs().mean().item():.6e}  池为0比例="
                  f"{(pv == 0).float().mean().item():.4f}")
    print("=" * 78)
    print("【读法】第一个 `完全相同=False` 的平面 = 坏掉的量 ✓")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
