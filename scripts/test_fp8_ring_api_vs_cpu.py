#!/usr/bin/env python
"""B916 判决:用【真正的环 API】`gpu_moe_layer_from_engine_tiled` 作对照。
若环 API 等价(~1e-2)而我手填 planes 是 806 ⇒ 池/手填的用法有问题;
若环 API 也是 806 ⇒ 环本身也错(⭐ A 项结论要重估)。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import xiaotu_moe  # noqa: E402
from vllm_xiaotu_moe.gpu_prefill_fp8 import (  # noqa: E402
    gpu_moe_layer_from_engine_tiled,
)

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

    print("=" * 74)
    print("【Q5】真·环 API `gpu_moe_layer_from_engine_tiled` 是否与 CPU 等价?")
    for tile in (e_shard, max(1, e_shard // 2), 1):
        try:
            out = gpu_moe_layer_from_engine_tiled(
                eng, x_b, ids_t, wts_t, H, I, E, K,
                device=dev, group_k=BLOCK, nslots=2, swiglu_limit=LIMIT)
            torch.cuda.synchronize()
            if out is None:
                print(f"  tile={tile}: 返回 None")
                continue
            r = rel(out.float().cpu(), a)
            flag = "✅" if r[0] < 2e-2 else "⛔"
            print(f"  {flag} 环 API(n_experts=E={E}): rms_rel={r[0]:.3e} "
                  f"max_abs={r[1]:.3e} ref_rms={r[2]:.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"  ⛔ 环 API: {type(e).__name__}: {str(e)[:130]}")
        break
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
