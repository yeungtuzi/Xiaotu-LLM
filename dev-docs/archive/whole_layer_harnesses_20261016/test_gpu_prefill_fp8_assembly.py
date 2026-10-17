#!/usr/bin/env python
"""Validate kmajor_from_engine_shards_fp8 against the weights the engine got.

Builds a real (sharded) MOE_FP8 engine from synthetic e4m3 weights, asks the FP8
assembly for K-major device tensors, and compares byte-for-byte (weights) /
exactly (scales) against the expected K-major transpose of the source arrays.

This is the piece that differs structurally from MXFP4: 1-byte rows and a
[N/128, K/128] fp32 block scale instead of [N, K/32] e8m0.

Usage: python scripts/test_gpu_prefill_fp8_assembly.py [E H I]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# ⛔⛔ 【测试隔离门 —— 必须在 import 包之前】本机 `/tmp/xiaotu_env` 是【生产】环境文件,
#    包内有一条"默认桥"会把它的值**补齐**进来(实测:生产正在用的
#    `XIAOTU_GPF_STAGE_TILE_E=96` 会被灌进本测试)⇒ 测试口径被生产污染 ✗
#    ⇒ 显式赋值(而非依赖缺省)即可,因为"桥"只补**缺失**的键 ✓
os.environ["XIAOTU_GPF_STAGE_TILE_E"] = "0"   # ⭐ 前置对拍必须先跑【整层】基线
os.environ["XIAOTU_GPF_RAW2_SHARE"] = "0"

import xiaotu_moe  # noqa: E402
from vllm_xiaotu_moe.byte_transpose import ktranspose_bytes  # noqa: E402
from vllm_xiaotu_moe.gpu_prefill_fp8 import (  # noqa: E402
    kmajor_from_engine_shards_fp8,
)

BLOCK = 128


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    args = [int(x) for x in sys.argv[1:]]
    E, H, I = args if len(args) == 3 else (4, 512, 256)
    dev = torch.device("cuda:0")
    g = np.random.default_rng(0)

    w13 = g.integers(0, 256, size=(E, 2 * I, H), dtype=np.uint8)
    w2 = g.integers(0, 256, size=(E, H, I), dtype=np.uint8)
    s13 = (10.0 ** g.uniform(-2, 1, size=(E, 2 * I // BLOCK, H // BLOCK))).astype(np.float32)
    s2 = (10.0 ** g.uniform(-2, 1, size=(E, H // BLOCK, I // BLOCK))).astype(np.float32)

    cfg = xiaotu_moe.MOEConfigV2()
    for k, v in dict(num_processes=1, process_id=0, gpu_id=0, has_gate_proj=True,
                     expert_num=E, top_k=1, hidden_size=H, intermediate_size=I,
                     max_batch_size=64, max_num_seqs=8, stride=BLOCK,
                     group_min_len=10, group_max_len=1024, groupN=BLOCK, groupK=BLOCK,
                     activation_type=1, swiglu_limit=10.0, swiglu_alpha=1.0,
                     swiglu_beta=0.0).items():
        setattr(cfg, k, v)
    eng = xiaotu_moe.MOE_FP8(cfg, w13, w2, s13, s2, 0, 0)
    geo = eng.shard_geometry()
    print(f"E={E} H={H} I={I} shards ns={geo['ns']} "
          f"w13_crows={geo['w13_crows']} w13_cbytes={geo['w13_cbytes']} "
          f"w2_crows={geo['w2_crows']} w2_cbytes={geo['w2_cbytes']}")
    if int(geo["ns"]) < 2:
        raise SystemExit("engine did not shard; cannot test the assembly path")

    built = kmajor_from_engine_shards_fp8(eng, dev, H, I, E)
    assert built is not None, "assembly returned None"
    w13t, s13t, w2t, s2t = built

    # Expected K-major, computed on the host from the SOURCE arrays.
    exp_w13t = torch.from_numpy(w13).transpose(1, 2).contiguous().to(dev)
    exp_w2t = torch.from_numpy(w2).transpose(1, 2).contiguous().to(dev)
    exp_s13t = torch.from_numpy(s13).transpose(1, 2).contiguous().to(dev)
    exp_s2t = torch.from_numpy(s2).transpose(1, 2).contiguous().to(dev)

    torch.cuda.synchronize()
    r = {}
    r["w13t"] = bool(torch.equal(w13t, exp_w13t))
    r["w2t"] = bool(torch.equal(w2t, exp_w2t))
    r["s13t"] = float((s13t - exp_s13t).abs().max().item())
    r["s2t"] = float((s2t - exp_s2t).abs().max().item())
    r["shape13"] = tuple(w13t.shape) == (E, H, 2 * I)
    r["shape2"] = tuple(w2t.shape) == (E, I, H)
    ok = r["w13t"] and r["w2t"] and r["shape13"] and r["shape2"] \
        and r["s13t"] == 0.0 and r["s2t"] == 0.0
    print(f"[{'OK ' if ok else 'BAD'}] assembly: w13t_bytes_equal={r['w13t']} "
          f"w2t_bytes_equal={r['w2t']} shapes_ok={r['shape13'] and r['shape2']} "
          f"scale_maxdiff=({r['s13t']:.1e}, {r['s2t']:.1e})")

    # ------------------------------------------------------------------
    # ⭐ 【微 ping/pong】tile 环:① 逐字节对拍(装配)② 数值对拍(整层 vs 环)
    # 这两条是本路径的**唯一防线** —— tile 只是"换个写法搬同样的字节",
    # 所以装配必须**逐字节相等**,而不是"差不多" ✗
    # ------------------------------------------------------------------
    from vllm_xiaotu_moe import gpu_prefill_fp8 as G
    from vllm_xiaotu_moe.gpu_prefill import stage_tiles, staging_tile_experts
    from vllm_xiaotu_moe.gpu_prefill_fp8 import (
        _fp8_kmajor_tiled, _fp8_tile_ring, gpu_moe_layer_from_engine_tiled,
    )

    tile = max(1, E // 2)
    os.environ["XIAOTU_GPF_STAGE_TILE_E"] = str(tile)
    if staging_tile_experts(E) != tile:
        raise SystemExit(f"tile env 未生效: {staging_tile_experts(E)} != {tile}")
    tiles = stage_tiles(E)
    ring = _fp8_tile_ring(dev, E, H, I, tile, 2)
    ok_t = True
    for (e0, e1) in tiles:
        Et = e1 - e0
        got = _fp8_kmajor_tiled(eng, dev, H, I, E, e0, e1,
                                tuple(b[:Et] for b in ring[0]))
        torch.cuda.synchronize()
        same13 = bool(torch.equal(got[0], exp_w13t[e0:e1]))
        same2 = bool(torch.equal(got[2], exp_w2t[e0:e1]))
        d13 = float((got[1] - exp_s13t[e0:e1]).abs().max().item())
        d2 = float((got[3] - exp_s2t[e0:e1]).abs().max().item())
        ok_t &= same13 and same2 and d13 == 0.0 and d2 == 0.0
        print(f"      tile[{e0}:{e1}) w13_bytes_equal={same13} w2_bytes_equal={same2} "
              f"scale_maxdiff=({d13:.1e}, {d2:.1e})")
    print(f"[{'OK ' if ok_t else 'BAD'}] tile 装配逐字节对拍 (E={E} tile={tile} "
          f"tiles={len(tiles)})")

    g2 = np.random.default_rng(1)
    T, K2 = 24, 4
    xs = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
    ids = torch.randint(0, E, (T, K2), dtype=torch.int32, device=dev)
    wts_r = torch.rand(T, K2, dtype=torch.float32, device=dev)
    ref = G.gpu_moe_layer_fp8(xs, ids, wts_r, w13t, s13t, w2t, s2t, H, I, K2,
                              device=dev)
    got_l = gpu_moe_layer_from_engine_tiled(eng, xs, ids, wts_r, H, I, E, K2,
                                            device=dev)
    torch.cuda.synchronize()
    if got_l is None:
        print("[BAD] tile 环整层调用返回 None")
        ok_n, dmax, rel = False, float("nan"), float("nan")
    else:
        dmax = float((ref - got_l).abs().max().item())
        rel = dmax / max(1e-9, float(ref.abs().max().item()))
        ok_n = rel < 1e-5
    print(f"[{'OK ' if ok_n else 'BAD'}] tile 环数值对拍:maxdiff={dmax:.3e} "
          f"rel={rel:.2e}")

    # 复位 env,避免污染同进程后续(整层路径必须仍是默认)✓
    os.environ["XIAOTU_GPF_STAGE_TILE_E"] = "0"
    return 0 if (ok and ok_t and ok_n) else 1


if __name__ == "__main__":
    raise SystemExit(main())
