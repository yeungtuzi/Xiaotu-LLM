#!/usr/bin/env python
"""离线判定:热专家池的 plane 装载 vs 环/整层路径 —— 到底哪一种数值等价?

⭐ 动机(2026-10-17 B914)
------------------------------------------------------------------
热专家池(`expert_pool.py`)开起来后 GLM 投机接受率恒为 **0.0%**,而
关池是 **82.0%**(B906)⇒ 池特有的数值缺陷。读码已否证 7 条嫌疑(尺度转置、
`e0` 全局/本地、`dst` 语义、尺度元素数、plane 形状、专家顺序、`inter` 行索引),
⇒ 必须从"读"转为"测"。

⚠️ 既有 harness(`test_gpu_prefill_fp8_vs_cpu.py`)有一个**致命盲点**:
   它用 `num_processes=1` ⇒ **无 EP ⇒ 全局 == 本地** ⇒
   **无法区分 `n_experts` 该传【全体】还是【分片】**,而这正是嫌疑的核心 ✗

本脚本回答两个问题(**都不需要真模型,纯合成权重**):

  Q1 ⭐ 引擎的尺度缓冲 `scale_w13_bytes()` 覆盖【本 rank 分片】还是【全体专家】?
     ⇒ `geo["w13_scale_bytes"] / 每专家尺度字节` ⇒ 商就是缓冲里的专家数

  Q2 ⭐ `n_experts` 到底该传哪个值?
     ⇒ 分别用 `E` 与 `E//NP` 跑 GPU 路径 ⇒ **哪一个与 CPU 引擎等价**
     (这是**直接实测**,不是推算 ✓)

  Q3 ⭐ 把权重经 `_fp8_kmajor_tiled` 装进【池形状的 planes】后,
     `gpu_moe_layer_fp8` 的输出是否仍与 CPU 等价?
     ⇒ 若 Q2 找出了正确 `n_experts`,而池形状仍偏 ⇒ **池的装载路径确凿有缺陷** ✓

用法:
    python scripts/test_fp8_pool_planes_vs_ring.py [E H I T K NP]
    NP = 进程数(EP 分片数);⭐ **NP=2 才能区分全局/本地**(默认 2)
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
    gpu_moe_layer_fp8,
    kmajor_from_engine_shards_fp8,
    _fp8_kmajor_tiled,
)

BLOCK = 128


def q(shape):
    """e4m3 字节(量级现实:随机 BYTES 会带出 448 指数,输出 ~1e5,微扰无意义)"""
    u8 = (torch.randn(shape, generator=torch.Generator().manual_seed(1)) * 0.1) \
        .to(torch.float8_e4m3fn).view(torch.uint8).numpy().copy()
    u8[(u8 == 0x7F) | (u8 == 0xFF)] = 0
    return u8


def rel(x, y):
    rms = y.pow(2).mean().sqrt().item()
    return ((x - y).pow(2).mean().sqrt() / (rms + 1e-12)).item(), (x - y).abs().max().item(), rms


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    args = [int(x) for x in sys.argv[1:]]
    # ⭐ 默认 E=16 ⇒ NP=2 ⇒ 分片 8 个,便于与"全体 16"区分
    E, H, I, T, K, NP = args if len(args) == 6 else (16, 2048, 1024, 64, 8, 2)
    dev = torch.device("cuda:0")
    g = np.random.default_rng(0)

    per_expert_s13 = (2 * I // BLOCK) * (H // BLOCK) * 4      # 字节/专家
    per_expert_s2 = (H // BLOCK) * (I // BLOCK) * 4

    w13 = q((E, 2 * I, H))
    w2 = q((E, H, I))
    s13 = (10.0 ** g.uniform(-0.5, 0.5, size=(E, 2 * I // BLOCK, H // BLOCK))).astype(np.float32)
    s2 = (10.0 ** g.uniform(-0.5, 0.5, size=(E, H // BLOCK, I // BLOCK))).astype(np.float32)

    cfg = xiaotu_moe.MOEConfigV2()
    for k, v in dict(num_processes=NP, process_id=0, gpu_id=0, has_gate_proj=True,
                     expert_num=E, top_k=K, hidden_size=H, intermediate_size=I,
                     max_batch_size=256, max_num_seqs=64, stride=BLOCK,
                     group_min_len=10, group_max_len=4096, groupN=BLOCK, groupK=BLOCK,
                     activation_type=1, swiglu_limit=10.0, swiglu_alpha=1.0,
                     swiglu_beta=0.0).items():
        setattr(cfg, k, v)
    eng = xiaotu_moe.MOE_FP8(cfg, w13, w2, s13, s2, 0, 0)
    geo = eng.shard_geometry()
    e_shard = E // NP

    print("=" * 78)
    print(f"E={E} H={H} I={I} T={T} K={K} NP={NP}  ⇒ 每 rank 分片专家数 = {e_shard}")
    print(f"  ns(shard 数)            = {geo['ns']}")
    print(f"  w13_scale_bytes()       = {geo['w13_scale_bytes']:,} B")
    print(f"  w2_scale_bytes()        = {geo['w2_scale_bytes']:,} B")
    print(f"  每专家 s13 尺度字节      = {per_expert_s13:,} B")
    print("=" * 78)

    # ---------------- Q1:尺度缓冲里有几个专家 ----------------
    n_in_s13 = geo["w13_scale_bytes"] / per_expert_s13
    n_in_s2 = geo["w2_scale_bytes"] / per_expert_s2
    print("\n【Q1】尺度缓冲覆盖几个专家?")
    print(f"  w13: {geo['w13_scale_bytes']:,} / {per_expert_s13:,} = {n_in_s13:.3f}")
    print(f"  w2 : {geo['w2_scale_bytes']:,} / {per_expert_s2:,} = {n_in_s2:.3f}")
    if abs(n_in_s13 - E) < 1e-6:
        print(f"  ⇒ ⭐ 覆盖【全体 {E}】⇒ `n_experts` 必须 = {E}  ★")
    elif abs(n_in_s13 - e_shard) < 1e-6:
        print(f"  ⇒ ⭐ 覆盖【分片 {e_shard}】⇒ `n_experts` 必须 = {e_shard}  ★")
    else:
        print(f"  ⇒ ⚠️ 两个都不是({n_in_s13});需人工看")

    # ---------------- 真值:CPU 引擎 ----------------
    xf = (g.standard_normal((T, H)) * 0.5).astype(np.float32)
    x_u16 = (xf.view(np.uint32) >> 16).astype(np.uint16)
    ids = g.integers(0, e_shard, size=(T, K)).astype(np.uint32)   # ⭐ 分片本地 id
    wts = g.uniform(0.05, 0.2, size=(T, K)).astype(np.float32)

    cpu_out = np.zeros((T, H), dtype=np.float32)
    eng.cpu_prefill(T, K, ids, wts, x_u16, cpu_out)
    a = torch.from_numpy(cpu_out)

    x_b = torch.from_numpy(x_u16).to(dev).view(torch.bfloat16).reshape(T, H)
    ids_t = torch.from_numpy(ids).to(dev).to(torch.int32)
    wts_t = torch.from_numpy(wts).to(dev)

    # ---------------- Q2:两种 n_experts 各跑一次 ----------------
    print("\n【Q2】`n_experts` 该传哪个?(⭐ 直接实测,哪個与 CPU 等价就是对的)")
    results = {}
    for tag, n_exp in (("全体 E", E), ("分片 E//NP", e_shard)):
        try:
            built = kmajor_from_engine_shards_fp8(eng, dev, H, I, n_exp)
            if built is None:
                print(f"  [{tag}] n_experts={n_exp}: 装配返回 None")
                results[tag] = None
                continue
            out = gpu_moe_layer_fp8(x_b, ids_t, wts_t, *built, H, I, K,
                                    device=dev, swiglu_limit=10.0)
            torch.cuda.synchronize()
            r = rel(out.float().cpu(), a)
            results[tag] = r
            flag = "✅" if r[0] < 2e-2 else "⛔"
            print(f"  {flag} [{tag}] n_experts={n_exp}: rms_rel={r[0]:.3e} "
                  f"max_abs={r[1]:.3e} ref_rms={r[2]:.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"  ⛔ [{tag}] n_experts={n_exp}: {type(e).__name__}: {str(e)[:110]}")
            results[tag] = None

    good = [t for t, r in results.items() if r is not None and r[0] < 2e-2]
    print(f"  ⇒ 等价的那一档: {good if good else '【两档都不等价】⭐ 说明问题不在 n_experts'}")

    # ---------------- Q3:池形状的 planes ----------------
    print("\n【Q3】经 `_fp8_kmajor_tiled` 装进【池形状 planes】是否仍等价?")
    # ⭐ 口径要求:让**全部本地专家**都进池(S = e_shard)⇒ 池路径应能【完整复刻】
    #    CPU 的结果 ⇒ 与 `a` 直接可比 ✓(否则只算了一部分,比了也没意义 ✗)
    if K < e_shard:
        print(f"  ⚠️ K={K} < 分片专家数 {e_shard} ⇒ 池装不下全部本地专家 ⇒ 跳过 Q3")
        print("     (⭐ 重跑时把 K 提到 >= 分片数,例如 `... 16 2048 1024 64 8 2`)")
        return 0
    S = e_shard
    # ⭐⭐ B915/B916:两种候选布局都要试 —— ⛔ 不许靠读 docstring 猜 ✓
    #   L1 = 环与池**当前实际**分配的形状:`(S, H, 2I)` / `(S, I, H)`
    #   L2 = `_fp8_kmajor_tiled` **docstring 自称**的形状:`(S, 2I, H)` / `(S, H, I)`
    layouts = {
        "L1 现状 (S,H,2I)/(S,I,H)": (
            (S, H, 2 * I), (S, H // BLOCK, (2 * I) // BLOCK * 4),
            (S, I, H), (S, I // BLOCK, H // BLOCK * 4),
        ),
        "L2 docstring (S,2I,H)/(S,H,I)": (
            (S, 2 * I, H), (S, H // BLOCK, (2 * I) // BLOCK * 4),
            (S, H, I), (S, H // BLOCK, I // BLOCK * 4),
        ),
    }
    print(f"  槽数 S={S};候选布局 = {list(layouts)}")
    for lname, lshapes in layouts.items():
        planes = tuple(torch.empty(sh, dtype=torch.uint8, device=dev) for sh in lshapes)
        print(f"\n  --- {lname} ---")
        print(f"      {[tuple(p.shape) for p in planes]}")
        for tag, n_exp in (("全体 E", E), ("分片 E//NP", e_shard)):
            if n_exp != E:
                continue          # ⭐ B914 已定案:只有全体成立;分片必崩,不再重复
            try:
                for s in range(S):
                    dst = tuple(p[s:s + 1] for p in planes)
                    _fp8_kmajor_tiled(eng, dev, H, I, n_exp, s, s + 1, dst)
                torch.cuda.synchronize()
                out = gpu_moe_layer_fp8(x_b, ids_t, wts_t, *planes, H, I, K,
                                        device=dev, swiglu_limit=10.0)
                torch.cuda.synchronize()
                r = rel(out.float().cpu(), a)
                flag = "✅" if r[0] < 2e-2 else "⛔"
                print(f"      {flag} n_experts={n_exp}: rms_rel={r[0]:.3e} "
                      f"max_abs={r[1]:.3e} ref_rms={r[2]:.4f}")
            except Exception as e:  # noqa: BLE001
                print(f"      ⛔ n_experts={n_exp}: {type(e).__name__}: {str(e)[:110]}")

    # ⭐ 同时判【环】:同一种直填法 + 两种布局,看环用的是哪一种
    print("\n【Q4】环(同一直填法)在两种布局下各是什么结果?")
    for lname, lshapes in layouts.items():
        planes = tuple(torch.empty(sh, dtype=torch.uint8, device=dev) for sh in lshapes)
        try:
            for s in range(S):
                _fp8_kmajor_tiled(eng, dev, H, I, E, s, s + 1,
                                  tuple(p[s:s + 1] for p in planes))
            torch.cuda.synchronize()
            out = gpu_moe_layer_fp8(x_b, ids_t, wts_t, *planes, H, I, K,
                                    device=dev, swiglu_limit=10.0)
            torch.cuda.synchronize()
            r = rel(out.float().cpu(), a)
            flag = "✅" if r[0] < 2e-2 else "⛔"
            print(f"  {flag} [{lname}]: rms_rel={r[0]:.3e} max_abs={r[1]:.3e}")
        except Exception as e:  # noqa: BLE001
            print(f"  ⛔ [{lname}]: {type(e).__name__}: {str(e)[:110]}")

    print("\n" + "=" * 78)
    print("【结论】Q1 给出 `n_experts` 的【正确值】;Q2 用实测交叉验证;")
    print("        Q3 若在正确 n_experts 下仍偏 ⇒ 池的装载路径确凿有缺陷 ★")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
