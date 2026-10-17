#!/usr/bin/env python
"""B940【P0】测 `T_cpu`(CPU MoE decode)与 `T_gpu`(池槽路径 GPU MoE)的单次成本。

⭐ 为什么要这个数(B932/B937/B939 的收敛点)
  B932 给出墙钟模型  `T = max(T_gpu(p), T_cpu(1-p)) + T_sync`
  B937/B938 实测:p=66.5% ⇒ 步时 132.2→124.5 ms(仅 −5.5%)
  B939 实测:p=100% 在显存预算内【不可达】(S=25 需 12.2 GiB,0.90 差 0.9 GiB;0.95 触发 cudagraph 崩)
  ⇒ ⭐ 于是唯一剩余的杠杆是【降低 T_gpu】还是【继续抬 p】,取决于
    ⭐ `T_gpu` 到底比 `T_cpu` 便宜多少 ⇒ **本脚本直接量这两个数** ✓

口径(必须与生产同形状)
  * GLM-5.3-Flash:`H=4096, I=2048`;`top_k K=8`(num_experts_per_tok)
  * `NP=1` ⇒ 本地专家 = 全部 E(消除 EP 全局/本地 id 歧义)
  * qlen ∈ {4, 8, 16} —— 生产 decode 每层每步的 token 数就在这个量级
    (SEQS=4 + 投机 k=4 ⇒ 每步 4..16 个 token)
  * 权重随机(fp8 e4m3 + 128×128 block scale),`swiglu_limit=10.0`(GLM 实际值)

输出
  ⭐ `T_cpu`  µs/次  (`engine.cpu_decode`,生产同款 API,含同步)
  ⭐ `T_gpu`  µs/次  (`gpu_moe_layer_fp8` 池路径,全部命中,含同步)
  ⭐ 比值,以及 `T = max(...)` 模型的预测 p* 阈值

⚠️ 这不是精度测试(B917/B923 已判精度);本脚本【只测时间】,但仍会报 rms_rel 以确认
   两条路径算的是同一个东西(避免"快是因为没算")。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import xiaotu_moe  # noqa: E402
from vllm_xiaotu_moe import expert_pool as xp  # noqa: E402
from vllm_xiaotu_moe.gpu_prefill_fp8 import gpu_moe_layer_fp8  # noqa: E402

BLOCK = 128
LIMIT = 10.0


def q8(shape):
    u8 = (torch.randn(shape, generator=torch.Generator().manual_seed(1)) * 0.1) \
        .to(torch.float8_e4m3fn).view(torch.uint8).numpy().copy()
    u8[(u8 == 0x7F) | (u8 == 0xFF)] = 0
    return u8


def rel(x, y):
    rms = y.pow(2).mean().sqrt().item()
    return ((x - y).pow(2).mean().sqrt() / (rms + 1e-12)).item()


def timeit(fn, iters=30, warmup=8):
    """返回 (中位 µs, 最小 µs)。⭐ 用中位:避免偶发调度尖峰污染判据。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def timeit_async(fn, iters=30, warmup=8):
    """⭐ B942:用 CUDA event 测「无同步」吞吐态成本。

    B941 的 T_gpu 是用 torch.cuda.synchronize() 夹的 ⇒ 含每次调用的同步开销。
    真机在 CUDA graph 里是 async 的(只有图末一次同步)⇒ 本函数把 8 次调用连续排进流、
    用两个 event 夹住全部、只在最后同步一次 ⇒ 得到吞吐态单次成本,更接近真机 ✓
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(8):
            fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e6 / 8.0)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def main() -> int:
    # ⭐ 生产形状(GLm-5.3-Flash),不是小号玩具
    E, H, I, K, NP = 288, 4096, 2048, 8, 1
    dev = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print("=" * 78)
    print("B940  T_cpu vs T_gpu  (GLM-5.3-Flash 形状: E=%d H=%d I=%d K=%d NP=%d)"
          % (E, H, I, K, NP))
    print("  ⭐ 判据:若 T_gpu ≪ T_cpu ⇒ 抬 p 仍有大空间;")
    print("        若 T_gpu ≳ T_cpu ⇒ ⭐ 池最多只能到 max() 的下界,必须改内核/减同步")
    print("=" * 78)

    g = np.random.default_rng(0)
    w13, w2 = q8((E, H, 2 * I)), q8((E, I, H))
    s13 = (10.0 ** g.uniform(-0.5, 0.5, size=(E, H // BLOCK, (2 * I) // BLOCK))).astype(np.float32)
    s2 = (10.0 ** g.uniform(-0.5, 0.5, size=(E, I // BLOCK, H // BLOCK))).astype(np.float32)

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

    # ⭐ 真 LayerPool:pin 【S 个】专家,且 ids 只取自这 S 个 ⇒ 无 miss ⇒ T_gpu 是【纯 GPU 成本】
    #    ⛔ B940 教训:S 必须与真机工作点一致(B937/B938 的 S=16),
    #       曾经用 S=288 ⇒ T_gpu 报 10.5 ms/层,而真机 124.5 ms/42 层 ≈ 2.96 ms/层 ⇒ 差 3.5× ⇒ 假数 ✗
    S_PIN = 16
    pinned = list(range(S_PIN))
    pool = xp.LayerPool(eng, dev, H, I, BLOCK, e_shard, pinned,
                        layer_idx=0, layout="fp8", topk=K)
    slot = pool.as_slot()
    planes = slot.bufs if hasattr(slot, "bufs") else slot[0]
    torch.cuda.synchronize()
    print("  池 planes: %s" % ([tuple(p.shape) for p in planes],))

    stream = torch.cuda.current_stream().cuda_stream
    # ⛔ B940 教训(根因):`binding.cpp:834-840` 用 **out 指针** 缓存行数
    #    (`if (stx->out_ptr_seen == op) rows = stx->out_rows;`)
    #    ⇒ 若对不同 qlen 复用【同一个】小 buffer,则第一次(qlen=4)缓存 rows=4,
    #      之后 qlen=16 被判 `rows=4 < qlen=16` ⇒ ⭐ **整层 skip**,T_cpu 变成空调用(5.1 µs 假数)✗
    #    ⇒ 修法:⭐ 按 `max_num_seqs`(=64)开一张,全程复用 ⇒ rows 恒 = 64 ≥ 任何 qlen ✓
    OUT_ROWS = 64
    out_cpu_dev = torch.zeros((OUT_ROWS, H), dtype=torch.float32, device=dev)
    rows = []
    for T in (4, 8, 16):
        xf = (g.standard_normal((T, H)) * 0.5).astype(np.float32)
        x_u16 = (xf.view(np.uint32) >> 16).astype(np.uint16)
        # ⭐ ids 只取自被 pin 的 S 个 ⇒ 全命中 ⇒ T_gpu 无 miss
        ids = g.integers(0, S_PIN, size=(T, K)).astype(np.uint32)
        wts = g.uniform(0.05, 0.2, size=(T, K)).astype(np.float32)

        # ---- T_cpu:生产同款 cpu_decode(device 入参,out 在 device)----
        x_dev = torch.from_numpy(x_u16).to(dev).view(torch.bfloat16).reshape(T, H)
        ids_t = torch.from_numpy(ids).to(dev).to(torch.int32)
        wts_t = torch.from_numpy(wts).to(dev)

        # ⛔ B941 教训:`cpu_decode` 的行数自检在本 harness 里恒报
        #    `out_gpu rows=0`(即使传 (64,H) 的 tensor 也如此)⇒ 整层 skip。
        #    ⇒ ⭐ 改用已验证可用的 `cpu_prefill`作 **CPU MoE 单次计算成本**的代理:
        #      它走同一个 `self.forward_many(qlen, top_k, ...)`(见 binding.cpp:1139-1152)⇒
        #      ⭐ 计算量与 decode 同形,用于回答「T_cpu 到底多少」是合法的。
        #    ⚠️ 它不含 decode 的 D2H/H2D 同步开销 ⇒ 报出的 T_cpu 是下界。
        out_np = np.zeros((T, H), dtype=np.float32)

        def f_cpu():
            out_np.fill(0.0)
            eng.cpu_prefill(T, K, ids, wts, x_u16, out_np)

        t_cpu, t_cpu_min = timeit(f_cpu)
        cpu_ref = torch.from_numpy(out_np.copy()).to(dev)  # ⭐ 与 gpu_out 同设备,否则 rel() 报 device 不匹配

        # ---- T_gpu:池槽路径(全命中)----
        def f_gpu():
            return gpu_moe_layer_fp8(x_dev, ids_t, wts_t, *planes, H, I, K,
                                     device=dev, swiglu_limit=LIMIT)

        t_gpu, t_gpu_min = timeit(f_gpu)
        # ⭐ B942 无同步(吞吐态,更接近真机图内 async)
        t_gpu_async, _ = timeit_async(f_gpu)
        gpu_out = f_gpu()
        torch.cuda.synchronize()
        r = rel(gpu_out.float(), cpu_ref)

        # ⛔ B940 教训(判据固化):⭐ 计时前必须先证明【真的算了】
        #    旧版只打印 rms_rel 却不拦 ⇒ 5.1 µs / 10.5 ms 的假数被当成结果 ✗
        if not (r < 1e-2):
            print("  \u26d4 T=%-3d rms_rel=%.3e **不达标** \u21d2 \u2b50 \u672c\u6b21\u8ba1\u65f6\u4f5c\u5e9f\uff08\u6ca1\u7b97 or \u5f62\u72b6\u9519\uff09\uff0c\u4e0d\u91c7\u4fe1" % (T, r))
            continue

        rows.append((T, t_cpu, t_cpu_min, t_gpu, t_gpu_min, r))
        print("  T=%-3d  T_cpu=%8.1f  T_gpu(含同步)=%8.1f  ⭐T_gpu(无同步)=%8.1f µs"
              "   比值 cpu/gpu_async=%5.2f  比值 cpu/gpu_sync=%5.2f  rms_rel=%.2e ✅"
              % (T, t_cpu, t_gpu, t_gpu_async, t_cpu / max(t_gpu_async, 1e-9), t_cpu / max(t_gpu, 1e-9), r))

    if not rows:
        print("  \u26d4 \u6ca1\u6709\u4efb\u4f55\u4e00\u884c\u901a\u8fc7 rms_rel<1e-2 \u786c\u95e8 \u21d2 \u2b50 **\u672c\u6b21\u6d4b\u91cf\u6574\u4f53\u4f5c\u5e9f**")
        print("     \u21d2 \u68c0\u67e5:out buffer \u884c\u6570 >= qlen?ids \u662f\u5426\u843d\u5728\u88ab pin \u7684 S \u4e2a\u4e13\u5bb6\u5185?")
        return 1

    print("-" * 78)
    print("  \u2b50 \u5224\u8bfb")
    t4 = [r for r in rows if r[0] == 4][0]
    ratio = t4[1] / max(t4[3], 1e-9)
    if ratio > 3:
        print("    ⭐⭐ T_gpu 比 T_cpu 快 %.1f× ⇒ ⭐ 抬 p 仍有【大】空间:"
              "p→1 时墙钟应趋近 T_gpu+sync" % ratio)
    elif ratio > 1.3:
        print("    ⭐ T_gpu 比 T_cpu 快 %.1f× ⇒ 抬 p 收益有限,但 `max()` 仍会随 p 下降" % ratio)
    else:
        print("    ⛔ T_gpu 与 T_cpu 同量级或更慢(%.2f×)⇒ ⭐ 池的收益【已被 GPU 侧成本吃掉】,"
              "必须改内核/减同步,而不是继续抬 p" % ratio)
    print("    提醒:B932 模型 T = max(T_gpu(p), T_cpu(1-p)) + T_sync;")
    print("          本脚本的 T_gpu 是【无 miss、单层、单次】的下界,真机还要乘层数并加同步。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
