#!/usr/bin/env python
"""⛔ B947:量 `T_gpu` 的**同步地板** —— 判断 B941/B942 的数是否代表图内真实成本。

## 为什么要做这件事

B941/B942 用 "调用 + `torch.cuda.synchronize()`" 的方式测出 `T_gpu`:
qlen=4 ⇒ **3.17 ms**、qlen=8 ⇒ 1.87–2.57 ms、qlen=16 ⇒ 1.96–2.37 ms。

⭐ 但那 3.17 ms 里**有多少是 GPU 真的在算**,有多少是
**主机↔设备同步的固定代价**,此前**从未测过**。

⭐ 这件事**决定性**:
  * 若同步地板 ≈ 2–3 ms  ⇒ ⭐ **B941/B942 的 `T_gpu` 几乎全是同步**
    ⇒ ⭐ 而真机在 CUDA graph 里**没有逐次同步** ⇒
    ⇒ ⭐ ⭐ **那些数【不代表】图内真实成本,池在小 qlen 未必是负收益** ✓
  * 若同步地板 ≪ 0.5 ms ⇒ ⭐ 3.17 ms 确实是 GPU 侧真实开销
    ⇒ ⭐ B944 的结论(必须降 `T_gpu`)成立 ✓

## 口径(必须写清,R43)

* 设备:GPU0(`CUDA_VISIBLE_DEVICES=0`),`XIAOTU_ENV_FILE=/dev/null` 清环境桥污染
* 三个量,**同一 `timeit` 函数**测,便于直接相减:
  1. ⭐ **空地板**:`torch.empty(1, device=cuda)` —— 一次分配(几乎无 GPU 工作)
  2. ⭐ **1 个小核**:`a.add_(1)` on a 1-element tensor —— 一次真实内核启动
  3. ⭐ **同步 + 64 MiB 拷贝**:`t.clone()`(纯带宽,便于判断"是不是带宽")
* ⭐ 判据:`T_floor` = 地板;`T_gpu − T_floor` = **扣除同步后的真实 GPU 成本**

## ⛔ 自检(吸收 B940 的教训)

⭐ 本脚本**不算任何 MoE**,所以**没有 rms_rel 门** —— 但**必须证明它真的启动了内核**:
⭐ 用 `torch.cuda.synchronize()` 后读回 `a.item()` 并 `assert` 值变了 ✓
⛔ 若 `assert` 失败 ⇒ **立即报错退出**,不输出任何时间 ✗
"""
import os
import sys
import time

import torch


def timeit(fn, iters=60, warmup=15):
    """(中位 µs, 最小 µs) —— 含同步,与 B941/B942 同口径 ✓"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()      # ⭐ 与 B941/B942 逐字相同的同步点
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def main() -> int:
    dev = torch.device("cuda:0")
    print("=" * 78)
    print("  B947  T_gpu 的【同步地板】测量  (判断 B941/B942 的数是否代表图内成本)")
    print("  ⭐ 判据:若地板 ≈ 2–3 ms ⇒ B941/B942 的 T_gpu 几乎全是同步 ⇒ 那些数不代表图内真实 ✓")
    print("=" * 78)
    print(f"  device={torch.cuda.get_device_name(0)}  torch={torch.__version__}")

    # ---- 1) 空地板:只有一次 cudaMalloc/cache 命中 ----
    warm = torch.empty(1, device=dev)
    t_empty, t_empty_min = timeit(lambda: torch.empty(1, device=dev))

    # ---- 2) 一次真实的小内核(必须证明它真的算了)----
    a = torch.zeros(1, device=dev)
    t_kernel, t_kernel_min = timeit(lambda: a.add_(1.0))
    torch.cuda.synchronize()
    got = float(a.item())
    # ⭐ 自检:60+15 次 add_ ⇒ 值必须 = iters+warmup
    expect = float(60 + 15 + 15)      # timeit 内的 warmup(15) + iters(60) + 上面那次 a.add_… 见下
    # ⚠️ 精确值依赖实现 ⇒ 只要求"远大于 0"且"非退化"
    assert got > 50.0, f"⛔ 自检失败:add_ 只累加到 {got} ⇒ 内核没真的跑,计时无效"
    print(f"  ✅ 自检:add_ 累加到 {got}(>50)⇒ 内核真的在跑,计时有效")

    # ---- 3) 纯带宽:64 MiB clone(64*1024*1024/4 = 16 Mi 个 fp32)----
    big = torch.empty(16 * 1024 * 1024, device=dev)
    t_copy, t_copy_min = timeit(lambda: big.clone(), iters=25, warmup=6)
    gib = big.numel() * 4 / (1024 ** 3)
    # clone 读写各一遍 ⇒ 有效带宽 = 2 * 字节 / 时间
    bw = (2 * gib) / (t_copy * 1e-6)
    print()

    # ---- 汇总 ----
    print("-" * 78)
    print(f"  ① 空地板(torch.empty(1))       中位={t_empty:9.1f} µs   min={t_empty_min:9.1f} µs")
    print(f"  ② 单内核(a.add_(1))            中位={t_kernel:9.1f} µs   min={t_kernel_min:9.1f} µs")
    print(f"  ③ 64MiB clone                  中位={t_copy:9.1f} µs   min={t_copy_min:9.1f} µs"
          f"   ⇒ 有效带宽 {bw:6.1f} GiB/s")

    print("-" * 78)
    print("  ⭐ 与 B941/B942 的 T_gpu 相减:")
    for q, tg in (("4", 3174.6), ("8", 1874.5), ("16", 1962.0)):
        net = tg - t_kernel
        print(f"     qlen={q:>2s}: T_gpu={tg:8.1f} µs  −  单内核地板 {t_kernel:7.1f}"
              f"  ⇒ 真实(多内核)≈ {net:8.1f} µs"
              f"   [地板占比 {100.0 * t_kernel / tg:5.1f}%]")

    print("-" * 78)
    # ⭐ 结论只用"地板 / T_gpu"的比值,不做超出测量的推断
    frac = t_kernel / 3174.6
    if frac > 0.5:
        print(f"  ⭐ ⭐ 判读:地板占 qlen=4 的 T_gpu 的 {100*frac:.0f}%"
              f" ⇒ **B941/B942 的 T_gpu 以同步开销为主** ⇒")
        print("           ⛔ 那些数【不代表】CUDA graph 内的真实成本")
        print("           ⇒ ⭐ B944「小 qlen 走池是负收益」的结论【不能只靠步时外的离线测量支撑】")
        print("           ⇒ ⭐ 必须改用【图内】口径复核(例如同 qlen 下 PIN=0 vs PIN=1 的步时差)✓")
    else:
        print(f"  ⭐ ⭐ 判读:地板只占 qlen=4 的 T_gpu 的 {100*frac:.0f}%"
              f" ⇒ **同步不是主因** ⇒")
        print("           ⭐ 3.17 ms 确实是 GPU 侧真实开销 ⇒ B944 的结论成立")
        print("           ⇒ ⭐ 方向确认:必须降低 GPU 侧内核/寻址成本 ✓")
    print("=" * 78)
    print("  ⛔ 证据等级:①②③ = 实测(本机 GPU0,env 干净,含 add_ 自检 ✓);")
    print("     与 B941/B942 相减 = 推算(⚠️ 两者时间不同轮,存在 run-to-run 方差 ✓)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
