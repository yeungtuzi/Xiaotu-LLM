"""定位引擎里 H2D 只有 5.1 GB/s 的原因（不启动 vLLM）。

引擎的实测事实（§9.54）：
    w13 单 tile 157 MB / 30.7 ms = 5.1 GB/s
    w2  单 tile 78.6 MB / 15.4 ms = 5.1 GB/s
而独立探针在**新进程主线程**里同样 157 MB 只要 2.8 ms（≈56 GiB/s）。

差异只在调用位置：引擎里 `_host_km_fill` 跑在 **CPT producer 线程**，
且用的是 `_prefetch_stream(device, slot)` 这条**非默认流**。

本探针做 3×2 拆解：
    (A) 主线程分配 pinned + 主线程默认流拷贝     —— 基线
    (B) 子线程分配 pinned + 子线程专用流拷贝     —— 模拟引擎
    (C) 主线程分配 pinned + 子线程专用流拷贝     —— 隔离"谁分配的"

判据：
    若 (B) 慢而 (C) 快  ⇒ 是"在子线程里 pin"失效
    若 (B)(C) 都慢      ⇒ 是"非默认流/子线程发 DMA"的问题
    若 (B)(C) 都快      ⇒ 引擎里还有别的原因（回到引擎内部查）
"""
import threading
import time

import torch

dev = torch.device("musa")
MB = 1024 * 1024
N = 1536 * MB // 8 * 8        # 与引擎 w13 tile 同量级：157 MB
N = 157 * MB
REPS = 12


def bw_run(dst, src, stream=None, reps=REPS):
    """在指定流上背靠背拷 reps 次，返回 GB/s（按 1 次读 + 1 次写不重复计，只算搬移字节）。"""
    def go():
        for _ in range(reps):
            dst.copy_(src, non_blocking=True)
    if stream is None:
        go()
    else:
        with torch.musa.stream(stream):
            go()
    torch.musa.synchronize()
    t0 = time.perf_counter()
    if stream is None:
        go()
    else:
        with torch.musa.stream(stream):
            go()
    torch.musa.synchronize()
    dt = time.perf_counter() - t0
    return reps * N / dt / 1e9, dt / reps * 1e3


def p(*a):
    print(*a, flush=True)


p(f"== H2D 5.1 GB/s 归因 ==  单次 {N/2**20:.0f} MiB, reps={REPS}\n")

# ── (A) 主线程分配 + 主线程默认流 ──
srcA = torch.empty(N, dtype=torch.uint8, pin_memory=True)
dstA = torch.empty(N, dtype=torch.uint8, device=dev)
p(f"A 主线程 pin: is_pinned={srcA.is_pinned()}  ptr%4096={srcA.data_ptr()%4096}")
gb, ms = bw_run(dstA, srcA)
resA = {"gb": gb, "ms": ms}
p(f"   -> {gb:6.2f} GB/s   {ms:7.2f} ms/次\n")

# ── (C) 主线程分配 + 子线程专用流 ──
resC = {}


def worker_C():
    st = torch.musa.Stream()
    dst = torch.empty(N, dtype=torch.uint8, device=dev)
    gb, ms = bw_run(dst, srcA, stream=st)
    resC["gb"], resC["ms"] = gb, ms


th = threading.Thread(target=worker_C)
th.start()
th.join()
p(f"C 主线程分配的 pin，子线程专用流拷:")
p(f"   -> {resC['gb']:6.2f} GB/s   {resC['ms']:7.2f} ms/次\n")

# ── (B) 子线程分配 pin + 子线程专用流（模拟引擎）──
resB = {}


def worker_B():
    st = torch.musa.Stream()
    try:
        src = torch.empty(N, dtype=torch.uint8, pin_memory=True)
        resB["pinned"] = src.is_pinned()
    except Exception as exc:                      # noqa: BLE001
        resB["err"] = repr(exc)
        return
    dst = torch.empty(N, dtype=torch.uint8, device=dev)
    gb, ms = bw_run(dst, src, stream=st)
    resB["gb"], resB["ms"] = gb, ms
    resB["ptr"] = src.data_ptr() % 4096


th = threading.Thread(target=worker_B)
th.start()
th.join()
if "err" in resB:
    p(f"B 子线程 pin: 分配失败 {resB['err']}")
else:
    p(f"B 子线程 pin: is_pinned={resB['pinned']}  ptr%4096={resB.get('ptr')}")
    p(f"   -> {resB['gb']:6.2f} GB/s   {resB['ms']:7.2f} ms/次\n")

# ── 汇总判据 ──
p("== 汇总 ==")
p(f"  A = 主线程pin + 默认流      : {resA['gb']:6.2f} GB/s")
p(f"  C = 主线程pin + 子线程流    : {resC['gb']:6.2f} GB/s")
if "gb" in resB:
    p(f"  B = 子线程pin+子线程流     : {resB['gb']:6.2f} GB/s")
    if resB["pinned"] is False:
        p("  ⛔ B 的 is_pinned 为 False ⇒ **在子线程里 pin_memory 没有真正生效**")
    elif resB["gb"] < 0.5 * resC["gb"]:
        p("  ⛔ B 明显慢于 C ⇒ 问题在“子线程里分配 pin”这一步")
    elif resC["gb"] < 0.5 * 56:
        p("  ⛔ C 也慢 ⇒ 问题在“子线程/非默认流发 DMA”，与谁分配无关")
    else:
        p("  ✅ B、C 都接近满速 ⇒ 引擎里的 5.1 GB/s 另有原因，需回引擎内部查")
