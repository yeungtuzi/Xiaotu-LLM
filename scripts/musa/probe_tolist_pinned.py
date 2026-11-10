"""验证假设：大 pinned 占用会让 `Tensor.tolist()` 变慢，而复用 pinned 缓冲能规避。

背景（§9.78）：引擎里 `seg_start.tolist()`（513 个 int32）卡 3 秒，而独立 probe 里只有 0.12 ms。
引擎与 probe 的差别是引擎已有约 87 GB pinned 缓存；`.tolist()` 内部的 `.cpu()`
**每次都要新分配一块 pinned 暂存**（`cudaHostAlloc`）。

本 probe 逐步抬高 pinned 占用，比较两条路径：
    A) `seg.tolist()`                          —— 每次新 pin
    B) 预分配 pinned 复用 + `copy_` + `tolist()` —— 不碰分配器
判据：
    * A 随 pinned 占用显著变慢（几十~几百 ms 以上）、B 保持微秒 ⇒ 假设成立，B 就是修复
    * A 始终 0.1 ms ⇒ 假设不成立
"""
import time

import torch

dev = torch.device("musa")
E = 512
N = E + 1
MB = 1024 * 1024
GB = 1024 * MB


def p(*a):
    print(*a, flush=True)


def med(fn, reps=15):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[-1]


seg = torch.arange(N, dtype=torch.int32, device=dev)
pin_reuse = torch.empty(N, dtype=torch.int32, pin_memory=True)

p(f"== pinned 占用对 `tolist()` 的影响 ==  {N} 个 int32\n")
p(f"{'pinned 占用':>14} | {'A 每次新 pin':>22} | {'B 复用缓冲':>22}")
p("-" * 66)

blocks = []
STEP = 4 * GB
for k in range(0, 18):                      # 0, 4, 8, ... 68 GB
    if k:
        try:
            blocks.append(torch.empty(STEP, dtype=torch.uint8, pin_memory=True))
        except Exception as exc:            # noqa: BLE001
            p(f"  ⚠️ pinned 分配到 {k*4} GB 时失败: {exc!r}")
            break
    used = len(blocks) * 4
    a_med, a_max = med(lambda: seg.tolist())
    b_med, b_max = med(lambda: (pin_reuse.copy_(seg), pin_reuse.tolist()))
    p(f"{used:>11} GB | {a_med:9.3f} / {a_max:8.3f} ms | {b_med:9.3f} / {b_max:8.3f} ms")
    if a_med > 200:
        p("  ⛔ A 已超过 200 ms ⇒ 假设成立方向")
        break

p("\n== 汇总 ==")
p("  A 的 median 随 pinned 占用上升、B 基本不变 ⇒ 修复方案 = 复用 pinned 缓冲")
p("  A 与 B 都平稳 ⇒ 假设不成立，需回到引擎流状态继续查")
