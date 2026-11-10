"""判别：MUSA 的同步 D2H（`.cpu()` / `.tolist()`）**是否等待其它流**。

为什么这个实验是关键（用户 2026-11-05 提问直接指向它）：
  引擎里 `bounds = seg_start.cpu()` 实测 30 / 462 / 3190 / 8470 ms，
  占每层墙钟的约 78%。而 `seg_start` 只有 513 个 int32。
  若这次同步会**等所有流**（legacy default stream 语义），
  那 `bounds` 等的就是 `_prefetch_stream` 上的预取 DMA 排空 ——
  也就是"整层权重流式传输"本身，而不是什么神秘的主机侧阻塞。

判据：
  * 侧流上排一个长拷贝，主流的 `.cpu()` 若跟着变慢  ⇒ **设备级同步** ⇒ bounds = DMA 排空
  * `.cpu()` 不受侧流影响                            ⇒ 流级同步   ⇒ 原结论（另有 90% 未知项）成立
"""
import time

import torch

dev = torch.device("musa")
MB = 1024 * 1024


def p(*a):
    print(*a, flush=True)


def med(fn, reps=9):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[-1]


small = torch.arange(513, dtype=torch.int32, device=dev)

side = torch.musa.Stream()
CAND = 1024 * MB          # 1 GB（引擎占着显存，只剩 ~2.9 GiB 可用）≈ 20 ms
h = torch.empty(CAND, dtype=torch.uint8, pin_memory=True)
d = torch.empty(CAND, dtype=torch.uint8, device=dev)

p("== MUSA 同步 D2H 是否等待其它流 ==\n")
a_med, a_max = med(lambda: small.cpu())
b_med, b_max = med(lambda: small.tolist())
p(f"  基线（侧流空闲）:  .cpu()={a_med:8.3f}ms   .tolist()={b_med:8.3f}ms")

# 侧流上排一个长拷贝，然后**立刻**在主流上做 D2H（不先同步）
res = []
for _ in range(6):
    with torch.musa.stream(side):
        for _ in range(3):
            d.copy_(h, non_blocking=True)
    t0 = time.perf_counter()
    small.cpu()
    res.append((time.perf_counter() - t0) * 1e3)
    torch.musa.synchronize()          # 收尾，保证下一次干净
res.sort()
c_med = res[len(res) // 2]
p(f"  侧流有 3 GB 在途:  .cpu()={c_med:8.3f}ms   （若≈60ms 即为设备级同步）")

res2 = []
for _ in range(6):
    with torch.musa.stream(side):
        for _ in range(3):
            d.copy_(h, non_blocking=True)
    t0 = time.perf_counter()
    small.tolist()
    res2.append((time.perf_counter() - t0) * 1e3)
    torch.musa.synchronize()
res2.sort()
d_med = res2[len(res2) // 2]
p(f"  侧流有 3 GB 在途:  .tolist()={d_med:8.3f}ms")

# 侧流上排一个长**计算**，同样测
M = 4096
A = torch.randn(M, M, device=dev, dtype=torch.bfloat16)
B = torch.randn(M, M, device=dev, dtype=torch.bfloat16)
res3 = []
for _ in range(6):
    with torch.musa.stream(side):
        for _ in range(4):
            A @ B
    t0 = time.perf_counter()
    small.cpu()
    res3.append((time.perf_counter() - t0) * 1e3)
    torch.musa.synchronize()
res3.sort()
e_med = res3[len(res3) // 2]
p(f"  侧流有 GEMM 在途:  .cpu()={e_med:8.3f}ms")

p("\n== 判据 ==")
p(f"  .cpu() 基线 {a_med:.3f}ms → 侧流有 DMA 时 {c_med:.3f}ms")
if c_med > max(10.0, 20 * a_med):
    p("  ⛔ **设备级同步**：`.cpu()` 会等侧流 ⇒ `bounds` 等的就是预取 DMA 排空")
    p("     ⇒ 用户判断正确：prefill 的主体就是整层流式传输")
elif c_med > 3 * a_med:
    p("  ⚠️ 部分等待，需要更细的判别")
else:
    p("  ✅ 流级同步：`.cpu()` 不等侧流 ⇒ 那 78% 另有来源")
p(f"  （参考：侧流 GEMM 时 {e_med:.3f}ms）")
