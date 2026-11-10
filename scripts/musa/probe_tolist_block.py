"""复现引擎里 `seg_start.tolist()` 的秒级阻塞（不启动 vLLM）。

引擎内已由栈 dump 定案（§9.78）：
    File ".../gpu_prefill_fp8.py", line 2178 in _gmlet_impl   ← 正是 `_bounds = seg_start.tolist()`
    同刻 `dev <= 2 ms`（设备空闲）、GC=0、磁盘读=0、producer 空闲、无其它线程在跑
    而 `seg_start` 只有 513 个 int32

⇒ 513 个 int32 的 `.tolist()` 在设备空闲时不可能要 3 秒。
本 probe 在**独立进程**里复现同一调用，并对比三种可替代写法，判据：
    * probe 里也是数秒  ⇒ 是 MUSA `tolist`/同步语义问题 ⇒ 换预分配 pinned + 显式 copy 即可绕开
    * probe 里是微秒    ⇒ 与引擎流状态有关 ⇒ 去量 current_stream 与 legacy default stream 语义
"""
import time

import torch

dev = torch.device("musa")
E = 512
N = E + 1
MB = 1024 * 1024


def p(*a):
    print(*a, flush=True)


def stat(name, fn, reps=20):
    """跑 reps 次，返回 (median, max) ms。"""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    p(f"  {name:<46} median={ts[len(ts)//2]:8.3f}ms  max={ts[-1]:8.3f}ms")
    return ts[len(ts) // 2], ts[-1]


p(f"== `seg_start.tolist()` 复现 ==  {N} 个 int32, E={E}\n")

seg = torch.arange(N, dtype=torch.int32, device=dev)
seg_cpu = seg.cpu()
pin = torch.empty(N, dtype=torch.int32, pin_memory=True)

p("[场景 A] 裸调用")
stat("A1 seg.tolist()", lambda: seg.tolist())
stat("A2 seg.cpu().tolist()", lambda: seg.cpu().tolist())
stat("A3 numpy().tolist()", lambda: seg.cpu().numpy().tolist())
stat("A4 预分配 pinned + copy_ + tolist",
     lambda: (pin.copy_(seg_cpu), pin.tolist()))
p("")

p("[场景 B] 另一条流上有排队 DMA（复现引擎条件）")
side = torch.musa.Stream()
big_h = torch.empty(256 * MB, dtype=torch.uint8, pin_memory=True)
big_d = torch.empty(256 * MB, dtype=torch.uint8, device=dev)
# 先在侧流上排 4 个 256 MB 拷贝（≈ 20 ms 的量级）
with torch.musa.stream(side):
    for _ in range(4):
        big_d.copy_(big_h, non_blocking=True)
stat("B1 侧流有排队时 seg.tolist()", lambda: seg.tolist(), reps=8)
torch.musa.synchronize()
p("")

p("[场景 C] 当前流是默认流 vs 非默认流")
s2 = torch.musa.Stream()
with torch.musa.stream(s2):
    stat("C1 非默认流上 seg.tolist()", lambda: seg.tolist())
p("")

p("[场景 D] 大量 tensor 存活（复现引擎的显存/对象压力）")
keep = [torch.empty(1024, dtype=torch.uint8, device=dev) for _ in range(20000)]
stat("D1 2 万个小 tensor 存活时 seg.tolist()", lambda: seg.tolist())
p("")

p("== 汇总判据 ==")
p("  若 A1/A4 都是微秒级、而引擎里是秒级 ⇒ 差异不在调用本身，而在引擎的流/上下文状态")
p("  若某条替代写法（尤其 A4 预分配 pinned）显著优于 A1 ⇒ 直接改用该写法")
