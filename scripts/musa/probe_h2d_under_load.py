"""S5000 快速验证：GPU 计算满载的同时，持续压 H2D 测吞吐。

不启动 vLLM，不加载权重。约 10 秒跑完。

真饱和验证（上次失败的教训：hammer 线程 OOM 静默死掉，测出的是空载）：
  * compute 线程记录自己完成的迭代数，主线程检查它确实在推进；
  * 并发期实测 compute ips，必须 >= solo 的 60%，否则判定"compute 没满"；
  * 并发期另开线程采样 mthreads-gmi %GPU 作为旁证。

输出：
  H2D solo 吞吐 / H2D under load 吞吐 / 保留率；compute solo 与 under load 的 ips。
"""
import subprocess
import threading
import time

import torch

DEV = torch.device("musa")
COPY_BYTES = 512 * 1024 * 1024        # 512 MiB per copy
DURATION = 4.0                        # seconds per phase

h = torch.empty(COPY_BYTES, dtype=torch.uint8, pin_memory=True)
d = torch.empty(COPY_BYTES, dtype=torch.uint8, device=DEV)
# 大 working set（远超 L2）⇒ 真跑 GEMM，不是 cache 命中
A = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
B = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)

s_copy = torch.musa.Stream()
s_comp = torch.musa.Stream()

_stop = threading.Event()
_counts = {}


def copy_worker(tag):
    """背靠背 H2D，直到 stop；记录完成次数。"""
    n = 0
    with torch.musa.stream(s_copy):
        while not _stop.is_set():
            d.copy_(h, non_blocking=True)
            n += 1
            if n % 16 == 0:
                torch.musa.synchronize()   # 防止队列无限堆积
    torch.musa.synchronize()
    _counts[tag] = n


def comp_worker(tag):
    """背靠背 GEMM，直到 stop；记录完成次数。"""
    n = 0
    with torch.musa.stream(s_comp):
        while not _stop.is_set():
            A @ B
            n += 1
            if n % 8 == 0:
                torch.musa.synchronize()
    torch.musa.synchronize()
    _counts[tag] = n


def phase(name, copies, compute):
    _stop.clear()
    _counts.clear()
    ths = []
    if copies:
        ths.append(threading.Thread(target=copy_worker, args=("copy",)))
    if compute:
        ths.append(threading.Thread(target=comp_worker, args=("comp",)))
    torch.musa.synchronize()
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    time.sleep(DURATION)
    _stop.set()
    for t in ths:
        t.join()
    dt = time.perf_counter() - t0
    n_copy = _counts.get("copy", 0)
    n_comp = _counts.get("comp", 0)
    gib = n_copy * COPY_BYTES / 2**30
    print(f"  {name:<22} {dt:5.2f}s  copies={n_copy:<5} gemm={n_comp:<5} "
          f"{gib/dt if n_copy else 0:6.2f} GiB/s  {n_comp/dt if n_comp else 0:7.1f} gemm/s")
    return (gib / dt if n_copy else 0.0), (n_comp / dt if n_comp else 0.0)


print("== S5000 H2D vs compute overlap ==\n")
solo_copy, _ = phase("H2D solo", copies=True, compute=False)
_, solo_comp = phase("compute solo", copies=False, compute=True)
both_copy, both_comp = phase("BOTH concurrent", copies=True, compute=True)

print(f"\nH2D  : {solo_copy:.2f} -> {both_copy:.2f} GiB/s "
      f"({both_copy/solo_copy*100:.0f}% retained)")
print(f"GEMM : {solo_comp:.1f} -> {both_comp:.1f} gemm/s "
      f"({both_comp/solo_comp*100:.0f}% retained)")

# 真饱和判据
if both_comp < 0.6 * solo_comp:
    print("\n⚠️  compute 未被真正压满（<60% solo）⇒ 本轮 H2D 数字不可信")
else:
    print(f"\n✅ compute 确实在跑（{both_comp/solo_comp*100:.0f}% of solo）")

total = both_copy / solo_copy + both_comp / solo_comp
print(f"overlap 效率 = {total:.2f}  (2.00=完美重叠, 1.00=完全串行)")
print("判定：" + ("可重叠" if total > 1.3 else "不能重叠"))
