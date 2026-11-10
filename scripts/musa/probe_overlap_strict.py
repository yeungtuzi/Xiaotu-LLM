"""S5000: 能否在 compute 运行时并发 H2D？严格重叠测试。

上一版 probe_cmp3 的 "GPU busy" 臂无效：hammer 线程 OOM 死了 ⇒ GPU 是空的。
本版：双线程各自喂满一条流，用「单独跑」与「并发跑」的墙钟比来判定重叠。

判据：
  eff = (t_copy_alone + t_compute_alone) / t_both
    eff ≈ 1.0 ⇒ 完全串行（不能重叠）
    eff ≈ 2.0 ⇒ 完全重叠
"""
import threading
import time

import torch


dev = torch.device("musa")

N = 512 * 1024 * 1024          # 512 MiB per H2D
M = 2048                       # matmul size

h = torch.empty(N, dtype=torch.uint8, pin_memory=True)
d = torch.empty(N, dtype=torch.uint8, device=dev)
a = torch.randn(M, M, device=dev, dtype=torch.bfloat16)
b = torch.randn(M, M, device=dev, dtype=torch.bfloat16)

s_copy = torch.musa.Stream()
s_comp = torch.musa.Stream()


def copy_loop(n):
    with torch.musa.stream(s_copy):
        for _ in range(n):
            d.copy_(h, non_blocking=True)


def comp_loop(n):
    with torch.musa.stream(s_comp):
        for _ in range(n):
            a @ b


def timeit(fn, n):
    fn(1)
    torch.musa.synchronize()
    t0 = time.perf_counter()
    fn(n)
    torch.musa.synchronize()
    return time.perf_counter() - t0


# ---- 标定速率 ----
t = timeit(copy_loop, 8)
copy_rate = 8 * N / t / 2**30          # GiB/s
print(f"copy alone : 8 x 512MiB in {t:.3f}s => {copy_rate:.2f} GiB/s")

t = timeit(comp_loop, 30)
comp_rate = 30 / t                      # matmul/s
print(f"comp alone : 30 matmul in {t:.3f}s => {comp_rate:.2f} matmul/s")

# 选 n 让两臂各约 2.0 s
n_copy = max(1, int(round(2.0 * copy_rate * 2**30 / N)))
n_comp = max(1, int(round(2.0 * comp_rate)))
print(f"\nchosen: n_copy={n_copy} (~{n_copy*N/2**30/copy_rate:.2f}s), "
      f"n_comp={n_comp} (~{n_comp/comp_rate:.2f}s)")

# ---- 单独跑 ----
t_copy = timeit(copy_loop, n_copy)
print(f"\ncopy alone    : {t_copy:.3f}s")
t_comp = timeit(comp_loop, n_comp)
print(f"comp alone    : {t_comp:.3f}s")

# ---- 并发跑 ----
torch.musa.synchronize()
t0 = time.perf_counter()
th_c = threading.Thread(target=copy_loop, args=(n_copy,))
th_m = threading.Thread(target=comp_loop, args=(n_comp,))
th_c.start()
th_m.start()
th_c.join()
th_m.join()
torch.musa.synchronize()
t_both = time.perf_counter() - t0
print(f"concurrent    : {t_both:.3f}s")

eff = (t_copy + t_comp) / t_both
print(f"\noverlap eff = ({t_copy:.2f} + {t_comp:.2f}) / {t_both:.2f} = {eff:.2f}")
print(f"  theoretical perfect = 2.00, fully serial = 1.00")
print(f"  => {'OVERLAP WORKS' if eff > 1.3 else 'NO OVERLAP (serialized)'}")

# ---- 并发期间各自的实测速率 ----
print(f"\nwhile concurrent:")
print(f"  copy achieved   : {n_copy*N/t_both/2**30:.2f} GiB/s "
      f"({n_copy*N/t_both/2**30/copy_rate*100:.0f}% of solo)")
print(f"  comp achieved   : {n_comp/t_both:.2f} matmul/s "
      f"({n_comp/t_both/comp_rate*100:.0f}% of solo)")
