"""S5000 overlap v2：确认 82% 效率可复现，并定位那 18% 损失。

假设 A：PCIe 与 HBM 争带宽（compute 的 working set 越大越差）
假设 B：固定开销（每次 copy 的提交/完成延迟）

做法：
  - 小 working set（L2 常驻的 matmul）vs 大 working set
  - 若小 working set 下 overlap 恢复到接近 2.0 ⇒ 是 HBM 争用
"""
import threading
import time

import torch

dev = torch.device("musa")
N = 512 * 1024 * 1024
MB = 1024 * 1024

h = torch.empty(N, dtype=torch.uint8, pin_memory=True)
d = torch.empty(N, dtype=torch.uint8, device=dev)

# 小 working set：512x512（每张 0.5 MiB，L2 常驻）
small = torch.randn(512, 512, device=dev, dtype=torch.bfloat16)
# 大 working set：4096x4096（每张 32 MiB，远超 L2）
big = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)

s_copy = torch.musa.Stream()
s_comp = torch.musa.Stream()


def copy_loop(n):
    with torch.musa.stream(s_copy):
        for _ in range(n):
            d.copy_(h, non_blocking=True)


def comp_small(n):
    with torch.musa.stream(s_comp):
        for _ in range(n):
            small @ small


def comp_big(n):
    with torch.musa.stream(s_comp):
        for _ in range(n):
            big @ big


def timed(fn, n):
    fn(1)
    torch.musa.synchronize()
    t0 = time.perf_counter()
    fn(n)
    torch.musa.synchronize()
    return time.perf_counter() - t0


def concurrent(fn_copy, n_copy, fn_comp, n_comp):
    torch.musa.synchronize()
    t0 = time.perf_counter()
    a = threading.Thread(target=fn_copy, args=(n_copy,))
    b = threading.Thread(target=fn_comp, args=(n_comp,))
    a.start(); b.start(); a.join(); b.join()
    torch.musa.synchronize()
    return time.perf_counter() - t0


copy_rate = 8 * N / timed(copy_loop, 8) / 2**30
print(f"copy solo: {copy_rate:.2f} GiB/s")
n_copy = max(1, int(round(2.0 * copy_rate * 2**30 / N)))

for label, fn in (("small-L2", comp_small), ("big-HBM", comp_big)):
    r = 40.0 / timed(fn, 40)
    n_comp = max(1, int(round(2.0 * r)))
    t_c = timed(copy_loop, n_copy)
    t_m = timed(fn, n_comp)
    t_b = concurrent(copy_loop, n_copy, fn, n_comp)
    eff = (t_c + t_m) / t_b
    print(f"\n[{label}] n_comp={n_comp}")
    print(f"  copy alone {t_c:.3f}s | comp alone {t_m:.3f}s | concurrent {t_b:.3f}s")
    print(f"  overlap eff = {eff:.2f}  (perfect 2.00 / serial 1.00)")
    print(f"  copy while busy = {n_copy*N/t_b/2**30:.2f} GiB/s "
          f"({n_copy*N/t_b/2**30/copy_rate*100:.0f}% of solo)")
    print(f"  comp while busy = {n_comp/t_b:.1f} ips ({n_comp/t_b/r*100:.0f}% of solo)")

# 复现性：同一配置跑 3 次
print("\n=== 复现性 (big-HBM x3) ===")
r = 40.0 / timed(comp_big, 40)
n_comp = max(1, int(round(2.0 * r)))
t_c = timed(copy_loop, n_copy)
t_m = timed(comp_big, n_comp)
for i in range(3):
    t_b = concurrent(copy_loop, n_copy, comp_big, n_comp)
    print(f"  run{i}: concurrent={t_b:.3f}s eff={(t_c+t_m)/t_b:.2f} "
          f"copy={n_copy*N/t_b/2**30:.1f}GiB/s")
