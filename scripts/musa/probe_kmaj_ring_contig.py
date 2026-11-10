"""Probe: 3 strided slice H2D vs 1 contiguous H2D into the same K-major ring.

Decides whether "C++ host K-major -> write contiguous ring" can pay off at all.
No vLLM, no weights: synthetic buffers, same shapes as qfn (H=2560 I=640 ns=2).
"""
import time
import torch

dev = torch.device("musa")
H, I, NS = 2560, 640, 2
cr13 = I // NS          # 320 rows per node
tile = 48               # experts per tile (default staging tile)

def bench(fn, n=20):
    fn()
    torch.musa.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.musa.synchronize()
    return (time.perf_counter() - t0) / n

# ---- ring slot: [tile, H, 2I] contiguous (K-major) ----
ring = torch.empty((tile, H, 2 * I), dtype=torch.uint8, device=dev)

# ---- host side A: 3 strided slices (what Python does today) ----
# pinned buffer holding [tile, H, 2, cr13]  (already transposed on CPU)
host_tr = torch.empty((tile, H, 2, cr13), dtype=torch.uint8, pin_memory=True)

def h2d_strided():
    for n in range(NS):
        c0 = n * cr13
        ring[:, :, c0:c0 + cr13].copy_(host_tr[:, :, 0, :], non_blocking=True)
        ring[:, :, I + c0:I + c0 + cr13].copy_(host_tr[:, :, 1, :], non_blocking=True)

# ---- host side B: full contiguous K-major tile ----
host_full = torch.empty((tile, H, 2 * I), dtype=torch.uint8, pin_memory=True)

def h2d_contig():
    ring[:, :].copy_(host_full, non_blocking=True)

nbytes = tile * H * 2 * I / 2**20
t_a = bench(h2d_strided)
t_b = bench(h2d_contig)
print(f"tile={tile}  bytes={nbytes:.1f} MiB per H2D")
print(f"3-slice strided : {t_a*1000:8.2f} ms  => {nbytes/1024/t_a:6.2f} GiB/s")
print(f"1 contiguous    : {t_b*1000:8.2f} ms  => {nbytes/1024/t_b:6.2f} GiB/s")
print(f"speedup = {t_a/t_b:.2f}x")

# ---- also measure the CPU transpose cost that the C++ path would pay ----
src = torch.empty((tile, 2, cr13, H), dtype=torch.uint8)   # pageable source (shard layout)
def cpu_tr():
    src.permute(0, 3, 1, 2).contiguous()
t_c = bench(cpu_tr, n=5)
print(f"CPU transpose (pageable->contig): {t_c*1000:8.2f} ms => {nbytes/1024/t_c:6.2f} GiB/s")
