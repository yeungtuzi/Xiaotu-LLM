"""验证 S5000 能否原生跑 FP8 矩阵乘（跳过 bf16 反量化）。

当前 _grouped_seq_matmul 链路：fp8 权重 -> decode 成 bf16 缓冲 -> bmm(bf16)
本探针测：bmm 是否接受 fp8，以及 fp8 直算与 decode+bf16 的耗时对比。

跑法：docker exec qfn-tp1 python3 -u /home/mccuseradmin/s5000/probe_fp8_grouped.py
"""
import time

import torch

dev = torch.device("musa")
FP8 = torch.float8_e4m3fn
H, I = 2560, 640
TWOH = 2 * I
ROWS = 320


def bench(fn, reps=50, warm=5):
    for _ in range(warm):
        fn()
    torch.musa.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.musa.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def show(tag, fn):
    try:
        out = fn()
        torch.musa.synchronize()
        print(f"  {tag:<56} OK  shape={tuple(out.shape)} dtype={out.dtype}")
        return out
    except Exception as e:                                   # noqa: BLE001
        print(f"  {tag:<56} FAIL {type(e).__name__}: {str(e).splitlines()[0][:90]}")
        return None


print("== S5000 FP8 matmul probe ==")

xB = torch.randn(ROWS, H, device=dev, dtype=torch.bfloat16)
wB = torch.randn(H, TWOH, device=dev, dtype=torch.bfloat16)
xF = xB.to(torch.float8_e4m3fn)
wF = xB.new_empty(0)  # placeholder to avoid lint
wF = wB.to(torch.float8_e4m3fn)

print("\nA. dtype support")
show("torch.mm(fp8, fp8)", lambda: torch.mm(xF, wF))

print("\nB. per-tile timing (qfn geometry, 320 rows)")
print(f"  fp8 mm : {bench(lambda: torch.mm(xF, wF)):8.3f} ms")
print(f"  bf16 mm: {bench(lambda: torch.mm(xB, wB)):8.3f} ms")

# C. 量化/反量化开销
print("\nC. decode cost")
print(f"  fp8->bf16 cast : {bench(lambda: xF.to(torch.bfloat16)):8.3f} ms")

# D. 大形状：模拟整层 GEMM
print("\nD. layer-scale GEMM (15k rows)")
xl_f8 = torch.randn(15000, H, device=dev, dtype=FP8)
xl_bf = torch.randn(15000, H, device=dev, dtype=torch.float16)
w_l_f8 = torch.randn(H, TWOH, device=dev, dtype=FP8)
w_l_bf = torch.randn(H, TWOH, device="musa", dtype=torch.float16)
print(f"  fp8  (15000x2560 @ 2560x1280): {bench(lambda: torch.mm(xl_f8, w_l_f8), reps=10):8.3f} ms")
print("  f16  (15000x2560 @ 2560x1280): " + f"{bench(lambda: torch.mm(xl_bf, w_l_bf), reps=10):8.3f} ms")
