"""测 grouped GEMM 的真实成本（每 tile / 每层）。

背景：引擎里 `_grouped_chunk` 的 bmm1/bmm2 只用 perf_counter 记了 **CPU 入队时间**，
`total=93ms/8tile` 那个数字不是 GPU 时间。这里用同步计时拿真值。

形状取引擎实测口径：E=512, tile=48, H=2560, I=640,
prefill 15394 tok × topk 10 = 153,940 行，均摊到 11 个 tile ⇒ 每 tile ≈ 14,000 行。
"""
import time

import torch

dev = torch.device("musa")
H, I = 2560, 640
ne = 48
N_TILES = 11
TOTAL_ROWS = 15394 * 10          # 153,940
rows = TOTAL_ROWS // N_TILES     # ~13995

print(f"== grouped GEMM 实测 ==  rows/tile={rows} ne={ne} H={H} I={I}\n")

gx = torch.randn(rows, H, device=dev, dtype=torch.bfloat16)
act = torch.randn(rows, I, device=dev, dtype=torch.bfloat16)

# w13: [ne, H, 2I]  (K-major 视图；引擎里是 reshape(ne,2I,H).transpose(1,2))
w13 = torch.randn(ne, H, 2 * I, device=dev, dtype=torch.bfloat16)
w2 = torch.randn(ne, I, H, device=dev, dtype=torch.bfloat16)

# offs = 每个专家的**结束行号**（升序，末项 = rows）
per = rows // ne
offs = torch.tensor([per * (k + 1) for k in range(ne)] + ([rows] if per * ne != rows else []),
                    dtype=torch.int32, device=dev)
if int(offs[-1]) != rows:
    offs = torch.cat([offs, torch.tensor([rows], dtype=torch.int32, device=dev)])
offs = offs[:ne]
offs[-1] = rows


def bench(fn, reps=5):
    fn()
    torch.musa.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.musa.synchronize()
    return (time.perf_counter() - t0) / reps


def bmm1():
    return torch._grouped_mm(gx, w13, offs=offs)


def bmm2():
    return torch._grouped_mm(act, w2, offs=offs)


t1 = bench(bmm1)
t2 = bench(bmm2)

fl1 = 2 * rows * H * (2 * I)
fl2 = 2 * rows * I * H

print(f"bmm1: {t1*1000:8.2f} ms/tile   {fl1/t1/1e12:6.2f} TFLOPS")
print(f"bmm2: {t2*1000:8.2f} ms/tile   {fl2/t2/1e12:6.2f} TFLOPS")
print(f"\n每层 ({N_TILES} tiles):")
print(f"  bmm1 {t1*N_TILES*1000:8.1f} ms")
print(f"  bmm2 {t2*N_TILES*1000:8.1f} ms")
print(f"  合计 {(t1+t2)*N_TILES*1000:8.1f} ms/layer")

# 其余小算子：gather / act / scatter
tok_sorted = torch.randint(0, 15394, (rows,), device=dev)
x = torch.randn(15394, H, device=dev, dtype=torch.bfloat16)


def gather():
    return x[tok_sorted.long()]


tg = bench(gather)


def activate():
    g = act[:, :I]
    u = act[:, I:]
    return (g * torch.sigmoid(g)) * u


ta = bench(activate)

out = torch.zeros(15394, H, device=dev, dtype=torch.float32)
wts = torch.rand(rows, device=dev)
dn = torch.randn(rows, H, device=dev, dtype=torch.float32)


def scatter():
    out.index_add_(0, tok_sorted.long(), dn * wts[:, None])


tsc = bench(scatter)

print(f"\n其他（每 tile）:")
print(f"  gather  {tg*1000:8.2f} ms  -> {tg*N_TILES*1000:7.1f} ms/layer")
print(f"  act     {ta*1000:8.2f} ms  -> {ta*N_TILES*1000:7.1f} ms/layer")
print(f"  scatter {tsc*1000:8.2f} ms  -> {tsc*N_TILES*1000:7.1f} ms/layer")
print(f"\n计算侧合计 ≈ {(t1+t2+tg+ta+tsc)*N_TILES*1000:.1f} ms/layer")

# 对照：peak bf16
M = 4096
a = torch.randn(M, M, device=dev, dtype=torch.bfloat16)
b = torch.randn(M, M, device=dev, dtype=torch.bfloat16)
tp = bench(lambda: a @ b, reps=20)
print(f"\n对照 4096 matmul 峰值: {2*M**3/tp/1e12:.2f} TFLOPS ({tp*1000:.2f} ms)")
