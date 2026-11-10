"""测 `_decode_ring_fp8_to_bf16` 的真实吞吐（GPU 时间）。

动机：引擎里 decode 只用 perf_counter 记了**CPU 入队时间**，没有 GPU event 埋点。
而每层要把 ~2.6 GB fp8 权重解码成 ~5.2 GB bf16（一次 tile 236 M 元素 × 11 tile）。
若它很慢，就是「604 ms/层 vs 事件只有 128 ms」那 476 ms 的真凶。

不启动 vLLM：按文件路径加载 `gpu_prefill_fp8`，绕开 __init__ 的 vllm 依赖。
"""
import importlib.util
import sys
import time
import types

import torch

ROOT = "/home/mccuseradmin/s5000/xiaotu-llm"
if "vllm_xiaotu_moe" not in sys.modules:
    pkg = types.ModuleType("vllm_xiaotu_moe")
    pkg.__path__ = [ROOT + "/vllm_xiaotu_moe"]
    sys.modules["vllm_xiaotu_moe"] = pkg
spec = importlib.util.spec_from_file_location(
    "gpf_dec", ROOT + "/vllm_xiaotu_moe/gpu_prefill_fp8.py")
gpf = importlib.util.module_from_spec(spec)
sys.modules["gpf_dec"] = gpf
spec.loader.exec_module(gpf)

dec = gpf._decode_ring_fp8_to_bf16
dev = torch.device("musa")

H, I = 2560, 640
ne = 48            # experts per tile
N_TILES = 11       # E=512 / tile=48

print("== decode fp8 -> bf16 吞吐 ==\n")


def bench(w, s, out, et, rows, n, reps=3):
    dec(w, s, out, et, rows, n)
    torch.musa.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        dec(w, s, out, et, rows, n)
    torch.musa.synchronize()
    return (time.perf_counter() - t0) / reps


trials = [
    ("w13 tile", ne, H, 2 * I),
    ("w2  tile", ne, I, H),
]

tot_sec = 0.0
tot_bytes = 0
for label, et, rows, n in trials:
    w = torch.randint(0, 200, (et, rows, n), dtype=torch.uint8, device=dev)
    s = torch.rand((et, rows // 128, n // 128), dtype=torch.float32, device=dev)
    out = torch.empty((et, rows, n), dtype=torch.bfloat16, device=dev)
    t = bench(w, s, out, et, rows, n)
    nbytes = et * rows * n
    rd = nbytes                    # fp8 1 B/elem
    wr = nbytes * 2                # bf16 2 B/elem
    tot = rd + wr
    per_layer = t * N_TILES
    tot_sec += per_layer
    tot_bytes += tot * N_TILES
    print(f"{label}: {et}x{rows}x{n} = {nbytes/2**20:.0f} M elem")
    print(f"   {t*1000:8.2f} ms/tile  HBM {tot/2**30/t:6.2f} GiB/s"
          f"  ({tot/t/1e9:6.1f} GB/s)")
    print(f"   x{N_TILES} tiles = {per_layer*1000:8.1f} ms/layer\n")

print(f"合计 decode: {tot_sec*1000:.1f} ms/layer   "
      f"平均 {tot_bytes/2**30/tot_sec:.2f} GiB/s")

# 对照：纯拷贝同样的 bf16 数据，看 HBM 上限
src = torch.empty((64 * 1024 * 1024,), dtype=torch.uint8, device=dev)
dst = torch.empty((64 * 1024 * 1024,), dtype=torch.uint8, device=dev)
dst.copy_(src)
torch.musa.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    dst.copy_(src)
torch.musa.synchronize()
dt = (time.perf_counter() - t0) / 5
print(f"\n对照 D2D copy 上限: {2*src.numel()/2**30/dt:.2f} GiB/s "
      f"(读+写都算)")
