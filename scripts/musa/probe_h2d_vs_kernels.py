"""验证：DMA 与 HBM 受限内核并发时，H2D 是否掉到 ~5 GB/s（不启动 vLLM）。

引擎实测（§9.54）：单 tile 157 MB H2D = 30.7 ms ⇒ **5.1 GB/s**。
独立探针：同一份 157 MB、pinned、子线程、非默认流 ⇒ **57.2 GB/s**（§9.55）。
两者剩下的唯一差异 = **同期另一条流上的真实内核在跑**。

本探针三档，隔离"是哪种内核把 DMA 拖慢"：
  P1  H2D 单独                       —— 应有 57 GB/s
  P2  H2D 与 **decode**(fp8->bf16) 并发  —— 引擎里每 tile 都跑，HBM 流量 708 MB
  P3  H2D 与 **grouped GEMM** 并发        —— 引擎里每 tile 都跑，算力受限

判据：哪一档把 H2D 压到 ~5 GB/s，就是它。
"""
import importlib.util
import sys
import threading
import time
import types

import torch

ROOT = "/home/mccuseradmin/s5000/xiaotu-llm"
if "vllm_xiaotu_moe" not in sys.modules:
    pkg = types.ModuleType("vllm_xiaotu_moe")
    pkg.__path__ = [ROOT + "/vllm_xiaotu_moe"]
    sys.modules["vllm_xiaotu_moe"] = pkg
spec = importlib.util.spec_from_file_location(
    "gpf_cc", ROOT + "/vllm_xiaotu_moe/gpu_prefill_fp8.py")
gpf = importlib.util.module_from_spec(spec)
sys.modules["gpf_cc"] = gpf
spec.loader.exec_module(gpf)

decode = gpf._decode_ring_fp8_to_bf16

dev = torch.device("musa")
MB = 1024 * 1024
H, I = 2560, 640
ne = 48
N13 = ne * H * 2 * I            # 157 MB  (fp8 w13 tile)
N2 = ne * I * H                 # 78.6 MB (fp8 w2 tile)


def p(*a):
    print(*a, flush=True)


p(f"== DMA 与内核并发对 H2D 的影响 ==  w13 tile = {N13/2**20:.0f} MiB\n")

# H2D 源/目的
hbuf = torch.empty(N13, dtype=torch.uint8, pin_memory=True)
dst = torch.empty(N13, dtype=torch.uint8, device=dev)

# decode 的输入/输出（与引擎同形状）
w13_fp8 = torch.randint(0, 200, (ne, H, 2 * I), dtype=torch.uint8, device=dev)
s13 = torch.rand((ne, H // 128, (2 * I) // 128), dtype=torch.float32, device=dev)
w13_bf = torch.empty((ne, H, 2 * I), dtype=torch.bfloat16, device=dev)

# grouped GEMM 的形状（每 tile ~14000 行）
rows = 14000
gx = torch.randn(rows, H, device=dev, dtype=torch.bfloat16)
gw = torch.randn(ne, H, 2 * I, device=dev, dtype=torch.bfloat16)
per = rows // ne
offs = torch.tensor([per * (k + 1) for k in range(ne)], dtype=torch.int32, device=dev)
offs[-1] = rows

sA = torch.musa.Stream()
sB = torch.musa.Stream()

DUR = 6.0


def run_phase(name, kernels):
    """sA 上背靠背 H2D；同时 sB 上背靠背跑 kernels（True=开）。"""
    stop = threading.Event()
    cnt = {"copy": 0, "k": 0}

    def copy_loop():
        with torch.musa.stream(sA):
            n = 0
            while not stop.is_set():
                dst.copy_(hbuf, non_blocking=True)
                n += 1
                cnt["copy"] = n
                if n % 8 == 0:
                    sA.synchronize()
            sA.synchronize()

    def kern_loop():
        with torch.musa.stream(sB):
            n = 0
            while not stop.is_set():
                if kernels == "decode":
                    decode(w13_fp8, s13, w13_bf, ne, H, 2 * I)
                elif kernels == "gemm":
                    torch._grouped_mm(gx, gw, offs=offs)
                n += 1
                cnt["k"] = n
                if n % 4 == 0:
                    sB.synchronize()
            sB.synchronize()

    torch.musa.synchronize()
    ths = [threading.Thread(target=copy_loop, daemon=True)]
    if kernels:
        ths.append(threading.Thread(target=kern_loop, daemon=True))
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    time.sleep(DUR)
    stop.set()
    for t in ths:
        t.join(timeout=30)
    dt = time.perf_counter() - t0
    gib = cnt["copy"] * N13 / 2**30
    gb = cnt["copy"] * N13 / dt / 1e9
    extra = ""
    if kernels:
        extra = f"   kernel={cnt['k']/dt:7.1f}/s"
    p(f"  {name:<34} {gb:6.2f} GB/s  ({cnt['copy']} 次拷贝){extra}")
    return gb


base = run_phase("P1 H2D 单独", None)
p("")
d2 = run_phase("P2 H2D + decode 并发", "decode")
p("")
d3 = run_phase("P3 H2D + grouped GEMM 并发", "gemm")

p("\n== 判据 ==")
p(f"  P1 单独              : {base:6.2f} GB/s")
p(f"  P2 + decode          : {d2:6.2f} GB/s  (保留 {d2/base*100:5.1f}%)")
p(f"  P3 + grouped GEMM    : {d3:6.2f} GB/s  (保留 {d3/base*100:5.1f}%)")
if min(d2, d3) < 0.3 * base:
    which = "decode" if d2 < d3 else "grouped GEMM"
    p(f"  ⛔ **{which} 并发时 H2D 崩到 {min(d2,d3)/base*100:.0f}%** ⇒ 这就是引擎 5.1 GB/s 的原因")
else:
    p("  ✅ 并发内核没有把 H2D 压垮 ⇒ 引擎的 5.1 GB/s 另有原因")
