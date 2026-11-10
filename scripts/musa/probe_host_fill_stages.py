"""拆解 `_host_km_fill` 冷路径：逐阶段计时，找出病态慢的那一步。

上一版整块量冷/热，结果 12 分钟无输出（RSS 2.7 GB）⇒ 说明冷路径里有**病态阶段**。
本版只做 1 个 tile，把 4 个阶段分开计时并**每步 flush**：

  A. `torch.empty((Et,H,2I))`          —— 主机普通内存分配 157 MB
  B. 两 node 的 strided 转置拷贝        —— permute 后写进 A
  C. `.pin_memory()`                   —— 再分配一块 pinned 并整块拷过去（怀疑这里）
  D. `w13t[:, :].copy_(h13)`           —— 一次连续 H2D

判据：若 C 占绝大部分 ⇒ 修法是**直接分配 pinned**（不经过 pin_memory 的二次拷贝）。
"""
import ctypes
import importlib.util
import sys
import time
import types

import numpy as _np
import torch

ROOT = "/home/mccuseradmin/s5000/xiaotu-llm"
if "vllm_xiaotu_moe" not in sys.modules:
    pkg = types.ModuleType("vllm_xiaotu_moe")
    pkg.__path__ = [ROOT + "/vllm_xiaotu_moe"]
    sys.modules["vllm_xiaotu_moe"] = pkg
spec = importlib.util.spec_from_file_location(
    "gpf_st", ROOT + "/vllm_xiaotu_moe/gpu_prefill_fp8.py")
gpf = importlib.util.module_from_spec(spec)
sys.modules["gpf_st"] = gpf
spec.loader.exec_module(gpf)


def p(*a):
    print(*a, flush=True)


dev = torch.device("musa")
H, I, NS = 2560, 640, 2
cr13, cr2 = I // NS, H // NS
Et = 48
c13, c2 = cr13 * H, cr2 * I

p(f"== 拆解冷路径 ==  Et={Et} H={H} I={I} ns={NS}")
p(f"w13 tile bytes = {Et*H*2*I/2**20:.0f} MiB, w2 tile bytes = {Et*I*H/2**20:.0f} MiB")

# host shard（每 node 一段）
bufs = {(w, n): torch.empty(sz, dtype=torch.uint8)
        for w, sz in ((0, Et * 2 * c13), (1, Et * c2))
        for n in range(NS)}

w13t = torch.zeros((Et, H, 2 * I), dtype=torch.uint8, device=dev)
torch.musa.synchronize()

# ── A: 普通主机内存分配 ──
t = time.perf_counter()
h13 = torch.empty((Et, H, 2 * I), dtype=torch.uint8)
p(f"A torch.empty(157MB host)      : {(time.perf_counter()-t)*1e3:8.1f} ms")

# ── B: 两 node 的 strided 转置写入 ──
t = time.perf_counter()
for n in range(NS):
    ptr = bufs[(0, n)].data_ptr()
    arr = _np.ctypeslib.as_array(
        (ctypes.c_uint8 * (Et * 2 * c13)).from_address(ptr))
    v = torch.from_numpy(arr).view(Et, 2, cr13, H).permute(0, 3, 1, 2)
    c0 = n * cr13
    h13[:, :, c0:c0 + cr13].copy_(v[:, :, 0, :])
    h13[:, :, I + c0:I + c0 + cr13].copy_(v[:, :, 1, :])
p(f"B 转置写入(2 node strided)     : {(time.perf_counter()-t)*1e3:8.1f} ms")

# ── C: pin_memory（怀疑的病态阶段）──
t = time.perf_counter()
h13p = h13.pin_memory()
p(f"C .pin_memory() 157MB          : {(time.perf_counter()-t)*1e3:8.1f} ms   <-- 怀疑")

# ── C2: 对照 —— 直接分配 pinned 再写入 ──
t = time.perf_counter()
direct = torch.empty((Et, H, 2 * I), dtype=torch.uint8, pin_memory=True)
p(f"C2 直接 torch.empty(pin=True)  : {(time.perf_counter()-t)*1e3:8.1f} ms   <-- 替代方案")

t = time.perf_counter()
direct.copy_(h13)
p(f"C3 普通->pinned 一次拷贝       : {(time.perf_counter()-t)*1e3:8.1f} ms")

# ── D: H2D ──
t = time.perf_counter()
w13t[:, :].copy_(h13p)
torch.musa.synchronize()
p(f"D H2D 157MB (连续)             : {(time.perf_counter()-t)*1e3:8.1f} ms")

# ── D2: 从直接分配的 pinned 再拷一次 ──
t = time.perf_counter()
w13t[:, :].copy_(direct)
torch.musa.synchronize()
p(f"D2 H2D 157MB (direct pinned)   : {(time.perf_counter()-t)*1e3:8.1f} ms")

# ── 热路径：再拷一次同一块 ──
t = time.perf_counter()
w13t[:, :].copy_(direct)
torch.musa.synchronize()
p(f"D3 H2D 再拷一次(稳态)          : {(time.perf_counter()-t)*1e3:8.1f} ms")

p("\n判据：C 很大而 C2 很小 ⇒ 改成直接分配 pinned 即可消除尖峰")
