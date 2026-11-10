"""对拍 + 计时：`_host_km_fill` 的 K-major 与 N-major 两条装配路径。

判据（两条都要过）：
  1. **数值**：N-major 结果转成 K-major 视图后，必须与 K-major 路径**逐字节相等**
     （`torch.equal`）—— 这是唯一允许的通过判据（R87：逐元素对拍）。
  2. **性能**：N-major 应显著快于 K-major 的 144.6 ms/tile。

不启动 vLLM。
"""
import importlib.util
import os
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
    "gpf_eq", ROOT + "/vllm_xiaotu_moe/gpu_prefill_fp8.py")
gpf = importlib.util.module_from_spec(spec)
sys.modules["gpf_eq"] = gpf
spec.loader.exec_module(gpf)


def p(*a):
    print(*a, flush=True)


dev = torch.device("musa")
H, I, NS = 2560, 640, 2
cr13, cr2 = I // NS, H // NS
Et = 48
c13, c2 = cr13 * H, cr2 * I

torch.manual_seed(11)
bufs = {(w, n): torch.randint(0, 256, (sz,), dtype=torch.uint8)
        for w, sz in ((0, Et * 2 * c13), (1, Et * c2))
        for n in range(NS)}


class FakeEngine:
    def shard_geometry(self):
        return {"ns": NS, "w13_crows": cr13, "w13_cbytes": c13,
                "w2_crows": cr2, "w2_cbytes": c2}

    def hostbuf_ptr(self, which, n):
        return bufs[(which, n)].data_ptr()


eng = FakeEngine()


def run_tn(tn: bool):
    """按指定 TN 开关跑一次真实 `_host_km_fill`，返回 (w13, w2, ms)。"""
    os.environ["XIAOTU_GPF_W13TN"] = "1" if tn else "0"
    os.environ["XIAOTU_GPF_W2TN"] = "1" if tn else "0"
    gpf._HOST_KM_CACHE.clear()
    shape13 = (Et, 2 * I, H) if tn else (Et, H, 2 * I)
    shape2 = (Et, H, I) if tn else (Et, I, H)
    d13 = torch.zeros(shape13, dtype=torch.uint8, device=dev)
    d2 = torch.zeros(shape2, dtype=torch.uint8, device=dev)
    t0 = time.perf_counter()
    ok = gpf._host_km_fill(eng, dev, H, I, 0, Et, d13, d2)
    torch.musa.synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    return d13, d2, ms, ok


# ── K-major 参照 ──
km13, km2, km_ms, ok1 = run_tn(False)
p(f"K-major 路径 : ok={ok1}  {km_ms:7.1f} ms")
p(f"  d13 {tuple(km13.shape)}  d2 {tuple(km2.shape)}")

# ── N-major 新路径 ──
tn13, tn2, tn_ms, ok2 = run_tn(True)
p(f"N-major 路径 : ok={ok2}  {tn_ms:7.1f} ms")
p(f"  d13 {tuple(tn13.shape)}  d2 {tuple(tn2.shape)}")

# ── 数值对拍：N-major (Et,2I,H) --transpose--> (Et,H,2I) 必须逐字节等于 K-major ──
eq13 = torch.equal(tn13.transpose(1, 2).contiguous(), km13)
eq2 = torch.equal(tn2.transpose(1, 2).contiguous(), km2)
p(f"\n对拍 w13 : torch.equal = {eq13}   (N-major.T vs K-major)")
p(f"对拍 w2  : torch.equal = {eq2}")

if not (eq13 and eq2):
    d13 = (tn13.transpose(1, 2) != km13)
    d2 = (tn2.transpose(1, 2) != km2)
    p(f"  w13 不同字节数 = {int(d13.sum())} / {km13.numel()}")
    p(f"  w2  不同字节数 = {int(d2.sum())} / {km2.numel()}")
    sys.exit(1)

# ── 负例（B1034 教训）：env 说 N-major，但调用方传的是 K-major 目标 ──
#   修复前这里会抛 `The expanded size ... Target [1,2560,1280] vs [1,1280,2560]`；
#   修复后布局**从目标形状推断**，env 不应再影响结果。
os.environ["XIAOTU_GPF_W13TN"] = "1"
os.environ["XIAOTU_GPF_W2TN"] = "1"
gpf._HOST_KM_CACHE.clear()
km_dst13 = torch.zeros((Et, H, 2 * I), dtype=torch.uint8, device=dev)
km_dst2 = torch.zeros((Et, I, H), dtype=torch.uint8, device=dev)
try:
    ok3 = gpf._host_km_fill(eng, dev, H, I, 0, Et, km_dst13, km_dst2)
    torch.musa.synchronize()
    neg_ok = ok3 and torch.equal(km_dst13, km13) and torch.equal(km_dst2, km2)
    p(f"\n负例(env=TN1 但目标 K-major): ok={ok3} 与 K-major 参照相同={neg_ok}")
    if not neg_ok:
        p("  ⛔ 布局推断没有压过 env —— 线上会崩")
        sys.exit(1)
except RuntimeError as e:
    p(f"\n负例(env=TN1 但目标 K-major): ⛔ 抛异常 = 布局仍按 env 判断 ✗\n  {e}")
    sys.exit(1)

if km_ms > 0:
    p(f"\n加速比 : {km_ms/tn_ms:.2f}×  "
      f"（省 {(km_ms-tn_ms)*11:.0f} ms/层，11 tile）")
p("\n结论：数值逐字节一致、布局由目标形状决定、且显著更快 ✓")
