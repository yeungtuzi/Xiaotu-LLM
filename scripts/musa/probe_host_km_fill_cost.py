"""量 `_host_km_fill` 的冷/热成本与缓存占用（不启动 vLLM）。

背景：引擎 TLAYER 计时显示分层双峰 ——
    多数层 kmajor≈1 ms、尖峰层 kmajor≈462 ms。
假设：尖峰 = `_HOST_KM_CACHE` 未命中，要在 CPU 上重建 225 MiB pinned 整块。

本 probe 直接回答三个问题：
  1. 冷调用的**真实成本**是多少 ms/tile？（假设中的 462 ms 是否对得上）
  2. 热命中成本多少 ms/tile？（对照 1.1 ms）
  3. 缓存**足迹**多大？（36 个 streaming 层 × 11 tile 是否会撑爆 pinned 内存）

几何照引擎实测：E=512 tile=48 ⇒ 11 tile；H=2560 I=640 ns=2。
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
    "gpf_hk", ROOT + "/vllm_xiaotu_moe/gpu_prefill_fp8.py")
gpf = importlib.util.module_from_spec(spec)
sys.modules["gpf_hk"] = gpf
spec.loader.exec_module(gpf)

host_km_fill = gpf._host_km_fill
CACHE = gpf._HOST_KM_CACHE

dev = torch.device("musa")
H, I, NS = 2560, 640, 2
cr13, cr2 = I // NS, H // NS
Et = 48
c13, c2 = cr13 * H, cr2 * I
E = 512
N_TILES = (E + Et - 1) // Et
print(f"== _host_km_fill 成本 ==  E={E} tile={Et} => {N_TILES} tiles  "
      f"H={H} I={I} ns={NS}\n")

w13t = torch.zeros((Et, H, 2 * I), dtype=torch.uint8, device=dev)
w2t = torch.zeros((Et, I, H), dtype=torch.uint8, device=dev)


class FakeEngine:
    """每层一份 host shard（指针不同 ⇒ 缓存键不同，模拟 36 个 streaming 层）。"""

    def __init__(self, tag):
        self.tag = tag
        self.bufs = {}
        for which, size in ((0, Et * 2 * c13), (1, Et * c2)):
            for n in range(NS):
                # 用不初始化的 uint8 省时间；内容不影响计时
                self.bufs[(which, n)] = torch.empty(size, dtype=torch.uint8)

    def shard_geometry(self):
        return {"ns": NS, "w13_crows": cr13, "w13_cbytes": c13,
                "w2_crows": cr2, "w2_cbytes": c2}

    def hostbuf_ptr(self, which, n):
        return self.bufs[(which, n)].data_ptr()


def cache_gib():
    tot = 0
    for v in CACHE.values():
        if hasattr(v, "numel"):
            tot += v.numel()
    return tot / 2**30


def run_layer(eng, label, tiles=N_TILES):
    t0 = time.perf_counter()
    per = []
    for i in range(tiles):
        e0 = i * Et
        a = time.perf_counter()
        ok = host_km_fill(eng, dev, H, I, e0, e0 + Et, w13t, w2t)
        torch.musa.synchronize()
        per.append((time.perf_counter() - a) * 1e3)
        if not ok:
            print(f"  {label} tile{i} FAILED")
    tot = (time.perf_counter() - t0) * 1e3
    print(f"  {label:<26} total={tot:8.1f} ms  "
          f"per-tile min={min(per):7.1f} med={sorted(per)[len(per)//2]:7.1f} "
          f"max={max(per):7.1f}  cache={cache_gib():6.2f} GiB")
    return tot, per


CACHE.clear()
print("── 第 1 层（全冷）──")
e1 = FakeEngine(1)
cold_tot, cold_per = run_layer(e1, "layer1 COLD")

print("\n── 同一层再跑（应全热命中）──")
warm_tot, warm_per = run_layer(e1, "layer1 WARM")

print("\n── 再追加 3 层（看 pin_memory 是否随缓存增长变慢）──")
extra = []
for k in range(2, 5):
    ek = FakeEngine(k)
    t, per = run_layer(ek, f"layer{k} COLD")
    extra.append(t)
    del ek

n_layer_cache = cache_gib()
print(f"\n── 汇总 ──")
print(f"冷层总耗时        : {cold_tot:8.1f} ms  ({cold_tot/N_TILES:.1f} ms/tile)")
print(f"热层总耗时        : {warm_tot:8.1f} ms  ({warm_tot/N_TILES:.1f} ms/tile)")
if warm_tot > 0:
    print(f"冷/热 倍数        : {cold_tot/warm_tot:6.1f}×")
print(f"每层缓存足迹      : {n_layer_cache/4:6.2f} GiB  (共 4 层)")
print(f"36 streaming 层推算: {n_layer_cache/4*36:6.1f} GiB pinned")
print(f"（如果每层只保留 1 个 tile 的缓冲，则为 "
      f"{(Et*H*2*I + Et*I*H)*36/2**30:.1f} GiB）")

print("\n每 tile 冷耗时明细 (layer1): "
      + " ".join(f"{v:.0f}" for v in cold_per))
print("每 tile 热耗时明细 (layer1): "
      + " ".join(f"{v:.1f}" for v in warm_per))
