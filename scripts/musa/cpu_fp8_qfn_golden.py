#!/usr/bin/env python3
"""CPU-only FP8 golden at qfn-like routing shape (top_k=10).

Uses the CPU-only engine build:
  NO_CUDA=1 ONLY=avx2 ... scripts/build_engine_variants.sh
No GPU, no CUDA runtime, no numpy.
"""
from __future__ import annotations
import array
import glob
import importlib.util
import math
import pathlib
import random
import struct

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load(variant: str = "avx2"):
    p = sorted(glob.glob(str(ROOT / "xiaotu_moe" / "build" / f"_xiaotu_moe_C_{variant}*.so")))[-1]
    name = f"_xiaotu_moe_C_{variant}"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def e4m3(b: int) -> float:
    s = -1.0 if b & 0x80 else 1.0
    e, m = (b >> 3) & 0xF, b & 7
    if e == 0:
        return s * (m / 8.0) * 2.0 ** -6
    if e == 0xF and m == 7:
        return math.nan
    return s * (1.0 + m / 8.0) * 2.0 ** (e - 7)


def bf16_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0] >> 16


def bf16_val(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b << 16))[0]


def silu(x: float) -> float:
    if x >= 0:
        return x / (1.0 + math.exp(-x))
    z = math.exp(x)
    return x * z / (1.0 + z)


def main() -> int:
    random.seed(11)
    E, H, I, T, K = 16, 2560, 640, 1, 10
    m = load("avx2")
    w13 = array.array("B", (random.randrange(8, 0x39) for _ in range(E * 2 * I * H)))
    w2 = array.array("B", (random.randrange(8, 0x39) for _ in range(E * H * I)))
    s13 = array.array("f", [1.0] * (E * (2 * I // 128) * (H // 128)))
    s2 = array.array("f", [1.0] * (E * (H // 128) * (I // 128)))
    cfg = m.MOEConfigV2()
    for k, v in dict(
        num_processes=1, process_id=0, gpu_id=0, has_gate_proj=True,
        expert_num=E, top_k=K, hidden_size=H, intermediate_size=I,
        max_batch_size=16, max_num_seqs=8, stride=128,
        group_min_len=10, group_max_len=4096, groupN=128, groupK=128,
        activation_type=0, swiglu_limit=0.0,
    ).items():
        setattr(cfg, k, v)
    eng = m.MOE_FP8(cfg, w13.buffer_info()[0], w2.buffer_info()[0],
                    s13.buffer_info()[0], s2.buffer_info()[0], 0, 0)

    xs = [[random.uniform(-0.5, 0.5) for _ in range(H)]]
    x_u16 = array.array("H", [bf16_bits(v) for v in xs[0]])
    ids = array.array("i", [random.randrange(E) for _ in range(T * K)])
    wts = array.array("f", [random.uniform(0.05, 0.2) for _ in range(T * K)])
    out = array.array("f", [0.0] * (T * H))
    eng.cpu_prefill(T, K, ids.buffer_info()[0], wts.buffer_info()[0],
                    x_u16.buffer_info()[0], out.buffer_info()[0])

    xf = [bf16_val(b) for b in x_u16]
    expect = [0.0] * H
    for j in range(K):
        e, wt = ids[j], wts[j]
        gate = [sum(xf[h] * e4m3(w13[e * 2 * I * H + o * H + h]) for h in range(H))
                for o in range(I)]
        up = [sum(xf[h] * e4m3(w13[e * 2 * I * H + (I + o) * H + h]) for h in range(H))
              for o in range(I)]
        act = [silu(gate[o]) * up[o] for o in range(I)]
        for o in range(H):
            expect[o] += wt * sum(act[i] * e4m3(w2[e * H * I + o * I + i])
                                  for i in range(I))
    max_abs = max(abs(a - b) for a, b in zip(out, expect))
    rms = math.sqrt(sum(v * v for v in expect) / H) or 1.0
    print(f"E={E} H={H} I={I} T={T} K={K} experts={len(set(ids))}")
    print(f"max_abs={max_abs:.6g} rms={rms:.6g} rel={max_abs / rms:.6g}")
    ok = max_abs / rms < 2e-2
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
