"""CPU-only FP8 engine golden test (no GPU, no CUDA runtime).

Requires a CPU-only build:
  NO_CUDA=1 ONLY=scalar ... scripts/build_engine_variants.sh
and numpy-free pure Python reference.
"""
from __future__ import annotations
import array
import glob
import importlib.util
import math
import os
import pathlib
import random
import struct
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SO_GLOB = str(ROOT / "xiaotu_moe" / "build" / "_xiaotu_moe_C_scalar*.so")


def _load_engine():
    paths = sorted(glob.glob(SO_GLOB))
    if not paths:
        raise unittest.SkipTest("CPU-only scalar .so not built")
    p = paths[-1]
    name = "_xiaotu_moe_C_scalar"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _e4m3(byte: int) -> float:
    sign = -1.0 if byte & 0x80 else 1.0
    exp = (byte >> 3) & 0xF
    man = byte & 0x7
    if exp == 0:
        return sign * (man / 8.0) * 2.0 ** -6
    if exp == 0xF and man == 0x7:
        return math.nan
    return sign * (1.0 + man / 8.0) * 2.0 ** (exp - 7)


def _bf16_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0] >> 16


def _ptr(a: array.array) -> int:
    return a.buffer_info()[0]


def _silu(x: float) -> float:
    if x >= 0:
        return x / (1.0 + math.exp(-x))
    e = math.exp(x)
    return x * e / (1.0 + e)


class TestCpuFp8Engine(unittest.TestCase):
    def test_cpu_prefill_matches_reference(self):
        # Tiny synthetic dims hit the N-sliced small-M path, whose row mapping
        # is not the target of this test. Force the legacy per-token path and
        # disable NUMA sharding so the reference is unambiguous.
        os.environ["XIAOTU_MOE_NOSHARD"] = "1"
        os.environ["XIAOTU_MOE_NSLICE_SMALL"] = "0"  # small-M N-slice crashes at real dims
        m = _load_engine()
        random.seed(7)
        E, H, I, T, K = 4, 128, 128, 4, 2
        # random e4m3 bytes, avoid NaN/Inf and signed zero weirdness
        w13 = array.array("B", (random.randrange(0x08, 0x39) for _ in range(E * 2 * I * H)))
        w2 = array.array("B", (random.randrange(0x08, 0x39) for _ in range(E * H * I)))
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
        eng = m.MOE_FP8(cfg, _ptr(w13), _ptr(w2), _ptr(s13), _ptr(s2), 0, 0)

        xs = [[random.uniform(-0.5, 0.5) for _ in range(H)] for _ in range(T)]
        ids = array.array("i", [random.randrange(E) for _ in range(T * K)])
        wts = array.array("f", [random.uniform(0.05, 0.2) for _ in range(T * K)])
        x_u16 = array.array("H", [_bf16_bits(v) for row in xs for v in row])
        out = array.array("f", [0.0] * (T * H))
        eng.cpu_prefill(T, K, _ptr(ids), _ptr(wts), _ptr(x_u16), _ptr(out))

        # reference: input bf16 -> float
        xf = [[struct.unpack("<f", struct.pack("<I", b << 16))[0] for b in row]
              for row in (x_u16[i * H:(i + 1) * H] for i in range(T))]
        expect = [0.0] * (T * H)
        for t in range(T):
            for j in range(K):
                e = ids[t * K + j]
                wt = wts[t * K + j]
                gate = [sum(xf[t][h] * _e4m3(w13[e * 2 * I * H + o * H + h])
                            for h in range(H)) for o in range(I)]
                up = [sum(xf[t][h] * _e4m3(w13[e * 2 * I * H + (I + o) * H + h])
                          for h in range(H)) for o in range(I)]
                act = [_silu(gate[o]) * up[o] for o in range(I)]
                for o in range(H):
                    expect[t * H + o] += wt * sum(
                        act[i] * _e4m3(w2[e * H * I + o * I + i]) for i in range(I))
        max_abs = max(abs(a - b) for a, b in zip(out, expect))
        rms = math.sqrt(sum(v * v for v in expect) / len(expect)) or 1.0
        self.assertLess(max_abs / rms, 2e-2, f"max_abs={max_abs:.5g} rms={rms:.5g}")


if __name__ == "__main__":
    unittest.main()
