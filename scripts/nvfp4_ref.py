#!/usr/bin/env python3
"""NVFP4(W4A16)反量化**参考实现** —— 纯 Python,不依赖 torch/numpy。

用途(2026-10-21 离线准备,T0.1/Phase 1):
  * 作为"正确性 oracle":任何 Marlin/CUTLASS/自写 dequant 内核的结果,
    都要能和本文件的 `dequant()` 对上;
  * 也可用来 `quantize_rne()` 造合成数据,做 round-trip 误差门禁。

格式(与 `lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-...` 的 `W4A16_NVFP4` 一致):
  * 值: **E2M1**(4 bit, 8 个正数 × 符号)⇒ {0, 0.5, 1, 1.5, 2, 3, 4, 6}
  * 每个 **16 个权重**一组,组内共享一个 **E4M3** block scale;
  * 另有 **FP32 global scale**(`weight_scale_2`);
  * 反量化: w = code(E2M1) × block_scale(E4M3) × global_scale(FP32)

⚠️ 本文件是**参考**,不是性能路径;不要拿它进服务。
"""
from __future__ import annotations

import random
import struct

# E2M1:4 bit 编码 = 1 sign + 2 exp + 1 mantissa;指数偏置 1(e2m1 表示 0..6)
_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def e2m1_to_float(code: int) -> float:
    """4bit 码 → float;bit3 = sign,bit2..0 = magnitude。"""
    code &= 0xF
    mag = _E2M1_MAG[code & 0x7]
    return -mag if (code & 0x8) else mag


def _e2m1_code(x: float) -> int:
    """float → 最近的 E2M1 码(round-to-nearest,平局取偶数 magnitude)。"""
    neg = x < 0
    a = abs(x)
    best, best_err = 0, float("inf")
    for i, v in enumerate(_E2M1_MAG):
        err = abs(a - v)
        if err < best_err or (err == best_err and i % 2 == 0):
            best, best_err = i, err
    return best | (0x8 if neg else 0)


def e4m3_to_float(bits: int) -> float:
    """E4M3(FN, bias 7,无 inf,0x7F=NaN)→ float。"""
    bits &= 0xFF
    sign = -1.0 if bits & 0x80 else 1.0
    exp = (bits >> 3) & 0xF
    man = bits & 0x7
    if exp == 0xF and man == 0x7:
        return float("nan")
    if exp == 0:
        return sign * (man / 8.0) * 2.0 ** (-6)      # subnormal: 2^(1-7) * m/8
    return sign * (1.0 + man / 8.0) * 2.0 ** (exp - 7)


def float_to_e4m3(x: float) -> int:
    """float → E4M3 bits(round-to-nearest-even);用于合成数据。"""
    if x != x:
        return 0x7F
    sign = 0x80 if x < 0 else 0
    a = abs(x)
    if a > 448.0:
        return sign | 0x7E
    # 找最近的 e4m3 值:暴力枚举(256 个)在本工具里足够
    best, best_err = 0, float("inf")
    for b in range(0x7F):
        v = e4m3_to_float(b)
        err = abs(a - v)
        if err < best_err:
            best, best_err = b, err
    return sign | best


def dequant(packed: bytes, block_scales: bytes, global_scale: float,
            numel: int, block: int = 16) -> list[float]:
    """按 NVFP4 规则反量化 numel 个权重。

    packed: 每字节 2 个 4bit 码(**低 nibble 在前**,与 GPTQ/Marlin 约定一致)
    block_scales: 每 block 一个 E4M3 字节
    """
    out: list[float] = []
    nb = (numel + block - 1) // block
    if len(block_scales) < nb:
        raise ValueError(f"block_scales 不足: {len(block_scales)} < {nb}")
    for bi in range(nb):
        s = e4m3_to_float(block_scales[bi]) * global_scale
        for j in range(block):
            idx = bi * block + j
            if idx >= numel:
                break
            byte = packed[idx // 2]
            code = (byte >> 4) if (idx % 2) else (byte & 0xF)
            out.append(e2m1_to_float(code) * s)
    return out


def quantize_rne(weights: list[float], block: int = 16) -> tuple[bytes, bytes, float]:
    """把 float 权重按 NVFP4 量化(per-block E4M3 scale + FP32 global scale)。

    返回 (packed, block_scales, global_scale)。仅用于合成/自检。
    """
    gmax = max((abs(w) for w in weights), default=0.0) or 1.0
    # 让 block scale 落在 E4M3 范围内:global 取"最大绝对值 / 448"以上
    gs = gmax / 448.0 if gmax > 0 else 1.0
    packed = bytearray((len(weights) + 1) // 2)
    scales = bytearray()
    for bi in range(0, len(weights), block):
        chunk = weights[bi:bi + block]
        bmax = max((abs(w) for w in chunk), default=0.0)
        s_float = (bmax / 6.0) if bmax > 0 else 1.0          # E2M1 最大 6
        s_bits = float_to_e4m3(s_float / gs)
        s = e4m3_to_float(s_bits) * gs
        scales.append(s_bits)
        for j, w in enumerate(chunk):
            code = _e2m1_code(w / s if s else 0.0)
            idx = bi + j
            if idx % 2 == 0:
                packed[idx // 2] = code & 0xF
            else:
                packed[idx // 2] |= (code & 0xF) << 4
    return bytes(packed), bytes(scales), gs


def _self_test() -> int:
    """已知向量 + round-trip 误差门禁(纯离线,不需要 GPU)。"""
    # 1) E4M3 已知值
    cases = {0x00: 0.0, 0x38: 1.0, 0x40: 2.0, 0x7E: 448.0}
    for bits, want in cases.items():
        got = e4m3_to_float(bits)
        assert abs(got - want) < 1e-9, f"e4m3 0x{bits:02x}: {got} != {want}"
    # 2) E2M1 已知值
    assert e2m1_to_float(0x0) == 0.0 and e2m1_to_float(0x2) == 1.0
    assert e2m1_to_float(0xA) == -1.0 and e2m1_to_float(0x7) == 6.0
    # 3) 量化 → 反量化 round-trip:相对误差应显著小于"纯 4bit"的直觉(有 block scale)
    rng = random.Random(0)
    w = [rng.gauss(0, 0.05) for _ in range(4096)]
    packed, scales, gs = quantize_rne(w)
    back = dequant(packed, scales, gs, len(w))
    err = [(a - b) for a, b in zip(w, back)]
    num = sum(e * e for e in err) ** 0.5
    den = sum(x * x for x in w) ** 0.5 or 1.0
    rel = num / den
    print(f"[nvfp4-ref] round-trip rms_rel = {rel:.3e} (n={len(w)})")
    if rel > 0.25:
        print("⛔ 误差过大,检查码表/尺度")
        return 1
    print("✓ nvfp4 参考实现自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
