#!/usr/bin/env python
"""⭐ B948:`gate_up_kernel_fp8` 的耗时是否真的 ∝ **grid 第一维(E)**?

## 为什么必须先做这个

B947 从读码推断:⭐ `gate_up_kernel_fp8[(E, cdiv(2I, bn))]`(`gpu_prefill_fp8.py:473`)
的 grid 第一维 = `E`(=288)**是常数、与 qlen 无关**,⇒ 这解释了 `T_gpu` 为何不随 qlen 增长。

⛔ **但那只是读码推断** —— 若"空专家的 block 内层 0 次迭代 ⇒ 几乎免费",
⭐ 那么 grid ∝ E **并不构成成本**,我的根因就**错了** ✗。
⇒ ⭐ 本脚本用**同一内核、同一批数据**,只改 grid 第一维 ⇒
   ⭐ **耗时随 dim0 增长 ⇒ 根因成立** ✓;⭐ **不增长 ⇒ 根因错,必须撤回** ✓

## ⭐ 关键设计(让"计算量相同、只有 grid 不同")

* 32 个赋值(模拟 32 个 `(token, expert)`)**全放在专家 0..7**(各 4 行)
* `seg_ptr` 在专家 8..287 上**全空**(`base == end`)
* ⇒ ⭐ `E_grid=8` 时一个不空;⭐ `E_grid=288` 时后面 **280 个全空**
* ⇒ ⭐ ⭐ **两次的计算量完全相同,唯一差别是 grid 大小** ✓✓✓

## 口径(R43)

* 设备 GPU0;`XIAOTU_ENV_FILE=/dev/null`;不加载模型(纯 Triton 微基准)
* GLM MoE 形状:`H=4096 I=2048`;`bn=BN=128` ⇒ grid dim1 = `cdiv(4096,128)` = **32**
* dim0 ∈ {8, 16, 64, 288};`BM=16 BK=128 NS=2 num_warps=4`(与生产同)
* ⚠️ 权重是 `zeros` + 常量尺度 ⇒ ⚠️ **只判"grid 维度是否收费",不代表真实权重的绝对耗时** ✓

## ⛔ 自检(B940 教训)

⭐ 必须证明内核**真的算了**:⭐ `E=8` 与 `E=16` 的输出**必须逐元素一致**
(因为赋值全在 0..7)⇒ `rms_rel < 1e-5` ✓
⛔ 不通过 ⇒ **立即退出,不输出任何时间** ✗
"""
import os
import sys
import time

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_xiaotu_moe.gpu_prefill_fp8 import (  # noqa: E402
    FP8_BLOCK, gate_up_kernel_fp8,
)


def timeit(fn, iters=40, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def main() -> int:
    dev = torch.device("cuda:0")
    H, I = 4096, 2048
    bn = BN = FP8_BLOCK          # 128
    BM, BK, NS, WARPS = 16, 128, 2, 4
    NLOC = 32
    E_MAX = 288
    NACT = 8                     # 赋值只用专家 0..7

    print("=" * 78)
    print("  B948  gate_up_kernel_fp8 耗时是否 ∝ grid 第一维(E)?")
    print("  ⭐ 32 个赋值全在专家 0..7 ⇒ E=288 时 280 个全空;E=8 时一个不空")
    print("     ⇒ ⭐ 计算量相同,只有 grid 大小不同 ✓")
    print("=" * 78)

    torch.manual_seed(0)
    n = NLOC
    x = torch.randn(n, H, device=dev, dtype=torch.bfloat16)
    toks = torch.arange(n, device=dev, dtype=torch.int32)

    # seg: 专家 0..7 各 4 行 ⇒ seg[0..8] = 0,4,8,...,32;专家 >=8 全空
    seg = torch.full((E_MAX + 1,), NLOC, device=dev, dtype=torch.int32)
    seg[:NACT + 1] = torch.arange(0, NLOC + 1, NLOC // NACT,
                                  device=dev, dtype=torch.int32)

    # ⛔ B948 修正:旧版用 `torch.zeros` 建 w13 ⇒ ⭐ **乘积恒为零** ⇒ 自检报
    #    "输出全零 ⇒ 内核没算" ✗(⭐ 自检发现了真问题,得分 ✓)
    #    ⇒ ⭐ 只把**用到的专家 0..NACT-1** 填成 fp8 e4m3 的 **1.0**(`0x38`)⇒
    #      ⭐ 其余专家在 seg 上是空的,根本不会被读 ⇒ 无需填(省 4.8 GB 写入)✓
    w13 = torch.zeros(E_MAX, H, 2 * I, device=dev, dtype=torch.uint8)
    w13[:NACT] = 0x38
    s13 = torch.full((E_MAX, H // bn, (2 * I) // bn), 0.02,
                     device=dev, dtype=torch.float32)
    inter = torch.zeros(n, 2 * I, device=dev, dtype=torch.float32)

    def launch(E_grid: int):
        gate_up_kernel_fp8[(E_grid, triton.cdiv(2 * I, bn))](
            x, x.stride(0), toks,
            seg,
            w13, w13.stride(1), s13, s13.stride(1),
            inter, inter.stride(0), w13.stride(0), s13.stride(0),
            H=H, BM=BM, BN=BN, BK=BK, NS=NS, num_warps=WARPS,
        )

    # ---- 自检:E=8 与 E=16 结果必须一致(赋值全在 0..7)----
    inter.zero_()
    launch(NACT)
    torch.cuda.synchronize()
    o8 = inter.clone()
    inter.zero_()
    launch(NACT * 2)
    torch.cuda.synchronize()
    o16 = inter.clone()
    rms = ((o8 - o16).pow(2).mean().sqrt()
           / (o8.pow(2).mean().sqrt() + 1e-12)).item()
    if not (rms < 1e-5):
        print(f"  ⛔ 自检失败:E=8 与 E=16 输出不一致(rms_rel={rms:.3e})")
        print("     ⇒ 赋值分布不在 0..7,或内核没按预期工作 ⇒ **计时作废**")
        return 1
    nz = int((o8.abs().sum(dim=1) > 0).sum().item())
    if nz == 0:
        print(f"  ⛔ 自检失败:输出全零 ⇒ 内核没算 ⇒ **计时作废**")
        return 1
    print(f"  ✅ 自检:E=8 与 E=16 输出逐元素一致(rms_rel={rms:.2e});非零行={nz}/{n}"
          f" ⇒ 内核真在算,且赋值确实全在专家 0..7 ✓")
    print(f"     ⇒ E=288 时后面 {E_MAX - NACT} 个专家是【空】的,而计算量不变 ✓")

    # ⭐ B952:BN 扫描(⭐ BN>128 时 128-block 尺度语义会错,⛔但【计时】有效
    #    ⇒ 只用它判断"每行连续字节数是否是带宽瓶颈")
    # ⭐ B956:验证 "sector 利用率" 假设 —— BN=512 且 NS=1(⭐ smem=128x512=64KB < 163KB)
    #    ⛔ BN=512 时 128-block 尺度语义会错 ⇒ 只用它判「每行连续 512B 能否带来 4x」
    print("─" * 78)
    print("  ⭐ B956:BN=512 + NS=1(每行连续字节 128 -> 512)")
    for bn2, ns2 in ((128, 2), (256, 1), (512, 1)):
        def launch_c(bn2=bn2, ns2=ns2):
            gate_up_kernel_fp8[(NACT, triton.cdiv(2 * I, bn2))](
                x, x.stride(0), toks,
                seg,
                w13, w13.stride(1), s13, s13.stride(1),
                inter, inter.stride(0), w13.stride(0), s13.stride(0),
                H=H, BM=BM, BN=bn2, BK=BK, NS=ns2, num_warps=WARPS,
            )
        try:
            tc, tcmin = timeit(launch_c, iters=25, warmup=6)
            smem = BK * bn2 * ns2 // 1024
            print(f"     BN={bn2:4d} NS={ns2}  smem≈{smem:3d}KB  中位={tc:8.1f} µs  [{608.1/tc:5.2f}× vs BN=128]")
        except Exception as e:
            print(f"     BN={bn2:4d} NS={ns2}  ⛔ {str(e)[:70]}")
    print("─" * 78)

    # ⭐ B955:扫 `num_warps` x `num_stages` x `BM`(⭐ 影响占用率与延迟隐藏的关键参数)
    print("─" * 78)
    print("  ⭐ B955:num_warps x num_stages x BM 扫描(BK=BN=128, E_grid=8)")
    best = (1e9, None)
    for nw in (4, 8):
        for ns2 in (2, 3, 4):
            for bm2 in (16, 32, 64):
                def launch_p(nw=nw, ns2=ns2, bm2=bm2):
                    gate_up_kernel_fp8[(NACT, triton.cdiv(2 * I, BN))](
                        x, x.stride(0), toks,
                        seg,
                        w13, w13.stride(1), s13, s13.stride(1),
                        inter, inter.stride(0), w13.stride(0), s13.stride(0),
                        H=H, BM=bm2, BN=BN, BK=BK, NS=ns2, num_warps=nw,
                    )
                try:
                    tp, _ = timeit(launch_p, iters=25, warmup=6)
                    star = ""
                    if tp < best[0]:
                        best = (tp, (nw, ns2, bm2)); star = "  ⭐ best"
                    print(f"     warps={nw} stages={ns2} BM={bm2:2d}  中位={tp:8.1f} µs  [{608.1/tp:5.2f}× vs 基准]{star}")
                except Exception as e:
                    print(f"     warps={nw} stages={ns2} BM={bm2:2d}  ⛔ {str(e)[:55]}")
    print("─" * 78)
    print(f"  ⭐ 最优组合 = {best[1]}  中位={best[0]:.1f} µs"
          f"  ⇒ 相对基准 608.1 µs 提速 {608.1/best[0]:.2f}×")
    print("─" * 78)

    res = {}
    print("─" * 78)
    print("  ⭐ B952:BN 扫描(权重每行连续字节 = BN)")
    for bn2 in (128,):
        def launch_bn(bn2=bn2):
            gate_up_kernel_fp8[(NACT, triton.cdiv(2 * I, bn2))](
                x, x.stride(0), toks,
                seg,
                w13, w13.stride(1), s13, s13.stride(1),
                inter, inter.stride(0), w13.stride(0), s13.stride(0),
                H=H, BM=BM, BN=bn2, BK=BK, NS=NS, num_warps=WARPS,
            )
        try:
            tb, tbmin = timeit(launch_bn)
            print(f"     BN={bn2:4d}  grid=(8, {triton.cdiv(2*I,bn2):3d})  中位={tb:8.1f} µs  min={tbmin:8.1f} µs"
                  f"  [vs BN=128: {res[NACT]/tb:5.2f}×]")
        except Exception as e:
            print(f"     BN={bn2:4d}  ⛔ 失败: {str(e)[:70]}")

    print("─" * 78)
    print("  ⭐ B953:BK 扫描(权重每行连续字节 = BK;⭐ BK>128 时 128-尺度语义会错,⛔但【计时】有效)")
    for bk2 in (128, 256, 512):
        def launch_bk(bk2=bk2):
            gate_up_kernel_fp8[(NACT, triton.cdiv(2 * I, BN))](
                x, x.stride(0), toks,
                seg,
                w13, w13.stride(1), s13, s13.stride(1),
                inter, inter.stride(0), w13.stride(0), s13.stride(0),
                H=H, BM=BM, BN=BN, BK=bk2, NS=NS, num_warps=WARPS,
            )
        try:
            tb, tbmin = timeit(launch_bk)
            print(f"     BK={bk2:4d}  中位={tb:8.1f} µs  min={tbmin:8.1f} µs  [vs BK=128: {608.1/tb:5.2f}×]")
        except Exception as e:
            print(f"     BK={bk2:4d}  ⛔ 失败: {str(e)[:75]}")
    print("─" * 78)
    for E_grid in (NACT, 16, 64, E_MAX):
        t, tmin = timeit(lambda: launch(E_grid))
        res[E_grid] = t
        print(f"  E_grid={E_grid:4d}  grid=({E_grid:3d}, 32)  中位={t:8.1f} µs  min={tmin:8.1f} µs")

    print("-" * 78)
    base = res[NACT]
    print("  ⭐ 相对 E_grid=8 的倍数:")
    for E_grid in (NACT, 16, 64, E_MAX):
        print(f"     E_grid={E_grid:4d}: {res[E_grid] / base:5.2f}×   ({res[E_grid]:8.1f} µs)")

    print("-" * 78)
    grow = res[E_MAX] / base
    if grow > 1.5:
        print(f"  ⭐ ⭐ 判读:E=288 比 E=8 慢 **{grow:.2f}×** ⇒ **根因成立** ✓")
        print("           ⭐ 空专家的 block 仍要付代价(grid ∝ E)")
        print(f"           ⇒ ⇒ **值得做「grid 只覆盖非空专家」**,gemm 部分理论上限降 {grow:.1f}× ✓")
    else:
        print(f"  ⭐ ⭐ 判读:E=288 只比 E=8 慢 {grow:.2f}×(<1.5×)⇒ ⛔ **根因【不成立】** ✗")
        print("           ⭐ 空 block 几乎免费 ⇒ B947 的「grid ∝ E」推断**必须撤回** ✗")
        print("           ⇒ 必须另找 3.17 ms 的来源(嫌疑:阈值/占用率/atomic 冲突)")
    print("=" * 78)
    print("  ⛔ 证据等级:各 µs = 实测(GPU0,env 干净,含 rms_rel + 非零行自检 ✓);")
    print("     ⚠️ 本基准用 zeros 权重 + 常量尺度 ⇒ **只判「grid 维度是否收费」**,不代表真实权重绝对耗时 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
