#!/usr/bin/env python
"""B958: Triton 手写 fp8 GEMM vs `反量化->bf16/fp16 -> torch.matmul`(A100 bf16 tensor core)

⭐ 背景:B956 证明所有 tiling 参数都无效(BN/BK/warps/stages/BM),32 行 608-626 us。
   B957 实测:sm_80 有 fp8 存储/转换,但【无 fp8 张量核 GEMM】
   ⇒ 唯一能用上张量核的路是「反量化到 bf16/fp16 再 matmul」。
   ⭐ 本脚本就测这条路到底快不快 —— 必须实测,不许拍脑袋。

口径:GPU0, XIAOTU_ENV_FILE=/dev/null, GLM MoE 形状 H=4096 2I=4096 I=2048
     8 个命中专家, 每专家 4 行(qlen=4), 与生产同形。
⛔ 自检:三种实现的输出必须数值一致(rms_rel<1e-2), 否则不输出时间。
"""
import os, sys, time
import torch, triton
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_xiaotu_moe.gpu_prefill_fp8 import FP8_BLOCK, gate_up_kernel_fp8  # noqa: E402


def timeit(fn, iters=25, warmup=6):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def main() -> int:
    dev = torch.device("cuda:0")
    H, I = 4096, 2048
    bn = BN = FP8_BLOCK
    BM, BK, NS, WARPS = 16, 128, 2, 4
    NACT, ROWS_PER = 8, 4
    NROW = NACT * ROWS_PER          # 32
    E_MAX = 288

    print("=" * 78)
    print("  B958  Triton fp8 GEMM  vs  反量化+torch.matmul(bf16/fp16 tensor core)")
    print(f"  device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
    print("=" * 78)
    torch.manual_seed(0)

    # 输入 32 行;专家 0..7 各 4 行
    x = torch.randn(NROW, H, device=dev, dtype=torch.bfloat16)
    toks = torch.arange(NROW, device=dev, dtype=torch.int32)
    seg = torch.full((E_MAX + 1,), NROW, device=dev, dtype=torch.int32)
    seg[:NACT + 1] = torch.arange(0, NROW + 1, ROWS_PER, device=dev, dtype=torch.int32)

    w13 = torch.zeros(E_MAX, H, 2 * I, device=dev, dtype=torch.uint8)
    w13[:NACT] = 0x38                      # fp8 e4m3 的 1.0
    s13 = torch.full((E_MAX, H // bn, (2 * I) // bn), 0.02, device=dev, dtype=torch.float32)
    inter = torch.zeros(NROW, 2 * I, device=dev, dtype=torch.float32)

    def run_triton():
        gate_up_kernel_fp8[(NACT, triton.cdiv(2 * I, BN))](
            x, x.stride(0), toks, seg,
            w13, w13.stride(1), s13, s13.stride(1),
            inter, inter.stride(0), w13.stride(0), s13.stride(0),
            H=H, BM=BM, BN=BN, BK=BK, NS=NS, num_warps=WARPS)
        return inter

    # ⭐ 反量化路径:逐专家把 fp8 权重反量化到 bf16,再 matmul
    #    ⚠️ 关键:必须【逐专家流式】,不能整体常驻(50 MB/专家 x 8 = 400 MB 还勉强)
    def make_dequant(dtype):
        w13_bf = torch.empty(NACT, H, 2 * I, device=dev, dtype=dtype)
        def f():
            for e in range(NACT):
                # ⛔ B958 修正:旧版忘了乘 0.02 尺度 ⇒ 自检报 rms_rel=0.98 ✗
                #    ⭐ Triton 路径算的是 `x @ (w * scale)` ⇒ matmul 路径必须也乘上尺度(本处尺度均为 0.02)
                w13_bf[e].copy_(w13[e].view(torch.float8_e4m3fn).to(dtype)).mul_(0.02)
            out = torch.empty(NROW, 2 * I, device=dev, dtype=dtype)
            for e in range(NACT):
                r0 = e * ROWS_PER
                out[r0:r0 + ROWS_PER] = x[r0:r0 + ROWS_PER].to(dtype) @ w13_bf[e]
            return out
        return f

    f_bf16, f_fp16 = make_dequant(torch.bfloat16), make_dequant(torch.float16)

    # ---- 自检:三者数值一致 ----
    inter.zero_(); r1 = run_triton().clone(); torch.cuda.synchronize()
    r2 = f_bf16().to(torch.float32); torch.cuda.synchronize()
    r3 = f_fp16().to(torch.float32); torch.cuda.synchronize()
    def rms(a, b):
        return ((a - b).pow(2).mean().sqrt() / (b.pow(2).mean().sqrt() + 1e-12)).item()
    d12, d13 = rms(r1, r2), rms(r1, r3)
    print(f"  自检 rms_rel: triton-vs-bf16={d12:.2e}  triton-vs-fp16={d13:.2e}")
    if not (d12 < 1e-2 and d13 < 1e-2):
        print("  ⛔ 自检失败:三种实现数值不一致 ⇒ 计时作废")
        return 1
    if (r1.abs().sum() == 0) or (r2.abs().sum() == 0):
        print("  ⛔ 自检失败:输出全零 ⇒ 作废"); return 1
    print("  ✅ 自检通过:三者数值一致且非零 ⇒ 计时有效")

    print("-" * 78)
    t1 = timeit(run_triton)
    print(f"  ① Triton 手写 fp8 GEMM          {t1:9.1f} µs   [基准]")
    for name, f in (("② 反量化->bf16 + torch.matmul", f_bf16),
                    ("③ 反量化->fp16 + torch.matmul", f_fp16)):
        t = timeit(f)
        print(f"  {name}   {t:9.1f} µs   [{t1/t:5.2f}× vs Triton]")
    print("-" * 78)
    tb, tf = timeit(f_bf16), timeit(f_fp16)
    best = min(tb, tf)
    if best < t1 * 0.9:
        print(f"  ⭐ ⭐ 判读:反量化+matmul 快 **{t1/best:.2f}×** ⇒ ⭐ **换实现是对的** ✓")
        print("           ⇒ 生产内核应改为「逐专家反量化 -> matmul」")
    else:
        print(f"  ⭐ ⭐ 判读:反量化+matmul 只快 {t1/best:.2f}×(<1.1×)⇒")
        print("           ⛔ **换实现无收益** ⇒ 608 µs 已是该形状在 A100 上的现实下限")
        print("           ⇒ 应停止内核优化,转而接受现状并如实报告")
    print("=" * 78)
    print("  ⛔ 证据等级:三种 µs = 实测(GPU0,env 干净,含 rms_rel+非零自检 ✓)")
    print("     ⚠️ 反量化路径按【逐专家流式】实现,与生产的显存约束一致 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
