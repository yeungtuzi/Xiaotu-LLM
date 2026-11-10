"""S5000 原生 FP8 计算能力探测 v2（修正 v1 的转置 bug，并测 `_scaled_grouped_mm`）。

v1 结论（已修正）：
  * `float8_e4m3fn` 可分配 ✓
  * `_scaled_mm` 失败是**我的转置写错**（应传 (2560,1280) 而非其转置），需重测
  * `_grouped_mm` 拒绝 fp8 是 PyTorch 的 dtype 白名单（"Expected mat_a to be
    Float32, BFloat16 or Float16"），**不是硬件不支持**
  * 朴素 fp8→bf16 全量转换 = 0.904 ms（w13 一个 tile）

本 v2 回答：S5000 上**原生 FP8 GEMM 到底能不能用、快多少**，以及引擎真正需要的
**grouped** 形式有没有 fp8 接口。
"""
import time

import torch

dev = torch.device("musa")
H, I, NE = 2560, 640, 48
TWOH = 2 * I
ROWS = 320
FP8 = torch.float8_e4m3fn


def p(*a):
    print(*a, flush=True)


def bench(fn, reps=20, warm=3):
    for _ in range(warm):
        try:
            fn()
        except Exception:
            return float("nan")
    torch.musa.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.musa.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


try_all = []


def attempt(tag, fn):
    """试一个调用，成功就计时。"""
    try:
        r = fn()
        torch.musa.synchronize()
        ms = bench(fn)
        dtype = getattr(r, "dtype", "?")
        p(f"  {tag:<44} ✓ {ms:8.3f} ms   out={dtype}")
        try_all.append((tag, ms, True))
        return True
    except Exception as e:                                   # noqa: BLE001
        msg = str(e).split("\n")[0][:110]
        p(f"  {tag:<44} ✗ {type(e).__name__}: {msg}")
        try_all.append((tag, float("nan"), False))
        return False


a_bf = torch.randn(ROWS, H, device=dev, dtype=torch.bfloat16)
w_bf = torch.randn(NE, H, TWOH, device=dev, dtype=torch.bfloat16)
a_f8 = a_bf.to(FP8)
w_f8 = w_bf.to(FP8)
sa1 = torch.tensor([1.0], device=dev)
sb1 = torch.tensor([1.0], device=dev)

p(f"== S5000 原生 FP8 探测 v2 == {torch.musa.get_device_name(0)}\n")

# ── A. dense fp8：修正转置 ──
p("A. dense FP8 矩阵乘（形状 A(320,2560) @ B(2560,1280)）")
b2d = w_f8[0]                            # (2560, 1280)  ← 正确形状，不转置
attempt("A1 _scaled_mm(a_f8, b_fp8)",
        lambda: torch._scaled_mm(a_f8, b2d, scale_a=sa1, scale_b=sb1,
                                 out_dtype=torch.bfloat16))
attempt("A2 _scaled_mm(..., use_fast_accum=True)",
        lambda: torch._scaled_mm(a_f8, b2d, scale_a=sa1, scale_b=sb1,
                                 out_dtype=torch.bfloat16, use_fast_accum=True))
attempt("A3 mm(a_f8, b_fp8) 直接调", lambda: torch.mm(a_f8, b2d))
attempt("A4 mm(a_bf16, b_bf16)  ← 现状基线", lambda: torch.mm(a_bf, w_bf[0]))

# ── B. grouped fp8：引擎真正需要的 ──
p("\nB. grouped FP8（引擎需要的形式）")
per = ROWS // NE
offs = torch.tensor([per * (k + 1) for k in range(NE)], dtype=torch.int32, device=dev)
attempt("B1 _grouped_mm(a_bf16, w_bf16)  ← 现状基线",
        lambda: torch._grouped_mm(a_bf, w_bf, offs=offs))
attempt("B2 _grouped_mm(a_fp8, w_fp8)", lambda: torch._grouped_mm(a_f8, w_f8, offs=offs))

has_sgm = hasattr(torch, "_scaled_grouped_mm")
p(f"  torch._scaled_grouped_mm 存在: {has_sgm}")
if has_sgm:
    attempt("B3 _scaled_grouped_mm(a_f8,w_f8,sa,sb,offs)",
            lambda: torch._scaled_grouped_mm(a_f8, w_f8, sa1, sb1, offs=offs,
                                             out_dtype=torch.bfloat16))
    attempt("B4 _scaled_grouped_mm(关键词形式)",
            lambda: torch._scaled_grouped_mm(
                a_f8, w_f8, scale_a=sa1, scale_b=sb1, offs=offs,
                out_dtype=torch.bfloat16))
else:
    p("  ⛔ 无 _scaled_grouped_mm ⇒ grouped fp8 只能自行拼（逐专家 _scaled_mm 循环）")

# ── C. 逐专家 _scaled_mm 循环（grouped 的替代实现）──
p("\nC. 若 grouped fp8 不可用：逐专家 _scaled_mm 循环（48 次）")


def loop_fp8():
    outs = []
    for k in range(NE):
        r0, r1 = per * k, per * (k + 1)
        outs.append(torch._scaled_mm(a_f8[r0:r1], w_f8[k], scale_a=sa1,
                                     scale_b=sb1, out_dtype=torch.bfloat16))
    return outs


attempt("C1 48× _scaled_mm 循环", loop_fp8)

# ── D. 现状路径 vs 原生路径 的端到端对比 ──
p("\nD. 关键对比（每 tile，w13 形状）")
ms_dec = bench(lambda: w_f8.to(torch.bfloat16))
p(f"  D1 decode fp8→bf16 全量            = {ms_dec:8.3f} ms")
bf_g = [x for x in try_all if x[0].startswith("B1")][0][1]
p(f"  D2 现状：D1 + bf16 grouped_mm       = {ms_dec + bf_g:8.3f} ms")
best = [(t, m) for t, m, ok in try_all
        if ok and (t.startswith("A1") or t.startswith("A2") or t.startswith("B3")
                   or t.startswith("C1"))]
for t, m in best:
    p(f"  D3 原生：{t:<34} = {m:8.3f} ms  （省 {(ms_dec + bf_g - m):.3f} ms）")

p("\n== 结论 ==")
ok_native = [t for t, m, ok in try_all if ok and not t.startswith(("A4", "B1"))]
p(f"  可用的原生 fp8 接口: {ok_native if ok_native else '无'}")
