"""v3：修 `_scaled_grouped_mm` 的布局要求（"Expected mat2 to be transposed"）。

v2 已证：
  * `_scaled_mm(a_fp8, b_fp8)` 在 S5000 上可用（0.028 ms）
  * `torch.mm(a_fp8, b_fp8)` 直接可用（0.021 ms，输出 fp8）
  * `torch._scaled_grouped_mm` **存在**，但要求 mat2 是转置布局
  * 逐专家 48× `_scaled_mm` 循环 = 1.347 ms，**比 bf16 grouped（0.820 ms）还慢**

目标：找出 `_scaled_grouped_mm` 的正确调用形式。若可用 ⇒ qfn 可**彻底免掉
fp8→bf16 全量 decode**，直接用 fp8 张量核做 grouped GEMM。

PyTorch 的 `_scaled_grouped_mm` 约定（按报错与文档推断，逐个试）：
  mat_a: (M, K) fp8
  mat_b: (G, N, K) fp8  ← N 在前、K 在后，即"转置"布局
  scale_a: (M, 1) 或 (M,) rowwise
  scale_b: (G, N) 或 swizzled blockwise
"""
import time

import torch

dev = torch.device("musa")
H, I, NE = 2560, 640, 48
TWOH = 2 * I
ROWS = 480                     # 480/48 = 10 行/专家，保证每个专家都非空
FP8 = torch.float8_e4m3fn


def p(*a):
    print(*a, flush=True)


def bench(fn, reps=20, warm=3):
    for _ in range(warm):
        fn()
    torch.musa.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.musa.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


per = ROWS // NE
offs = torch.tensor([per * (k + 1) for k in range(NE)], dtype=torch.int32, device=dev)

a_bf = torch.randn(ROWS, H, device=dev, dtype=torch.bfloat16)
w_bf = torch.randn(NE, H, TWOH, device=dev, dtype=torch.bfloat16)
a_f8 = a_bf.to(FP8)
w_f8 = w_bf.to(FP8)
w_f8_T = w_f8.transpose(1, 2).contiguous()      # (G, N, K) = (48, 1280, 2560)

p(f"== _scaled_grouped_mm 布局探测 ==  A{tuple(a_f8.shape)} W{tuple(w_f8.shape)}"
  f" W_T{tuple(w_f8_T.shape)}\n")

results = []


def attempt(tag, fn):
    try:
        r = fn()
        torch.musa.synchronize()
        ms = bench(fn)
        p(f"  {tag:<52} ✓ {ms:8.3f} ms  out={getattr(r,'dtype','?')}")
        results.append((tag, ms, True))
        return True
    except Exception as e:                                   # noqa: BLE001
        p(f"  {tag:<52} ✗ {str(e).split(chr(10))[0][:94]}")
        results.append((tag, float('nan'), False))
        return False


# 基线
ms_bf = bench(lambda: torch._grouped_mm(a_bf, w_bf, offs=offs))
p(f"  基线 bf16 _grouped_mm（含 decode 前的现状 GEMM）  {ms_bf:8.3f} ms\n")

sa_row = torch.ones(ROWS, 1, device=dev)
sb_gn = torch.ones(NE, TWOH, device=dev)
sa1 = torch.tensor([1.0], device=dev)
sb1 = torch.tensor([1.0], device=dev)

p("尝试各种 mat2 / scale 组合：")
# 1) mat2 已转置
attempt("S1 w_T, sa1, sb1, offs=",
        lambda: torch._scaled_grouped_mm(a_f8, w_f8_T, sa1, sb1, offs=offs,
                                         out_dtype=torch.bfloat16))
# 2) 传原 w 但让库自己转
attempt("S2 w, sa1, sb1, offs=",
        lambda: torch._scaled_grouped_mm(a_f8, w_f8, sa1, sb1, offs=offs,
                                         out_dtype=torch.bfloat16))
# 3) rowwise scale_a + per-(G,N) scale_b
attempt("S3 w_T, sa_row, sb_gn, offs=",
        lambda: torch._scaled_grouped_mm(a_f8, w_f8_T, sa_row, sb_gn, offs=offs,
                                         out_dtype=torch.bfloat16))
# 4) 关键词 + 转置
attempt("S4 kw=, w_T, sa1, sb1",
        lambda: torch._scaled_grouped_mm(
            mat_a=a_f8, mat_b=w_f8_T, scale_a=sa1, scale_b=sb1, offs=offs,
            out_dtype=torch.bfloat16))
# 5) 3D mat_a（部分版本要 (M,K) 展开）
attempt("S5 w_T, sa1, sb1, use_fast_accum",
        lambda: torch._scaled_grouped_mm(a_f8, w_f8_T, sa1, sb1, offs=offs,
                                         out_dtype=torch.bfloat16,
                                         use_fast_accum=True))
# 6) bf16 版 _scaled_grouped_mm（确认接口本身可用）
attempt("S6 bf16 输入 w_T(bf16)",
        lambda: torch._scaled_grouped_mm(a_bf, w_bf.transpose(1, 2).contiguous(),
                                         sa1, sb1, offs=offs,
                                         out_dtype=torch.bfloat16))

p("\n== 数值校验（S1 若可用，与 bf16 grouped 比）==")
try:
    ref = torch._grouped_mm(a_bf, w_bf, offs=offs).float()
    got = torch._scaled_grouped_mm(a_f8, w_f8_T, sa1, sb1, offs=offs,
                                   out_dtype=torch.bfloat16).float()
    d = (got - ref).abs().max().item()
    scale = ref.abs().max().item()
    p(f"  max|Δ|={d:.6g}  max|ref|={scale:.6g}  相对={d/max(scale,1e-9):.3e}")
except Exception as e:                                       # noqa: BLE001
    p(f"  校验跳过：{str(e).split(chr(10))[0][:90]}")

ok = [t for t, m, o in results if o]
p(f"\n== 结论 ==\n  可用形式: {ok if ok else '无（grouped fp8 需自研或改上游）'}")
best = [(t, m) for t, m, o in results if o]
if best:
    t, m = min(best, key=lambda x: x[1])
    p(f"  最快: {t} = {m:.3f} ms  vs bf16 基线 {ms_bf:.3f} ms ⇒ {ms_bf/m:.2f}×")
