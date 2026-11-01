"""对改过的内核做 AOT 编译冒烟（不占显存、不需要起服务）。

constexpr 形参由 Triton 自己暴露的 `kernel.constexprs`（参数下标）自动识别，
再按名字给一个**合法**取值（BLOCK 类取 16/32，布尔语义取 0/1，跨距取 8）。
只验证"能否编译"，不执行任何 kernel。
"""
import sys, traceback
import triton
from triton.compiler import ASTSource, compile as tcompile
from triton.backends.compiler import GPUTarget

TARGETS = [
    ("vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa", [
        "_fp8_mqa_logits_kernel",
        "_fp8_paged_mqa_logits_kernel",
        "_fp8_paged_mqa_logits_rowwise_kernel",
    ]),
    ("vllm.models.deepseek_v4.nvidia.ops.fp8_einsum", [
        "_deepseek_v4_sm12x_fp8_einsum_kernel",
    ]),
]

def const_value(name: str):
    u = name.upper()
    if "BLOCK" in u or u.startswith("UPCAST"):
        return 16
    if any(u.startswith(p) for p in ("IS_", "USE_", "HAS_", "STORE_", "QUANT", "DECODE", "CLEAN", "DOT_", "B_BF16", "SANITIZE", "PAD_IS", "TMA_", "LAUNCH")):
        return 0
    if "DIM" in u or "HEAD" in u or "HIDDEN" in u or "RANK" in u or "GROUP" in u:
        return 128
    return 8

def build(k):
    names = list(k.arg_names)
    cidx = set(getattr(k, "constexprs", []) or [])
    sig, ce = {}, {}
    for i, n in enumerate(names):
        if i in cidx:
            v = const_value(n)
            sig[n] = "constexpr"; ce[n] = v
        elif n.endswith("_ptr") or n.endswith("_ptr,"):
            # 名字里带这些词的指针装的是整数（块表/长度/索引/槽位），其余按 fp32
            lowered = n.lower()
            if any(t in lowered for t in ("block_tables", "block_table", "seqlen", "lens",
                                          "idx", "ids", "pos", "slot", "indices", "rows")):
                sig[n] = "*i32"
            else:
                sig[n] = "*fp32"
        else:
            sig[n] = "i32"
    return sig, ce

def main():
    print("triton:", triton.__version__)
    total = ok = 0
    fails = []
    for modname, kernels in TARGETS:
        mod = __import__(modname, fromlist=["*"])
        for kn in kernels:
            k = getattr(mod, kn, None)
            if k is None:
                print(f"  - {kn}: 不存在"); continue
            sig, ce = build(k); total += 1
            try:
                tcompile(ASTSource(k, sig, constexprs=ce),
                         target=GPUTarget("cuda", 80, 32))
                print(f"  ✅ sm80 {kn}"); ok += 1
            except Exception as e:
                fails.append((kn, f"{type(e).__name__}: {e}"))
                print(f"  ❌ sm80 {kn}: {str(e)[:160]}")
    print(f"\n结果: {ok}/{total} 通过")
    if fails and len(sys.argv) > 1:
        print("\n--- 首个失败详情 ---")
        print(fails[0][1][:3000])
    return 0 if ok == total else 1

if __name__ == "__main__":
    sys.exit(main())
