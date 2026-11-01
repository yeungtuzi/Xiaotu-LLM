"""真机验证：降级 constexpr 后，内核变体是否还会随请求长度增长。

在空闲 GPU 上直接调用 `fp8_mqa_logits_triton`，用 **多种 num_q / seq_len_kv** 组合，
统计 Triton 缓存目录增量。
期望（修复生效）：只新增 **1 份**变体（首个组合），之后长度变化不再新增。
修复前行为：每换一个长度 +1 份（实测该内核历史累计 1,121 份）。
"""
import os, glob, json, collections, itertools, sys

def variant_count():
    T = os.path.expanduser("~/.triton/cache")
    return len([d for d in os.listdir(T) if os.path.isdir(os.path.join(T, d))])

def main():
    import torch
    from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import fp8_mqa_logits_triton as f

    dev = "cuda:0"
    H, D = 16, 128
    # 30 组互不相同的长度（覆盖 16 倍数/非倍数、==1 类、大数）
    import random as _r
    _r.seed(7)
    combos = [(64, 128), (100, 128), (129, 200), (200, 333), (333, 512), (512, 777)]
    combos += [(_r.choice([65, 79, 97, 113, 128, 144, 160, 176]), _r.choice([321, 444, 555, 666, 888, 999, 1111, 2222])) for _ in range(24)]
    torch.cuda.init()

    base = variant_count()
    print(f"起始缓存变体数: {base}")

    first_delta = None
    for i, (nq, nkv) in enumerate(combos):
        q = torch.randn(nq, H, D, device=dev, dtype=torch.bfloat16)
        k = torch.randn(nkv, D, device=dev, dtype=torch.bfloat16)
        # 该路径要求 uint8 视图（Ampere 无 fp8e4nv）
        q_u8 = q.view(torch.uint8).contiguous()
        k_u8 = k.view(torch.uint8).contiguous()
        scale = torch.ones(nkv, 1, device=dev, dtype=torch.float32)
        w = torch.randn(nq, H, device=dev, dtype=torch.float32)
        ks = torch.zeros(nq, device=dev, dtype=torch.int32)
        ke = torch.full((nq,), nkv, device=dev, dtype=torch.int32)
        try:
            f(q_u8, (k_u8, scale), w, ks, ke)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  ⚠️ ({nq},{nkv}) 调用失败: {type(e).__name__}: {str(e)[:120]}")
            continue
        cur = variant_count()
        d = cur - base
        if first_delta is None:
            first_delta = d
        print(f"  ({nq:>4},{nkv:>4}) 完成 → 累计新增变体 {d}")
    final = variant_count() - base
    print(f"\n结论：{len(combos)} 种不同长度组合共新增 **{final}** 份变体")
    print("  ✅ PASS（形状爆炸已消除）" if final <= 2 else
          f"  ❌ FAIL：仍有按长度新增（{final} 份）")

if __name__ == "__main__":
    sys.exit(main())
