#!/usr/bin/env python3
"""校验**派生 ckpt**（`...-NVFP4-attnBF16`）里的 attention 权重是否忠实于原始模型。

背景：这个派生目录是上一轮"把 attention 离线反量化成 BF16"的产物
（`attn_bf16.safetensors` 5.35 GB，专家分片是指向原始 121 GB 模型的符号链接）。
如果这一步反量化做错了，MoE 再怎么对，输出也一定是乱码 —— 而且**任何日志都不报错** ✗

做法：从原始模型取同一个 attention 张量的 FP8 权重 + 块尺度 ⇒ 反量化 ⇒
与派生目录的 BF16 逐元素比；打印 max|diff| 与相对误差。

用法：
    /root/venvs/vllm310/bin/python scripts/musa/verify_derived_attn_bf16.py
    ... --key model.language_model.layers.0.linear_attn.out_proj.weight
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ORIG = "/root/models/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4"
DERIVED = "/root/models/Qwen3.8-Flash-Next-NVFP4-attnBF16"


def _find_shard(root: str, key: str) -> str | None:
    idx = os.path.join(root, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        if key in wm:
            return os.path.join(root, wm[key])
    return None


def _load(root: str, key: str):
    from safetensors import safe_open

    shard = _find_shard(root, key)
    if shard is None:
        return None
    with safe_open(shard, framework="pt") as f:
        if key not in f.keys():
            return None
        return f.get_tensor(key)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="model.language_model.layers.0.linear_attn.out_proj.weight")
    ap.add_argument("--block", type=int, default=128)
    args = ap.parse_args()
    import torch

    key = args.key
    print(f"[verify] key = {key}")

    d = _load(DERIVED, key)
    if d is None:
        # 派生目录把 attention 放在一个单独的 safetensors 里
        from safetensors import safe_open

        p = os.path.join(DERIVED, "attn_bf16.safetensors")
        with safe_open(p, framework="pt") as f:
            ks = list(f.keys())
            cand = [k for k in ks if k == key or k.endswith(key)]
            if not cand:
                # 尝试去掉 model.language_model. 前缀
                short = key.replace("model.language_model.", "").replace("model.", "")
                cand = [k for k in ks if k == short or k.endswith(short)]
            if not cand:
                print(f"[verify] ⛔ 派生目录里找不到该键；样例键: {ks[:3]}")
                return 2
            with safe_open(p, framework="pt") as f2:
                d = f2.get_tensor(cand[0])
            print(f"[verify] 派生键 = {cand[0]}")
    print(f"[verify] derived: dtype={d.dtype} shape={tuple(d.shape)} "
          f"mean={d.float().mean():.4e} std={d.float().std():.4e} "
          f"absmax={d.float().abs().max():.4e}")

    w = _load(ORIG, key)
    if w is None:
        print("[verify] ⛔ 原始模型里找不到该键（可能原名不同）")
        return 2
    print(f"[verify] orig(fp8): dtype={w.dtype} shape={tuple(w.shape)} absmax={w.float().abs().max():.4e}")

    sc = _load(ORIG, key + "_scale_inv")
    if sc is None:
        sc = _load(ORIG, key.replace(".weight", ".weight_scale_inv"))
    if sc is None:
        print("[verify] ⛔ 原始模型里没有块尺度（不是 FP8 块量化？）")
        return 2
    print(f"[verify] scale: shape={tuple(sc.shape)} absmax={sc.float().abs().max():.4e}")

    s = sc.float()
    if s.dim() == 4:
        s = s.squeeze(1).squeeze(-1)
    blk = args.block
    up = s.repeat_interleave(blk, 0).repeat_interleave(blk, 1)[: w.shape[0], : w.shape[1]]
    ref = (w.float() * up).to(torch.bfloat16).float()
    got = d.float()
    if ref.shape != got.shape:
        print(f"[verify] ⛔ 形状不一致 {tuple(ref.shape)} vs {tuple(got.shape)}")
        return 1
    diff = (got - ref).abs()
    rel = diff.mean() / ref.abs().mean().clamp_min(1e-8)
    print(f"[verify] ref std={ref.std():.4e} | max|diff|={diff.max():.4e} "
          f"| 相对误差={float(rel):.4e}")
    ok = float(rel) < 5e-2
    print("[verify]", "✅ 派生 attention 与原始一致" if ok else "⛔ 派生 attention **不一致**（离线反量化有错）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
