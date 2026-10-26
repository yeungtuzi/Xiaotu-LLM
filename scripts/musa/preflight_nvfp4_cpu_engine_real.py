#!/usr/bin/env python3
"""用**真实 checkpoint 的专家权重字节**验证 CPU 引擎的 NVFP4 数学习惯。

为什么这一步不能省：`preflight_nvfp4_cpu_engine.py` 用的是**我按内核约定自己打包**的
随机字节，所以它只能证明"引擎与我的打包约定自洽"，**不能**证明
"checkpoint 的字节 == 引擎期望的布局" ✗ —— 两者不一致时同样**不报错、只出垃圾数**。

本脚本取真实 ckpt 的前 N 个专家（layer 0），
  * 用引擎算一次（`MOE_NVFP4` + `cpu_prefill`）
  * 用 **torch 参考**按"NVIDIA NVFP4 约定：低 nibble = 偶数 k、组大小 16、fp32 组尺度 × scale_2"解码同一批字节
  * 对比两者（同样用随机隐藏态，避免依赖前向环境）
两者一致 ⇒ 布局/尺度约定正确；不一致 ⇒ 布局或尺度约定与引擎期望不同（**必须先定位，别再跑 e2e**）。

用法：
    /root/venvs/vllm310/bin/python scripts/musa/preflight_nvfp4_cpu_engine_real.py --experts 4
"""

from __future__ import annotations

import argparse
import sys

M = "/root/models/Qwen3.8-Flash-Next-NVFP4-attnBF16"
PREFIX = "model.language_model.layers.0.mlp.experts."


def _load(key: str):
    import json
    import os

    from safetensors import safe_open

    idx = json.load(open(os.path.join(M, "model.safetensors.index.json")))["weight_map"]
    shard = idx[key]
    with safe_open(os.path.join(M, shard), framework="pt") as f:
        return f.get_tensor(key)


def _unpack(packed, low_first=True):
    """[E, N, K/2] uint8 -> [E, N, K] float32（NVFP4 约定：低 nibble = 偶数 k）。"""
    import torch

    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    lo = (packed & 0x0F).long()
    hi = ((packed >> 4) & 0x0F).long()
    out = torch.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=torch.float32)
    if low_first:
        out[..., 0::2] = lut[lo]
        out[..., 1::2] = lut[hi]
    else:
        out[..., 0::2] = lut[hi]
        out[..., 1::2] = lut[lo]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--low-first", type=int, default=1, help="1=低 nibble 是偶数 k（NVFP4 约定）")
    args = ap.parse_args()

    import torch
    import xiaotu_moe

    global PREFIX
    PREFIX = f"model.language_model.layers.{args.layer}.mlp.experts."
    n = args.experts
    # ⚠️ checkpoint 是**逐专家**存的（`experts.<id>.{gate,up,down}_proj.*`），
    #    插件在加载时按 `w13 = [gate; up]` 融合 ⇒ 这里必须用**同样的融合顺序**
    _g, _u, _d = [], [], []
    _gs, _us, _ds = [], [], []
    _g2, _u2, _d2 = [], [], []
    for e in range(n):
        p = PREFIX + f"{e}."
        _g.append(_load(p + "gate_proj.weight"))
        _u.append(_load(p + "up_proj.weight"))
        _d.append(_load(p + "down_proj.weight"))
        _gs.append(_load(p + "gate_proj.weight_scale"))
        _us.append(_load(p + "up_proj.weight_scale"))
        _ds.append(_load(p + "down_proj.weight_scale"))
        _g2.append(_load(p + "gate_proj.weight_scale_2").reshape(-1)[0])
        _u2.append(_load(p + "up_proj.weight_scale_2").reshape(-1)[0])
        _d2.append(_load(p + "down_proj.weight_scale_2").reshape(-1)[0])
    import torch as _tt
    w13 = _tt.stack([_tt.cat([g, u], dim=0) for g, u in zip(_g, _u)]).contiguous()   # [n,2I,H/2]
    w2 = _tt.stack(_d).contiguous()                                                  # [n,H,I/2]
    s13b = _tt.stack([_tt.cat([g, u], dim=0) for g, u in zip(_gs, _us)]).contiguous()  # e4m3 字节
    s2b = _tt.stack(_ds).contiguous()
    ts13 = _tt.stack(_g2).reshape(n, 1).contiguous()     # 与插件一致：只取 gate 那一列
    ts2 = _tt.stack(_d2).reshape(n, 1).contiguous()
    print(f"[real] gate/up 的 scale_2 是否相同: "
          f"{'是' if bool((_tt.stack(_g2) == _tt.stack(_u2)).all()) else '否'}")
    E = int(w13.shape[0])
    I = int(w13.shape[1]) // 2
    H = int(w13.shape[2]) * 2
    GKS = 16
    print(f"[real] E={E} H={H} I={I} w13={tuple(w13.shape)} s13={tuple(s13b.shape)} "
          f"ts13={tuple(ts13.shape)}")

    import torch as _t
    s13 = s13b.view(_t.float8_e4m3fn).to(_t.float32) * ts13[:, 0].view(E, 1, 1)
    s2 = s2b.view(_t.float8_e4m3fn).to(_t.float32) * ts2[:, 0].view(E, 1, 1)
    s13 = s13.contiguous()
    s2 = s2.contiguous()

    # ── torch 参考（按 NVFP4 约定解码**同一批字节**）────────────────────────
    d13 = _unpack(w13, bool(args.low_first)) * s13.repeat_interleave(GKS, dim=2)
    d2 = _unpack(w2, bool(args.low_first)) * s2.repeat_interleave(GKS, dim=2)

    torch.manual_seed(args.seed)
    T, K = args.T, min(args.K, E)
    x = (torch.randn(T, H) * 0.02).to(torch.bfloat16)
    wts, ids = torch.topk(torch.randn(T, E).softmax(-1), K, dim=-1)
    wts = wts.to(torch.float32).contiguous()
    ids = ids.to(torch.int32).contiguous()
    xf = x.to(torch.float32)
    ref = torch.zeros((T, H), dtype=torch.float32)
    for t in range(T):
        acc = torch.zeros(H)
        for j in range(K):
            e = int(ids[t, j])
            gate = xf[t] @ d13[e, :I, :].T
            up = xf[t] @ d13[e, I:, :].T
            acc += float(wts[t, j]) * ((torch.nn.functional.silu(gate) * up) @ d2[e].T)
        ref[t] = acc

    # ── 引擎 ────────────────────────────────────────────────────────────────
    cfg = xiaotu_moe.MOEConfigV2()
    cfg.num_processes, cfg.process_id, cfg.gpu_id = 1, 0, 0
    cfg.has_gate_proj = True
    cfg.expert_num, cfg.top_k = E, K
    cfg.hidden_size, cfg.intermediate_size = H, I
    cfg.max_batch_size, cfg.max_num_seqs = max(64, T + 8), 64
    cfg.stride, cfg.group_min_len = 32, 10
    cfg.group_max_len = max(4096, T) + 128
    cfg.groupN, cfg.groupK = 1, GKS
    cfg.activation_type, cfg.use_gpu_prefill, cfg.int8_activation = 0, False, 0
    eng = xiaotu_moe.MOE_NVFP4(cfg, w13.data_ptr(), w2.data_ptr(),
                               s13.data_ptr(), s2.data_ptr(), 0, 0)
    out = torch.zeros((T, H), dtype=torch.float32)
    xh = x.contiguous()
    eng.cpu_prefill(T, K, ids.data_ptr(), wts.data_ptr(), xh.data_ptr(), out.data_ptr())

    rel = (out - ref).abs().mean() / ref.abs().mean().clamp_min(1e-8)
    print(f"[real] ref|mean|={ref.abs().mean():.4e} got|mean|={out.abs().mean():.4e} "
          f"NaN={int(torch.isnan(out).sum())}")
    print(f"[real] 相对误差 = {float(rel):.4e}")
    ok = float(rel) < 5e-2
    print("[real]", "✅ 真实 ckpt 字节与引擎约定一致" if ok
          else "⛔ 不一致 ⇒ checkpoint 布局/尺度约定与引擎期望不同（先定位，别跑 e2e）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
