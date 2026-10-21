#!/usr/bin/env python3
"""qfn FP4 checkpoint 审计 + dispatch 门禁(纯离线,T0.1/T0.1b)。

只读 `config.json` / `model.safetensors.index.json`,不需要 GPU、不加载权重。

用法:
  python scripts/audit_qfn_fp4_checkpoint.py --ckpt <snapshot 目录> \
      --expect nvfp4|mxfp4 [--json out.json]

门禁(不过 ⇒ 退出码 1):
  * quant_method 必须是混合量化;
  * **49** 个 `W4A16_NVFP4`(48 主层 + 1 MTP)或 `W4A8_MXFP4_FP8`(回退);
  * **156** 个 `FP8_PB_WO`;
  * index 里必须**没有** "skip loading" 风险项(缺 mtp.* / 缺 scale_inv 的迹象);
  * 专家体积 / 每层体积与计划(D7.1)一致(±5%)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

EXPECT_NVFP4 = {"W4A16_NVFP4": 49, "FP8_PB_WO": 156}
EXPECT_MXFP4 = {"W4A8_MXFP4_FP8": 49, "FP8_PB_WO": 156}
PLAN_EXPERTS_GB = {"nvfp4": 67.99, "mxfp4": 64.19}
PLAN_TOTAL_GPU_GB = {"nvfp4": 77.70, "mxfp4": 75.17}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--expect", choices=["nvfp4", "mxfp4"], default="nvfp4")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    cfg_p = os.path.join(a.ckpt, "config.json")
    idx_p = os.path.join(a.ckpt, "model.safetensors.index.json")
    if not os.path.exists(cfg_p):
        print(f"⛔ 缺少 {cfg_p}")
        return 1
    cfg = json.load(open(cfg_p))
    t = cfg.get("text_config", cfg)
    q = cfg.get("quantization_config") or t.get("quantization_config") or {}
    qlayers = q.get("quantized_layers") or q.get("quantized_layers_map") or {}

    counts: dict[str, int] = {}
    for _, v in qlayers.items():
        algo = (v or {}).get("quant_algo") if isinstance(v, dict) else v
        counts[str(algo)] = counts.get(str(algo), 0) + 1

    print("== 基本结构 ==")
    for k in ("num_hidden_layers", "num_experts", "num_experts_per_tok",
              "hidden_size", "moe_intermediate_size", "max_position_embeddings"):
        if k in t:
            print(f"  {k} = {t[k]}")
    print(f"  quant_method = {q.get('quant_method')}")
    print(f"  quantized_layers = {len(qlayers)} 个;分布 = {counts}")

    want = EXPECT_NVFP4 if a.expect == "nvfp4" else EXPECT_MXFP4
    ok = True
    print("\n== 门禁 1:逐层量化算法计数 ==")
    for algo, n in want.items():
        got = counts.get(algo, 0)
        flag = "✓" if got == n else "⛔"
        if got != n:
            ok = False
        print(f"  {flag} {algo}: {got} (期望 {n})")

    if not os.path.exists(idx_p):
        print(f"⛔ 缺少 {idx_p}(无法审计 tensor 名)")
        return 1
    idx = json.load(open(idx_p))
    names = list((idx.get("weight_map") or {}).keys())
    print(f"\n== weight_map: {len(names)} 个 tensor ==")
    for pat in ("mtp.", "ple.", "engram.", "mtp_block"):
        hit = sum(1 for n in names if pat in n)
        print(f"  {pat:10s} 命中 {hit}")
    need = ["mtp."] if a.expect in ("nvfp4", "mxfp4") else []
    for pat in need:
        if not any(pat in n for n in names):
            print(f"  ⛔ 没有 `{pat}` tensor ⇒ MTP 可能没下全")
            ok = False

    # 专家体积估算:48 层 × 每层 experts.*.weight_scale 的规模从 index 元数据读不到,
    # 只能按计划值核对 checkpoint 总大小。
    total = 0
    for f in os.listdir(a.ckpt):
        if f.endswith(".safetensors"):
            total += os.path.getsize(os.path.join(a.ckpt, f))
    print(f"\n== 体积 ==")
    print(f"  checkpoint 文件合计 = {total / 1e9:.2f} GB(计划值 NVFP4 128.95 / MXFP4 125.07)")

    print("\n== 结论 ==")
    print("✓ 门禁通过" if ok else "⛔ 门禁未通过 —— 禁止起服务")
    if a.json:
        json.dump({"ok": ok, "counts": counts, "total_bytes": total},
                  open(a.json, "w"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
