#!/usr/bin/env python3
"""qfn 3×A100 层分区容量计算器(纯离线,T0.2)。

不是 TP=3:按**层**切到各卡(PP 风格),每卡只放自己那段层的专家 + 本地 KV。

用法:
  python scripts/plan_qfn_3gpu_budget.py                       # NVFP4 默认参数
  python scripts/plan_qfn_3gpu_budget.py --fmt mxfp4 --ctx 262144
  python scripts/plan_qfn_3gpu_budget.py --split 14,17,17 --kv-dtype bf16
"""
from __future__ import annotations

import argparse
import json
import os


def kv_bytes_per_token(ctx_layers: int, kv_heads: int, head_dim: int, dtype_bytes: int) -> int:
    """每 token 的 KV 字节:只算 full-attention 层(混合模型里 linear 层不存 KV)。"""
    return ctx_layers * 2 * kv_heads * head_dim * dtype_bytes          # K + V


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", choices=["nvfp4", "mxfp4"], default="nvfp4")
    ap.add_argument("--cards", default="40,40,40", help="每卡可用 GB(逗号)")
    ap.add_argument("--split", default="16,16,16", help="每卡层数(逗号)")
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--experts-gb", type=float, default=None)
    ap.add_argument("--other-gb", type=float, default=8.20, help="BF16 非专家(全模型)")
    ap.add_argument("--mtp-gb", type=float, default=1.52)
    ap.add_argument("--full-attn-layers", type=int, default=12)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--config", default="", help="可选:从 config.json 读 num_key_value_heads/head_dim")
    ap.add_argument("--kv-dtype", choices=["fp8", "bf16"], default="fp8",
                    help="qfn 的既成配置是 fp8(见 USER_QA_LEDGER:958,342→1,688,025 token,"
                         "256K 四针 4/4 + GSM8K 25/25 逐题一致);bf16 只作对照")
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--margin-gb", type=float, default=3.0)
    a = ap.parse_args()

    if a.config:
        import json as _j
        _c = _j.load(open(a.config))
        _t = _c.get("text_config", _c)
        a.kv_heads = int(_t.get("num_key_value_heads") or a.kv_heads)
        a.head_dim = int(_t.get("head_dim") or a.head_dim)
        print(f"[config] kv_heads={a.kv_heads} head_dim={a.head_dim}")
    experts_gb = a.experts_gb or (67.99 if a.fmt == "nvfp4" else 64.19)
    caps = [float(x) for x in a.cards.split(",")]
    split = [int(x) for x in a.split.split(",")]
    if len(caps) != len(split):
        raise SystemExit("--cards 与 --split 数量必须一致")
    if sum(split) != a.layers:
        raise SystemExit(f"层数之和 {sum(split)} != {a.layers}")

    per_layer_expert = experts_gb / a.layers
    per_layer_other = (a.other_gb + a.mtp_gb) / a.layers
    kvb = kv_bytes_per_token(a.full_attn_layers, a.kv_heads, a.head_dim,
                             2 if a.kv_dtype == "bf16" else 1)
    kv_total = kvb * a.ctx / 1e9

    print(f"== qfn {a.fmt.upper()} 3GPU 层分区预算 ==")
    print(f"  专家 {experts_gb:.2f} GB({per_layer_expert * 1000:.0f} MB/层);"
          f"其它 {a.other_gb + a.mtp_gb:.2f} GB;KV {kvb}B/token × {a.ctx} = {kv_total:.2f} GB")
    print(f"  {'卡':<4}{'层':>4}{'专家':>9}{'其它':>9}{'权重':>9}{'KV':>8}{'合计':>9}{'余量':>8}")
    ok = True
    for i, (cap, n) in enumerate(zip(caps, split)):
        ex = per_layer_expert * n
        ot = per_layer_other * n
        w = ex + ot
        kv = kv_total                          # 每卡本地 KV(同样的上下文)
        tot = w + kv
        left = cap - tot
        flag = "✓" if left >= a.margin_gb else "⛔"
        if left < a.margin_gb:
            ok = False
        print(f"  gpu{i:<2}{n:>4}{ex:>9.2f}{ot:>9.2f}{w:>9.2f}{kv:>8.2f}{tot:>9.2f}"
              f"{left:>+8.2f} {flag}")
    print("\n✓ 全部满足 ≥%.1f GB 余量" % a.margin_gb if ok
          else "\n⛔ 至少一卡余量不足 —— 减层 / 减 context / 改 KV dtype")

    # 方案 B 提醒
    print("\n提示:方案 B(GPU0 单卡 + GPU1/2 TP=2)需要按 1:2 的算力/显存重新算,")
    print("      本工具只算方案 A(等分或给定切分)。")


if __name__ == "__main__":
    main()
