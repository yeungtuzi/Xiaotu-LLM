#!/usr/bin/env python3
"""辅助 GPU（挎斗）显存预算计算器（纯 Python，无 GPU 依赖）。

用途：给定模型/槽数/上下文/并发，算出
  - GPU0 挎斗占多少、剩多少；
  - GPU1/2 的 KV 池需要多少、余量多少；
  - GPU1/2 还能留多少 PIN 槽（不 OOM）。
用法示例：
  python scripts/plan_sidecar_budget.py --slots 1900 --ctx 524288 --conc 4
"""
import argparse
import json

GB = 1e9
GIB = 1024 ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=1900, help="GPU0 整专家槽数")
    ap.add_argument("--expert-mb", type=float, default=18.81, help="一个整专家 MB（ds41f）")
    ap.add_argument("--gpu0-gb", type=float, default=40.0)
    ap.add_argument("--gpu0-reserve-gb", type=float, default=4.0)
    ap.add_argument("--kv-pool-gib", type=float, default=3.43, help="util 0.70、PIN=0 时的 KV 池")
    ap.add_argument("--kv-kb-per-token", type=float, default=1.35, help="fp8_ds_mla")
    ap.add_argument("--ctx", type=int, default=524288)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--pin-slot-mb", type=float, default=9.40, help="GPU1/2 每 rank 半专家 MB")
    a = ap.parse_args()

    expert_gb = a.expert_mb / 1000.0
    gpu0_used = a.slots * expert_gb
    gpu0_free = a.gpu0_gb - a.gpu0_reserve_gb - gpu0_used

    kv_need_gb = a.ctx * a.conc * a.kv_kb_per_token / 1e6
    kv_pool_gb = a.kv_pool_gib * GIB / GB
    kv_margin_gb = kv_pool_gb - kv_need_gb
    pin_slots_max = max(0, int(kv_margin_gb * 1000 / a.pin_slot_mb * 0.9))  # 10% 安全

    out = {
        "gpu0": {
            "slots": a.slots,
            "experts_gb": round(gpu0_used, 2),
            "reserve_gb": a.gpu0_reserve_gb,
            "free_gb": round(gpu0_free, 2),
            "fits": gpu0_free >= 0,
            "max_slots": int((a.gpu0_gb - a.gpu0_reserve_gb) / expert_gb),
        },
        "kv": {
            "ctx": a.ctx,
            "conc": a.conc,
            "need_gb": round(kv_need_gb, 2),
            "pool_gb": round(kv_pool_gb, 2),
            "margin_gb": round(kv_margin_gb, 2),
            "fits": kv_margin_gb >= 0,
            "max_conc": int(kv_pool_gb * 1e6 / (a.ctx * a.kv_kb_per_token)),
        },
        "gpu12_pin": {
            "max_slots_10pct_margin": pin_slots_max,
            "chosen": 0,
        },
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["gpu0"]["fits"] and out["kv"]["fits"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
