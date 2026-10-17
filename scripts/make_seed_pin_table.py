#!/usr/bin/env python3
# scripts/make_seed_pin_table.py —— 生成【冷启动种子】钉选表(XTUP v0)
"""为什么需要"种子表"(2026-10-15 用户裁定"动态换入"后新增):

`expert_pool.build_layer_pool` 要求表里**每层至少有一条**才建池 ⇒ 要跑"动态换入"
就必须**先有一张表** ✗。而真表的正规来源是路由轨迹(`route_probe` dump)⇒ 鸡生蛋 ✗。

⇒ 本脚本生成一张**尺寸正确、内容随意**的种子表:
  * 每层钉**相同数量的槽**(`--slots-per-layer`,缺省 8)⇒ 池大小可预测 ✓;
  * 初始专家号取**该层前 `n` 个**(任意选择即可)✓;
  * ⭐ 装载后由 `XIAOTU_PIN_WINDOW` 驱动的 `plan_swap`/`swap_in` **自动把冷槽换成热专家**
    ⇒ 种子内容对最终稳态**几乎无影响**,它只决定"从多冷开始"✓
      (旁证:`STRATA_ENGINE_LESSONS` §9.4 —— 静态全局先到先得只 2.97% 命中,
       而带自适应的一旦跑起来就靠**在线热度**收敛 ✓)

⚠️ **池大小必须显式算清**:池是**从 KV 预算里扣**的(SPEC §3/§4)⇒ 槽数给太大
   会把 KV 挤到不可用 ✗。用 `--print-gib` 先看代价:

    每槽字节 = H·2I  +  (H/128)·(2I/128)·4  +  I·H  +  (I/128)·(H/128)·4      [FP8, e4m3+fp32尺度]
               (H/2)·2I + (H/gk)·2I         +  (I/2)·H + (I/gk)·H              [MXFP4, 4bit+e8m0]

XTUP v0 格式(与 `expert_pool.load_pin_table` 逐字对齐,头长 **20** 字节 ✗ 不是 16):
    header  = struct.pack("<4sIIII", b"XTUP", 1, n_layers, e_max, n_entries)
    entry   = struct.pack("<HH", layer, expert)      # 按全局频率降序
"""
from __future__ import annotations

import argparse
import struct
import sys

# 各模型的 MoE 层数 / 专家数(用最小公倍数式的保守默认;可用 --modules 直接给)
MODELS = {
    # name: (层数, 每层专家数)
    "glm":  (62, 288),      # GLM-5.3-Flash:FP8 专家(block 128x128)
    "ds41f": (40, 128),     # DeepSeek-V4.1-Flash:MXFP4 专家(gk=32)
    "qfn":  (48, 512),      # Qwen3.8-Flash-Next:FP8 专家
}


def slot_bytes_fp8(H: int, I: int, block: int = 128) -> int:
    return (H * 2 * I
            + (H // block) * (2 * I // block) * 4
            + I * H
            + (I // block) * (H // block) * 4)


def slot_bytes_mxfp4(H: int, I: int, gk: int = 32) -> int:
    return ((H // 2) * 2 * I + (H // gk) * 2 * I
            + (I // 2) * H + (I // gk) * H)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="输出 .bin(XTUP v0)")
    ap.add_argument("--model", default="glm", choices=sorted(MODELS))
    ap.add_argument("--layers", type=int, default=0, help="0 = 用 --model 的默认层数")
    ap.add_argument("--experts", type=int, default=0, help="0 = 用 --model 的默认专家数")
    ap.add_argument("--slots-per-layer", type=int, default=8)
    ap.add_argument("--layer-name", default="layers.%d.mlp.experts")
    ap.add_argument("--print-gib", action="store_true",
                    help="只打印显存代价(用 --hidden/--inter/--gk 算),不写文件")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--inter", type=int, default=1024, help="**每 rank** 的 I(TP=2 时已折半)")
    ap.add_argument("--gk", type=int, default=128)
    ap.add_argument("--layout", default="fp8", choices=["fp8", "mxfp4"])
    a = ap.parse_args()

    dl, de = MODELS[a.model]
    L = int(a.layers or dl)
    E = int(a.experts or de)
    S = int(a.slots_per_layer)

    if a.print_gib:
        per = (slot_bytes_fp8(a.hidden, a.inter) if a.layout == "fp8"
               else slot_bytes_mxfp4(a.hidden, a.inter, a.gk))
        tot = per * S * L
        print(f"  layout={a.layout} H={a.hidden} I={a.inter} gk={a.gk}")
        print(f"  每槽 = {per/2**20:.2f} MiB")
        print(f"  池   = {S} 槽/层 × {L} 层 = {S*L} 槽 ⇒ **{tot/2**30:.2f} GiB**")
        return 0

    entries = []
    order = 0
    for l in range(L):
        for s in range(min(S, E)):
            # 反向写:让 layer0 的最后一条排在最前(顺序对种子无实质影响,只求确定)
            entries.append((l, s, order))
            order += 1
    # 按"层内槽序"升序写出即可(种子不依赖频率序;load_pin_table 按层归组)
    entries.sort(key=lambda t: (t[0], t[1]))

    with open(a.out, "wb") as f:
        f.write(struct.pack("<4sIIII", b"XTUP", 1, L, max(E, max((e for _, e, _ in entries), default=0) + 1),
                            len(entries)))
        for l, e, _ in entries:
            f.write(struct.pack("<HH", l, e))
    print(f"  emit seed-table: {len(entries)} 条目({S} 槽/层 × {L} 层)⇒ {a.out}")
    print(f"  ⚠️ 池大小请用 --print-gib 先算;池从 KV 预算扣(SPEC §3)✗")
    return 0


if __name__ == "__main__":
    sys.exit(main())
