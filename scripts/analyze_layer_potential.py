#!/usr/bin/env python3
"""从一次带 `XIAOTU_PIN_DEBUG=1` + `XIAOTU_CD_TIMING=1` 的日志里,
算出**每层的常驻收益潜力**:CPU 计算时间 vs 路由集中度(活跃专家数/top8 覆盖)。

判据(2026-10-19):
  * 常驻一层省下的 = 该层 CPU MoE(`compute`)+ 一次 D2H/H2D 握手;
  * 常驻一层付出的 = GPU pair 内核(其成本随**活跃专家数**增长,集中度越高越便宜);
  * ⇒ 潜力 ≈ compute / 活跃专家数 排序(越大越该常驻)。

用法:
  python scripts/analyze_layer_potential.py --log <日志> [--top 10]
"""
from __future__ import annotations

import argparse
import collections
import re

DIST = re.compile(
    r"\[xtu-pin\]\[dist\] layer(\d+): experts=(\d+)/\d+ events=(\d+) slots=\d+ "
    r"top8=([\d.]+)% top16=([\d.]+)%")
TIMING = re.compile(
    r"\[cd-timing\] layers=\d+ qlen=(\d+) k=\d+ period=([\d.]+)ms compute=([\d.]+)ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()

    dist = collections.defaultdict(list)
    acc = collections.defaultdict(lambda: [0.0, 0.0, 0])  # compute sum, period sum, n
    n = 0
    for line in open(a.log, errors="ignore"):
        m = DIST.search(line)
        if m:
            dist[int(m.group(1))].append(
                (int(m.group(2)), float(m.group(4)), float(m.group(5))))
        m = TIMING.search(line)
        if m and float(m.group(1)) <= 4:
            layer = n % 48
            n += 1
            r = acc[layer]
            r[0] += float(m.group(3))
            r[1] += float(m.group(2))
            r[2] += 1

    rows = []
    for layer in range(48):
        if layer not in dist or not acc[layer][2]:
            continue
        k = len(dist[layer])
        ex = sum(x[0] for x in dist[layer]) / k
        t8 = sum(x[1] for x in dist[layer]) / k
        t16 = sum(x[2] for x in dist[layer]) / k
        comp = acc[layer][0] / acc[layer][2]
        rows.append((layer, comp, ex, t8, t16, comp / max(ex, 1.0)))

    print(f"样本: {n} 次 MoE 调用, {len(rows)} 层有完整数据")
    print(f"{'layer':>5} {'cpu_ms':>7} {'experts':>7} {'top8%':>6} {'top16%':>7} {'gain_idx':>9}")
    for r in sorted(rows, key=lambda x: -x[5]):
        print(f"{r[0]:5d} {r[1]:7.3f} {r[2]:7.1f} {r[3]:6.1f} {r[4]:7.1f} {r[5]:9.4f}")
    best = sorted(rows, key=lambda x: -x[5])[: a.top]
    print("\n建议常驻(按潜力排序):", ", ".join(str(r[0]) for r in best))
    print("平均 cpu_ms/层 = %.3f;平均活跃专家 = %.1f" % (
        sum(r[1] for r in rows) / max(len(rows), 1),
        sum(r[2] for r in rows) / max(len(rows), 1)))


if __name__ == "__main__":
    raise SystemExit(main())
