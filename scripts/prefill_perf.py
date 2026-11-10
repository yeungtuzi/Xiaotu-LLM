#!/usr/bin/env python3
"""prefill_perf.py —— 从复现输出里算**纯 prefill 吞吐**，并做修前/修后对照。

为什么要单独算：验收要求"prefill 性能不许牺牲"（≥95% 基线）。复现输出每行已给出
`cached=<n> ptok=<m> <秒>s`，其中 `cached=0` 的请求就是**纯预填充**（无缓存命中）；
把它们按 `ptok/秒` 汇总取中位数，即得到可比的 prefill tok/s。

⚠️ 只统计 `cached == 0` 且 `秒 > 0` 的行：
   * 命中请求（`cached>0`）测的是重放/复用，不是预填充；
   * `秒 == 0` 的行（亚秒）会把吞吐算成无穷大。

用法：
    python3 scripts/prefill_perf.py <out1.txt> [<out2.txt> ...]        # 各自打印
    python3 scripts/prefill_perf.py --compare <修复前> <修复后>          # 对照 + 95% 判据
退出码：0 = 通过（或只统计）；1 = 低于 95%。
"""
from __future__ import annotations

import re
import statistics
import sys

LINE = re.compile(r"cached=(\d+)\s+ptok=(\d+)\s+([0-9.]+)s")


def parse(path: str) -> dict:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            m = LINE.search(ln)
            if not m:
                continue
            cached, ptok, sec = int(m.group(1)), int(m.group(2)), float(m.group(3))
            if cached == 0 and sec > 0 and ptok > 0:
                rows.append((ptok, sec, ptok / sec))
    if not rows:
        return {"path": path, "n": 0}
    tps = [r[2] for r in rows]
    ptoks = [r[0] for r in rows]
    return {
        "path": path,
        "n": len(rows),
        "median_tps": statistics.median(tps),
        "min_tps": min(tps),
        "max_tps": max(tps),
        # ⚠️ 必须记长度：tok/s 与 prompt 长度强相关，两条臂长度不同就没有可比性
        #    （短提示会天然更高 ⇒ 会把"换了口径"误判成"性能更好"）。台账 #227。
        "median_ptok": statistics.median(ptoks),
        "min_ptok": min(ptoks),
        "max_ptok": max(ptoks),
    }


def show(st: dict) -> None:
    if not st.get("n"):
        print(f"{st['path']}: 没有 cached=0 的纯预填充行（无法评估）")
        return
    print(f"{st['path']}: n={st['n']}  中位 {st['median_tps']:.1f} tok/s  "
          f"(min {st['min_tps']:.1f} / max {st['max_tps']:.1f})  "
          f"prompt 中位 {st['median_ptok']:,} tokens({st['min_ptok']:,}–{st['max_ptok']:,})")


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    if args[0] == "--compare":
        rest = [x for x in args[1:] if x != "--allow-len-mismatch"]
        if len(rest) != 2:
            print("用法: --compare <修复前.txt> <修复后.txt> [--allow-len-mismatch]")
            return 2
        args = ["--compare"] + rest + (["--allow-len-mismatch"] if "--allow-len-mismatch" in args else [])
        before, after = parse(args[1]), parse(args[2])
        show(before)
        show(after)
        if not before.get("n") or not after.get("n"):
            print("❌ 有一侧没有可评估的纯预填充行，无法对照")
            return 1
        # 长度一致性门槛：±5% 之外直接拒绝比较（除非显式 --allow-len-mismatch）
        allowance = 0.05
        lb, la = before["median_ptok"], after["median_ptok"]
        drift = abs(la - lb) / max(lb, 1)
        if drift > allowance and "--allow-len-mismatch" not in args:
            print(f"❌ 两条臂的 prompt 长度差 {drift * 100:.1f}%（{lb:,} vs {la:,} tokens）> "
                  f"{allowance * 100:.0f}% ⇒ **不可比**，不给出通过/不通过结论")
            print("   要么用同一 prompt 重跑一条臂，要么显式加 --allow-len-mismatch 自行承担误判风险")
            return 2
        ratio = after["median_tps"] / before["median_tps"]
        ok = ratio >= 0.95
        print(f"\n修后/修前 = {ratio * 100:.1f}%（要求 ≥95%）  ⇒ {'✅ 通过' if ok else '❌ 未达标'}")
        return 0 if ok else 1
    rc = 0
    for p in args:
        st = parse(p)
        show(st)
        if not st.get("n"):
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
