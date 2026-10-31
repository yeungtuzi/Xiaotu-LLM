#!/usr/bin/env python3
"""算实例的**显存增长斜率**与"还能活多久"。

数据源(按可用性自动挑):
  1. `resmon.csv`(有 `ts` 与 `gN_mib` 列) —— 首选,带时间戳;
  2. vLLM 日志里的 `[xtu-vram] steady ... free=` 行 —— 只用来提供**当前池内余量**。

用法:
    python scripts/vram_slope.py --csv resmon.csv --gpu 1 [--since HH:MM] [--cap-mib 40400]
    python scripts/vram_slope.py --log <vllm 日志>          # 仅报当前 free 余量

只读文件,不动任何进程。
"""
import argparse
import csv
import sys
from datetime import datetime, timedelta


def load_csv(path, gpu):
    col = f"g{gpu}_mib"
    rows = []
    with open(path, newline="", errors="ignore") as fh:
        for r in csv.DictReader(fh):
            v, ts = r.get(col), r.get("ts")
            if not v or not ts:
                continue
            try:
                rows.append((ts, float(v)))
            except ValueError:
                continue
    return rows


def minutes(ts):
    h, m, s = (int(x) for x in ts.split(":"))
    return h * 60 + m + s / 60.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--last-min", type=float, default=0.0,
                    help="只用最后 N 分钟的样本(0=全部);跨天安全,优于按时钟过滤")
    ap.add_argument("--cap-mib", type=float, default=40400.0, help="单卡可用上限 MiB")
    ap.add_argument("--log", help="vLLM 日志,用于读取当前池内 free 余量")
    a = ap.parse_args()

    if a.log:
        free = None
        with open(a.log, errors="ignore") as fh:
            for line in fh:
                if "[xtu-vram] steady" in line and "free=" in line:
                    for tok in line.split():
                        if tok.startswith("free="):
                            free = float(tok.split("=")[1])
        if free is None:
            print("日志里没有 [xtu-vram] steady free= 行")
        else:
            print(f"池内可用 free = {free:.3f} GiB")

    if not a.csv:
        return 0

    rows = load_csv(a.csv, a.gpu)
    if len(rows) < 3:
        print(f"样本不足({len(rows)})")
        return 1

    x = [minutes(t) for t, _ in rows]
    # 跨天纠正:时间回退则 +1440
    for i in range(1, len(x)):
        while x[i] < x[i - 1]:
            x[i] += 1440.0
    if a.last_min > 0:
        keep = [i for i, v in enumerate(x) if v >= x[-1] - a.last_min]
        x = [x[i] for i in keep]
        rows = [rows[i] for i in keep]
    y = [v for _, v in rows]
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    den = sum((v - mx) ** 2 for v in x)
    slope = sum((v - mx) * (w - my) for v, w in zip(x, y)) / den if den else 0.0
    span_h = (x[-1] - x[0]) / 60.0
    print(f"样本 {n}, 跨度 {span_h:.2f} h: {rows[0][0]} {y[0]:.0f} -> {rows[-1][0]} {y[-1]:.0f} MiB")
    print(f"斜率 {slope:+.2f} MiB/分钟 ({slope * 60:+.0f} MiB/小时)")
    head = a.cap_mib - y[-1]
    print(f"距单卡上限({a.cap_mib:.0f} MiB) 余量 {head:.0f} MiB", end="")
    if slope > 0.05:
        eta_h = head / (slope * 60)
        print(f" ⇒ 约 {eta_h:.1f} 小时后触顶")
    else:
        print(" ⇒ 无增长趋势")
    return 0


if __name__ == "__main__":
    sys.exit(main())
