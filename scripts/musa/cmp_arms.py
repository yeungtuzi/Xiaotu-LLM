#!/usr/bin/env python3
"""对比两个基准臂日志的四项指标（prefill / TTFT / decode / 步长）。

用法：
    python3 cmp_arms.py <基线.log> <对比.log> [--label-a 基线] [--label-b 对比]

只读日志，不碰 GPU。缺失的档位打印为 `—` 而不是报错（便于实验中途查看）。
"""
from __future__ import annotations

import argparse
import re
import sys

ROW = re.compile(
    r"^\[e2e\]\s*C=(\d+)\s+TTFT=\s*(\d+)\s*ms\s+prefill=\s*([\d.]+)\s*tok/s\s+"
    r"decode=\s*([\d.]+)\s*tok/s\s+步长=\s*([\d.]+)\s*ms"
)


def parse(path: str) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    try:
        fh = open(path, errors="ignore")
    except OSError as exc:
        print(f"⚠️ 读不到 {path}: {exc}", file=sys.stderr)
        return out
    with fh:
        for ln in fh:
            m = ROW.match(ln.strip())
            if not m:
                continue
            c = int(m.group(1))
            out[c] = {
                "ttft": float(m.group(2)),
                "prefill": float(m.group(3)),
                "decode": float(m.group(4)),
                "step": float(m.group(5)),
            }
    return out


def fmt(v: float | None, spec: str) -> str:
    return "—" if v is None else format(v, spec)


def ratio(b: float | None, a: float | None) -> str:
    if not a or b is None or a is None or a == 0:
        return "—"
    return f"{b / a:.2f}×"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("other")
    ap.add_argument("--label-a", default="基线")
    ap.add_argument("--label-b", default="对比")
    args = ap.parse_args()

    a = parse(args.base)
    b = parse(args.other)
    if not a:
        print(f"⛔ {args.base} 里没有 [e2e] 汇总行", file=sys.stderr)
        return 2

    print(f"{args.label_a}: {args.base}")
    print(f"{args.label_b}: {args.other}\n")

    hdr = (
        f"{'C':>2} | {'TTFT ms':>9} {'TTFT ms':>9} {'':>6} | "
        f"{'prefill':>8} {'prefill':>8} {'':>6} | "
        f"{'decode':>7} {'decode':>7} {'':>6} | "
        f"{'step ms':>8} {'step ms':>8} {'':>6}"
    )
    print(hdr)
    print(f"{'':>2} | {args.label_a[:9]:>9} {args.label_b[:9]:>9} {'B/A':>6} | "
          f"{args.label_a[:8]:>8} {args.label_b[:8]:>8} {'B/A':>6} | "
          f"{args.label_a[:7]:>7} {args.label_b[:7]:>7} {'B/A':>6} | "
          f"{args.label_a[:8]:>8} {args.label_b[:8]:>8} {'B/A':>6}")
    print("-" * len(hdr))

    for c in sorted(set(a) | set(b)):
        ra, rb = a.get(c, {}), b.get(c, {})
        print(
            f"{c:>2} | "
            f"{fmt(ra.get('ttft'), '.0f'):>9} {fmt(rb.get('ttft'), '.0f'):>9} "
            f"{ratio(rb.get('ttft'), ra.get('ttft')):>6} | "
            f"{fmt(ra.get('prefill'), '.1f'):>8} {fmt(rb.get('prefill'), '.1f'):>8} "
            f"{ratio(rb.get('prefill'), ra.get('prefill')):>6} | "
            f"{fmt(ra.get('decode'), '.1f'):>7} {fmt(rb.get('decode'), '.1f'):>7} "
            f"{ratio(rb.get('decode'), ra.get('decode')):>6} | "
            f"{fmt(ra.get('step'), '.2f'):>8} {fmt(rb.get('step'), '.2f'):>8} "
            f"{ratio(rb.get('step'), ra.get('step')):>6}"
        )
    print("\n提示：TTFT 越小越好；prefill/decode tok/s 越大越好；step 越小越好。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
