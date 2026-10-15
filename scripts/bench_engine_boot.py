#!/usr/bin/env python3
"""引擎侧 `Avg generation throughput` 按 Running 分档 + bootstrap 置信区间。

数据源:vLLM 每 ~10s 的 `Engine 000:` 统计行 —— **引擎自己测的聚合产出速率**,
不经过客户端重建、AL 天然计入(它数的是真正发出的 token)。
判据:同一 Running 档下 PIN=1 与 PIN=0 的**中位数差的 95% CI 是否排除 0**。

用法: python engine_boot.py [--prefill-max 50] [--reps 4000] --pair "协议::pin0.log::pin1.log" ...
"""
import random
import re
import statistics
import sys

PREFILL_MAX = 50.0
REPS = 4000
if "--prefill-max" in sys.argv:
    PREFILL_MAX = float(sys.argv[sys.argv.index("--prefill-max") + 1])
if "--reps" in sys.argv:
    REPS = int(sys.argv[sys.argv.index("--reps") + 1])

PAT = re.compile(
    r"Engine 000: Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, Waiting: (\d+) reqs")


def buckets(path):
    b = {}
    for line in open(path, errors="replace"):
        m = PAT.search(line)
        if not m:
            continue
        p, g, run = float(m.group(1)), float(m.group(2)), int(m.group(3))
        if p > PREFILL_MAX or run == 0:
            continue
        b.setdefault(run, []).append(g)
    return b


def ci(v):
    v = sorted(v)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


random.seed(20261014)
print("=== 引擎侧聚合产出速率(Engineering `Avg generation throughput`,prompt>%.0f 的窗口已滤)===" % PREFILL_MAX)
print()
print("| 协议 | Running | n0 | PIN=0 中位 [95% CI] | n1 | PIN=1 中位 [95% CI] | Δ中位 [95% CI] | P(Δ>0) | 判读 |")
print("|---|---:|---:|---|---:|---|---|---:|---|")
for a in [x for x in sys.argv[1:] if x.startswith("--pair")]:
    pass
PAIRS = [x.split("::", 1) for x in sys.argv[1:] if "::" in x]
for lab, spec in PAIRS:
    parts = spec.split("::")
    if len(parts) != 2:
        continue
    f0, f1 = parts
    b0, b1 = buckets(f0), buckets(f1)
    for run in sorted(set(b0) & set(b1)):
        v0, v1 = b0[run], b1[run]
        if len(v0) < 5 or len(v1) < 5:
            continue
        m0, m1 = statistics.median(v0), statistics.median(v1)
        ds = []
        for _ in range(REPS):
            s0 = [v0[random.randrange(len(v0))] for _ in range(len(v0))]
            s1 = [v1[random.randrange(len(v1))] for _ in range(len(v1))]
            ds.append(statistics.median(s1) - statistics.median(s0))
        lo, hi = ci(ds)
        p = sum(1 for x in ds if x > 0) / len(ds)
        v = "**显著为正**" if lo > 0 else ("显著为负" if hi < 0 else "不显著(跨 0)")
        c0 = ci([statistics.median([v0[random.randrange(len(v0))] for _ in range(len(v0))])
                 for _ in range(1000)])
        c1 = ci([statistics.median([v1[random.randrange(len(v1))] for _ in range(len(v1))])
                 for _ in range(1000)])
        print("| %s | %d | %d | %.1f [%.1f, %.1f] | %d | %.1f [%.1f, %.1f] | %+.1f [%+.1f, %+.1f] | %.3f | %s |" % (
            lab, run, len(v0), m0, c0[0], c0[1], len(v1), m1, c1[0], c1[1], m1 - m0, lo, hi, p, v))
