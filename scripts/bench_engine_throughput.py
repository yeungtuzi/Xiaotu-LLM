#!/usr/bin/env python3
"""直接从 vLLM 引擎日志取**引擎侧聚合输出速度**(不经过客户端重建,AL 天然计入)。

引擎每 ~10s 打一行:
  Engine 000: Avg prompt throughput: P tokens/s, Avg generation throughput: G tokens/s,
              Running: N reqs, Waiting: M reqs, GPU KV cache usage: U%
其中 **G = 该窗口内所有 running 请求的聚合产出速率**(引擎侧墙钟),
`Running: N` 直接给出该窗口的并发 ⇒ **按 N 分档就是 c1/c2/c4 的聚合速度**。

⚠️ 只有"以解码为主"的窗口才代表解码速度 ⇒ 默认过滤 `P <= PREFILL_MAX`(预填主导的窗口会被计入分母而假低)。
⚠️ 窗口跨越轮次边界时同样会稀释 ⇒ 另给"P==0 窗口"的严格子集。

用法: python engine_throughput.py <arm=logfile> ... [--prefill-max 50]
"""
import collections
import re
import statistics
import sys

PREFILL_MAX = 50.0
if "--prefill-max" in sys.argv:
    PREFILL_MAX = float(sys.argv[sys.argv.index("--prefill-max") + 1])
ARGS = [a for a in sys.argv[1:] if "::" in a]

PAT = re.compile(
    r"Engine 000: Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, Waiting: (\d+) reqs")

print("引擎侧聚合解码吞吐(按 Running 分档;已过滤 prompt>%.0f tok/s 的预填主导窗口)" % PREFILL_MAX)
print()
print("| 臂 | Running | 窗口数 | 聚合 tok/s 中位 [P25, P75] | 严格子集(P=0) 中位 [n] |")
print("|---|---:|---:|---|---|")
res = {}
for a in ARGS:
    lab, path = a.split("::", 1)
    buckets = collections.defaultdict(list)
    for line in open(path, errors="replace"):
        m = PAT.search(line)
        if not m:
            continue
        p, g, run, wait = float(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4))
        if p > PREFILL_MAX or run == 0:
            continue
        buckets[run].append((p, g))
    for run in sorted(buckets):
        vals = [g for _, g in buckets[run]]
        zeros = [g for p, g in buckets[run] if p == 0.0]
        q = statistics.quantiles(vals, n=4) if len(vals) >= 4 else [min(vals), statistics.median(vals), max(vals)]
        zs = ("%.1f [%d]" % (statistics.median(zeros), len(zeros))) if zeros else "—"
        print("| %s | %d | %d | %.1f [%.1f, %.1f] | %s |" % (
            lab, run, len(vals), statistics.median(vals), q[0], q[2], zs))
        res[(lab, run)] = (statistics.median(vals), statistics.median(zeros) if zeros else None, len(vals))

print()
print("=== 同协议 PIN=1 vs PIN=0(Running 档对齐)===")
print("| 协议 | Running | PIN=0 聚合 | PIN=1 聚合 | Δ |")
print("|---|---:|---:|---:|---:|")
for proto, a, b in [("冷", "冷 PIN=0", "冷 PIN=1 v1_700"), ("暖", "暖 PIN=0", "暖 PIN=1 v1_700"),
                    ("暖²", "暖² PIN=0", "暖² PIN=1 v1_700")]:
    live = sorted({r for (l, r) in res if l in (a, b)})
    for run in live:
        if (a, run) not in res or (b, run) not in res:
            continue
        ga, gb = res[(a, run)][0], res[(b, run)][0]
        print("| %s | %d | %.1f | %.1f | %+.1f%% |" % (proto, run, ga, gb, (gb / ga - 1) * 100))
