#!/usr/bin/env python3
"""① 引擎每 10s 报的 AL 在一次运行**内部**的波动(⇒ 加长输出能不能摊平?)
   ② 按"文本体制"分层后的臂差(⇒ 臂差是不是体制混合造成的假象?)

用法: python al_strat.py
"""
import collections
import glob
import json
import re
import statistics
import sys

BASE = "/root/lvllm/tmp/bench_20261014"
ALPAT = re.compile(r"Mean acceptance length: ([\d.]+)")

print("=== ① 引擎侧 AL 的**运行内**分布(每 10s 一个窗口)===")
print("| 臂 | 窗口数 | AL 中位 | AL 标准差 | AL P5~P95 | 极差 |")
print("|---|---:|---:|---:|---|---:|")
for lab, f in (("暖2 PIN=0", "eng_v6w2_pin0.log"), ("暖2 PIN=1v1", "eng_v6w2_pin1.log"),
               ("暖 PIN=0", "eng_v6w_pin0.log"), ("暖 PIN=1v1", "eng_v6w_pin1.log"),
               ("冷 PIN=0", "eng_pin0.log"), ("冷 PIN=1v1", "eng_pin1v1.log")):
    p = "%s/logs/%s" % (BASE, f)
    vals = [float(m.group(1)) for m in (ALPAT.search(l) for l in open(p, errors="replace")) if m]
    if len(vals) < 5:
        continue
    q = statistics.quantiles(vals, n=20)
    print("| %s | %d | %.2f | %.2f | %.2f~%.2f | %.2f |" % (
        lab, len(vals), statistics.median(vals), statistics.pstdev(vals),
        q[0], q[18], max(vals) - min(vals)))

print()
print("=== ② 按体制分层的臂差(体制 = 该样本字符 8-gram 复制率 <=/> 0.30)===")
ARMS = [("冷 PIN=0", "v6_pin0", False), ("冷 PIN=1v1", "v6_pin1v1", False),
        ("暖 PIN=0", "v6w_pin0", True), ("暖 PIN=1v1", "v6w_pin1", True),
        ("暖2 PIN=0", "v6w2_pin0", True), ("暖2 PIN=1v1", "v6w2_pin1", True)]
DATA = collections.defaultdict(list)
for lab, d, warm in ARMS:
    for st in ("c1", "c2", "c4"):
        for f in sorted(glob.glob("%s/%s/%s_*.jsonl" % (BASE, d, st))):
            for l in open(f):
                r = json.loads(l)
                m = r.get("marks") or []
                n = r.get("tokens") or 0
                if len(m) < 5 or n < 5 or not r.get("last_wall"):
                    continue
                if not warm and r.get("cached") != 0:
                    continue
                iv = [m[i + 1] - m[i] for i in range(len(m) - 1)]
                DATA[(st, lab)].append({
                    "regime": "退化(复制循环)" if (r.get("deg") or 0) > 0.30 else "正常续写",
                    "al": n / len(m), "period": statistics.median(iv) * 1000.0,
                    "tok_s": (n - 1) / (m[-1] - m[0])})

print("| 档 | 体制 | 臂 | n | AL | period(ms) | tok/s 中位 |")
print("|---|---|---|---:|---:|---:|---:|")
for st in ("c1", "c2", "c4"):
    for reg in ("正常续写", "退化(复制循环)"):
        for lab, _, _ in ARMS:
            S = [x for x in DATA[(st, lab)] if x["regime"] == reg]
            if len(S) < 3:
                continue
            print("| %s | %s | %s | %d | %.2f | %.2f | %.1f |" % (
                st, reg, lab, len(S), statistics.median(x["al"] for x in S),
                statistics.median(x["period"] for x in S),
                statistics.median(x["tok_s"] for x in S)))
        print("| | | | | | | |")

print("=== ③ 同一体制内 PIN=1 vs PIN=0(消掉体制混合)===")
print("| 档 | 体制 | 协议 | Δperiod | Δtok/s |")
print("|---|---|---|---:|---:|")
for st in ("c1", "c2", "c4"):
    for reg in ("正常续写", "退化(复制循环)"):
        for proto, a, b in (("冷", "冷 PIN=0", "冷 PIN=1v1"), ("暖", "暖 PIN=0", "暖 PIN=1v1"),
                            ("暖²", "暖2 PIN=0", "暖2 PIN=1v1")):
            A = [x for x in DATA[(st, a)] if x["regime"] == reg]
            B = [x for x in DATA[(st, b)] if x["regime"] == reg]
            if len(A) < 3 or len(B) < 3:
                continue
            pa = statistics.median(x["period"] for x in A)
            pb = statistics.median(x["period"] for x in B)
            ta = statistics.median(x["tok_s"] for x in A)
            tb = statistics.median(x["tok_s"] for x in B)
            print("| %s | %s | %s | %+.1f%% | %+.1f%% |" % (
                st, reg, proto, (pb / pa - 1) * 100, (tb / ta - 1) * 100))
