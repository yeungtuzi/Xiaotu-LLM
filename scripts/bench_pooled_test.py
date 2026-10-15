#!/usr/bin/env python3
"""把所有臂的样本**跨协议合并**后,对每个分量做 bootstrap(利用已有全部数据,不新增测量)。

为什么可以合并:
  * `period`(每 token 时延)是**引擎属性** ⇒ 实测在三种协议下几乎相同
    (c1:PIN=0 80.4/81.1/81.2ms、PIN=1 79.3/79.3/79.6ms)⇒ 可池化;
  * `AL`(接受长度)是**文本属性** ⇒ 池化后正好看它有没有系统性偏移;
  * `tok/s = AL/period` ⇒ 两个分量分别检验后,吞吐的期望增益 = 步时增益(若 E[ΔAL]=0)。

用法: python pooled_test.py
"""
import glob
import json
import random
import statistics
import sys

BASE = "/root/lvllm/tmp/bench_20261014"
# (标签, 目录, 是否 warm)
ARM0 = [("cold", "v6_pin0", False), ("warm", "v6w_pin0", True), ("warm2", "v6w2_pin0", True)]
ARM1 = [("cold", "v6_pin1v1", False), ("warm", "v6w_pin1", True), ("warm2", "v6w2_pin1", True)]


def samples(spec, st):
    out = []
    for tag, d, warm in spec:
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
                out.append({"al": n / len(m),
                            "period": statistics.median(iv) * 1000.0,
                            "tok_s": (n - 1) / (m[-1] - m[0]),
                            "tag": tag})
    return out


def ci(v):
    v = sorted(v)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


random.seed(20261014)
print("=== 跨协议池化的分量检验(已有全部数据的合并,无新增测量)===")
print()
print("| 档 | 分量 | PIN=0 中位 [95% CI] (n) | PIN=1 中位 [95% CI] (n) | Δ中位 [95% CI] | 判读 |")
print("|---|---|---|---|---|---|")
for st in ("c1", "c2", "c4"):
    s0, s1 = samples(ARM0, st), samples(ARM1, st)
    if not s0 or not s1:
        continue
    for key, unit in (("period", "ms"), ("al", "tok/步"), ("tok_s", "tok/s")):
        v0 = [x[key] for x in s0]
        v1 = [x[key] for x in s1]
        m0, m1 = statistics.median(v0), statistics.median(v1)
        ds = []
        for _ in range(4000):
            a = [v0[random.randrange(len(v0))] for _ in range(len(v0))]
            b = [v1[random.randrange(len(v1))] for _ in range(len(v1))]
            ds.append(statistics.median(b) - statistics.median(a))
        lo, hi = ci(ds)
        pct = (m1 / m0 - 1) * 100
        lo_p, hi_p = lo / m0 * 100, hi / m0 * 100
        v = "**PIN=1 显著更优**" if hi < 0 else ("**PIN=1 显著更差**" if lo > 0 else "不显著")
        if key == "al":
            v = "**AL 显著偏移**" if (lo > 0 or hi < 0) else "AL 无系统偏移"
        if key == "tok_s":
            v = "**吞吐显著为正**" if lo > 0 else ("吞吐显著为负" if hi < 0 else "吞吐不显著")
        print("| %s | %s(%s) | %.2f [%.2f, %.2f] (%d) | %.2f [%.2f, %.2f] (%d) | %+.1f%% [%+.1f, %+.1f] | %s |" % (
            st, key, unit, m0, ci([statistics.median([v0[random.randrange(len(v0))]
                                                      for _ in range(len(v0))]) for _ in range(800)])[0],
            ci([statistics.median([v0[random.randrange(len(v0))] for _ in range(len(v0))])
                for _ in range(800)])[1], len(v0),
            m1, ci([statistics.median([v1[random.randrange(len(v1))] for _ in range(len(v1))])
                    for _ in range(800)])[0],
            ci([statistics.median([v1[random.randrange(len(v1))] for _ in range(len(v1))])
                for _ in range(800)])[1], len(v1),
            pct, lo_p, hi_p, v))
    print("| | | | | | |")
