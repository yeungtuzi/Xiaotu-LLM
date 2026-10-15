#!/usr/bin/env python3
"""保留 AL 的聚合 tok/s + 不确定度(配对 bootstrap)。

口径(用户裁定,AL 必须计入):
    聚合 = Σ(所有请求的输出 token) ÷ Σ(每轮的墙钟窗口)      ← 总产出/总时间
    窗口 = max(last_wall) - min(first_wall)  over 该轮全部请求

因为文本轨迹(⇒ AL)是**真实产出的一部分**,不做任何归一化;
改用**统计**处理它的高方差:
  * 对**轮次**(= 客户端一次调用,同一文件名在两臂都出现)做**配对 bootstrap**;
  * 报告 聚合 的 95% 置信区间 与 两臂差的 95% CI。
⇒ 判据变成"**差的 95% CI 是否排除 0**",而不是比单次数值大小。

用法: python agg_boot.py [--deg-gate] [--reps 20000]
"""
import glob
import json
import random
import statistics
import sys

BASE = "/root/lvllm/tmp/bench_20261014"
DEG_GATE = "--deg-gate" in sys.argv
REPS = 20000
if "--reps" in sys.argv:
    REPS = int(sys.argv[sys.argv.index("--reps") + 1])
DEG = 0.30

ARMS = {
    "cold_pin0": ("v6_pin0", False),
    "cold_pin1_v0": ("v6_pin1", False),
    "cold_pin1_v1": ("v6_pin1v1", False),
    "warm_pin0": ("v6w_pin0", True),
    "warm_pin1_v1": ("v6w_pin1", True),
    "warm2_pin0": ("v6w2_pin0", True),
    "warm2_pin1_v1": ("v6w2_pin1", True),
}
PAIRS = [("冷", "cold_pin0", "cold_pin1_v1"), ("暖", "warm_pin0", "warm_pin1_v1"),
         ("暖²", "warm2_pin0", "warm2_pin1_v1")]
README = {"c1": 48.7, "c2": 70.4, "c4": 95.2}


def rounds(arm, st):
    """{文件名: (Σtok, 窗口s)}"""
    d, warm = ARMS[arm]
    out = {}
    for f in sorted(glob.glob("%s/%s/%s_*.jsonl" % (BASE, d, st))):
        rows = []
        for l in open(f):
            r = json.loads(l)
            if not (r.get("marks") and r.get("last_wall")):
                continue
            if not warm and r.get("cached") != 0:
                continue
            if DEG_GATE and (r.get("deg") or 0) > DEG:
                continue
            rows.append(r)
        if not rows:
            continue
        t0 = min(r["first_wall"] for r in rows)
        t1 = max(r["last_wall"] for r in rows)
        if t1 - t0 < 0.5:
            continue
        out[f.split("/")[-1]] = (sum(r["tokens"] for r in rows), t1 - t0)
    return out


def agg(items):
    tok = sum(t for t, _ in items)
    win = sum(w for _, w in items)
    return tok / win if win else 0.0


def ci(vals):
    vals = sorted(vals)
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals)) - 1]
    return lo, hi


random.seed(20261014)
print("=== 保留 AL 的聚合 tok/s(口径:Σ输出token ÷ Σ墙钟;%s)===" % (
    "已过退化门" if DEG_GATE else "含全部样本"))
print()
print("| 档 | 臂 | 轮 | **聚合 tok/s [95% CI]** | vs README |")
print("|---|---|---:|---|---:|")
res = {}
for st in ("c1", "c2", "c4"):
    for arm in ARMS:
        rr = rounds(arm, st)
        if not rr:
            continue
        base = agg(list(rr.values()))
        vals = [agg([list(rr.values())[i] for i in
                     (random.randrange(len(rr)) for _ in range(len(rr)))])
                for _ in range(2000)]
        lo, hi = ci(vals)
        res[(st, arm)] = (base, rr, lo, hi)
        print("| %s | %s | %d | **%.1f** [%.1f, %.1f] | %+.1f%% |" % (
            st, arm, len(rr), base, lo, hi, (base / README[st] - 1) * 100))

print()
print("=== PIN=1(v1_700) − PIN=0:**配对 bootstrap**(按轮次同名配对)===")
print("| 档 | 协议 | Δ聚合 | 95% CI | P(Δ>0) | 结论 |")
print("|---|---|---:|---|---:|---|")
for st in ("c1", "c2", "c4"):
    for pname, a, b in PAIRS:
        if (st, a) not in res or (st, b) not in res:
            continue
        ra, rb = res[(st, a)][1], res[(st, b)][1]
        keys = sorted(set(ra) & set(rb))
        if len(keys) < 4:
            print("| %s | %s | — | 配对轮数不足(%d) | | |" % (st, pname, len(keys)))
            continue
        d0 = (agg([rb[k] for k in keys]) / agg([ra[k] for k in keys]) - 1) * 100
        ds = []
        for _ in range(REPS):
            ks = [keys[random.randrange(len(keys))] for _ in range(len(keys))]
            da = agg([ra[k] for k in ks])
            db = agg([rb[k] for k in ks])
            if da > 0:
                ds.append((db / da - 1) * 100)
        lo, hi = ci(ds)
        p = sum(1 for x in ds if x > 0) / len(ds)
        verdict = "**显著为正**" if lo > 0 else ("显著为负" if hi < 0 else "不显著(跨 0)")
        print("| %s | %s | %+.1f%% | [%+.1f%%, %+.1f%%] | %.3f | %s |" % (
            st, pname, d0, lo, hi, p, verdict))
