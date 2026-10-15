#!/usr/bin/env python3
"""把聚合 tok/s 拆成 (AL, period) 并对**每个分量**做配对 bootstrap。

逻辑(AL 必须计入产出,所以不归一化;但要判定它是否是被测对象的属性):
  观测到的 Δtok/s  = Δ(AL/period)
  若 ΔAL 的 95% CI **含 0**(⇒ 该特征在平均意义上不改变接受长度),
  则 Δtok/s 的期望 = Δ(1/period),即**步时收益就是吞吐收益**,原始数的散布只是抽样噪声;
  若 ΔAL 的 CI **不含 0**,则该特征真的改变了产出效率,必须按原始数定价。

用法: python agg_boot2.py [--reps N]
"""
import glob
import json
import random
import statistics
import sys

BASE = "/root/lvllm/tmp/bench_20261014"
REPS = 4000
if "--reps" in sys.argv:
    REPS = int(sys.argv[sys.argv.index("--reps") + 1])

ARMS = {"cold_pin0": ("v6_pin0", False), "cold_pin1_v1": ("v6_pin1v1", False),
        "warm_pin0": ("v6w_pin0", True), "warm_pin1_v1": ("v6w_pin1", True),
        "warm2_pin0": ("v6w2_pin0", True), "warm2_pin1_v1": ("v6w2_pin1", True)}
PAIRS = [("冷", "cold_pin0", "cold_pin1_v1"), ("暖", "warm_pin0", "warm_pin1_v1"),
         ("暖²", "warm2_pin0", "warm2_pin1_v1")]


def per_round(arm, st):
    """{文件名: {tok, win, steps}}  —— AL = tok/steps, period = win/steps"""
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
            rows.append(r)
        if not rows:
            continue
        t0 = min(r["first_wall"] for r in rows)
        t1 = max(r["last_wall"] for r in rows)
        if t1 - t0 < 0.5:
            continue
        tok = sum(r["tokens"] for r in rows)
        steps = sum(len(r["marks"]) for r in rows)
        out[f.split("/")[-1]] = {"tok": tok, "win": t1 - t0, "steps": steps}
    return out


def agg_tok(rs):
    tok = sum(x["tok"] for x in rs)
    win = sum(x["win"] for x in rs)
    return tok / win if win else 0.0


def agg_al(rs):
    tok = sum(x["tok"] for x in rs)
    st = sum(x["steps"] for x in rs)
    return tok / st if st else 0.0


def agg_per(rs):
    win = sum(x["win"] for x in rs)
    st = sum(x["steps"] for x in rs)
    return win / st * 1000.0 if st else 0.0


def ci(v):
    v = sorted(v)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


random.seed(20261014)
print("=== 配对 bootstrap(按轮次同名配对):Δtok/s 分解为 ΔAL 与 Δperiod ===")
print("| 档 | 协议 | 轮 | Δ聚合tok/s [95% CI] | ΔAL [95% CI] | Δperiod(ms/token) [95% CI] | 判读 |")
print("|---|---|---:|---|---|---|---|")
for st in ("c1", "c2", "c4"):
    for pname, a, b in PAIRS:
        ra, rb = per_round(a, st), per_round(b, st)
        keys = sorted(set(ra) & set(rb))
        if len(keys) < 4:
            continue
        f_t, f_al, f_p = [], [], []
        for _ in range(REPS):
            ks = [keys[random.randrange(len(keys))] for _ in range(len(keys))]
            sa = [ra[k] for k in ks]
            sb = [rb[k] for k in ks]
            ta, tb = agg_tok(sa), agg_tok(sb)
            if ta > 0:
                f_t.append((tb / ta - 1) * 100)
            aa, ab = agg_al(sa), agg_al(sb)
            if aa > 0:
                f_al.append((ab / aa - 1) * 100)
            pa, pb = agg_per(sa), agg_per(sb)
            if pa > 0:
                f_p.append((pb / pa - 1) * 100)
        if not (f_t and f_al and f_p):
            continue
        t0 = (agg_tok([rb[k] for k in keys]) / agg_tok([ra[k] for k in keys]) - 1) * 100
        a0 = (agg_al([rb[k] for k in keys]) / agg_al([ra[k] for k in keys]) - 1) * 100
        p0 = (agg_per([rb[k] for k in keys]) / agg_per([ra[k] for k in keys]) - 1) * 100
        tl, th = ci(f_t)
        al, ah = ci(f_al)
        pl, ph = ci(f_p)
        verdict = []
        if ah < 0 or al > 0:
            verdict.append("**AL 被改变**")
        else:
            verdict.append("AL 无系统性变化")
        if ph < 0:
            verdict.append("步时显著更短")
        elif pl > 0:
            verdict.append("步时显著更长")
        else:
            verdict.append("步时无显著差")
        print("| %s | %s | %d | %+.1f%% [%+.1f, %+.1f] | %+.1f%% [%+.1f, %+.1f] | %+.1f%% [%+.1f, %+.1f] | %s |" % (
            st, pname, len(keys), t0, tl, th, a0, al, ah, p0, pl, ph, ";".join(verdict)))

print()
print("=== 同表:各臂的 AL(tok/步)与 period(ms/token),按协议分列 ===")
print("| 档 | 臂 | AL | period(ms/token) | 隐含 tok/s |")
print("|---|---|---:|---:|---:|")
for st in ("c1", "c2", "c4"):
    for arm in ARMS:
        rr = per_round(arm, st)
        if not rr:
            continue
        v = list(rr.values())
        print("| %s | %s | %.2f | %.2f | %.1f |" % (st, arm, agg_al(v), agg_per(v), agg_tok(v)))
