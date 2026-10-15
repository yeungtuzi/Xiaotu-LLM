#!/usr/bin/env python3
"""轨迹归一化聚合(科学口径):把 AL 与 period 分开后,用共同参考 AL* 重算聚合 tok/s。

为什么:tok/s = AL ÷ period。**period 是引擎速度**,**AL 是输出文本的属性**
(draft/target 一致度,会被 ULP 级差异翻转,QA #183)⇒ 直接比 tok/s 等于把两种东西混在一起。
本脚本保留各臂**实测步时**,把每臂的 AL 归到同一参考值 AL*,得到
    agg_norm = C × AL* ÷ period_arm
即"**同一段输出文本下各臂的吞吐**"。同时给出原始实测聚合与 AL/步时分解便于归因。

用法: python agg_corrected.py [--deg-gate]
"""
import glob
import json
import statistics
import sys

BASE = "/root/lvllm/tmp/bench_20261014"
DEG_GATE = "--deg-gate" in sys.argv
DEG = 0.30
C = {"c1": 1, "c2": 2, "c4": 4}
README = {"c1": 48.7, "c2": 70.4, "c4": 95.2}

ARMS = [
    ("冷 PIN=0", "v6_pin0", False),
    ("冷 PIN=1 v0", "v6_pin1", False),
    ("冷 PIN=1 v1_700", "v6_pin1v1", False),
    ("暖 PIN=0", "v6w_pin0", True),
    ("暖 PIN=1 v1_700", "v6w_pin1", True),
    ("暖² PIN=0", "v6w2_pin0", True),
    ("暖² PIN=1 v1_700", "v6w2_pin1", True),
]


def samples(d, st, warm):
    """每条样本:(AL, 稳健步时ms, 原始tok/s, out_tokens, first_wall, last_wall)"""
    out = []
    for f in sorted(glob.glob("%s/%s/%s_*.jsonl" % (BASE, d, st))):
        for l in open(f):
            r = json.loads(l)
            m = r.get("marks") or []
            n = r.get("tokens") or 0
            if len(m) < 5 or n < 5 or not r.get("last_wall"):
                continue
            if not warm and r.get("cached") != 0:
                continue
            if DEG_GATE and (r.get("deg") or 0) > DEG:
                continue
            iv = [m[i + 1] - m[i] for i in range(len(m) - 1)]
            out.append({"al": n / len(m),
                        "itl_ms": statistics.median(iv) * 1000.0,
                        "raw": (n - 1) / (m[-1] - m[0]),
                        "n": n, "steps": len(m),
                        "first": r["first_wall"], "last": r["last_wall"]})
    return out


def agg_raw(d, st, warm):
    tok = win = 0.0
    byround = {}
    for f in sorted(glob.glob("%s/%s/%s_*.jsonl" % (BASE, d, st))):
        rows = []
        for l in open(f):
            r = json.loads(l)
            m = r.get("marks") or []
            if len(m) < 5 or not r.get("last_wall"):
                continue
            if not warm and r.get("cached") != 0:
                continue
            if DEG_GATE and (r.get("deg") or 0) > DEG:
                continue
            rows.append(r)
        if not rows:
            continue
        byround[f] = rows
        tok += sum(r["tokens"] for r in rows)
        t0 = min(r["first_wall"] for r in rows)
        t1 = max(r["last_wall"] for r in rows)
        if t1 > t0 + 0.5:
            win += t1 - t0
    return (tok / win if win else 0.0), byround


print("=== 轨迹归一化聚合 tok/s(科学口径%s)===" % (",已过退化门" if DEG_GATE else ""))
print()
print("| 档 | 臂 | n | AL(tok/步) | 稳健步时(ms) | 原始实测聚合 | **归一化聚合(AL*)**: | 相对 PIN=0 |")
print("|---|---|---:|---:|---:|---:|---:|---:|")
store = {}
for st in ("c1", "c2", "c4"):
    arms = [(lab, d, w) for lab, d, w in ARMS if glob.glob("%s/%s/%s_*.jsonl" % (BASE, d, st))]
    data = {lab: samples(d, st, w) for lab, d, w in arms}
    # 参考 AL* = 两臂合并后的中位数(不偏向任一方)
    allal = [s["al"] for v in data.values() for s in v]
    alref = statistics.median(allal) if allal else 0.0
    print("**%s:共同参考 AL* = %.2f tok/步**(两臂合并中位)\n" % (st, alref))
    for lab, d, w in arms:
        v = data[lab]
        if not v:
            continue
        al = statistics.median(s["al"] for s in v)
        itl = statistics.median(s["itl_ms"] for s in v)
        raw = statistics.median(s["raw"] for s in v)
        wall, _ = agg_raw(d, st, w)
        norm = C[st] * alref / (itl / 1000.0)
        store[(st, lab)] = (al, itl, raw, wall, norm, len(v))
        print("| %s | %s | %d | %.2f | %.2f | %.1f | **%.1f** | |" % (
            st, lab, len(v), al, itl, wall, norm))

print()
print("=== PIN=1(v1_700) 相对 PIN=0:{三个口径的对照} ===")
print("| 档 | 协议 | 原始实测聚合 | 原始每请求中位 | **归一化聚合** | ΔAL | Δ步时 |")
print("|---|---|---:|---:|---:|---:|---:|")
PAIRS = [("冷", "冷 PIN=0", "冷 PIN=1 v1_700", False),
         ("暖", "暖 PIN=0", "暖 PIN=1 v1_700", True),
         ("暖²", "暖² PIN=0", "暖² PIN=1 v1_700", True)]
for st in ("c1", "c2", "c4"):
    for pname, a, b, _ in PAIRS:
        ka, kb = (st, a), (st, b)
        if ka not in store or kb not in store:
            continue
        pc = lambda x: (x - 1) * 100  # noqa: E731
        da = pc(store[kb][0] / store[ka][0])
        di = pc(store[kb][1] / store[ka][1])
        dn = pc(store[kb][4] / store[ka][4])
        dw = pc(store[kb][3] / store[ka][3])
        dr = pc(store[kb][2] / store[ka][2])
        print("| %s | %s | %+.1f%% | %+.1f%% | **%+.1f%%** | %+.1f%% | %+.1f%% |" % (
            st, pname, dw, dr, dn, da, di))
