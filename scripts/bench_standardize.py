#!/usr/bin/env python3
"""AL / 体制的**直接标准化(direct standardization)** —— 正确的"对齐 AL"。

思路(实验设计标准手段,不是 B411 那种除法):
  ① 把每条样本按 (体制 = deg 档, 输出长度桶) 分层;
  ② 算每臂在**各层**的 tok/s 均值;
  ③ 把两臂**重新加权到同一分层分布**(用两臂合并后的层占比)⇒
     standardized = Σ_strata w_s × rate_arm(s)
  ⇒ 这就等于"两臂的体制/长度构成相同"时的 tok/s,即用户要的"计入 AL 但对齐了构成"的数。
  ④ 对层内样本 bootstrap ⇒ CI。

用法: python standardize.py [--level c1,c2,c4]
"""
import glob
import json
import random
import statistics
import sys

BASE = "/root/lvllm/tmp/bench_20261014"
ARMS = [("冷 PIN=0", "v6_pin0", False), ("冷 PIN=1v1", "v6_pin1v1", False),
        ("暖 PIN=0", "v6w_pin0", True), ("暖 PIN=1v1", "v6w_pin1", True),
        ("暖2 PIN=0", "v6w2_pin0", True), ("暖2 PIN=1v1", "v6w2_pin1", True)]
LEVELS = (sys.argv[sys.argv.index("--level") + 1].split(",")
          if "--level" in sys.argv else ["c1", "c2", "c4"])
README = {"c1": 48.7, "c2": 70.4, "c4": 95.2}
C = {"c1": 1, "c2": 2, "c4": 4}
random.seed(20261015)


def load():
    out = {}
    for lab, d, warm in ARMS:
        for st in LEVELS:
            for f in sorted(glob.glob("%s/%s/%s_*.jsonl" % (BASE, d, st))):
                for l in open(f):
                    r = json.loads(l)
                    m = r.get("marks") or []
                    n = r.get("tokens") or 0
                    if len(m) < 5 or n < 5 or not r.get("last_wall"):
                        continue
                    if not warm and r.get("cached") != 0:
                        continue
                    strata = ("退化" if (r.get("deg") or 0) > 0.30 else "正常",
                              "≤300" if n <= 300 else ("≤700" if n <= 700 else ">700"))
                    out.setdefault((st, lab), []).append(
                        ((n - 1) / (m[-1] - m[0]), strata))
    return out


D = load()
print("=== 直接标准化后的 tok/s(两臂重加权到同一体制/长度构成)===")
print("| 档 | 协议 | 臂 | 原始中位 tok/s | **标准化 tok/s [95% CI]** | n | 分层构成(正常≤300/正常≤700/正常>700/退化) |")
print("|---|---|---|---:|---|---:|---|")
res = {}
for st in LEVELS:
    for proto, a, b in (("冷", "冷 PIN=0", "冷 PIN=1v1"), ("暖", "暖 PIN=0", "暖 PIN=1v1"),
                        ("暖²", "暖2 PIN=0", "暖2 PIN=1v1")):
        ka, kb = (st, a), (st, b)
        if ka not in D or kb not in D:
            continue
        # 共同分层权重 = 两臂合并后的层占比
        allrows = D[ka] + D[kb]
        w = {}
        for _, s in allrows:
            w[s] = w.get(s, 0) + 1
        tot = sum(w.values())
        w = {k: v / tot for k, v in w.items()}

        def std(rows):
            by = {}
            for r, s in rows:
                by.setdefault(s, []).append(r)
            return sum(w[s] * statistics.mean(v) for s, v in by.items() if s in w), by

        sa, _ = std(D[ka])
        sb, _ = std(D[kb])
        # bootstrap(层内重采样)
        ds, sas, sbs = [], [], []
        for _ in range(3000):
            ra, rb = {}, {}
            for r, s in D[ka]:
                ra.setdefault(s, []).append(r)
            for r, s in D[kb]:
                rb.setdefault(s, []).append(r)
            va = sum(w[s] * statistics.mean([v[random.randrange(len(v))] for v in [ra[s]]] * 1)
                     for s in ra if s in w)
            # 逐层 bootstrap:层内重采样取均值
            va = sum(w[s] * statistics.mean([ra[s][random.randrange(len(ra[s]))]
                                             for _ in range(len(ra[s]))]) for s in ra if s in w)
            vb = sum(w[s] * statistics.mean([rb[s][random.randrange(len(rb[s]))]
                                             for _ in range(len(rb[s]))]) for s in rb if s in w)
            sas.append(va)
            sbs.append(vb)
            ds.append(vb - va)
        q = lambda v: (sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v)) - 1])  # noqa: E731
        la, ha = q(sas)
        lb, hb = q(sbs)
        ld, hd = q(ds)
        res[(st, proto)] = (sa, sb)
        comp = {}
        for _, s in D[ka]:
            comp[s] = comp.get(s, 0) + 1
        n0 = len(D[ka])
        print("| %s | %s | **PIN=0** | %.1f | **%.1f [%.1f, %.1f]** | %d | %s |" % (
            st, proto, statistics.median(r for r, _ in D[ka]), sa, la, ha, n0,
            "/".join(str(comp.get(k, 0)) for k in
                     [("正常", "≤300"), ("正常", "≤700"), ("正常", ">700"), ("退化", "≤300"),
                      ("退化", "≤700"), ("退化", ">700")] if comp.get(k, 0))))
        print("| %s | %s | **PIN=1v1** | %.1f | **%.1f [%.1f, %.1f]** | %d | Δ=%+.1f%% [%+.1f, %+.1f] |" % (
            st, proto, statistics.median(r for r, _ in D[kb]), sb, lb, hb, len(D[kb]),
            (sb / sa - 1) * 100, ld / sa * 100, hd / sa * 100))

print()
print("=== 标准化 tok/s 与 README(×C 口径)对照 ===")
print("| 档 | README | 冷 PIN=0 | 冷 PIN=1 | 暖 PIN=0 | 暖 PIN=1 | 暖² PIN=0 | 暖² PIN=1 |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for st in LEVELS:
    row = [C[st] * res.get((st, p), (0, 0))[i] for p in ("冷", "暖", "暖²") for i in (0, 1)]
    print("| %s | %.1f | %.1f | **%.1f** | %.1f | **%.1f** | %.1f | **%.1f** |" % tuple(
        [st, README[st]] + row))
