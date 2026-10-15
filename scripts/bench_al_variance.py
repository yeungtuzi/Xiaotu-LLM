#!/usr/bin/env python3
"""AL / tok-s 的方差来自哪里?—— "输出只有 1024 token 太少" 还是 "文本体制二元"?

对每条样本算:steps(=len(marks)≈解码步数)、AL=tokens/steps、tok/s,并按
  ① 输出长度分桶   ② 退化(复制循环)标记   ③ finish 类型
看 AL 的散布由什么驱动;并给出"只用长样本 / 只用过退化门样本"时的聚合与离散度。

用法: python al_variance.py
"""
import glob
import json
import statistics
import sys

BASE = "/root/lvllm/tmp/bench_20261014"
ARMS = [("冷 PIN=0", "v6_pin0", False), ("冷 PIN=1v1", "v6_pin1v1", False),
        ("暖 PIN=0", "v6w_pin0", True), ("暖 PIN=1v1", "v6w_pin1", True),
        ("暖2 PIN=0", "v6w2_pin0", True), ("暖2 PIN=1v1", "v6w2_pin1", True)]


def rows():
    out = []
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
                    out.append({"arm": lab, "st": st, "tok": n, "steps": len(m),
                                "al": n / len(m), "tok_s": (n - 1) / (m[-1] - m[0]),
                                "deg": r.get("deg") or 0.0, "finish": r.get("finish"),
                                "itl": statistics.median(iv) * 1000.0})
    return out


R = rows()
print("总样本 = %d" % len(R))
print()
print("=== ① 每条样本有多少步?(说明 1024 token 是否偏少)===")
for st in ("c1", "c2", "c4"):
    S = [x for x in R if x["st"] == st]
    print("%s: 样本=%d, 输出 token 中位=%.0f, 步数中位=%.0f, AL 中位=%.2f" % (
        st, len(S), statistics.median(x["tok"] for x in S),
        statistics.median(x["steps"] for x in S), statistics.median(x["al"] for x in S)))

print()
print("=== ② AL 的散布:组间(不同体制)vs 组内(同长度)===")
print("| 输出长度桶 | n | AL 中位 | AL 标准差 | AL 极差 | tok/s 中位 | 退化占比 |")
print("|---|---:|---:|---:|---:|---:|---:|")
for lo, hi in ((0, 300), (300, 700), (700, 1023), (1023, 99999)):
    S = [x for x in R if lo <= x["tok"] <= hi]
    if len(S) < 3:
        continue
    al = [x["al"] for x in S]
    print("| %d~%s | %d | %.2f | %.2f | %.2f | %.1f | %.0f%% |" % (
        lo, "1024" if hi == 1023 else (str(hi) if hi < 99999 else "inf"), len(S),
        statistics.median(al), statistics.pstdev(al), max(al) - min(al),
        statistics.median(x["tok_s"] for x in S),
        100 * sum(1 for x in S if x["deg"] > 0.30) / len(S)))

print()
print("=== ③ 退化(复制循环)是否就是 AL 高的那一档?===")
for tag, sel in (("非退化 deg<=0.30", lambda x: x["deg"] <= 0.30),
                 ("退化 deg>0.30", lambda x: x["deg"] > 0.30)):
    S = [x for x in R if sel(x)]
    al = [x["al"] for x in S]
    ts = [x["tok_s"] for x in S]
    print("%-18s n=%3d  AL 中位=%.2f (标准差 %.2f)  tok/s 中位=%.1f (标准差 %.1f)  长度中位=%.0f" % (
        tag, len(S), statistics.median(al), statistics.pstdev(al),
        statistics.median(ts), statistics.pstdev(ts), statistics.median(x["tok"] for x in S)))

print()
print("=== ④ 只保留够长且非退化的样本时,聚合的离散度(按档)===")
print("| 档 | 筛选 | n | 聚合 tok/s | 样本间 tok/s 标准差 | 样本间 AL 标准差 |")
print("|---|---|---:|---:|---:|---:|")
for st in ("c1", "c2", "c4"):
    for tag, sel in (("全部", lambda x: True),
                     ("仅 tok>=900", lambda x: x["tok"] >= 900),
                     ("仅非退化", lambda x: x["deg"] <= 0.30),
                     ("tok>=900 且非退化", lambda x: x["tok"] >= 900 and x["deg"] <= 0.30)):
        S = [x for x in R if x["st"] == st and sel(x)]
        if len(S) < 3:
            print("| %s | %s | %d | — | — | — |" % (st, tag, len(S)))
            continue
        ts = [x["tok_s"] for x in S]
        al = [x["al"] for x in S]
        print("| %s | %s | %d | %.1f | %.2f | %.2f |" % (
            st, tag, len(S), statistics.median(ts),
            statistics.pstdev(ts), statistics.pstdev(al)))
