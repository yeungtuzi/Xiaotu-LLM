#!/usr/bin/env python3
"""从客户端 jsonl **已有的数据**里把 AL 与步时拆开(不需要引擎侧探针)。

依据:投机解码下 vLLM 的 SSE 是「一步一个 delta」(该步接受的全部 token 一起发)
⇒ len(marks) ≈ 解码步数,于是
    AL_c     = tokens / len(marks)                  ← 平均接受长度(每步产出 token 数)
    period_c = (m[-1]-m[0]) / (len(marks)-1)        ← 平均步时(含被预填饿死的大间隔)
    med_itl  = median(相邻 mark 间隔)               ← **稳健步时**(剔除少数被预填占掉的步)
恒等式 tok/s = AL_c / period_c ⇒ 二者可分别比较,PIN 的判据取 period(AL 无关)。

用法: python al_period_split.py [--deg-gate]
"""
import glob
import json
import statistics
import sys

BASE = "/root/lvllm/tmp/bench_20261014"
DEG_GATE = "--deg-gate" in sys.argv
DEG = 0.30

ARMS = [
    ("冷 PIN=0", "v6_pin0"),
    ("冷 PIN=1 v0", "v6_pin1"),
    ("冷 PIN=1 v1_700", "v6_pin1v1"),
    ("暖 PIN=0", "v6w_pin0"),
    ("暖 PIN=1 v1_700", "v6w_pin1"),
    ("暖² PIN=0", "v6w2_pin0"),
    ("暖² PIN=1 v1_700", "v6w2_pin1"),
]


def collect(d, st):
    out = []
    for f in sorted(glob.glob("%s/%s/%s_*.jsonl" % (BASE, d, st))):
        for l in open(f):
            r = json.loads(l)
            m = r.get("marks") or []
            n = r.get("tokens") or 0
            if len(m) < 5 or n < 5:
                continue
            if DEG_GATE and (r.get("deg") or 0) > DEG:
                continue
            iv = [m[i + 1] - m[i] for i in range(len(m) - 1)]
            if not iv:
                continue
            out.append({
                "al": n / len(m),
                "period": (m[-1] - m[0]) / (len(m) - 1) * 1000.0,
                "med_itl": statistics.median(iv) * 1000.0,
                "tok_s": (n - 1) / (m[-1] - m[0]),
            })
    return out


print("=== AL / 步时 分解(客户端已有数据%s)===" % (" ,已过滤退化样本" if DEG_GATE else ""))
print("| 档 | 臂 | n | AL_c(tok/步) | 平均步时(ms) | **稳健步时中位(ms)** | 每请求中位 tok/s |")
print("|---|---|---:|---:|---:|---:|---:|")
res = {}
for st in ("c1", "c2", "c4"):
    for lab, d in ARMS:
        rows = collect(d, st)
        if not rows:
            continue
        al = statistics.median(r["al"] for r in rows)
        per = statistics.median(r["period"] for r in rows)
        itl = statistics.median(r["med_itl"] for r in rows)
        ts = statistics.median(r["tok_s"] for r in rows)
        res[(st, lab)] = (al, per, itl, ts, len(rows))
        print("| %s | %s | %d | %.2f | %.2f | **%.2f** | %.1f |" % (st, lab, len(rows), al, per, itl, ts))

print("\n=== PIN=1 vs PIN=0 配对(同一协议,同池)===")
print("| 档 | 协议 | Δ稳健步时 | ΔAL | Δtok/s |")
print("|---|---|---:|---:|---:|")
PAIRS = [("冷", "冷 PIN=0", "冷 PIN=1 v1_700"), ("暖", "暖 PIN=0", "暖 PIN=1 v1_700"),
         ("暖²", "暖² PIN=0", "暖² PIN=1 v1_700")]
for st in ("c1", "c2", "c4"):
    for pname, a, b in PAIRS:
        ka, kb = (st, a), (st, b)
        if ka not in res or kb not in res:
            continue
        dal = (res[kb][0] / res[ka][0] - 1) * 100
        ditl = (res[kb][2] / res[ka][2] - 1) * 100
        dts = (res[kb][3] / res[ka][3] - 1) * 100
        print("| %s | %s | **%+.1f%%** | %+.1f%% | %+.1f%% |" % (st, pname, ditl, dal, dts))
