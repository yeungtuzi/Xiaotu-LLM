#!/usr/bin/env python3
"""从 v2 基准的 jsonl 出**最终对比表 + 配对 bootstrap 检验**。

每个格子的 jsonl 由 `scripts/bench_agent_throughput.py --jsonl` 写出,字段:
  tok / ttft / tlast / dt / chunks / deg / ptok / tag
用法:
  python bench_final_table.py --dir /tmp/v2json --arms pin0,pin1 --glm glm
"""
import argparse
import glob
import json
import os
import random
import statistics as st

random.seed(20261015)
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/tmp/v2json")
ap.add_argument("--arms", default="pin0,pin1")
ap.add_argument("--glm", default="")
ap.add_argument("--boot", type=int, default=4000)
A = ap.parse_args()


def load(p):
    out = []
    for ln in open(p, encoding="utf-8"):
        try:
            r = json.loads(ln)
        except Exception:  # noqa: BLE001
            continue
        if r.get("err") or not r.get("ttft") or not r.get("tlast"):
            continue
        if r["tlast"] <= r["ttft"] or r["tok"] < 2:
            continue
        out.append(r)
    return out


def cells(prefix):
    res = {}
    for p in glob.glob(os.path.join(A.dir, "%s_N*_C*.jsonl" % prefix)):
        base = os.path.basename(p)[len(prefix) + 1:]
        try:
            n = int(base.split("_")[0][1:])
            c = int(base.split("_")[1][1:].split(".")[0])
        except Exception:  # noqa: BLE001
            continue
        rows = load(p)
        if rows:
            res[(n, c)] = rows
    return res


dec = lambda r: (r["tok"] - 1) / (r["tlast"] - r["ttft"])            # noqa: E731 解码段速率
tpot = lambda r: (r["tlast"] - r["ttft"]) / (r["tok"] - 1) * 1000   # noqa: E731 ms/token
outr = lambda r: r["tok"] / r["dt"]                                  # noqa: E731 含预填的端到端速率


def agg_dec(rows):
    """⭐ 聚合解码吞吐 = Σ(tok−1) / Σ(解码段)—— 与客户端日志、README 口径**同一估计量**(比值),不是'速率的均值'。"""
    n = len(rows)
    dur = sum(r["tlast"] - r["ttft"] for r in rows)
    return (sum(r["tok"] for r in rows) - n) / dur if dur else float("nan")


def summary(rows, conc):
    tok = sum(r["tok"] for r in rows)
    wall = sum(r["dt"] for r in rows) / conc        # 连续灌入下的墙钟近似
    d = agg_dec(rows)
    return {"n": len(rows), "out": tok / wall, "dec": d, "decxC": d * conc,
            "tpot": st.median(tpot(r) for r in rows), "ttft": st.median(r["ttft"] * 1000 for r in rows),
            "deg": st.median(r["deg"] for r in rows),
            "deg30": sum(1 for r in rows if r["deg"] > 0.30) / len(rows)}


def boot_ratio_sums(ra, rb, B):
    """Σ(tok−1)/Σ(解码段) 的 bootstrap 比值(两臂各自格内重采样)。"""
    ds = []
    for _ in range(B):
        sa = [ra[random.randrange(len(ra))] for _ in range(len(ra))]
        sb = [rb[random.randrange(len(rb))] for _ in range(len(rb))]
        va, vb = agg_dec(sa), agg_dec(sb)
        if va and va == va:
            ds.append((vb / va - 1) * 100)
    if not ds:
        return (float("nan"), float("nan"))
    ds.sort()
    return (ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds)) - 1])


def boot_ratio(ra, rb, fn, B):
    """Δ% = mean_b/mean_a − 1 的 bootstrap CI(两臂各自格内重采样)。"""
    ds = []
    for _ in range(B):
        va = st.mean(fn(ra[random.randrange(len(ra))]) for _ in range(len(ra)))
        vb = st.mean(fn(rb[random.randrange(len(rb))]) for _ in range(len(rb)))
        if va:
            ds.append((vb / va - 1) * 100)
    if not ds:
        return (float("nan"), float("nan"))
    ds.sort()
    return (ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds)) - 1])


arms = [x for x in A.arms.split(",") if x]
data = {x: cells(x) for x in arms}
glm = cells(A.glm) if A.glm else {}

print("## 最终对比表(v2 口径:满 C 连续灌入;固定输出长度;真实 DSH 载荷)")
print()
print("| 臂 | N | C | n | 聚合输出 tok/s(含预填) | **聚合解码 tok/s(仅解码段)** | ×C 外推 | **TPOT 中位 ms** | TTFT 中位 ms | deg 中位 | deg>0.3 |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for (n, c) in sorted(set().union(*[set(d) for d in data.values()]) if data else set(), key=lambda k: (k[0], k[1])):
    for x in arms:
        if (n, c) not in data[x]:
            continue
        m = summary(data[x][(n, c)], c)
        print("| **%s** | %d | %d | %d | %.1f | **%.1f** | %.1f | **%.1f** | %.0f | %.2f | %.0f%% |" % (
            x, n, c, m["n"], m["out"], m["dec"], m["decxC"], m["tpot"], m["ttft"], m["deg"], 100 * m["deg30"]))
for (n, c) in sorted(glm, key=lambda k: (k[0], k[1])):
    m = summary(glm[(n, c)], c)
    print("| **%s** | %d | %d | %d | %.1f | **%.1f** | %.1f | **%.1f** | %.0f | %.2f | %.0f%% |" % (
        A.glm, n, c, m["n"], m["out"], m["dec"], m["decxC"], m["tpot"], m["ttft"], m["deg"], 100 * m["deg30"]))

if len(arms) >= 2:
    x0, x1 = arms[0], arms[1]
    print()
    print("## 配对检验(%s 为基线,%s 相对它;bootstrap %d 次)" % (x0, x1, A.boot))
    print()
    print("| N | C | Δ端到端 tok/s | Δ解码 tok/s | **ΔTPOT(负=更快)** | **Δ步时** | deg 基线/对照 |")
    print("|---|---:|---|---:|---|---:|---|")
    for (n, c) in sorted(set(data[x0]) & set(data[x1]), key=lambda k: (k[0], k[1])):
        ra, rb = data[x0][(n, c)], data[x1][(n, c)]
        b_out = boot_ratio(ra, rb, outr, A.boot)
        b_dec = boot_ratio(ra, rb, dec, A.boot)
        b_tpot = boot_ratio(ra, rb, tpot, A.boot)
        # 步时 = tlast-ttft(每步),与 TPOT 同向
        b_step = boot_ratio(ra, rb, lambda r: (r["tlast"] - r["ttft"]) / max(1, r["chunks"] - 1) * 1000, A.boot)
        pt = (st.mean(tpot(r) for r in rb) / st.mean(tpot(r) for r in ra) - 1) * 100
        mark = " ⭐显著更快" if b_tpot[1] < 0 else (" ⚠️显著更慢" if b_tpot[0] > 0 else "")
        print("| %d | %d | %+.1f%% | %+.1f%% [%+.1f,%+.1f] | **%+.1f%% [%+.1f,%+.1f]**%s | %+.1f%% [%+.1f,%+.1f] | %.2f / %.2f |" % (
            n, c,
            (st.mean(outr(r) for r in rb) / st.mean(outr(r) for r in ra) - 1) * 100,
            (agg_dec(rb) / agg_dec(ra) - 1) * 100, b_dec[0], b_dec[1],
            pt, b_tpot[0], b_tpot[1], mark,
            (st.mean((r["tlast"] - r["ttft"]) / max(1, r["chunks"] - 1) for r in rb) /
             st.mean((r["tlast"] - r["ttft"]) / max(1, r["chunks"] - 1) for r in ra) - 1) * 100,
            b_step[0], b_step[1],
            st.median(r["deg"] for r in ra), st.median(r["deg"] for r in rb)))
