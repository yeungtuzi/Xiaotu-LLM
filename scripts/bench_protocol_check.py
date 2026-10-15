#!/usr/bin/env python3
"""口径**校验脚本**:检查一次 v2 基准的 jsonl 是否满足 `docs/BENCH_PROTOCOL.md` §2 的硬要求。

为什么需要它:本仓的失败模式不是"不知道口径",而是"**跑完才发现口径没满足**"。
⇒ 把 §2 的每条要求落成**可机械判定**的规则,任何要写进 README/发行说明的数字,先过这里。

用法:
  python scripts/bench_protocol_check.py --jsonl /tmp/v2json/pin0_N1024_C4.jsonl --outlen 1024 --conc 4
  python scripts/bench_protocol_check.py --dir /tmp/v2json --glob 'pin0_N1024_C4*'
退出码:0 = 全过;1 = 有 FAIL(不得入表)。
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--jsonl", default="")
ap.add_argument("--dir", default="")
ap.add_argument("--glob", default="*.jsonl")
ap.add_argument("--outlen", type=int, default=0, help="本次请求的固定输出长度 N(用于校验 §2.2)")
ap.add_argument("--conc", type=int, default=0, help="本次并发 C(用于校验 §2.4 的 n 下限)")
ap.add_argument("--wall", type=float, default=0.0,
                help="客户端打印的**真实墙钟**(秒)。给定它才能校验 §2.4 的'满 C 连续灌入':"
                     "槽位利用率 = Σ端到端耗时 / (C×墙钟),连续灌入应≈1,波次同步会明显<1")
A = ap.parse_args()

paths = [A.jsonl] if A.jsonl else sorted(glob.glob("%s/%s" % (A.dir, A.glob)))
if not paths or not paths[0]:
    print("未指定 jsonl"); sys.exit(2)

fails, warns = [], []


def chk(cond, cid, desc, detail=""):
    print("%s [%s] %s%s" % ("✅" if cond else "❌", cid, desc, ("  ← " + detail) if detail else ""))
    if not cond:
        fails.append(cid)
    return cond


def warn(cond, cid, desc, detail=""):
    if not cond:
        print("⚠️ [%s] %s%s" % (cid, desc, ("  ← " + detail) if detail else ""))
        warns.append(cid)
    else:
        print("✅ [%s] %s" % (cid, desc))


for p in paths:
    rows = []
    for ln in open(p, encoding="utf-8"):
        try:
            rows.append(json.loads(ln))
        except Exception:  # noqa: BLE001
            pass
    good = [r for r in rows if not r.get("err")]
    print("\n=== %s : 记录 %d 条(失败 %d)===" % (p, len(rows), len(rows) - len(good)))
    chk(bool(good), "S0", "至少有一条有效样本")
    if not good:
        continue
    # ---- §2.2 固定长度 ⇒ 服务端 token 数应恒等于 N(允许 ±1 的 usage 计数边界)----
    if A.outlen:
        toks = [r["tok"] for r in good]
        bad = [t for t in toks if abs(t - A.outlen) > 1]
        chk(not bad, "S1", "§2.2 固定长度:全部 tok == N=%d" % A.outlen,
            "偏离的记录 %d 条,样例 %s" % (len(bad), bad[:5]) if bad else "")
    # ---- §2.3 计数须来自服务端 usage ----
    u = sum(1 for r in good if r.get("n_src") == "usage")
    chk(u == len(good), "S2", "§2.3 token 计数全部来自服务端 usage(非 SSE chunk 数)",
        "usage=%d / 总 %d" % (u, len(good)))
    # ---- §2.1 必须流式(要 ttft/tlast 才能算 TPOT)----
    s = sum(1 for r in good if r.get("ttft") and r.get("tlast") and r["tlast"] > r["ttft"])
    chk(s == len(good), "S3", "§2.1 全部有 ttft/tlast(流式 ⇒ 可算 TPOT)",
        "有效 %d / 总 %d" % (s, len(good)))
    # ---- §2.4 请求数下限 ----
    n = len(good)
    if A.conc:
        need = 16
        chk(n >= need, "S4", "§2.4 n ≥ %d(并发 C=%d)" % (need, A.conc), "实际 n=%d" % n)
    # ---- §2.7 健康列必须可得 ----
    chk(all("deg" in r for r in good), "S5", "§2.7 每条都有 deg(健康列可算)")
    deg = st.median(r["deg"] for r in good)
    hi = sum(1 for r in good if r["deg"] > 0.30) / n
    warn(hi < 0.5, "W1", "deg>0.30 占比 %.0f%%(<50%% 时结论更可信;偏高 ⇒ 绝对 tok/s 偏乐观)" % (100 * hi))
    # ---- §2.4 ⭐ 满 C 连续灌入的**定量判据**:槽位利用率 ----
    #   ⚠️ 早先版本用"完成时刻是否聚簇"判,但 jsonl 里的 ttft/tlast 都是**相对各自提交时刻**的,
    #      互不可比 ⇒ 该判据无效(已删)。正确做法:拿客户端打印的**真实墙钟**比:
    #      利用率 = Σ(dt) / (C × 墙钟);连续灌入 ≈ 1(每槽几乎不满即补),波次同步会明显 < 1(槽位空转)。
    if A.wall and A.conc:
        util = sum(r["dt"] for r in good) / (A.conc * A.wall)
        warn(util >= 0.70, "W2",
             "§2.4 槽位利用率 = Σ端到端/(C×墙钟) = %.2f(≥0.70 才算'满 C 连续灌入';偏低 ⇒ 波次同步/槽位空转)" % util)
    else:
        print("⚠️ [W2] 未给 --wall ⇒ **无法校验** §2.4 的'满 C 连续灌入'(jsonl 里无绝对时刻)")
    print("   ℹ️ n=%d, tok 中位=%d, TPOT 中位=%.1f ms, deg 中位=%.2f, deg>0.3=%.0f%%" % (
        n, st.median(r["tok"] for r in good),
        st.median((r["tlast"] - r["ttft"]) / max(1, r["tok"] - 1) * 1000 for r in good),
        deg, 100 * hi))

print()
if fails:
    print("⛔ 口径校验**未通过**:失败项 %s ⇒ **该数字不得写进 README/发行说明**" % fails)
    sys.exit(1)
print("✅ 口径校验通过(警告 %d 项:%s —— 警告不阻塞,但须随数字一起写进健康列)" % (len(warns), warns or "无"))
