#!/usr/bin/env python3
"""⭐ agent 编程场景的标准吞吐基准(行业口径:固定输出长度 + 大量请求 + Σtoken÷墙钟)。

为什么这样设计(对齐 vLLM/SGLang/TRT-LLM 的 bench 口径,也是"AL 按构造对齐"的正解):
  * `max_tokens = min_tokens = N` + `ignore_eos=True` ⇒ **每个请求恰好吐 N 个 token**
    ⇒ **分子按构造固定** ⇒ 两臂的 tok/s 差异只反映"产出 N 个 token 要多久",不受 AL 抽样影响;
  * 聚合 = **Σ输出token ÷ 墙钟**(并发下所有请求一起算)⇒ 与 `vllm bench serve` 的
    `Output token throughput` 同口径;
  * 请求数足够多(每个并发档 ≥ 数十~数百)⇒ 把 AL/体制的残余波动平均掉;
  * 副作用要一起报:强制长度会落进复制循环(AL 虚高)⇒ **同时报 AL 与 deg**,说明数是在哪个体制下拿到的。

用法(远端 gpu):
  PY=/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python
  $PY scripts/bench_agent_throughput.py --conc 1 --rounds 3 --outlen 1024
  $PY scripts/bench_agent_throughput.py --conc 4 --rounds 2 --outlen 1024
数据集:process_data/data/dsh_assembled_bench.jsonl(真实 DSH 载荷,预装配 <|im_start|> 模板)
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import threading
import time
import urllib.request

DS = "/home/user/lvllm/process_data/data/dsh_assembled_bench.jsonl"
ALPAT = re.compile(r"Mean acceptance length: ([\d.]+)")


def deg(t, n=8):
    if len(t) < n * 2:
        return 0.0
    seen, dup, tot = set(), 0, 0
    for i in range(len(t) - n + 1):
        g = t[i:i + n]
        tot += 1
        if g in seen:
            dup += 1
        else:
            seen.add(g)
    return dup / tot


def one(prompt, outlen, sink, tag):
    body = {"model": "DeepSeek-V4.1-Flash", "prompt": prompt,
            "max_tokens": outlen, "min_tokens": outlen, "ignore_eos": True,
            "temperature": 0.0, "stream": False}
    req = urllib.request.Request("http://127.0.0.1:8070/v1/completions",
                                data=json.dumps(body).encode(),
                                headers={"content-type": "application/json"})
    t0 = time.time()
    try:
        j = json.load(urllib.request.urlopen(req, timeout=3600))
    except Exception as e:  # noqa: BLE001
        sink[tag] = {"err": repr(e)}
        return
    ch = (j["choices"] or [{}])[0]
    t = ch.get("text") or ""
    sink[tag] = {"tok": j["usage"]["completion_tokens"], "ptok": j["usage"]["prompt_tokens"],
                 "dt": time.time() - t0, "deg": deg(t),
                 "chunks": None, "text": t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--outlen", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="每轮取多少条载荷(0=全部 50)")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(DS, encoding="utf-8") if l.strip()]
    if a.limit:
        recs = recs[:a.limit]

    allr, wall_tot, tok_tot, ptok_tot, degs = [], 0.0, 0, 0, []
    for rd in range(a.rounds):
        wave = recs[rd::a.rounds]           # 轮换,别总用同几条
        sink = {}
        T0 = time.time()
        for i in range(0, len(wave), a.conc):
            grp = wave[i:i + a.conc]
            ths = [threading.Thread(target=one, args=(r["prompt"], a.outlen, sink,
                                                      "r%d_%d" % (rd, i + k)))
                   for k, r in enumerate(grp)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
        T1 = time.time()
        good = [v for v in sink.values() if not v.get("err")]
        if not good:
            continue
        wall_tot += T1 - T0
        tok_tot += sum(v["tok"] for v in good)
        ptok_tot += sum(v["ptok"] for v in good)
        degs += [v["deg"] for v in good]
        allr += good
        for v in good:
            v["al"] = v["tok"] / v["dt"]
        print("  round%d: n=%d 墙钟=%.1fs 该轮聚合=%.1f tok/s" % (
            rd, len(good), T1 - T0, sum(v["tok"] for v in good) / (T1 - T0)), flush=True)

    if not allr:
        print("无有效样本"); return
    agg = tok_tot / wall_tot
    print()
    print("===== agent 编程场景(真实 DSH 载荷)标准吞吐 =====")
    print("并发=%d 载荷=%d 条 × %d 轮  输出长度=固定 %d token(ignore_eos)" % (
        a.conc, len(recs), a.rounds, a.outlen))
    print("⭐ **聚合输出吞吐 = Σ输出token ÷ Σ墙钟 = %d / %.1fs = %.1f tok/s**" % (
        tok_tot, wall_tot, agg))
    print("   输入 token 合计=%d ⇒ 总吞吐(含输入)= %.1f tok/s" % (
        ptok_tot, (ptok_tot + tok_tot) / wall_tot))
    print("   每请求中位耗时=%.1fs  中位吞吐(单流)=%.1f tok/s" % (
        statistics.median(v["dt"] for v in allr),
        statistics.median(v["tok"] / v["dt"] for v in allr)))
    print("   健康列:deg 中位=%.2f ; deg>0.30 占比=%.0f%% ; 失败=%d" % (
        statistics.median(degs), 100 * sum(1 for d in degs if d > 0.30) / len(degs),
        len([v for v in allr if v.get("err")])))


if __name__ == "__main__":
    main()
