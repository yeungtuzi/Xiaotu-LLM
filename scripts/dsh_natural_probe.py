#!/usr/bin/env python3
"""真实 DSH 载荷(预装配 <|im_start|> 模板)在 **自然结束** 下的行为:
会不会产生复制循环?与"基准转录续写"对比。

要点:⛔ 不设 min_tokens / ignore_eos ⇒ 让模型自己停。
用法: python dsh_natural_probe.py <n> [--conc N]
"""
import json
import statistics
import sys
import threading
import time
import urllib.request

DS = "/home/user/lvllm/process_data/data/dsh_assembled_bench.jsonl"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 1
recs = [json.loads(l) for l in open(DS, encoding="utf-8") if l.strip()][:N]


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


sink = {}


def one(i):
    r = recs[i]
    body = {"model": "DeepSeek-V4.1-Flash", "prompt": r["prompt"], "max_tokens": 1024,
            "temperature": 0, "stream": False}
    req = urllib.request.Request("http://127.0.0.1:8070/v1/completions",
                                data=json.dumps(body).encode(),
                                headers={"content-type": "application/json"})
    t0 = time.time()
    try:
        j = json.load(urllib.request.urlopen(req, timeout=1800))
    except Exception as e:  # noqa: BLE001
        sink[i] = {"err": repr(e)}
        return
    ch = (j["choices"] or [{}])[0]
    t = ch.get("text") or ""
    sink[i] = {"tok": j["usage"]["completion_tokens"], "finish": ch.get("finish_reason"),
               "ptok": j["usage"]["prompt_tokens"], "deg": deg(t), "dt": time.time() - t0,
               "text": t}


ths = []
for i in range(len(recs)):
    t = threading.Thread(target=one, args=(i,))
    t.start()
    ths.append(t)
    if len(ths) % CONC == 0:
        for x in ths:
            x.join()
        ths = []
for x in ths:
    x.join()

print("| # | prompt tok | 输出 tok | finish | deg(8-gram) | 耗时(s) |")
print("|---:|---:|---:|---|---:|---:|")
ok = []
for i in range(len(recs)):
    v = sink.get(i, {})
    if v.get("err"):
        print("| %d | — | — | ERROR %s | | |" % (i, v["err"][:40]))
        continue
    ok.append(v)
    print("| %d | %d | %d | %s | **%.2f** | %.1f |" % (
        i, v["ptok"], v["tok"], v["finish"], v["deg"], v["dt"]))
if ok:
    print()
    print("真实 DSH 载荷(自然结束,n=%d,conc=%d):" % (len(ok), CONC))
    print("  输出 token 中位=%.0f (min %d / max %d)" % (
        statistics.median(v["tok"] for v in ok), min(v["tok"] for v in ok),
        max(v["tok"] for v in ok)))
    print("  finish 分布=%s" % dict((k, sum(1 for v in ok if v["finish"] == k))
                                 for k in {v["finish"] for v in ok}))
    print("  deg 中位=%.2f ; deg>0.30 的比例=%d/%d" % (
        statistics.median(v["deg"] for v in ok),
        sum(1 for v in ok if v["deg"] > 0.30), len(ok)))
    longtext = max(ok, key=lambda v: v["tok"])
    print("  最长一条(%d tok)尾部 160 字符: %r" % (longtext["tok"], longtext["text"][-160:]))
