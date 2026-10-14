#!/usr/bin/env python3
"""并发解码扫描:conc=N 并行解码请求,聚合 tok/s + 平均单请求 ITL。
用法: p2_sweep.py <port> <model> [conc列表,如 1 4 16 32]"""
import json, sys, threading, time, urllib.request

PORT = int(sys.argv[1]); MODEL = sys.argv[2]
CONCS = [int(x) for x in (sys.argv[3:] or ["1", "4", "16", "32"])]
PROMPT = "请用中文写一段关于分布式系统一致性的技术说明,尽量详细。"

def one(tok):
    body = json.dumps({"model": MODEL, "prompt": PROMPT, "max_tokens": tok,
                       "temperature": 0.7}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions",
                                 data=body, headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        n = json.load(r)["usage"]["completion_tokens"]
    return n, time.perf_counter() - t0

for c in CONCS:
    # 预热 KV 后计时;每线程 128 token
    ths, res = [], []
    def worker():
        res.append(one(128))
    for _ in range(c):
        t = threading.Thread(target=worker); ths.append(t)
    t0 = time.perf_counter()
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.perf_counter() - t0
    tot = sum(n for n, _ in res)
    itl = sum(dt / n for n, dt in res) / len(res) * 1000
    print(f"conc={c:3d}  agg={tot/wall:6.1f} tok/s  mean_ITL={itl:7.1f} ms")
    time.sleep(8)
