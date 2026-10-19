#!/usr/bin/env python3
"""qfn 解码步时 / 吞吐探针(流式,预热后取数)。

用途:QFN 接入 + 热专家池的**同口径**判据。
  * `--warm N` 先发 N 条丢弃(默认 1 条),再取数 —— 冷启动首轮含 CUDA graph/惰性分配,
    绝不能当基线(见 HANDOFF_GLM_PREFILL_2026-10-17 §0)。
  * 输出:每请求 TTFT、解码 tok/s、**步时中位**(相邻 chunk 间隔)、deg、文本前 160 字。
  * `--dump-tag` 只用于人工比对,不写盘。

用法:
  PY=/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python
  $PY scripts/qfn_decode_probe.py --port 8140 --model Qwen3.8-Flash-Next \
      --conc 1 --n 3 --outlen 256 --warm 1
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request

PROMPT = (
    "用 Python 实现归并排序,要求:1) 递归版;2) 处理空数组和单元素;"
    "3) 给出一个测试用例并说明输出。只给代码和必要注释。"
)


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


def one(port, model, outlen):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": outlen,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"content-type": "application/json"})
    t0 = time.time()
    ts, txt = [], []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                d = json.loads(data)
            except Exception:
                continue
            ch = d.get("choices") or []
            if not ch:
                continue
            delta = (ch[0].get("delta") or {})
            piece = delta.get("content") or ""
            if piece:
                ts.append(time.time() - t0)
                txt.append(piece)
    if len(ts) < 2:
        return None
    gaps = [(ts[i + 1] - ts[i]) * 1000 for i in range(len(ts) - 1)]
    text = "".join(txt)
    return {
        "ttft_ms": ts[0] * 1000,
        "tokens": len(ts),
        "decode_s": ts[-1] - ts[0],
        "tok_s": (len(ts) - 1) / (ts[-1] - ts[0]) if ts[-1] > ts[0] else 0.0,
        "step_ms": statistics.median(gaps),
        "deg": deg(text),
        "text": text[:160].replace("\n", " "),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--n", type=int, default=3, help="取数请求数(每档)")
    ap.add_argument("--outlen", type=int, default=256)
    ap.add_argument("--warm", type=int, default=1, help="先发多少条丢弃")
    a = ap.parse_args()
    for _ in range(a.warm):
        one(a.port, a.model, min(32, a.outlen))
        print("[warm] 丢弃首轮", flush=True)
    res, lock = [], threading.Lock()

    def worker(k):
        for _ in range(k):
            r = one(a.port, a.model, a.outlen)
            if r:
                with lock:
                    res.append(r)

    per = max(1, a.n // a.conc)
    ths = [threading.Thread(target=worker, args=(per,)) for _ in range(a.conc)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    if not res:
        print("⛔ 无有效响应")
        return 1
    tput = sum(r["tok_s"] for r in res) / len(res)
    agg = sum(r["tokens"] - 1 for r in res) / wall
    print(f"== conc={a.conc} 请求={len(res)} 墙钟={wall:.2f}s ==")
    print(f"  TTFT 中位    = {statistics.median([r['ttft_ms'] for r in res]):8.1f} ms")
    print(f"  步时 中位    = {statistics.median([r['step_ms'] for r in res]):8.2f} ms")
    print(f"  每请求解码   = {tput:8.2f} tok/s(中位 {statistics.median([r['tok_s'] for r in res]):.2f})")
    print(f"  聚合解码     = {agg:8.2f} tok/s(×C 口径 = {agg * a.conc:.2f})")
    print(f"  deg 中位     = {statistics.median([r['deg'] for r in res]):8.3f}")
    print(f"  样例: {res[0]['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
