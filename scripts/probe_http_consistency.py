#!/usr/bin/env python3
"""HTTP 侧前缀缓存一致性探针（对已就绪的服务，不负责起停）。

判据（生产判据的微缩版）：**同一段 token 序列在同一服务生命周期内重复发送，输出必须逐 token 相同。**
结构照生产复现器：一条共享长文档 + 每轮唯一后缀（派生部分命中 ⇒ 触发 decoder-side SWA
bounded replay），每条 prompt 连发两次（第一次部分命中，第二次全命中），两次必须一致。

用法：
    python scripts/probe_http_consistency.py --port 8710 --len 12000 --rounds 8
退出码：0 = 一致；1 = 出现不一致；2 = 服务不可用。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.error
import urllib.request

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
         "mike november oscar papa quebec romeo sierra tango uniform victor whiskey").split()


def make_doc(n_tokens: int, seed: int) -> str:
    rnd = random.Random(seed)
    return " ".join(rnd.choice(WORDS) for _ in range(n_tokens))


def post(port: int, model: str, prompt: str, max_tokens: int) -> str:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return d["choices"][0]["text"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8710)
    ap.add_argument("--len", type=int, default=12000)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=1)
    a = ap.parse_args()

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=5) as r:
            model = json.loads(r.read())["data"][0]["id"]
    except urllib.error.URLError as e:
        print(f"⛔ 服务不可用：{e}")
        return 2
    print(f"model={model} port={a.port} len={a.len} rounds={a.rounds}", flush=True)

    doc = make_doc(a.len, seed=999)
    # 播种：先把 doc 的 KV 写进缓存
    post(a.port, model, doc, a.max_tokens)
    print("seeded", flush=True)

    if a.concurrency > 1:
        import concurrent.futures as cf

        def one(i: int) -> tuple[int, bool, str]:
            suf = make_doc(8, seed=20000 + i)
            text = f"{doc} {suf}"
            o1 = post(a.port, model, text, a.max_tokens)
            o2 = post(a.port, model, text, a.max_tokens)
            return i, o1 == o2, o1

        bad = []
        with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            for i, same, o1 in ex.map(one, range(1, a.rounds + 1)):
                print(f"conc round {i}: same={same} first={o1[:20]!r}", flush=True)
                if not same:
                    bad.append((i, o1, ""))
        if bad:
            for i, x, y in bad[:4]:
                print(f"❌ conc round {i}: hit-arm={x[:40]!r} 与复发不一致")
            return 1
        print(f"✅ 并发 {a.concurrency} 下全部一致")
        return 0

    bad = []
    for i in range(1, a.rounds + 1):
        suf = make_doc(8, seed=10000 + i)
        text = f"{doc} {suf}"
        o1 = post(a.port, model, text, a.max_tokens)
        o2 = post(a.port, model, text, a.max_tokens)
        same = o1 == o2
        print(f"round {i}: same={same} first={o1[:24]!r}", flush=True)
        if not same:
            bad.append((i, o1, o2))

    if bad:
        for i, x, y in bad[:4]:
            print(f"❌ round {i}: hit-arm={x[:40]!r} repeat={y[:40]!r}")
        return 1
    print("✅ 全部一致（部分命中臂 == 全命中臂）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
