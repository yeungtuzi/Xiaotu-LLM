#!/usr/bin/env python3
"""前缀缓存正确性 oracle（跨实例对比，dummy 权重也可用）。

思路：同一份随机权重（同 seed 的 dummy 加载是确定性的，已实测 A/B 两进程输出一致），
在**两个实例**上跑同一批 prompt：

* 臂 ON ：`enable_prefix_caching=True` —— 第 2..N 条会命中共享文档前缀，触发
  decoder-side SWA bounded replay；
* 臂 OFF：`NO_PREFIX_CACHE=1` —— 每条都从头算，是**参考答案**。

同一 prompt 在两臂上的输出必须逐字相同；不同 ⇒ 缓存/回放把 KV 搞坏了（就是生产 bug 的形态）。

用法：
    python scripts/probe_pc_oracle.py --port 8710 --phase on  --out /path/on.json
    （重启成 NO_PREFIX_CACHE=1 后）
    python scripts/probe_pc_oracle.py --port 8710 --phase off --out /path/off.json
    python scripts/probe_pc_oracle.py --compare /path/on.json /path/off.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
         "mike november oscar papa quebec romeo sierra tango uniform victor whiskey").split()


def doc(n_words: int, seed: int) -> str:
    rnd = random.Random(seed)
    return " ".join(rnd.choice(WORDS) for _ in range(n_words))


def post(port: int, prompt: str, max_tokens: int, logprobs: int = 0) -> str:
    """返回指纹：默认是生成文本；logprobs>0 时附带每步 top-k (token, logprob)。

    ⭐ 只看文本不够灵敏：argmax 相同但 KV 已漂移的情况用 logprob 才能抓到。
    """
    payload = {
        "model": "DeepSeek-V4.1-Flash", "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "seed": 0, "ignore_eos": True,
    }
    if logprobs > 0:
        payload["logprobs"] = logprobs
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        ch = json.loads(r.read())["choices"][0]
    text = ch["text"]
    if logprobs <= 0:
        return text
    picked = []
    for step in (ch.get("logprobs") or {}).get("content") or []:
        top = step.get("top_logprobs") or []
        picked.append([[t.get("token"), round(float(t.get("logprob", 0.0)), 6)] for t in top])
    return json.dumps({"text": text, "lp": picked}, ensure_ascii=False)


def run(port: int, out: str, n_words: int, rounds: int, max_tokens: int,
        logprobs: int = 0) -> int:
    shared = doc(n_words, seed=999)
    results: dict[str, str] = {}
    # 播种：先把共享文档的 KV 写进缓存（OFF 臂也照做，保证请求序列一致）
    post(port, shared, max_tokens, logprobs)
    for i in range(1, rounds + 1):
        suffix = doc(6, seed=5000 + i)
        prompt = f"{shared} {suffix}"
        results[f"r{i}"] = post(port, prompt, max_tokens, logprobs)
        print(f"r{i}: {results[f'r{i}'][:32]!r}", flush=True)
    json.dump(results, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"saved {out}")
    return 0


def compare(a: str, b: str) -> int:
    ra, rb = json.load(open(a)), json.load(open(b))
    bad = [k for k in ra if ra[k] != rb.get(k)]
    if bad:
        for k in bad[:6]:
            print(f"❌ {k}: ON={ra[k][:40]!r} OFF={rb[k][:40]!r}")
        print(f"\n❌ {len(bad)}/{len(ra)} 条不一致 ⇒ 缓存/回放破坏了 KV")
        return 1
    print(f"✅ {len(ra)}/{len(ra)} 条与无缓存参考逐字一致")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8710)
    ap.add_argument("--phase", choices=("on", "off"))
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--len", type=int, default=3000)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--logprobs", type=int, default=0)
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)
    if not (a.phase and a.out):
        print("需要 --phase 与 --out（或 --compare A B）")
        return 2
    return run(a.port, a.out, a.len, a.rounds, a.max_tokens, a.logprobs)


if __name__ == "__main__":
    sys.exit(main())
