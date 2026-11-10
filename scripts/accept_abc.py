#!/usr/bin/env python3
"""验收 A/B 的 HTTP 执行器（只发请求，不碰起停）。

背景（`dev-docs/DEGEN_STAGE_COMPARISON.md`）：生产口径下 prefix cache 常开、decoder-side
SWA bounded replay 生效时，长会话会退化。验收要求：

* **A**：同一 prompt 连发 10 次，输出签名必须**唯一**（10 次完全一致）；
* **B**：交错 shared/unique 四臂，**unique 命中率 100%**（每条带唯一标记的请求都能拿到自己的针尖）。

本脚本实现 A 与 B；C（142K 针尖 >4h 后 ≥2/3）与最小复现（8×131K）分别用
`scripts/degen_repro.sh` 与 `scripts/probe_dump_long.py`。

用法：
    python scripts/accept_abc.py --port 8700 --a-rounds 10 --b-rounds 8 --len 4000
退出码：0 = A、B 全过；1 = 有失败；2 = 服务不可用。
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


def doc(n_words: int, seed: int) -> str:
    rnd = random.Random(seed)
    return " ".join(rnd.choice(WORDS) for _ in range(n_words))


def ask(port: int, prompt: str, max_tokens: int) -> str:
    body = json.dumps({
        "model": "DeepSeek-V4.1-Flash", "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "seed": 0, "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())["choices"][0]["text"]


def check_alive(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status == 200
    except urllib.error.URLError:
        return False


def test_a(port: int, rounds: int, n_words: int, max_tokens: int) -> bool:
    """A：同一 prompt ×N，输出签名必须唯一。"""
    prompt = f"{doc(n_words, seed=4242)} -- project Q access code?"
    outs = [ask(port, prompt, max_tokens) for _ in range(rounds)]
    uniq = len(set(outs))
    print(f"[A] 同 prompt ×{rounds} ⇒ 不同输出 {uniq} 种（要求 1）", flush=True)
    if uniq != 1:
        for i, o in enumerate(outs[:4], 1):
            print(f"    #{i}: {o[:48]!r}")
        print("[A] ❌ 输出不唯一")
        return False
    print("[A] ✅ 输出签名唯一")
    return True


def test_b(port: int, rounds: int, n_words: int, max_tokens: int) -> bool:
    """B：交错 shared/unique 四臂，unique 命中率必须 100%。

    每条请求 = 共享文档 + **唯一标记**；答案里必须出现自己的标记。
    四臂指交错发送的 4 种前缀长度，模拟生产里的混合命中形态。
    """
    shared = doc(n_words, seed=777)
    ok = 0
    total = 0
    for i in range(1, rounds + 1):
        for depth in (0.25, 0.5, 0.75, 0.98):
            cut = int(len(shared) * depth)
            mark = f"MARK-{i}-{int(depth * 100)}-{random.randint(1000, 9999)}"
            prompt = (f"{shared[:cut]}\n\n{mark}\n\n"
                      "Repeat the MARK string exactly, nothing else.")
            total += 1
            got = ask(port, prompt, max_tokens)
            hit = mark in got
            ok += int(hit)
            if not hit:
                print(f"    ✗ r{i} depth={depth}: 期望 {mark} 实得 {got[:40]!r}")
    rate = 100.0 * ok / max(total, 1)
    print(f"[B] unique 命中 {ok}/{total} = {rate:.0f}%（要求 100%）", flush=True)
    if ok != total:
        print("[B] ❌ 未达 100%")
        return False
    print("[B] ✅ unique 命中率 100%")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--a-rounds", type=int, default=10)
    ap.add_argument("--b-rounds", type=int, default=8)
    ap.add_argument("--len", type=int, default=4000)
    ap.add_argument("--max-tokens", type=int, default=32)
    a = ap.parse_args()

    if not check_alive(a.port):
        print(f"⛔ 服务未就绪（port={a.port}）；本脚本不负责起停")
        return 2
    ok_a = test_a(a.port, a.a_rounds, a.len, a.max_tokens)
    ok_b = test_b(a.port, a.b_rounds, a.len, a.max_tokens)
    print(f"\n=== 汇总 A={'✅' if ok_a else '❌'} B={'✅' if ok_b else '❌'}")
    return 0 if (ok_a and ok_b) else 1


if __name__ == "__main__":
    sys.exit(main())
