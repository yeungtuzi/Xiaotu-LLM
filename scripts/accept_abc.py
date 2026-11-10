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


def ask(port: int, prompt: str, max_tokens: int) -> tuple[str, dict]:
    # ⛔ 2026-11-10 实测更正（台账 #274）：本模型是指令/对话形态，必须走 chat；
    #    /v1/completions 在真权重下立刻 EOS（text=''）⇒ A 臂会全部"空输出"。
    #    注意：**不再**传 ignore_eos（chat 下正常 stop 即可；ignore_eos 会让它继续胡写）。
    body = json.dumps({
        "model": "DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "seed": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    # 带上 usage：B 臂必须能证明"共享前缀真的命中了缓存"，
    # 否则"unique 命中率 100%"可以在**完全没用缓存**的情况下假通过（同 #222 那类漏洞）。
    ch = d["choices"][0]
    text = (ch.get("message") or {}).get("content") or ""
    return text, (d.get("usage") or {})


def check_alive(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status == 200
    except urllib.error.URLError:
        return False


def test_a(port: int, rounds: int, n_words: int, max_tokens: int,
           require_long_tokens: int = 0) -> bool:
    """A：同一 prompt ×N，输出签名必须唯一。"""
    prompt = f"{doc(n_words, seed=4242)} -- project Q access code?"
    pairs = [ask(port, prompt, max_tokens) for _ in range(rounds)]
    outs = [p[0] for p in pairs]
    ptoks = sorted(u.get("prompt_tokens") for _, u in pairs
                   if isinstance(u.get("prompt_tokens"), int))
    if ptoks:
        med = ptoks[len(ptoks) // 2]
        print(f"[A] 上下文 tokens 中位 {med:,}（最小 {ptoks[0]:,} 最大 {ptoks[-1]:,}）")
        # 退化只在长上下文（触发回放/pad）出现 ⇒ 在 5K 上跑"唯一性"不能代表 131K 的情形
        if require_long_tokens and med < require_long_tokens:
            print(f"[A] ❌ 口径不足：中位 {med:,} < 要求 {require_long_tokens:,} "
                  f"⇒ 这个通过不能代表长上下文（加 --len 或去掉 --require-long-tokens）")
            return False
    elif require_long_tokens:
        print("[A] ❌ 服务没回 usage.prompt_tokens，无法自证长上下文 ⇒ 不能算通过")
        return False
    uniq = len(set(outs))
    print(f"[A] 同 prompt ×{rounds} ⇒ 不同输出 {uniq} 种（要求 1）", flush=True)
    # ⛔ 空输出会让 uniq=1 变成"假通过"（全是 "" ⇒ 集合大小为 1）
    if any(not o.strip() for o in outs):
        print(f"[A] ❌ 有 {sum(1 for o in outs if not o.strip())} 条是**空输出** ⇒ uniq=1 无意义，不算通过")
        return False
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
    cached_hits = 0          # 共享前缀命中缓存的次数（行为证据）
    ptoks: list[int] = []
    for i in range(1, rounds + 1):
        for depth in (0.25, 0.5, 0.75, 0.98):
            cut = int(len(shared) * depth)
            mark = f"MARK-{i}-{int(depth * 100)}-{random.randint(1000, 9999)}"
            prompt = (f"{shared[:cut]}\n\n{mark}\n\n"
                      "Repeat the MARK string exactly, nothing else.")
            total += 1
            got, usage = ask(port, prompt, max_tokens)
            hit = mark in got
            ok += int(hit)
            if not hit:
                print(f"    ✗ r{i} depth={depth}: 期望 {mark} 实得 {got[:40]!r}")
            c = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
            cached_hits += int(c > 0)
            if isinstance(usage.get("prompt_tokens"), int):
                ptoks.append(usage["prompt_tokens"])
    rate = 100.0 * ok / max(total, 1)
    if ptoks:
        ptoks.sort()
        print(f"[B] 上下文 tokens 中位 {ptoks[len(ptoks)//2]:,}（最小 {ptoks[0]:,} 最大 {ptoks[-1]:,}）")
    print(f"[B] 共享前缀命中缓存 {cached_hits}/{total} 次（**行为证据**，0 次说明根本没走缓存）")
    print(f"[B] unique 命中 {ok}/{total} = {rate:.0f}%（要求 100%）", flush=True)
    # ⛔ 关键：若一次缓存都没命中，这个 100% 是"没走缓存也能 100%"，不能算 B 通过。
    if cached_hits == 0:
        print("[B] ❌ 全程没有发生缓存命中 ⇒ B 的结论无意义（先查 enable_prefix_caching / 前缀长度是否够一个 block）")
        return False
    if ok != total:
        print("[B] ❌ 未达 100%")
        return False
    print(f"[B] ✅ unique 命中率 100%（且共享前缀确实命中缓存 {cached_hits}/{total}）")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--a-rounds", type=int, default=10)
    ap.add_argument("--b-rounds", type=int, default=8)
    ap.add_argument("--len", type=int, default=4000)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--a-only", action="store_true", help="只跑 A（B 用 probe_degen_arms.py 的既有四臂口径）")
    ap.add_argument("--require-long-tokens", type=int, default=0,
                    help="A 臂要求上下文至少这么多 tokens（验收 A 建议 120000：退化只在长上下文出现）")
    ap.add_argument("--b-only", action="store_true", help="只跑本脚本的 B 变体（辅助，不替代既有口径）")
    a = ap.parse_args()
    if a.a_only and a.b_only:
        print("⛔ --a-only 与 --b-only 不能同时给")
        return 2

    if not check_alive(a.port):
        print(f"⛔ 服务未就绪（port={a.port}）；本脚本不负责起停")
        return 2
    # ⚠️ 验收 B 的**既有定义**是四臂（shared/unique/tail/suffix）：
    #    `PORT=8700 ROUNDS=3 python3 scripts/probe_degen_arms.py`。
    #    本脚本的 B 是"4 个前缀深度"的另一种构造，只作辅助，不替代既有口径（见台账 #155）。
    ok_a = True if a.b_only else test_a(a.port, a.a_rounds, a.len, a.max_tokens,
                                       a.require_long_tokens)
    ok_b = True if a.a_only else test_b(a.port, a.b_rounds, a.len, a.max_tokens)
    print(f"\n=== 汇总 A={'✅' if ok_a else '❌'} B={'✅' if ok_b else '❌'}")
    return 0 if (ok_a and ok_b) else 1


if __name__ == "__main__":
    sys.exit(main())
