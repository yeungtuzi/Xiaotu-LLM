#!/usr/bin/env python3
"""验收 C 的长时探针：在**不重启服务**的前提下，按固定间隔打 142K 针尖，统计命中率。

验收 C 的定义（见 dev-docs/PLAN_SM80_REPLAY_2026-11-10.md）：
服务起来 **>4 小时后**，142K 上下文里的针尖命中率仍须 **≥ 2/3**。

本脚本只发请求、只写自己的日志，**不启停服务**；适合 `nohup` 长跑。
它把每次探测写成一行 JSON（便于事后按小时聚合），最后打印按小时的命中率表。

用法：
    nohup python scripts/long_uptime_needle.py --port 8700 --hours 6 \
        --interval-min 10 --rounds 3 --tokens 142000 \
        > dev-docs/report/tuning/logs/needle_c.jsonl 2>&1 &

    # 只看聚合结果（不探测）：
    python scripts/long_uptime_needle.py --aggregate dev-docs/report/tuning/logs/needle_c.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
         "mike november oscar papa quebec romeo sierra tango uniform victor whiskey").split()


def make_prompt(tokens: int, needle: int, rnd: random.Random) -> str:
    # ⚠️ 用**单词**做填充会让 token 数比 `tokens` 大 1.2~1.4 倍（BPE 切分），
    #    而验收 C 要的是"142K 上下文"⇒ 改成空白分隔的**单个字母/数字**，
    #    每项约 1 token，`--tokens` 才近似等于真实 token 数。
    body = " ".join(rnd.choice("abcdefghijklmnopqrstuvwxyz0123456789")
                    for _ in range(max(tokens - 64, 64)))
    row = f"project Q access code {needle} end"
    return (body + "\n\n" + row + "\n\n"
            + "What is the access code for project Q? Answer with the number only.")


def ask(port: int, prompt: str, max_tokens: int) -> tuple[str, float]:
    payload = {"model": "DeepSeek-V4.1-Flash", "prompt": prompt,
               "max_tokens": max_tokens, "temperature": 0.0, "seed": 0}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    return d["choices"][0]["text"], time.time() - t0


def probe_once(port: int, tokens: int, max_tokens: int) -> dict:
    needle = random.randint(100000, 999999)
    prompt = make_prompt(tokens, needle, random.Random(needle))
    try:
        text, dt = ask(port, prompt, max_tokens)
        hit = str(needle) in text
        return {"t": time.time(), "needle": needle, "hit": hit, "dt": round(dt, 1),
                "out": text[:40]}
    except urllib.error.URLError as e:
        return {"t": time.time(), "needle": needle, "hit": False, "dt": None,
                "error": str(e)[:80]}


def aggregate(path: str) -> int:
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        print("⛔ 没有可聚合的记录")
        return 1
    t0 = rows[0]["t"]
    buckets: dict[int, list[bool]] = defaultdict(list)
    for r in rows:
        buckets[int((r["t"] - t0) // 3600)].append(bool(r["hit"]))
    print(f"{'小时':>4}  {'命中/总数':>10}  {'命中率':>7}")
    for h in sorted(buckets):
        hits, total = sum(buckets[h]), len(buckets[h])
        print(f"{h:>4}  {hits:>4}/{total:<5}  {100.0 * hits / total:>6.0f}%")
    tail = rows[-min(9, len(rows)):]
    later = [r for r in rows if (r["t"] - t0) >= 4 * 3600]
    if later:
        rate = 100.0 * sum(bool(r["hit"]) for r in later) / len(later)
        print(f"\n>4h 段命中率 {rate:.0f}%（{sum(bool(r['hit']) for r in later)}/{len(later)}）"
              f"  判据 ≥67% ⇒ {'✅ 通过' if rate >= 66.67 else '❌ 未通过'}")
    else:
        print(f"\n⚠️ 还没有 >4h 的数据（已覆盖 {tail[-1]['t'] - t0:.0f}s）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--interval-min", type=float, default=10.0)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=142000)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--aggregate")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.aggregate:
        return aggregate(a.aggregate)
    if a.dry_run:
        n = int(a.hours * 60 / max(a.interval_min, 0.1)) + 1
        print(f"计划：{n} 轮 × {a.rounds} 次探测，每次 {a.tokens} tokens，"
              f"间隔 {a.interval_min} 分钟，共 {a.hours} 小时；端口 {a.port}")
        return 0

    deadline = time.time() + a.hours * 3600
    n = 0
    while time.time() < deadline:
        for _ in range(a.rounds):
            r = probe_once(a.port, a.tokens, a.max_tokens)
            n += 1
            print(json.dumps(r, ensure_ascii=False), flush=True)
        time.sleep(max(a.interval_min * 60, 1))
    print(f"# 结束：共 {n} 次探测", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
