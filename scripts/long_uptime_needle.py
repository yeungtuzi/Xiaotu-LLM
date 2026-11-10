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


# 【2026-11-10 实测·台账 #256】"单个字母/数字 + 空格" **不是** 1 token/项：
#   本模型 tokenizer 实测 ratio = 1.279（20000 项 ⇒ 25582 tokens）。
#   原代码按 1 项≈1 token 估 ⇒ `--tokens 142000` 实际生成 **181,394 tokens（+27.7%）**，
#   已经超出 142K±15% 的口径带。⇒ 改成**先向服务端标定 ratio**，再按 ratio 反算项数。
FILLER_RATIO_FALLBACK = 1.279     # 标定失败时的兜底（偏保守；聚合阶段还有 ±15% 自证兜底）


def make_prompt_items(items: int, needle: int, rnd: random.Random) -> str:
    body = " ".join(rnd.choice("abcdefghijklmnopqrstuvwxyz0123456789")
                    for _ in range(max(items, 64)))
    row = f"project Q access code {needle} end"
    return (body + "\n\n" + row + "\n\n"
            + "What is the access code for project Q? Answer with the number only.")


def make_prompt(tokens: int, needle: int, rnd: random.Random,
                ratio: float = FILLER_RATIO_FALLBACK) -> str:
    """按 `ratio`（每项填充项消耗的 token 数）反算填充项数，使真实 prompt ≈ tokens。"""
    if ratio <= 0:
        ratio = FILLER_RATIO_FALLBACK
    items = max(int(round(tokens / ratio)), 64)
    return make_prompt_items(items, needle, rnd)


CAL_ITEMS = 2000


def measure_ratio(port: int) -> float | None:
    """向服务端发一条**小**标定请求，用 usage.prompt_tokens / 项数 得到真实 ratio。

    为什么必须标定：ratio 依赖 tokenizer 与填充串形态（单词 1.57 / 单字母 1.279 / 逗号 2.69），
    写死常量在下一次换模型/换填充串时就会悄悄失准。
    """
    try:
        txt = make_prompt_items(CAL_ITEMS, 987654, random.Random(1))
        _, _, usage = ask(port, txt, 1)
        ptok = usage.get("prompt_tokens")
        if isinstance(ptok, int) and ptok > CAL_ITEMS * 0.5:
            return ptok / CAL_ITEMS
    except Exception:
        pass
    return None


def ask(port: int, prompt: str, max_tokens: int) -> tuple[str, float, dict]:
    payload = {"model": "DeepSeek-V4.1-Flash", "prompt": prompt,
               "max_tokens": max_tokens, "temperature": 0.0, "seed": 0}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    # ⚠️ 必须带出 usage.prompt_tokens：填充用的"单字母"并不严格 1 token/项，
    #    不看这个数就无法证明本轮真的是 142K 上下文（判据会变成"我以为的 142K"）。
    return d["choices"][0]["text"], time.time() - t0, (d.get("usage") or {})


def probe_once(port: int, tokens: int, max_tokens: int,
               ratio: float = FILLER_RATIO_FALLBACK) -> dict:
    needle = random.randint(100000, 999999)
    prompt = make_prompt(tokens, needle, random.Random(needle), ratio)
    try:
        text, dt, usage = ask(port, prompt, max_tokens)
        hit = str(needle) in text
        return {"t": time.time(), "needle": needle, "hit": hit, "dt": round(dt, 1),
                "ptok": usage.get("prompt_tokens"), "out": text[:40]}
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
    # 时间戳归一化：新记录写的是 epoch 浮点；手写/旧文件可能是 ISO 字符串。
    # 不归一化时 `r["t"] - t0` 会抛 TypeError，把 4 小时的数据全部作废（实测踩过）。
    def _ts(v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                pass
            try:
                from datetime import datetime
                return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return None
        return None

    norm, dropped = [], 0
    for r in rows:
        ts = _ts(r.get("t"))
        if ts is None:
            dropped += 1
            continue
        r = dict(r); r["t"] = ts
        norm.append(r)
    if dropped:
        print(f"⚠️ 跳过 {dropped} 条时间戳无法解析的记录")
    if not norm:
        print("⛔ 没有可聚合的记录（时间戳全部无法解析）")
        return 1
    rows = norm
    t0 = rows[0]["t"]
    buckets: dict[int, list[bool]] = defaultdict(list)
    for r in rows:
        buckets[int((r["t"] - t0) // 3600)].append(bool(r["hit"]))
    ptoks = sorted(r["ptok"] for r in rows if isinstance(r.get("ptok"), int) and r["ptok"] > 0)
    if ptoks:
        med = ptoks[len(ptoks) // 2]
        print(f"上下文实际 tokens（usage.prompt_tokens）：中位 {med:,}  最小 {ptoks[0]:,}  最大 {ptoks[-1]:,}")
        if med < 120_000 or med > 170_000:
            print(f"⚠️ 口径存疑：中位 {med:,} 不在 142K±15% 内 ⇒ 别把结论当作 142K 上下文")
        else:
            print("✅ 口径符合 142K±15%")
    else:
        print("⚠️ 记录里没有 ptok 字段（旧格式）⇒ 无法自证是 142K 上下文")

    print(f"{'小时':>4}  {'命中/总数':>10}  {'命中率':>7}")
    for h in sorted(buckets):
        hits, total = sum(buckets[h]), len(buckets[h])
        print(f"{h:>4}  {hits:>4}/{total:<5}  {100.0 * hits / total:>6.0f}%")
    tail = rows[-min(9, len(rows)):]
    later = [r for r in rows if (r["t"] - t0) >= 4 * 3600]
    # ⚠️ 2026-11-10 补（台账 #260）：**结论必须进退出码**。
    #    原实现无论命中率多少都 `return 0`，于是"C 未通过"在自动化里等于"通过"（同 #259 的形态）。
    #    返回码：0=通过 / 1=命中率不达标 / 2=口径存疑（不是 142K）/ 4=>4h 数据不足
    if not later:
        print(f"\n⚠️ 还没有 >4h 的数据（已覆盖 {tail[-1]['t'] - t0:.0f}s）⇒ 不足以判定")
        return 4
    hits = sum(bool(r["hit"]) for r in later)
    rate = 100.0 * hits / len(later)
    ok_rate = rate >= 66.67
    print(f"\n>4h 段命中率 {rate:.0f}%（{hits}/{len(later)}）"
          f"  判据 ≥67% ⇒ {'✅ 通过' if ok_rate else '❌ 未通过'}")

    caliber_bad = False
    if ptoks:
        med = ptoks[len(ptoks) // 2]
        if med < 120_000 or med > 170_000:
            caliber_bad = True
    else:
        caliber_bad = True          # 无 ptok ⇒ 无法自证是 142K 上下文
    if caliber_bad:
        print("⛔ 口径存疑 ⇒ 本轮**不算通过**（先把上下文做成 142K±15%，或让服务回 usage.prompt_tokens）")
        return 2
    if not ok_rate:
        return 1
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

    # 先用一条**小**请求标定"每项填充项 = 多少 token"，再按 ratio 反算 --tokens 对应的项数。
    # 不标定就会像 #256 那样：--tokens 142000 实际生成 181,394 tokens。
    ratio = measure_ratio(a.port)
    if ratio:
        print(f"# 标定：{CAL_ITEMS} 项 ⇒ ratio={ratio:.4f} tokens/项 ⇒ "
              f"{a.tokens} tokens 用 {int(round(a.tokens / ratio))} 项", flush=True)
    else:
        ratio = FILLER_RATIO_FALLBACK
        print(f"# ⚠️ 标定失败（拿不到 usage.prompt_tokens）⇒ 用兜底 ratio={ratio:.3f}", flush=True)

    deadline = time.time() + a.hours * 3600
    n = 0
    while time.time() < deadline:
        for _ in range(a.rounds):
            r = probe_once(a.port, a.tokens, a.max_tokens, ratio)
            n += 1
            print(json.dumps(r, ensure_ascii=False), flush=True)
        time.sleep(max(a.interval_min * 60, 1))
    print(f"# 结束：共 {n} 次探测", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
